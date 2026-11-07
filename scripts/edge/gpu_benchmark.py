#!/usr/bin/env python3
"""
Benchmark the ncnn encoders on the UNO Q: Adreno 702 GPU (Vulkan) against the CPU.

Runs on the board itself, not in a container, because it needs the board's
Vulkan driver (Mesa Turnip). Needs numpy and ncnn, installed by
"run_on_board.py gpu-setup". For every model it times single MBIC sentences
and whole articles, sentence by sentence, on:
  - the GPU through Vulkan (ncnn uses FP16 there when the GPU supports it)
  - the CPU with ncnn, 1 and 4 threads, for comparison with ONNX Runtime
It also checks that the GPU gives the same scores as the CPU, and records CPU
and GPU temperature, GPU frequency, and how much CPU time a GPU run still uses.

Usage, from the folder that holds ncnn/ and board_benchmark.py:
    python3 gpu_benchmark.py
    python3 gpu_benchmark.py --models bert-tiny --budget 10
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from board_benchmark import (  # noqa: E402  (same folder on the board)
    Sampler, Workload, cpu_temp, find_zones, memory_mb, read_number, summarize, wait_to_cool,
)

GPU_FREQ = "/sys/class/devfreq/5900000.gpu/cur_freq"


class GpuSampler(Sampler):
    """Adds GPU temperature and frequency to the CPU sampler."""

    def __init__(self, cpu_zones, gpu_zones, interval=0.5):
        super().__init__(cpu_zones, interval)
        self.gpu_zones = gpu_zones
        self.gpu_temps: List[float] = []
        self.gpu_freqs: List[float] = []

    def sample(self):
        super().sample()
        temp, freq = cpu_temp(self.gpu_zones), read_number(GPU_FREQ)
        if temp is not None:
            self.gpu_temps.append(temp)
        if freq is not None:
            self.gpu_freqs.append(freq / 1e6)

    def finish(self):
        report = super().finish()
        report.update({"gpu_temp_max_c": max(self.gpu_temps) if self.gpu_temps else None,
                       "gpu_freq_mean_mhz": float(np.mean(self.gpu_freqs)) if self.gpu_freqs else None,
                       "gpu_freq_max_mhz": max(self.gpu_freqs) if self.gpu_freqs else None})
        return report


def make_net(model_dir: str, device: str, threads: int):
    import ncnn

    net = ncnn.Net()
    net.opt.num_threads = threads
    net.opt.use_vulkan_compute = device == "gpu"
    if device == "gpu":
        net.set_vulkan_device(0)
    started = time.perf_counter()
    net.load_param(os.path.join(model_dir, "model.ncnn.param"))
    net.load_model(os.path.join(model_dir, "model.ncnn.bin"))
    return net, time.perf_counter() - started


def infer(net, ids: np.ndarray):
    import ncnn

    extractor = net.create_extractor()
    extractor.input("in0", ncnn.Mat(ids.astype(np.int32)))
    _, out = extractor.extract("out0")  # blocks until the result is back in CPU memory
    return out


def run_one(models_dir: str, name: str, device: str, threads: int, budget_s: float, warmup: int) -> dict:
    import ncnn

    model_dir = os.path.join(models_dir, name)
    work = Workload(model_dir)
    cpu_zones, gpu_zones = find_zones(), find_zones(names=("gpu-thermal",))
    temp_start = cpu_temp(cpu_zones)
    net, load_s = make_net(model_dir, device, threads)
    for ids in work.sentences[:warmup]:  # GPU: also compiles the shaders
        infer(net, ids)

    check = {}
    if device == "gpu":  # compare GPU scores with the CPU's on the first 20 sentences
        reference, _ = make_net(model_dir, "cpu", 4)
        diffs = [float(np.max(np.abs(np.array(infer(net, ids)) - np.array(infer(reference, ids)))))
                 for ids in work.sentences[:20]]
        check = {"gpu_vs_cpu_max_abs_diff": max(diffs), "gpu_output_nan": bool(np.isnan(diffs).any())}
        del reference

    sampler = GpuSampler(cpu_zones, gpu_zones)
    sampler.start()
    cpu_began, wall_began = time.process_time(), time.perf_counter()
    sentence_s: List[float] = []
    for ids in work.sentences:
        t = time.perf_counter()
        infer(net, ids)
        sentence_s.append(time.perf_counter() - t)
        if time.perf_counter() - wall_began > budget_s:
            break
    sequential_s: List[float] = []
    words: List[int] = []
    began = time.perf_counter()
    for article, n_words in zip(work.articles, work.article_words):
        t = time.perf_counter()
        for ids in article:
            infer(net, ids)
        sequential_s.append(time.perf_counter() - t)
        words.append(n_words)
        if time.perf_counter() - began > budget_s:
            break
    cpu_used = time.process_time() - cpu_began
    wall = time.perf_counter() - wall_began
    thermal = sampler.finish()

    return {
        "model": name, "device": device, "threads": threads, "params_m": work.manifest["params"] / 1e6,
        "model_mb": work.manifest.get("ncnn_mb"), "load_s": load_s,
        "peak_rss_mb": memory_mb()["peak_rss_mb"],
        "sentence": summarize(sentence_s),
        "article_sequential": summarize(sequential_s),
        "articles_words_total": int(sum(words)),
        "words_per_s_sequential": float(sum(words) / sum(sequential_s)) if sequential_s else None,
        "cpu_cores_busy": cpu_used / wall if wall else None,  # 1.0 = one core fully busy
        "temp_start_c": temp_start, **thermal, **check,
        "vulkan_gpus": int(ncnn.get_gpu_count()), "ncnn": getattr(ncnn, "__version__", None),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", default="ncnn")
    parser.add_argument("--results", default="results")
    parser.add_argument("--models", nargs="+", help="default: every model in ncnn/index.json")
    parser.add_argument("--devices", nargs="+", default=["gpu", "cpu1", "cpu4"],
                        help="gpu, or cpuN for the CPU with N threads")
    parser.add_argument("--budget", type=float, default=90.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cool-to", type=float, default=50.0)
    parser.add_argument("--max-cooldown", type=float, default=180.0)
    parser.add_argument("--config-timeout", type=float, default=900.0)
    parser.add_argument("--one", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.one:
        name, device = args.one.split(":")
        threads = 1 if device == "gpu" else int(device[3:])
        result = run_one(args.models_dir, name, "gpu" if device == "gpu" else "cpu", threads, args.budget, args.warmup)
        print("RESULT " + json.dumps(result), flush=True)
        return

    import ncnn

    os.makedirs(args.results, exist_ok=True)
    progress = open(os.path.join(args.results, "gpu_progress.log"), "a", encoding="utf-8")

    def log(message: str) -> None:
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + message
        print(line, flush=True)
        progress.write(line + "\n")
        progress.flush()

    gpus = ncnn.get_gpu_count()
    log(f"ncnn {getattr(ncnn, '__version__', '?')}, Vulkan GPUs found: {gpus}")
    devices = [d for d in args.devices if d != "gpu" or gpus > 0]
    if args.models:
        models = args.models
    else:
        with open(os.path.join(args.models_dir, "index.json"), encoding="utf-8") as f:
            models = json.load(f)["models"]
    zones = find_zones()
    out_path = os.path.join(args.results, "gpu_results.jsonl")
    configs = [(m, d) for m in models for d in devices]
    for i, (name, device) in enumerate(configs, 1):
        waited = wait_to_cool(zones, args.cool_to, args.max_cooldown)
        cmd = [sys.executable, os.path.abspath(__file__), "--one", f"{name}:{device}", "--models-dir", args.models_dir,
               "--budget", str(args.budget), "--warmup", str(args.warmup)]
        started = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.config_timeout)
            lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
            result = (json.loads(lines[-1][len("RESULT "):]) if lines else
                      {"model": name, "device": device, "error": (proc.stderr or proc.stdout)[-2000:]})
        except subprocess.TimeoutExpired:
            result = {"model": name, "device": device, "error": "timeout"}
        result["cooldown_s"], result["wall_s"] = waited, time.time() - started
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        if "error" in result:
            log(f"[{i}/{len(configs)}] {name} {device}: ERROR {result['error'][-300:]}")
        else:
            check = (f", GPU vs CPU {result['gpu_vs_cpu_max_abs_diff']:.1e}" if "gpu_vs_cpu_max_abs_diff" in result else "")
            log(f"[{i}/{len(configs)}] {name} {device}: sentence {result['sentence']['median_ms']:.1f} ms, "
                f"article {result['article_sequential']['median_ms'] / 1000:.2f} s, "
                f"CPU cores busy {result['cpu_cores_busy']:.2f}, GPU max {result['gpu_freq_max_mhz']} MHz{check}")
    log("all GPU runs done")


if __name__ == "__main__":
    main()
