#!/usr/bin/env python3
"""
Benchmark exported encoders on the Arduino UNO Q's CPU.

Needs only numpy and onnxruntime, so it runs inside App Lab's Python image
(scripts/edge/run_on_board.py starts it there). For every model, precision and
thread count, a fresh process loads the model and times:
  - single MBIC sentences, one at a time (batch size 1)
  - whole MBIC articles, sentence by sentence and as one padded batch
It records load time, peak memory, CPU temperature and CPU frequency, and waits
for the CPU to cool between runs so that every run starts from a similar state.

Usage, from the folder that holds models/:
    python3 board_benchmark.py                                   # the full matrix
    python3 board_benchmark.py --models bert-tiny --threads 1    # a quick check
    python3 board_benchmark.py --sustained minilm-l6:int8:4      # 5 minutes non-stop
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

THERMAL_ROOT = "/sys/class/thermal"
CPUFREQ = "/sys/devices/system/cpu/cpufreq/policy0"
CPU_ZONES = ("cpuss0-thermal", "cpuss1-thermal")


# ---------------------------------------------------------------- helpers

def unpack(flat: np.ndarray, offsets: np.ndarray) -> List[np.ndarray]:
    """Inverse of export_encoders.pack: split a flat array at the given offsets."""
    return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def summarize(seconds: Sequence[float]) -> Optional[Dict[str, float]]:
    """Median, 95th percentile, mean, min and max, in milliseconds."""
    ms = np.asarray(seconds, dtype=float) * 1000.0
    if ms.size == 0:
        return None
    return {"n": int(ms.size), "median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
            "mean_ms": float(ms.mean()), "min_ms": float(ms.min()), "max_ms": float(ms.max())}


def find_zones(root: str = THERMAL_ROOT, names: Sequence[str] = CPU_ZONES) -> Dict[str, str]:
    """Paths of the temperature files for the named thermal zones."""
    zones: Dict[str, str] = {}
    if not os.path.isdir(root):
        return zones
    for entry in sorted(os.listdir(root)):
        if not entry.startswith("thermal_zone"):
            continue
        try:
            with open(os.path.join(root, entry, "type")) as f:
                kind = f.read().strip()
        except OSError:
            continue
        if kind in names:
            zones[kind] = os.path.join(root, entry, "temp")
    return zones


def read_number(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def cpu_temp(zones: Dict[str, str]) -> Optional[float]:
    """Hottest CPU sensor, in degrees C (sysfs reports millidegrees)."""
    readings = [read_number(p) for p in zones.values()]
    readings = [r / 1000.0 for r in readings if r is not None]
    return max(readings) if readings else None


def cpu_freq_mhz() -> Optional[float]:
    khz = read_number(os.path.join(CPUFREQ, "scaling_cur_freq"))
    return khz / 1000.0 if khz is not None else None


def memory_mb() -> Dict[str, Optional[float]]:
    """Current (VmRSS) and peak (VmHWM) resident memory of this process."""
    found: Dict[str, float] = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, value = line.split(":", 1)
                    found[key] = int(value.split()[0]) / 1024.0
    except OSError:
        pass
    return {"rss_mb": found.get("VmRSS"), "peak_rss_mb": found.get("VmHWM")}


class Sampler(threading.Thread):
    """Samples CPU temperature and frequency in the background while a test runs."""

    def __init__(self, zones: Dict[str, str], interval: float = 0.5):
        super().__init__(daemon=True)
        self.zones, self.interval = zones, interval
        self.temps: List[float] = []
        self.freqs: List[float] = []
        self._halt = threading.Event()  # not "_stop": threading.Thread uses that name itself

    def sample(self) -> None:
        temp, freq = cpu_temp(self.zones), cpu_freq_mhz()
        if temp is not None:
            self.temps.append(temp)
        if freq is not None:
            self.freqs.append(freq)

    def run(self) -> None:
        while not self._halt.is_set():
            self.sample()
            self._halt.wait(self.interval)

    def finish(self) -> Dict[str, Optional[float]]:
        self._halt.set()
        self.join()
        self.sample()
        return {"temp_max_c": max(self.temps) if self.temps else None,
                "temp_end_c": self.temps[-1] if self.temps else None,
                "freq_mean_mhz": float(np.mean(self.freqs)) if self.freqs else None,
                "freq_min_mhz": float(min(self.freqs)) if self.freqs else None}


# ---------------------------------------------------------------- workload and model

class Workload:
    """The tokenized MBIC sentences and articles saved next to each model."""

    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)
        data = np.load(os.path.join(model_dir, "inputs.npz"))
        self.sentences = unpack(data["sentence_ids"], data["sentence_offsets"])
        article_sentences = unpack(data["article_sentence_ids"], data["article_sentence_offsets"])
        bounds = data["article_offsets"]
        self.articles = [article_sentences[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
        self.article_words = [int(w) for w in data["article_words"]]


class Runner:
    """One ONNX Runtime session on the CPU with a fixed number of threads."""

    def __init__(self, model_path: str, threads: int, pad_id: int):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        started = time.perf_counter()
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        self.load_s = time.perf_counter() - started
        self.pad_id = pad_id
        self.version = ort.__version__

    def sentence(self, ids: np.ndarray) -> None:
        ids = ids.reshape(1, -1)
        self.session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})

    def batch(self, sequences: Sequence[np.ndarray]) -> None:
        width = max(len(s) for s in sequences)
        ids = np.full((len(sequences), width), self.pad_id, dtype=np.int64)
        mask = np.zeros_like(ids)
        for row, s in enumerate(sequences):
            ids[row, :len(s)] = s
            mask[row, :len(s)] = 1
        self.session.run(None, {"input_ids": ids, "attention_mask": mask})


def run_one(models_dir: str, name: str, precision: str, threads: int, budget_s: float, warmup: int) -> dict:
    """Time one model at one precision and thread count. Each phase stops after budget_s."""
    model_dir = os.path.join(models_dir, name)
    work = Workload(model_dir)
    zones = find_zones()
    before = memory_mb()
    temp_start = cpu_temp(zones)
    path = os.path.join(model_dir, f"{precision}.onnx")
    runner = Runner(path, threads, work.manifest["pad_token_id"])
    loaded = memory_mb()
    for ids in work.sentences[:warmup]:
        runner.sentence(ids)

    sampler = Sampler(zones)
    sampler.start()
    sentence_s: List[float] = []
    began = time.perf_counter()
    for ids in work.sentences:
        t = time.perf_counter()
        runner.sentence(ids)
        sentence_s.append(time.perf_counter() - t)
        if time.perf_counter() - began > budget_s:
            break

    sequential_s: List[float] = []
    batched_s: List[float] = []
    words: List[int] = []
    began = time.perf_counter()
    for article, n_words in zip(work.articles, work.article_words):
        t = time.perf_counter()
        for ids in article:
            runner.sentence(ids)
        sequential_s.append(time.perf_counter() - t)
        t = time.perf_counter()
        runner.batch(article)
        batched_s.append(time.perf_counter() - t)
        words.append(n_words)
        if time.perf_counter() - began > budget_s:
            break
    thermal = sampler.finish()
    after = memory_mb()

    tokens = [len(s) for s in work.sentences[:len(sentence_s)]]
    return {
        "model": name, "precision": precision, "threads": threads,
        "params_m": work.manifest["params"] / 1e6,
        "file_mb": os.path.getsize(path) / 2**20,
        "load_s": runner.load_s,
        "rss_before_mb": before["rss_mb"], "rss_loaded_mb": loaded["rss_mb"], "peak_rss_mb": after["peak_rss_mb"],
        "sentence": summarize(sentence_s),
        "sentence_tokens_median": float(np.median(tokens)) if tokens else None,
        "article_sequential": summarize(sequential_s),
        "article_batched": summarize(batched_s),
        "articles_words_total": int(sum(words)),
        "words_per_s_sequential": float(sum(words) / sum(sequential_s)) if sequential_s else None,
        "words_per_s_batched": float(sum(words) / sum(batched_s)) if batched_s else None,
        "temp_start_c": temp_start, **thermal,
        "onnxruntime": runner.version,
    }


def run_sustained(models_dir: str, name: str, precision: str, threads: int, duration_s: float, out_path: str) -> dict:
    """Process articles back to back for duration_s, logging speed, temperature and frequency."""
    model_dir = os.path.join(models_dir, name)
    work = Workload(model_dir)
    zones = find_zones()
    runner = Runner(os.path.join(model_dir, f"{precision}.onnx"), threads, work.manifest["pad_token_id"])
    for ids in work.sentences[:10]:
        runner.sentence(ids)
    rows = []
    began = time.perf_counter()
    with open(out_path, "a", encoding="utf-8") as out:
        k = 0
        while time.perf_counter() - began < duration_s:
            article = work.articles[k % len(work.articles)]
            n_words = work.article_words[k % len(work.articles)]
            t = time.perf_counter()
            for ids in article:
                runner.sentence(ids)
            elapsed = time.perf_counter() - t
            row = {"model": name, "precision": precision, "threads": threads,
                   "t_s": time.perf_counter() - began, "article_s": elapsed, "words": n_words,
                   "words_per_s": n_words / elapsed, "temp_c": cpu_temp(zones), "freq_mhz": cpu_freq_mhz()}
            out.write(json.dumps(row) + "\n")
            out.flush()
            rows.append(row)
            k += 1
    first = [r["words_per_s"] for r in rows if r["t_s"] <= 60]
    last = [r["words_per_s"] for r in rows if r["t_s"] >= duration_s - 60]
    temps = [r["temp_c"] for r in rows if r["temp_c"] is not None]
    freqs = [r["freq_mhz"] for r in rows if r["freq_mhz"] is not None]
    return {"model": name, "precision": precision, "threads": threads, "duration_s": duration_s,
            "articles": len(rows),
            "words_per_s_first_minute": float(np.median(first)) if first else None,
            "words_per_s_last_minute": float(np.median(last)) if last else None,
            "temp_max_c": max(temps) if temps else None,
            "freq_min_mhz": min(freqs) if freqs else None}


# ---------------------------------------------------------------- driver

def environment() -> dict:
    info = {"python": platform.python_version(), "machine": platform.machine(),
            "kernel": platform.release(), "cpu_count": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("features"):
                    info["cpu_features"] = line.split(":", 1)[1].split()
                    break
    except OSError:
        pass
    for key, leaf in (("governor", "scaling_governor"), ("max_freq_khz", "cpuinfo_max_freq")):
        try:
            with open(os.path.join(CPUFREQ, leaf)) as f:
                info[key] = f.read().strip()
        except OSError:
            pass
    try:
        with open("/proc/meminfo") as f:
            info["mem_total_mb"] = int(f.readline().split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
    except ImportError:
        pass
    return info


def wait_to_cool(zones: Dict[str, str], cool_to: float, max_wait: float) -> float:
    """Sleep until the CPU is at or below cool_to degrees C, for at most max_wait seconds."""
    waited = 0.0
    while waited < max_wait:
        temp = cpu_temp(zones)
        if temp is None or temp <= cool_to:
            break
        time.sleep(2.0)
        waited += 2.0
    return waited


def parse_config(text: str):
    name, precision, threads = text.split(":")
    return name, precision, int(threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results", default="results")
    parser.add_argument("--models", nargs="+", help="default: every model in models/index.json")
    parser.add_argument("--precisions", nargs="+", default=["fp32", "int8"], choices=["fp32", "int8"])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--budget", type=float, default=90.0, help="seconds per phase (sentences, articles)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cool-to", type=float, default=50.0, help="start each run at or below this CPU temperature")
    parser.add_argument("--max-cooldown", type=float, default=180.0)
    parser.add_argument("--config-timeout", type=float, default=900.0)
    parser.add_argument("--sustained", metavar="MODEL:PRECISION:THREADS")
    parser.add_argument("--duration", type=float, default=300.0, help="seconds, with --sustained")
    parser.add_argument("--one", metavar="MODEL:PRECISION:THREADS", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.one:  # child process: one configuration, result on stdout
        name, precision, threads = parse_config(args.one)
        print("RESULT " + json.dumps(run_one(args.models_dir, name, precision, threads, args.budget, args.warmup)),
              flush=True)
        return

    os.makedirs(args.results, exist_ok=True)
    progress = open(os.path.join(args.results, "progress.log"), "a", encoding="utf-8")

    def log(message: str) -> None:
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + message
        print(line, flush=True)
        progress.write(line + "\n")
        progress.flush()

    with open(os.path.join(args.results, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(environment(), f, indent=2)
    zones = find_zones()

    if args.sustained:
        name, precision, threads = parse_config(args.sustained)
        waited = wait_to_cool(zones, args.cool_to, args.max_cooldown)
        log(f"sustained {name} {precision} x{threads} for {args.duration:.0f} s (cooled {waited:.0f} s)")
        summary = run_sustained(args.models_dir, name, precision, threads, args.duration,
                                os.path.join(args.results, "sustained.jsonl"))
        with open(os.path.join(args.results, "sustained_summary.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        log("sustained done: " + json.dumps(summary))
        return

    if args.models:
        models = args.models
    else:
        with open(os.path.join(args.models_dir, "index.json"), encoding="utf-8") as f:
            models = json.load(f)["models"]
    configs = [(m, p, t) for m in models for p in args.precisions for t in args.threads]
    out_path = os.path.join(args.results, "results.jsonl")
    log(f"starting {len(configs)} runs")
    for i, (name, precision, threads) in enumerate(configs, 1):
        waited = wait_to_cool(zones, args.cool_to, args.max_cooldown)
        cmd = [sys.executable, os.path.abspath(__file__), "--one", f"{name}:{precision}:{threads}",
               "--models-dir", args.models_dir, "--budget", str(args.budget), "--warmup", str(args.warmup)]
        started = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.config_timeout)
            lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
            result = (json.loads(lines[-1][len("RESULT "):]) if lines else
                      {"model": name, "precision": precision, "threads": threads,
                       "error": (proc.stderr or proc.stdout)[-2000:]})
        except subprocess.TimeoutExpired:
            result = {"model": name, "precision": precision, "threads": threads, "error": "timeout"}
        result["cooldown_s"] = waited
        result["wall_s"] = time.time() - started
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        if "error" in result:
            log(f"[{i}/{len(configs)}] {name} {precision} x{threads}: ERROR {result['error'][-300:]}")
        else:
            log(f"[{i}/{len(configs)}] {name} {precision} x{threads}: sentence {result['sentence']['median_ms']:.1f} ms, "
                f"article {result['article_sequential']['median_ms'] / 1000:.2f} s one by one / "
                f"{result['article_batched']['median_ms'] / 1000:.2f} s batched, "
                f"peak {result['peak_rss_mb']:.0f} MB, max {result['temp_max_c']} C")
    log("all runs done")


if __name__ == "__main__":
    main()
