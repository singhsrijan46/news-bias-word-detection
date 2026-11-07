#!/usr/bin/env python
"""
Run the encoder benchmark on an Arduino UNO Q over USB, using adb.

    python scripts/edge/run_on_board.py setup     # copy the script and models, install onnxruntime
    python scripts/edge/run_on_board.py start     # start the full benchmark in the background
    python scripts/edge/run_on_board.py status    # show progress
    python scripts/edge/run_on_board.py fetch     # copy results to outputs/edge/results/

Arguments after "start --" go to board_benchmark.py, for example:
    python scripts/edge/run_on_board.py start -- --models bert-tiny --threads 1

The benchmark runs inside App Lab's Python image, which is already on the board.
onnxruntime is installed into /home/arduino/edge-bench/pylib, so nothing in the
board's system changes. adb comes with Arduino App Lab; this script finds App
Lab's copy when adb is not on PATH.
"""

import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REMOTE = "/home/arduino/edge-bench"
IMAGE = "ghcr.io/arduino/app-bricks/python-apps-base:0.5.0"  # App Lab's Python image on the board
ONNXRUNTIME = "onnxruntime==1.28.0"  # same version used for export and quantization
CONTAINER = "edge-bench"


def find_adb(explicit: str = None) -> str:
    if explicit:
        return explicit
    on_path = shutil.which("adb")
    if on_path:
        return on_path
    for root in (os.path.join(os.environ.get("LOCALAPPDATA", ""), "Arduino15"),
                 os.path.expanduser("~/.arduino15"), os.path.expanduser("~/Library/Arduino15")):
        hits = [h for h in sorted(glob.glob(os.path.join(root, "packages", "arduino", "tools", "adb", "*", "adb*")))
                if os.path.basename(h) in ("adb", "adb.exe")]
        if hits:
            return hits[-1]
    sys.exit("adb not found: install Arduino App Lab, or pass --adb PATH")


class Board:
    def __init__(self, adb: str, serial: str = None):
        self.adb, self.serial = adb, serial

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        cmd = [self.adb] + (["-s", self.serial] if self.serial else []) + list(args)
        return subprocess.run(cmd, check=check, text=True)

    def shell(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        return self.run("shell", command, check=check)


def docker(command: str, detach: bool = False, network: str = "none") -> str:
    """A docker run line for App Lab's image, with the benchmark folder mounted at /work.

    The benchmark itself runs with no network. pip uses the board's own network
    ("host"): through Docker's default bridge, large downloads stalled on our Wi-Fi.
    """
    mode = f"-d --name {CONTAINER}" if detach else "--rm"
    return (f"docker run {mode} --network {network} -v {REMOTE}:/work -w /work -e PYTHONPATH=/work/pylib "
            f"-e PYTHONUNBUFFERED=1 --entrypoint python3 {IMAGE} {command}")


def setup(board: Board, models_dir: str, models=None) -> None:
    index_path = os.path.join(models_dir, "index.json")
    if not os.path.exists(index_path):
        sys.exit(f"{index_path} not found: run scripts/edge/export_encoders.py first")
    with open(index_path, encoding="utf-8") as f:
        names = models or json.load(f)["models"]
    board.shell(f"mkdir -p {REMOTE}/models {REMOTE}/results")
    board.run("push", os.path.join(HERE, "board_benchmark.py"), f"{REMOTE}/board_benchmark.py")
    board.run("push", index_path, f"{REMOTE}/models/index.json")
    for name in names:
        board.shell(f"rm -rf {REMOTE}/models/{name}")
        board.run("push", os.path.join(models_dir, name), f"{REMOTE}/models/{name}")
    version = ONNXRUNTIME.split("==")[1]
    installed = board.shell(f"test -d {REMOTE}/pylib/onnxruntime-{version}.dist-info", check=False).returncode == 0
    if not installed:
        board.shell(f"rm -rf {REMOTE}/pylib")
        board.shell(docker(f"-m pip install --quiet --no-cache-dir --progress-bar off --target /work/pylib "
                           f"{ONNXRUNTIME}", network="host"))
    board.shell(docker("-c \"import onnxruntime as o; print('onnxruntime', o.__version__, 'ready')\""))


def start(board: Board, extra) -> None:
    board.shell(f"docker rm -f {CONTAINER} >/dev/null 2>&1; true")
    board.shell(docker("board_benchmark.py " + " ".join(shlex.quote(a) for a in extra), detach=True))
    print("Started. Check progress with: python scripts/edge/run_on_board.py status")


def gpu_setup(board: Board, ncnn_dir: str, models=None) -> None:
    """The GPU benchmark runs on the board's own Python, which can reach the Vulkan driver.

    The board's Python has no pip, so App Lab's image installs ncnn into a folder
    (/home/arduino/edge-bench/gpulib) that the board's Python then imports from.
    Both are Python 3.13, so the same wheels work.
    """
    index_path = os.path.join(ncnn_dir, "index.json")
    if not os.path.exists(index_path):
        sys.exit(f"{index_path} not found: run scripts/edge/export_ncnn.py first")
    with open(index_path, encoding="utf-8") as f:
        names = models or json.load(f)["models"]
    board.shell(f"mkdir -p {REMOTE}/ncnn {REMOTE}/results")
    for script in ("board_benchmark.py", "gpu_benchmark.py"):
        board.run("push", os.path.join(HERE, script), f"{REMOTE}/{script}")
    board.run("push", index_path, f"{REMOTE}/ncnn/index.json")
    for name in names:
        board.shell(f"rm -rf {REMOTE}/ncnn/{name}")
        board.run("push", os.path.join(ncnn_dir, name), f"{REMOTE}/ncnn/{name}")
    if board.shell(f"test -d {REMOTE}/gpulib/ncnn", check=False).returncode != 0:
        board.shell(f"docker run --rm --network host -v {REMOTE}:/work --entrypoint python3 {IMAGE} "
                    f"-m pip install --quiet --no-cache-dir --progress-bar off --target /work/gpulib ncnn numpy")
    board.shell(f"cd {REMOTE} && PYTHONPATH={REMOTE}/gpulib python3 -c "
                f"\"import ncnn; print('ncnn', ncnn.__version__, '- Vulkan GPUs:', ncnn.get_gpu_count())\"")


def gpu_start(board: Board, extra) -> None:
    args = " ".join(shlex.quote(a) for a in extra)
    board.shell(f"cd {REMOTE} && setsid sh -c 'PYTHONPATH={REMOTE}/gpulib python3 gpu_benchmark.py {args} "
                f">> results/gpu_stdout.log 2>&1' < /dev/null > /dev/null 2>&1 &")
    print("Started. Check progress with: python scripts/edge/run_on_board.py status")


def status(board: Board) -> None:
    board.shell(f"docker ps -a --filter name=^{CONTAINER}$ --format '{{{{.Names}}}}: {{{{.Status}}}}'; "
                f"tail -n 15 {REMOTE}/results/progress.log 2>/dev/null; "
                f"pgrep -f '[g]pu_benchmark.py' >/dev/null && echo 'GPU benchmark: running'; "
                f"tail -n 8 {REMOTE}/results/gpu_progress.log 2>/dev/null; true")


def fetch(board: Board, out_parent: str) -> None:
    os.makedirs(out_parent, exist_ok=True)
    board.run("pull", f"{REMOTE}/results", out_parent)
    print(f"Results in {os.path.normpath(os.path.join(out_parent, 'results'))}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["setup", "start", "gpu-setup", "gpu-start", "status", "fetch"])
    parser.add_argument("extra", nargs="*", help="after --: arguments for the benchmark script (start, gpu-start)")
    parser.add_argument("--adb", help="path to adb (default: PATH, then Arduino App Lab's copy)")
    parser.add_argument("--serial", help="adb serial number, if more than one device is connected")
    parser.add_argument("--models-dir", default=os.path.join(ROOT, "outputs", "edge", "models"))
    parser.add_argument("--ncnn-dir", default=os.path.join(ROOT, "outputs", "edge", "ncnn"))
    parser.add_argument("--models", nargs="+", help="setup, gpu-setup: copy just these models")
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "edge"), help="fetch: results go in OUT/results")
    args = parser.parse_args(argv)

    board = Board(find_adb(args.adb), args.serial)
    if args.action == "setup":
        setup(board, args.models_dir, args.models)
    elif args.action == "start":
        start(board, args.extra)
    elif args.action == "gpu-setup":
        gpu_setup(board, args.ncnn_dir, args.models)
    elif args.action == "gpu-start":
        gpu_start(board, args.extra)
    elif args.action == "status":
        status(board)
    else:
        fetch(board, args.out)


if __name__ == "__main__":
    main()
