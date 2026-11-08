#!/usr/bin/env python
"""
Put the Bias Lens app on an Arduino UNO Q over USB (adb), and run it.

    python scripts/app/deploy_app.py install             # build the app folder and copy it to the board
    python scripts/app/deploy_app.py start               # start (or restart) it with App Lab's CLI
    python scripts/app/deploy_app.py logs                # the Python side's log
    python scripts/app/deploy_app.py stop
    python scripts/app/deploy_app.py forward             # open http://localhost:7000 on this PC

On the home network the app is at http://<board name>.local:7000.

install puts together, from this repository:
    app/bias-lens/                      the App Lab app: web page, Python, LED matrix sketch
    src/models/onnx_token_model.py      the inference code, copied into the app's python/
    outputs/student/final/              the trained student (scripts/train_student.py --final)
    data/raw/*lexicon*, seed words      the paper's bias lexicon, as python/model/lexicon.txt
App Lab installs an app's Python packages through Docker's default network, where large
downloads stalled on our Wi-Fi, so onnxruntime and tokenizers are installed once over the
board's own network into /home/arduino/.cache/bias-lens/vendor and copied into the app.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = "bias-lens"
REMOTE_APPS = "/home/arduino/ArduinoApps"
REMOTE_CACHE = "/home/arduino/.cache/bias-lens"
IMAGE = "ghcr.io/arduino/app-bricks/python-apps-base:0.5.0"
PACKAGES = ["onnxruntime==1.28.0", "tokenizers==0.23.3", "rbcl==1.1.2", "barriers==1.2.0"]
VENDOR_CHECK = ["onnxruntime", "tokenizers", "rbcl", "barriers"]  # numpy and cryptography come with App Lab's image
MODEL_FILES = ["model_int8.onnx", "tokenizer.json", "model_info.json"]
LEXICONS = ["data/raw/bised_words_lexicon.xlsx", "data/raw/seed_biased_words.xlsx"]
PORT = 7000
# adb sessions on the board set TMPDIR to Android's /data/local/tmp, which does not exist there.
CLI = "TMPDIR=/tmp arduino-app-cli"


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

    def run(self, *args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        cmd = [self.adb] + (["-s", self.serial] if self.serial else []) + list(args)
        # App Lab's output has emoji icons; Windows' default code page cannot decode them.
        return subprocess.run(cmd, check=check, text=True, capture_output=capture, encoding="utf-8", errors="replace")

    def shell(self, command: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        return self.run("shell", command, check=check, capture=capture)


def write_lexicon(path: str) -> int:
    import pandas as pd

    words = set()
    for name in LEXICONS:
        frame = pd.read_excel(os.path.join(ROOT, name), header=None)
        words.update(str(w).strip().lower() for w in frame.iloc[:, 0].dropna() if str(w).strip())
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(words)) + "\n")
    return len(words)


def build(model_dir: str) -> str:
    """Assemble the app folder in outputs/app/bias-lens, ready to copy."""
    missing = [f for f in MODEL_FILES if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        sys.exit(f"{model_dir} lacks {missing}: run scripts/train_student.py --final first")
    stage = os.path.join(ROOT, "outputs", "app", APP)
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, "app", APP), stage)
    shutil.copy(os.path.join(ROOT, "src", "models", "onnx_token_model.py"), os.path.join(stage, "python"))
    for name in ("biaslens-client.js", "noble-bundle.js", "overlay-core.js"):  # pairing, encryption, phone bookmark
        shutil.copy(os.path.join(ROOT, "clients", "shared", name), os.path.join(stage, "assets"))
    model_out = os.path.join(stage, "python", "model")
    os.makedirs(model_out)
    for name in MODEL_FILES:
        shutil.copy(os.path.join(model_dir, name), model_out)
    n_words = write_lexicon(os.path.join(model_out, "lexicon.txt"))
    with open(os.path.join(model_dir, "model_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    print(f"Built {stage}: {info.get('model_name')}, flag threshold {info.get('flag_threshold')}, "
          f"{n_words} lexicon words")
    return stage


def install(board: Board, model_dir: str) -> None:
    stage = build(model_dir)
    target = f"{REMOTE_APPS}/{APP}"
    # Paired devices and settings live in data/, which survives reinstalling.
    board.shell(f"mkdir -p {REMOTE_APPS} {REMOTE_CACHE} && rm -rf {REMOTE_CACHE}/data && "
                f"if [ -d {target}/data ]; then mv {target}/data {REMOTE_CACHE}/data; fi && rm -rf {target}")
    board.run("push", stage, target)
    check = " && ".join(f"test -d {REMOTE_CACHE}/vendor/{name}" for name in VENDOR_CHECK)
    if board.shell(check, check=False).returncode != 0:
        board.shell(f"rm -rf {REMOTE_CACHE}/vendor && docker run --rm --network host -v {REMOTE_CACHE}:/cache "
                    f"--entrypoint python3 {IMAGE} -m pip install --quiet --no-cache-dir --progress-bar off "
                    f"--only-binary=:all: --no-deps --target /cache/vendor {' '.join(PACKAGES)}")
    board.shell(f"cp -r {REMOTE_CACHE}/vendor {target}/python/vendor && "
                f"if [ -d {REMOTE_CACHE}/data ]; then mv {REMOTE_CACHE}/data {target}/data; fi")
    print(f"Installed to {target}")


def app_id(board: Board) -> str:
    """App Lab names apps in ~/ArduinoApps "user:<folder>" (its JSON listing gives the IDs in base64)."""
    return f"user:{APP}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["install", "start", "stop", "logs", "forward", "build"])
    parser.add_argument("--model-dir", default=os.path.join(ROOT, "outputs", "student", "final"))
    parser.add_argument("--adb", help="path to adb (default: PATH, then Arduino App Lab's copy)")
    parser.add_argument("--serial", help="adb serial number, if more than one device is connected")
    args = parser.parse_args(argv)

    if args.action == "build":
        build(args.model_dir)
        return
    board = Board(find_adb(args.adb), args.serial)
    if args.action == "install":
        install(board, args.model_dir)
    elif args.action == "start":
        # "app stop" sometimes reports success but leaves the container running, after which
        # "app start" refuses with "App is running"; removing the container clears that.
        app = app_id(board)
        board.shell(f"{CLI} app stop {app} >/dev/null 2>&1; docker rm -f {APP}-main-1 >/dev/null 2>&1; "
                    f"{CLI} app start {app}")
    elif args.action == "stop":
        board.shell(f"{CLI} app stop {app_id(board)}", check=False)
    elif args.action == "logs":
        board.shell(f"TMPDIR=/tmp timeout 8 arduino-app-cli app logs {app_id(board)}", check=False)
    else:
        board.run("forward", f"tcp:{PORT}", f"tcp:{PORT}")
        print(f"Open http://localhost:{PORT} on this PC")


if __name__ == "__main__":
    main()
