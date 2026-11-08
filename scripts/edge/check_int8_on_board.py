#!/usr/bin/env python
"""
Check that the board's INT8 model gives the scores that were measured on the PC.

Every accuracy and audit number for the INT8 student was measured with ONNX Runtime on the PC.
The board runs the same model file with the same ONNX Runtime version, but on its Arm cores, whose
integer kernels differ from the PC's x86 ones. This script scores the same sentences on both and
compares every word's score and flag:
  - MBIC's 1,700 sentences;
  - BABE's 2,418 sentences that are not in MBIC (scripts/evaluate_babe.py), where the experts' word
    labels also give the INT8 student's F1 as the board computes it.

The board uses the model deployed with Bias Lens (scripts/app/deploy_app.py), after checking that it
is the same file as outputs/student/final, and the ONNX Runtime that the deployment installed. It
runs in App Lab's Python image with no network.

Usage:
    python scripts/edge/check_int8_on_board.py
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from run_on_board import IMAGE, Board, find_adb  # noqa: E402
from score_sentences import score_all  # noqa: E402
from src.data.dataset_loader import MBICDataLoader  # noqa: E402
from src.evaluation.word_metrics import detection_metrics  # noqa: E402
from src.models.onnx_token_model import OnnxTokenModel  # noqa: E402

APP_MODEL = "/home/arduino/ArduinoApps/bias-lens/python/model"  # where deploy_app.py puts the model
VENDOR = "/home/arduino/.cache/bias-lens/vendor"  # and the packages it installs
REMOTE = "/home/arduino/edge-bench/int8-check"
AGREED_SHARE = 0.6  # the app's "most readers" band starts here (app/bias-lens/python/analyzer.py)


def _evaluate_babe():
    spec = importlib.util.spec_from_file_location("evaluate_babe", os.path.join(ROOT, "scripts", "evaluate_babe.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def board_output(board: Board, command: str) -> str:
    cmd = [board.adb] + (["-s", board.serial] if board.serial else []) + ["shell", command]
    return subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8").stdout


def flatten(scores: List[List]) -> np.ndarray:
    return np.array([np.nan if s is None else s for sentence in scores for s in sentence], dtype=np.float64)


def band(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(scores < threshold, 0, np.where(scores < AGREED_SHARE, 1, 2))


def compare(a: np.ndarray, b: np.ndarray, threshold: float) -> Dict:
    """How far scores b are from scores a, and how many flags and display bands change."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    diff = np.abs(a - b)
    return {"words": int(ok.sum()), "identical_share": float(np.mean(diff == 0)),
            "mean_abs_diff": float(diff.mean()), "p999_abs_diff": float(np.quantile(diff, 0.999)),
            "max_abs_diff": float(diff.max()), "spearman": float(spearmanr(a, b).correlation),
            "flags_a": int((a >= threshold).sum()), "flags_b": int((b >= threshold).sum()),
            "flag_changes": int(((a >= threshold) != (b >= threshold)).sum()),
            "band_changes": int((band(a, threshold) != band(b, threshold)).sum())}


def babe_metrics(scores: Dict[str, List[List]], new, offset: int, threshold: float) -> Dict:
    """F1 and ROC AUC against BABE's expert word labels, for each set of scores."""
    babe = _evaluate_babe()
    labels, picked = [], {name: [] for name in scores}
    for i, row in enumerate(new.itertuples(index=False)):
        if row.words is None:
            continue
        labels.append(babe.expert_labels(str(row.text), row.words))
        for name, s in scores.items():
            picked[name].append(np.array([np.nan if x is None else x for x in s[offset + i]]))
    y = np.concatenate(labels)
    out = {}
    for name, parts in picked.items():
        s = np.concatenate(parts)
        ok = ~np.isnan(s)
        d = detection_metrics(y[ok], s[ok], (s[ok] >= threshold).astype(int))
        out[name] = {k: d[k] for k in ("f1", "precision", "recall", "roc_auc")}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adb", help="path to adb (default: PATH, then Arduino App Lab's copy)")
    parser.add_argument("--serial", help="adb serial number, if more than one device is connected")
    parser.add_argument("--student", default=os.path.join(ROOT, "outputs", "student", "final"))
    parser.add_argument("--threads", type=int, default=4, help="threads, on the board and on the PC")
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "edge", "int8_check"))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    board = Board(find_adb(args.adb), args.serial)
    local = sha256(os.path.join(args.student, "model_int8.onnx"))
    deployed = board_output(board, f"sha256sum {APP_MODEL}/model_int8.onnx").split()[0]
    if deployed != local:
        sys.exit(f"The board's model ({deployed[:12]}) is not {args.student} ({local[:12]}): deploy it first.")

    labeled = MBICDataLoader().load_raw_sentences()
    babe = _evaluate_babe()
    _, new, _ = babe.split_by_mbic(babe.load_babe(), labeled)
    sentences = labeled.sentence.astype(str).tolist() + new.text.astype(str).tolist()
    print(f"{len(sentences)} sentences: {len(labeled)} from MBIC, {len(new)} new ones from BABE")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sentences.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sentences, f)
        board.shell(f"rm -rf {REMOTE} && mkdir -p {REMOTE}")
        for local_file in (path, os.path.join(HERE, "score_sentences.py"),
                           os.path.join(ROOT, "src", "models", "onnx_token_model.py")):
            board.run("push", local_file, f"{REMOTE}/{os.path.basename(local_file)}")
    print("Scoring on the board...")
    # As the board's own user, so the results can be removed without root on the next run.
    board.shell(f"docker run --rm --network none --user \"$(id -u):$(id -g)\" -v {REMOTE}:/work "
                f"-v {APP_MODEL}:/model:ro -v {VENDOR}:/vendor:ro -w /work -e PYTHONPATH=/vendor "
                f"-e PYTHONUNBUFFERED=1 --entrypoint python3 {IMAGE} score_sentences.py --model /model "
                f"--sentences sentences.json --out board_scores.json --threads {args.threads}")
    board.run("pull", f"{REMOTE}/board_scores.json", os.path.join(args.out, "board_scores.json"))
    with open(os.path.join(args.out, "board_scores.json"), encoding="utf-8") as f:
        on_board = json.load(f)

    print("Scoring on the PC...")
    pc_int8 = score_all(OnnxTokenModel(args.student, "int8", args.threads), sentences)
    pc_fp32 = score_all(OnnxTokenModel(args.student, "fp32", args.threads), sentences)
    with open(os.path.join(args.student, "model_info.json"), encoding="utf-8") as f:
        threshold = json.load(f)["flag_threshold"]
    lengths_match = all(len(a) == len(b) for a, b in zip(on_board["scores"], pc_int8))
    board_flat, int8_flat, fp32_flat = flatten(on_board["scores"]), flatten(pc_int8), flatten(pc_fp32)
    summary = {
        "model_sha256": local, "threshold": threshold, "sentences": len(sentences), "word_counts_match": lengths_match,
        "board": {k: on_board[k] for k in ("onnxruntime", "machine", "threads", "seconds")},
        "board_ms_per_sentence": 1000 * on_board["seconds"] / len(sentences),
        "board_int8_vs_pc_int8": compare(int8_flat, board_flat, threshold),
        "board_int8_vs_pc_fp32": compare(fp32_flat, board_flat, threshold),
        "babe_new_vs_experts": babe_metrics({"board_int8": on_board["scores"], "pc_int8": pc_int8, "pc_fp32": pc_fp32},
                                            new, len(labeled), threshold),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
