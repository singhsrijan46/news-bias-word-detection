#!/usr/bin/env python
"""
Run the on-board explainer on a fixed sample of flagged words, to see what it gets right and wrong.

The sample is 40 words that the board's model flags in BABE's sentences that are not in MBIC
(scripts/evaluate_babe.py), one word per sentence, drawn with a fixed seed. For each word it
records the model's reason, its suggested replacement, and whether the replacement passed the
checks. The explainer code is the app's own (app/bias-lens/python/explainer.py).

With --board, the language model runs on the board: llama.cpp's server, started in the app's
container image with the app's settings, and reached through an adb-forwarded port. The bias-word
model runs on the PC, which gives the board's scores (docs/EDGE_BENCHMARK.md). Without --board,
everything runs on the PC.

docs/EXPLANATIONS.md reports the results.

Usage:
    python scripts/app/fetch_llm.py
    python scripts/app/deploy_app.py install          # puts the model on the board, for --board
    python scripts/app/evaluate_explainer.py --board --suggestions
"""

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app", "bias-lens", "python"))
sys.path.insert(0, HERE)

from analyzer import band, load_lexicon  # noqa: E402
from deploy_app import write_lexicon  # noqa: E402
from explainer import Explainer, LlamaServer, load_model, read_info, server_args  # noqa: E402
from src.data.dataset_loader import MBICDataLoader  # noqa: E402
from src.models.onnx_token_model import OnnxTokenModel  # noqa: E402


def babe_module():
    spec = importlib.util.spec_from_file_location("evaluate_babe", os.path.join(ROOT, "scripts", "evaluate_babe.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PORT = 8090
CONTAINER = "bias-lens-llm-test"


class Timed:
    """The language model, timing each call: the reason, then the replacement word."""

    def __init__(self, lm):
        self.lm, self.info, self.calls, self.texts = lm, lm.info, [], []

    def complete(self, prompt, stops, max_new_tokens):
        started = time.perf_counter()
        text = self.lm.complete(prompt, stops, max_new_tokens)
        self.calls.append(round(time.perf_counter() - started, 2))
        self.texts.append(text)  # before the checks, to see why a reason was not shown
        return text

    def alive(self):
        return True

    def close(self):
        self.lm.close()


def board_server(llm_dir: str, adb: str = None, serial: str = None):
    """llama.cpp's server on the board, run as the app runs it: the client, and a function that stops it."""
    from deploy_app import IMAGE, REMOTE_CACHE, Board, find_adb

    info = read_info(llm_dir)
    if info.get("runtime") != "llama.cpp":
        sys.exit("--board needs a model that runs with llama.cpp")
    board = Board(find_adb(adb), serial)
    args = " ".join(server_args(f"/llm/{info['model_file']}", f"127.0.0.1:{PORT}", 4))
    board.shell(f"docker rm -f {CONTAINER} >/dev/null 2>&1; docker run -d --rm --name {CONTAINER} --network host "
                f"-v {REMOTE_CACHE}/llm:/llm:ro --entrypoint /llm/llama/llama-server {IMAGE} {args}", capture=True)
    board.run("forward", f"tcp:{PORT}", f"tcp:{PORT}", capture=True)

    def stop():
        board.shell(f"docker rm -f {CONTAINER}", check=False, capture=True)
        board.run("forward", "--remove", f"tcp:{PORT}", check=False, capture=True)

    server = LlamaServer(f"127.0.0.1:{PORT}", info)
    try:
        server.wait_until_ready()
    except Exception:
        stop()
        raise
    return server, stop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", default=os.path.join(ROOT, "outputs", "student", "final"))
    parser.add_argument("--llm-dir", default=os.path.join(ROOT, "outputs", "llm", "qwen2.5-1.5b-instruct"))
    parser.add_argument("--board", action="store_true", help="run the language model on the board (llama.cpp)")
    parser.add_argument("--adb", help="path to adb, for --board")
    parser.add_argument("--serial", help="adb serial number, for --board")
    parser.add_argument("--suggestions", action="store_true",
                        help="also ask for replacement words, even if the model's llm_info.json turns them off")
    parser.add_argument("--repeat-penalty", type=float, help="instead of the one in llm_info.json (1.0: none)")
    parser.add_argument("--words", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="default: outputs/explanations/<model folder>, with -board for --board")
    args = parser.parse_args(argv)
    name = os.path.basename(os.path.normpath(args.llm_dir)) + ("-board" if args.board else "")
    args.out = args.out or os.path.join(ROOT, "outputs", "explanations", name)
    os.makedirs(args.out, exist_ok=True)

    scorer = OnnxTokenModel(args.student, "int8", 4)
    threshold = float(scorer.info["flag_threshold"])
    with tempfile.TemporaryDirectory() as tmp:
        write_lexicon(os.path.join(tmp, "lexicon.txt"))
        lexicon = load_lexicon(os.path.join(tmp, "lexicon.txt"))
    babe = babe_module()
    _, new, _ = babe.split_by_mbic(babe.load_babe(), MBICDataLoader().load_raw_sentences())

    rng = np.random.default_rng(args.seed)
    candidates = []
    for text in new.text.astype(str).sample(frac=1.0, random_state=args.seed):
        spans, shares = scorer.score(text)
        flagged = [i for i, s in enumerate(shares) if not np.isnan(s) and s >= threshold]
        if flagged:
            i = int(rng.choice(flagged))
            candidates.append((text, spans[i], float(shares[i])))
        if len(candidates) == args.words:
            break

    if args.board:
        lm, stop = board_server(args.llm_dir, args.adb, args.serial)
    elif read_info(args.llm_dir).get("runtime") == "llama.cpp":
        sys.exit("This model's llama.cpp build is for the board: add --board")
    else:
        lm = load_model(args.llm_dir, 4)
        stop = lm.close
    if args.suggestions:
        lm.info = {**lm.info, "suggest_alternatives": True}
    if args.repeat_penalty is not None:
        lm.info = {**lm.info, "repeat_penalty": args.repeat_penalty}
    timed = Timed(lm)
    explainer = Explainer(args.llm_dir, lm=timed)
    rows = []
    try:
        for n, (sentence, (start, end), share) in enumerate(candidates, 1):
            word = sentence[start:end]
            timed.calls, timed.texts = [], []
            result = explainer.explain(sentence, start, end, share, band(share, threshold), word.lower() in lexicon,
                                       scorer.score, threshold, lexicon)
            asked = len(timed.calls) > 1  # a replacement word too
            rows.append({"sentence": sentence, **result, "reason_seconds": timed.calls[0],
                         "alternative_seconds": timed.calls[1] if asked else None, "reason_raw": timed.texts[0],
                         "alternative_raw": timed.texts[1] if asked else None})
            print(f"{n:>2}. {word} ({result['seconds']} s): {result['alternative'] or '-'} | "
                  f"{result['reason'] or '(no reason)'}")
    finally:
        stop()
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "sample.csv"), index=False)
    kept = table.alternative.notna()
    summary = {
        "model": {**explainer.describe(), "repeat_penalty": timed.info.get("repeat_penalty")},
        "words": len(table), "threshold": threshold,
        "reason_kept": int(table.reason.notna().sum()), "alternative_kept": int(kept.sum()),
        "alternative_rejections": table.alternative_rejected.value_counts().to_dict(),
        "mean_share_flagged_word": float(table.share.mean()),
        "mean_share_kept_alternative": float(table.alternative_share[kept].mean()) if kept.any() else None,
        "where": "board" if args.board else "pc",
        "mean_seconds": float(table.seconds.mean()), "median_seconds": float(table.seconds.median()),
        "max_seconds": float(table.seconds.max()), "median_reason_seconds": float(table.reason_seconds.median()),
        "median_alternative_seconds": (float(table.alternative_seconds.median())
                                       if table.alternative_seconds.notna().any() else None),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
