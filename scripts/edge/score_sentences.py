#!/usr/bin/env python3
"""
Score sentences with an exported bias-word model and save every word's score.

This is the board side of check_int8_on_board.py. It needs only numpy, onnxruntime,
tokenizers and onnx_token_model.py, which is copied next to it on the board.

    python3 score_sentences.py --model DIR --sentences sentences.json --out scores.json
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    from onnx_token_model import OnnxTokenModel  # on the board: copied next to this script
except ImportError:  # on the PC: the repository's copy
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    from src.models.onnx_token_model import OnnxTokenModel


def score_all(model, sentences):
    """Every word's score, sentence by sentence; None where truncation cut the word."""
    return [[None if np.isnan(s) else float(s) for s in model.score(text)[1]] for text in sentences]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="an exported model folder (onnx_token_model.export)")
    parser.add_argument("--sentences", required=True, help="a JSON list of sentences")
    parser.add_argument("--out", required=True)
    parser.add_argument("--precision", default="int8", choices=["int8", "fp32"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    import onnxruntime

    with open(args.sentences, encoding="utf-8") as f:
        sentences = json.load(f)
    model = OnnxTokenModel(args.model, args.precision, args.threads)
    started = time.time()
    scores = score_all(model, sentences)
    result = {"precision": args.precision, "threads": args.threads, "onnxruntime": onnxruntime.__version__,
              "machine": platform.machine(), "sentences": len(sentences), "seconds": time.time() - started,
              "scores": scores}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print(f"{len(sentences)} sentences in {result['seconds']:.0f} s on {result['machine']}")


if __name__ == "__main__":
    main()
