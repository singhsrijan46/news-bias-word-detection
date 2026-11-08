#!/usr/bin/env python
"""
Run the on-board explainer on a fixed sample of flagged words, to see what it gets right and wrong.

The sample is 40 words that the board's model flags in BABE's sentences that are not in MBIC
(scripts/evaluate_babe.py), one word per sentence, drawn with a fixed seed. For each word it
records the model's reason, its suggested replacement, and whether the replacement passed the
checks. The explainer code is the app's own (app/bias-lens/python/explainer.py), on the PC.

docs/EXPLANATIONS.md reports the results.

Usage:
    python scripts/app/fetch_llm.py
    python scripts/app/evaluate_explainer.py
"""

import argparse
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app", "bias-lens", "python"))
sys.path.insert(0, HERE)

from analyzer import band, load_lexicon  # noqa: E402
from deploy_app import write_lexicon  # noqa: E402
from explainer import Explainer  # noqa: E402
from src.data.dataset_loader import MBICDataLoader  # noqa: E402
from src.models.onnx_token_model import OnnxTokenModel  # noqa: E402


def babe_module():
    spec = importlib.util.spec_from_file_location("evaluate_babe", os.path.join(ROOT, "scripts", "evaluate_babe.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", default=os.path.join(ROOT, "outputs", "student", "final"))
    parser.add_argument("--llm-dir", default=os.path.join(ROOT, "outputs", "llm", "qwen2.5-0.5b-instruct"))
    parser.add_argument("--words", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "explanations"))
    args = parser.parse_args(argv)
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

    explainer = Explainer(args.llm_dir, threads=4)
    rows = []
    for n, (sentence, (start, end), share) in enumerate(candidates, 1):
        word = sentence[start:end]
        result = explainer.explain(sentence, start, end, share, band(share, threshold), word.lower() in lexicon,
                                   scorer.score, threshold, lexicon)
        rows.append({"sentence": sentence, **result})
        print(f"{n:>2}. {word}: {result['alternative'] or '-'} | {result['reason'] or '(no reason)'}")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "sample.csv"), index=False)
    kept = table.alternative.notna()
    summary = {
        "model": explainer.describe(), "words": len(table), "threshold": threshold,
        "reason_kept": int(table.reason.notna().sum()), "alternative_kept": int(kept.sum()),
        "alternative_rejections": table.alternative_rejected.value_counts().to_dict(),
        "mean_share_flagged_word": float(table.share.mean()),
        "mean_share_kept_alternative": float(table.alternative_share[kept].mean()) if kept.any() else None,
        "mean_seconds_pc": float(table.seconds.mean()),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
