#!/usr/bin/env python
"""
Compare the token models fold by fold, from train_token_models.py's saved predictions.

For each pair (M1 against M0, M2 against M1) and each metric: both models' mean
over the folds, the difference, in how many folds the second model is higher,
and paired tests across folds (t-test and Wilcoxon signed-rank). With 10 folds,
0.002 is the smallest Wilcoxon p-value possible. Also checks how often the
paper's label4 agrees with the label rebuilt from the raw annotations.

Usage:
    python scripts/compare_token_models.py                   # outputs/models/cv/
    python scripts/compare_token_models.py --cv-dir DIR
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.word_metrics import paired_comparison, per_fold_metrics  # noqa: E402


def compare(cv_dir: str) -> str:
    words = pd.read_csv(os.path.join(cv_dir, "paper_words_oof.csv"))
    models = [m for m in ("m0", "m1", "m2") if m in words.columns]
    words = words[(words.word_index >= 0) & words[models].notna().all(axis=1)]
    per_fold = per_fold_metrics(words, models)
    per_fold.to_csv(os.path.join(cv_dir, "per_fold_metrics.csv"), index=False)

    parts = []
    for a, b in (("m0", "m1"), ("m1", "m2")):
        if a in models and b in models:
            table = paired_comparison(per_fold, a, b)
            table.to_csv(os.path.join(cv_dir, f"paired_{a}_{b}.csv"), index=False)
            parts += [f"## {b.upper()} against {a.upper()}, paired by fold", "",
                      table.round(4).to_markdown(index=False), ""]

    tokens = pd.read_csv(os.path.join(cv_dir, "tokens_oof.csv"))
    both = words.merge(tokens[["sentence_id", "word_index", "label"]], on=["sentence_id", "word_index"])
    parts.append(f"The paper's label4 against the label rebuilt from the raw annotations (at least 4 votes), "
                 f"on {len(both)} words: agreement {(both.label4 == both.label).mean():.4f}, "
                 f"Cohen's kappa {cohen_kappa_score(both.label4, both.label):.3f}.")
    text = "\n".join(parts)
    with open(os.path.join(cv_dir, "paired.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cv-dir", default=os.path.join(ROOT, "outputs", "models", "cv"))
    args = parser.parse_args(argv)
    print(compare(args.cv_dir))


if __name__ == "__main__":
    main()
