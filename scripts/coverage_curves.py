#!/usr/bin/env python
"""
Accuracy against coverage: what each model gains by not deciding on its least certain words.

A model that knows when it is unsure can abstain on those words and be right more often on the
rest. Here each model abstains on the words whose score is closest to its decision threshold (its
cross-validated threshold for that fold), and is scored on the words it keeps, against the
paper's 4-vote labels. This is RQ1's last measure: whether training on vote shares (M2) makes
abstaining more useful than training on yes/no labels (M1).

Also reported: how many of the abstained words are contested, with 2 to 5 of 10 annotators
highlighting them, and the per-fold comparison of M1 and M2 at 90% coverage.

Needs the cross-validated predictions of train_token_models.py and train_student.py.

Usage:
    python scripts/coverage_curves.py
"""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.word_metrics import detection_metrics  # noqa: E402

COVERAGES = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)
CONTESTED = (0.2, 0.6)  # a word is contested when 2 to 5 of 10 annotators highlighted it
MODELS = {"m0": "M0, the paper's XGBoost", "m1": "M1, hard labels", "m2": "M2, vote shares",
          "s_distil": "Student", "s_distil_int8": "Student, INT8"}


def fold_thresholds(words: pd.DataFrame, model: str) -> pd.Series:
    """Each fold's decision threshold, recovered from its flags: halfway between the highest
    unflagged score and the lowest flagged one."""
    def one(part):
        flagged = part.loc[part[f"{model}_flag"] == 1, model]
        unflagged = part.loc[part[f"{model}_flag"] == 0, model]
        return (flagged.min() + unflagged.max()) / 2
    return words.groupby("fold").apply(one, include_groups=False)


def confidence(words: pd.DataFrame, model: str) -> np.ndarray:
    """Distance of each score from its fold's threshold: larger means more certain."""
    t = words.fold.map(fold_thresholds(words, model))
    return np.abs(words[model].to_numpy() - t.to_numpy())


def keep_most_certain(conf: np.ndarray, coverage: float) -> np.ndarray:
    """A mask keeping the `coverage` share of words with the highest confidence."""
    n_keep = int(round(coverage * len(conf)))
    keep = np.zeros(len(conf), dtype=bool)
    keep[np.argsort(-conf, kind="stable")[:n_keep]] = True
    return keep


def is_contested(share: np.ndarray) -> np.ndarray:
    return (share >= CONTESTED[0]) & (share < CONTESTED[1])


def curve(words: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    rows = []
    contested = is_contested(words.share.to_numpy())
    for m in models:
        conf = confidence(words, m)
        for c in COVERAGES:
            keep = keep_most_certain(conf, c)
            part = words[keep]
            d = detection_metrics(part.label4, part[m], part[f"{m}_flag"])
            dropped = ~keep
            rows.append({"model": m, "coverage": c, "words": int(keep.sum()), "f1": d["f1"],
                         "precision": d["precision"], "recall": d["recall"],
                         "error_rate": float((part[f"{m}_flag"] != part.label4).mean()),
                         "positives_kept": float(part.label4.sum() / words.label4.sum()),
                         "contested_share_of_abstained": float(contested[dropped].mean()) if dropped.any() else None,
                         "contested_abstained": float(contested[dropped].sum() / contested.sum())})
    return pd.DataFrame(rows)


def paired_folds(words: pd.DataFrame, a: str, b: str, coverage: float) -> Dict:
    """F1 at `coverage` per fold for models a and b, each abstaining within the fold."""
    f1 = {a: [], b: []}
    for _, part in words.groupby("fold"):
        for m in (a, b):
            keep = keep_most_certain(confidence(part, m), coverage)
            kept = part[keep]
            f1[m].append(detection_metrics(kept.label4, kept[m], kept[f"{m}_flag"])["f1"])
    x, y = np.array(f1[a]), np.array(f1[b])
    full = {m: [detection_metrics(p.label4, p[m], p[f"{m}_flag"])["f1"] for _, p in words.groupby("fold")] for m in (a, b)}
    gain_a, gain_b = x - np.array(full[a]), y - np.array(full[b])
    return {"coverage": coverage, "folds": int(len(x)), f"{a}_f1": float(x.mean()), f"{b}_f1": float(y.mean()),
            f"{b}_minus_{a}": float((y - x).mean()), "p_ttest": float(ttest_rel(y, x).pvalue),
            "p_wilcoxon": float(wilcoxon(y - x).pvalue), f"{b}_better_folds": int((y > x).sum()),
            f"{a}_gain_from_abstaining": float(gain_a.mean()), f"{b}_gain_from_abstaining": float(gain_b.mean()),
            "gain_difference_p_ttest": float(ttest_rel(gain_b, gain_a).pvalue)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-cv", default=os.path.join(ROOT, "outputs", "models", "cv"))
    parser.add_argument("--student-cv", default=os.path.join(ROOT, "outputs", "student", "cv"))
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "coverage"))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    # Each file is scored on its own rows, the paper's words as in its cross-validation, so that
    # full coverage reproduces the published results.
    base = pd.read_csv(os.path.join(args.models_cv, "paper_words_oof.csv"))
    base = base[(base.fold >= 0) & base.m1.notna()]
    student = pd.read_csv(os.path.join(args.student_cv, "paper_words_oof.csv"))
    student = student[(student.fold >= 0) & student.s_distil.notna()]
    print(f"{len(base)} paper words for M0 to M2, {len(student)} for the student")

    table = pd.concat([curve(base, ["m0", "m1", "m2"]), curve(student, ["s_distil", "s_distil_int8"])],
                      ignore_index=True)
    paired = pd.DataFrame([paired_folds(base, "m1", "m2", c) for c in (0.9, 0.8)])
    table.to_csv(os.path.join(args.out, "coverage_curves.csv"), index=False)
    paired.to_csv(os.path.join(args.out, "m1_vs_m2.csv"), index=False)
    wide = table.pivot(index="model", columns="coverage", values="f1").loc[list(MODELS)]
    wide = wide[sorted(wide.columns, reverse=True)]
    wide.index = [MODELS[m] for m in wide.index]
    wide.columns = [f"F1 at {int(c * 100)}%" for c in wide.columns]
    report = "\n".join([
        "## F1 on the words each model keeps", "", wide.round(3).to_markdown(), "",
        "## All measures", "", table.round(3).to_markdown(index=False), "",
        "## M1 against M2, per fold", "", paired.round(4).to_markdown(index=False), "",
    ])
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
