"""
Word-level metrics for bias-word detectors.

Detection follows the paper: precision, recall and F1 for the "at least 4 votes"
label at a decision threshold, plus threshold-free ROC AUC and average precision.
Accuracy is left out on purpose: with about 9.5% positives, never flagging
anything scores 90.5%.

Calibration asks whether a predicted probability means what it says: against
the hard label (expected calibration error, Brier score), and against the vote
share (how far predicted shares are from the actual ones, and whether words are
ranked by how many annotators agreed: Spearman's rho).
"""

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def best_f1_threshold(y_true, scores) -> float:
    """The decision threshold with the highest F1."""
    precision, recall, thresholds = precision_recall_curve(np.asarray(y_true), np.asarray(scores))
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))])  # the curve's last point has no threshold


def detection_metrics(y_true, scores, flagged) -> Dict[str, float]:
    """Precision, recall and F1 of the 0/1 `flagged` decisions; ROC AUC and AP of the scores."""
    y_true, scores, flagged = (np.asarray(a) for a in (y_true, scores, flagged))
    tp = int(((flagged == 1) & (y_true == 1)).sum())
    precision = tp / max(int(flagged.sum()), 1)
    recall = tp / max(int(y_true.sum()), 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "flagged_share": float(flagged.mean()),
    }


def expected_calibration_error(targets, probs, bins: int = 10) -> float:
    """Mean gap between predicted probability and observed rate over equal-width bins, weighted by bin size.

    `targets` may be 0/1 labels or vote shares.
    """
    targets, probs = np.asarray(targets, dtype=float), np.asarray(probs, dtype=float)
    which = np.clip(np.digitize(probs, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    gap = 0.0
    for b in range(bins):
        in_bin = which == b
        if in_bin.any():
            gap += in_bin.mean() * abs(probs[in_bin].mean() - targets[in_bin].mean())
    return float(gap)


def calibration_metrics(labels, shares, probs) -> Dict[str, float]:
    """How well predicted probabilities match the hard labels and the vote shares."""
    labels, shares, probs = (np.asarray(a, dtype=float) for a in (labels, shares, probs))
    clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    highlighted = shares > 0
    return {
        "brier": float(np.mean((probs - labels) ** 2)),
        "ece": expected_calibration_error(labels, probs),
        "share_mse": float(np.mean((probs - shares) ** 2)),
        "share_ece": expected_calibration_error(shares, probs),
        "share_cross_entropy": float(np.mean(-(shares * np.log(clipped) + (1 - shares) * np.log(1 - clipped)))),
        "spearman_vs_share": float(spearmanr(probs, shares).correlation),
        "spearman_vs_share_highlighted": float(spearmanr(probs[highlighted], shares[highlighted]).correlation),
    }


PAIRED_METRICS = ("f1", "roc_auc", "average_precision", "brier", "ece", "share_ece", "share_cross_entropy",
                  "spearman_vs_share")


def per_fold_metrics(words: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    """Detection and calibration metrics for every fold and model.

    `words` has columns fold, label4, share, and for each model its score and
    `<model>_flag` (the 0/1 decision at that fold's threshold).
    """
    rows = []
    for fold, group in words.groupby("fold"):
        for m in models:
            metrics = {**detection_metrics(group.label4, group[m], group[f"{m}_flag"]),
                       **calibration_metrics(group.label4, group.share, group[m])}
            rows.append({"fold": int(fold), "model": m, **{k: metrics[k] for k in PAIRED_METRICS}})
    return pd.DataFrame(rows)


def paired_comparison(per_fold: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Model b minus model a on each metric, paired by fold, with a paired t-test and a Wilcoxon test."""
    wide = per_fold.pivot(index="fold", columns="model")
    rows = []
    for metric in PAIRED_METRICS:
        diff = wide[(metric, b)] - wide[(metric, a)]
        rows.append({"metric": metric, f"{a}_mean": wide[(metric, a)].mean(), f"{b}_mean": wide[(metric, b)].mean(),
                     "difference": diff.mean(), f"folds_{b}_higher": int((diff > 0).sum()), "folds": int(len(diff)),
                     "t_test_p": float(ttest_rel(wide[(metric, b)], wide[(metric, a)]).pvalue),
                     "wilcoxon_p": float(wilcoxon(diff).pvalue)})
    return pd.DataFrame(rows)
