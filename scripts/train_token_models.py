#!/usr/bin/env python
"""
Train and compare the first bias-word models with sentence-grouped cross-validation.

  M0  the paper's tuned XGBoost on its word features (dt_final)
  M1  RoBERTa-base token classifier trained on the paper's hard labels (at least 4 votes)
  M2  the same model trained on vote shares (soft labels)

Folds are grouped by sentence, so no sentence is in both training and test. (The
earlier random split over words put words of one sentence on both sides.) Within
each fold, 10% of the training sentences are held out to choose each model's
decision threshold: the one with the best F1 on the paper's words. Every model is
scored on its out-of-fold predictions for the paper's words and labels; M1 and M2
also on every token, against the annotators' vote shares.

With --split topic, each of MBIC's 14 topics is held out in turn instead, so every
prediction is for a topic the model never trained on.

Usage:
    python scripts/train_token_models.py               # 10 folds, about 30 minutes on a 4 GB GPU
    python scripts/train_token_models.py --quick       # 2 of the 10 folds, 1 epoch: checks that everything runs
    python scripts/train_token_models.py --models m0   # the XGBoost baseline only
    python scripts/train_token_models.py --split topic # 14 folds, one per topic (outputs/models/cv_topic)
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset_loader import MBICDataLoader  # noqa: E402
from src.data.word_targets import align_paper_words, build_sentence_targets  # noqa: E402
from src.evaluation.word_metrics import best_f1_threshold, calibration_metrics, detection_metrics  # noqa: E402
from src.models.all_models import NON_FEATURE_COLS  # noqa: E402
from src.utils.helpers import setup_logger  # noqa: E402

logger = setup_logger("token_models")

MIN_VOTES = 4  # the paper's label: at least 4 annotators highlighted the word
NATURAL_SHARE = 0.4  # 4 of 10 annotators: M2's own reading of the paper's cutoff
# The paper's tuned XGBoost, as in src/models/all_models.py.
M0_PARAMS = dict(objective="binary:logistic", max_depth=6, min_child_weight=18, subsample=1.0, colsample_bytree=1.0,
                 learning_rate=0.2, scale_pos_weight=9.5, n_estimators=150, eval_metric="logloss")
NAMES = {"m0": "M0 XGBoost (paper features)", "m1": "M1 RoBERTa, hard labels", "m2": "M2 RoBERTa, vote shares"}


def make_folds(n_sentences: int, folds: int, seed: int) -> np.ndarray:
    """Fold number of every sentence, in a seeded random order."""
    order = np.random.default_rng(seed).permutation(n_sentences)
    fold_of = np.empty(n_sentences, dtype=int)
    for k in range(folds):
        fold_of[order[k::folds]] = k
    return fold_of


def topic_folds(annotations: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Fold number of every sentence when each topic is held out in turn, and the topic of each fold.

    Sentences are in the order of build_sentence_targets: first appearance in the annotations.
    """
    topics = annotations.groupby("text", sort=False)["topic"].first().astype(str).str.strip().str.lower()
    names = sorted(set(topics))
    return np.array([names.index(t) for t in topics]), names


def split_validation(train_ids: np.ndarray, share: float, seed: int):
    """Hold out `share` of the training sentences for choosing the threshold."""
    shuffled = np.random.default_rng(seed).permutation(train_ids)
    n_val = max(1, int(round(share * len(shuffled))))
    return np.sort(shuffled[n_val:]), np.sort(shuffled[:n_val])


def word_scores(paper: pd.DataFrame, rows: np.ndarray, predictions: Dict[int, np.ndarray]) -> np.ndarray:
    """Look up a token model's probability for each listed paper word."""
    return np.array([predictions[s][w] for s, w in zip(paper.sentence_id.values[rows], paper.word_index.values[rows])])


def evaluate(paper: pd.DataFrame, scored: np.ndarray, models: List[str], oof: dict, flags: dict,
             token_frame: pd.DataFrame) -> dict:
    results = {}
    for m in models:
        rows = scored
        y, s, share = paper.label4.values[rows], oof[m][rows], paper.share.values[rows]
        entry = {"paper_words": {**detection_metrics(y, s, flags[m][rows]), **calibration_metrics(y, share, s)}}
        per_fold = [detection_metrics(y[f], s[f], flags[m][rows][f])["f1"]
                    for f in (paper.fold.values[rows] == k for k in sorted(set(paper.fold.values[rows])))]
        entry["paper_words"]["f1_fold_mean"] = float(np.mean(per_fold))
        entry["paper_words"]["f1_fold_sd"] = float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0
        if m == "m2":
            entry["paper_words_at_share_0.4"] = detection_metrics(y, s, (s >= NATURAL_SHARE).astype(int))
        if m in token_frame.columns:
            t = token_frame.dropna(subset=[m])
            entry["all_tokens"] = {"tokens": int(len(t)),
                                   "roc_auc": detection_metrics(t.label, t[m], (t[m] >= 0.5).astype(int))["roc_auc"],
                                   **calibration_metrics(t.label, t.share, t[m])}
        results[m] = entry
    return results


def summary_table(results: dict, thresholds: dict) -> str:
    rows = []
    for m, r in results.items():
        p = r["paper_words"]
        rows.append({"model": NAMES[m], "F1": p["f1"], "F1 per fold": f"{p['f1_fold_mean']:.3f} ± {p['f1_fold_sd']:.3f}",
                     "precision": p["precision"], "recall": p["recall"], "ROC AUC": p["roc_auc"],
                     "avg precision": p["average_precision"], "Brier": p["brier"], "ECE": p["ece"],
                     "share ECE": p["share_ece"], "Spearman vs share": p["spearman_vs_share"],
                     "threshold (mean)": float(np.mean(thresholds[m]))})
    return pd.DataFrame(rows).round(3).to_markdown(index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["m0", "m1", "m2"], choices=["m0", "m1", "m2"])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--max-folds", type=int, help="run only the first N folds")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--model-name", default="FacebookAI/roberta-base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="2 folds, 1 epoch")
    parser.add_argument("--split", choices=["random", "topic"], default="random",
                        help="random: sentences in seeded random folds; topic: each topic held out in turn")
    parser.add_argument("--out", help="default: outputs/models/cv, or outputs/models/cv_topic with --split topic")
    args = parser.parse_args(argv)
    if args.quick:
        args.max_folds, args.epochs = 2, 1
    args.out = args.out or os.path.join(ROOT, "outputs", "models", "cv" if args.split == "random" else "cv_topic")
    os.makedirs(args.out, exist_ok=True)

    loader = MBICDataLoader()
    annotations = loader.load_annotations()
    targets = build_sentence_targets(annotations)
    dt = loader.load_token_dataset(drop_missing_tfidf=False).reset_index(drop=True)
    paper = align_paper_words(targets, dt).set_index("dt_index").loc[dt.index].reset_index()
    found = (paper.sentence_id >= 0) & (paper.word_index >= 0)
    logger.info(f"{len(targets)} sentences, {sum(len(t.spans) for t in targets)} tokens; "
                f"{int(found.sum())} of {len(paper)} paper words matched to tokens")
    paper["share"] = [targets[s].shares[w] if ok else np.nan
                      for s, w, ok in zip(paper.sentence_id, paper.word_index, found)]
    fold_names = None
    if args.split == "topic":
        fold_of, fold_names = topic_folds(annotations)
        args.folds = len(fold_names)
    else:
        fold_of = make_folds(len(targets), args.folds, args.seed)
    paper["fold"] = np.where(paper.sentence_id >= 0, fold_of[paper.sentence_id.clip(lower=0)], -1)

    features = [c for c in dt.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(dt[c])]
    X = dt[features].fillna(0).values
    oof = {m: np.full(len(paper), np.nan) for m in args.models}
    flags = {m: np.full(len(paper), -1) for m in args.models}
    thresholds: Dict[str, List[float]] = {m: [] for m in args.models}
    token_preds: Dict[str, Dict[int, np.ndarray]] = {m: {} for m in args.models if m != "m0"}

    n_run = args.max_folds or args.folds
    for k in range(n_run):
        test_ids = np.where(fold_of == k)[0]
        train_ids, val_ids = split_validation(np.where(fold_of != k)[0], 0.1, args.seed + k)
        in_set = lambda ids, matched=True: np.where(np.isin(paper.sentence_id, ids) & (found if matched else True))[0]
        val_rows, test_rows = in_set(val_ids), in_set(test_ids)
        for m in args.models:
            started = time.time()
            if m == "m0":
                import xgboost as xgb

                train_rows = in_set(train_ids, matched=False)  # every paper word of the training sentences
                model = xgb.XGBClassifier(**M0_PARAMS, random_state=args.seed, n_jobs=-1)
                model.fit(X[train_rows], dt.label4.values[train_rows])
                val_scores = model.predict_proba(X[val_rows])[:, 1]
                test_scores = model.predict_proba(X[test_rows])[:, 1]
            else:
                from src.models.token_transformer import TokenModel, TrainConfig

                token_model = TokenModel(TrainConfig(model_name=args.model_name, epochs=args.epochs, seed=args.seed))
                train = [targets[i] for i in train_ids]
                y = [t.labels(MIN_VOTES).astype(np.float32) if m == "m1" else t.shares.astype(np.float32) for t in train]
                token_model.fit(train, y)
                predicted = dict(zip(val_ids, token_model.predict([targets[i] for i in val_ids])))
                tested = dict(zip(test_ids, token_model.predict([targets[i] for i in test_ids])))
                token_model.release()
                token_preds[m].update(tested)
                val_scores, test_scores = word_scores(paper, val_rows, predicted), word_scores(paper, test_rows, tested)
            threshold = best_f1_threshold(paper.label4.values[val_rows], val_scores)
            thresholds[m].append(threshold)
            oof[m][test_rows] = test_scores
            flags[m][test_rows] = (test_scores >= threshold).astype(int)
            fold_f1 = detection_metrics(paper.label4.values[test_rows], test_scores, flags[m][test_rows])["f1"]
            held_out = f" ({fold_names[k]} held out)" if fold_names else ""
            logger.info(f"fold {k + 1}/{n_run}{held_out} {m}: test F1 {fold_f1:.3f} at threshold {threshold:.3f} "
                        f"({time.time() - started:.0f} s)")

    scored = np.where(found & (paper.fold < n_run) & (paper.fold >= 0))[0]
    token_rows = []
    for t in targets:
        if fold_of[t.sentence_id] >= n_run:
            continue
        for w in range(len(t.spans)):
            row = {"sentence_id": t.sentence_id, "word_index": w, "fold": int(fold_of[t.sentence_id]),
                   "votes": int(t.votes[w]), "n_annotators": t.n_annotators, "share": float(t.shares[w]),
                   "label": int(t.votes[w] >= MIN_VOTES)}
            for m, preds in token_preds.items():
                row[m] = float(preds[t.sentence_id][w])
            token_rows.append(row)
    token_frame = pd.DataFrame(token_rows)
    results = evaluate(paper, scored, args.models, oof, flags, token_frame)
    report = {"settings": {**vars(args), "paper_words_scored": int(len(scored)),
                           "paper_words_unmatched": int((~found).sum()), "fold_topics": fold_names},
              "thresholds": thresholds, "results": results}
    if fold_names:  # each held-out topic's own result
        rows = []
        for k, name in enumerate(fold_names[:n_run]):
            r = scored[paper.fold.values[scored] == k]
            for m in args.models:
                d = detection_metrics(paper.label4.values[r], oof[m][r], flags[m][r])
                rows.append({"topic": name, "model": m, "words": int(len(r)),
                             "positive_share": float(paper.label4.values[r].mean()), "f1": d["f1"],
                             "precision": d["precision"], "recall": d["recall"], "roc_auc": d["roc_auc"]})
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "per_topic.csv"), index=False)

    paper_out = paper.copy()
    for m in args.models:
        paper_out[m], paper_out[f"{m}_flag"] = oof[m], flags[m]
    paper_out.to_csv(os.path.join(args.out, "paper_words_oof.csv"), index=False)
    token_frame.to_csv(os.path.join(args.out, "tokens_oof.csv"), index=False)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    table = summary_table(results, thresholds)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)
    if "m2" in results:
        r = results["m2"]["paper_words_at_share_0.4"]
        print(f"\nM2 at its own cutoff (predicted share >= {NATURAL_SHARE}): F1 {r['f1']:.3f}, "
              f"precision {r['precision']:.3f}, recall {r['recall']:.3f}")
    for m in token_preds:
        a = results[m]["all_tokens"]
        print(f"{NAMES[m]}, all {a['tokens']} tokens: share ECE {a['share_ece']:.3f}, "
              f"Spearman vs share {a['spearman_vs_share']:.3f} (highlighted words only: "
              f"{a['spearman_vs_share_highlighted']:.3f})")
    if n_run > 1:
        from compare_token_models import compare

        print("\n" + compare(args.out))
    print(f"\nSaved to {os.path.normpath(args.out)}")


if __name__ == "__main__":
    main()
