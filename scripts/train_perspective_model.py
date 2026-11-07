#!/usr/bin/env python
"""
Train M3, the perspective model, and test whether it captures how liberal and
conservative annotators differ (RQ2).

M3 predicts three shares per word: all annotators, liberal annotators and conservative
annotators (the paper's grouping). It is scored on the same 10 sentence-grouped folds as
train_token_models.py, against M2's single share as the baseline that ignores perspective:

  - Log loss per annotator judgement, for each group's votes: does a group-specific share
    explain that group's choices better than one share for everyone?
  - The liberal-conservative gap: does M3's predicted gap follow the gap the annotators show?
  - The "all" output on the paper's words: does adding groups cost detection?

Usage:
    python scripts/train_perspective_model.py              # 10 folds, about 15 minutes on a 4 GB GPU
    python scripts/train_perspective_model.py --quick      # 2 folds, 1 epoch
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset_loader import MBICDataLoader  # noqa: E402
from src.data.word_targets import align_paper_words, build_sentence_targets  # noqa: E402
from src.evaluation.word_metrics import best_f1_threshold, detection_metrics  # noqa: E402
from src.models.perspective_model import (  # noqa: E402
    GroupTokenModel, build_group_targets, group_log_loss, summarize_groups,
)
from src.models.token_transformer import TrainConfig  # noqa: E402
from src.utils.helpers import setup_logger  # noqa: E402
from train_token_models import make_folds, split_validation, word_scores  # noqa: E402

logger = setup_logger("perspective_model")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--model-name", default="FacebookAI/roberta-base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--m2-tokens", default=os.path.join(ROOT, "outputs", "models", "cv", "tokens_oof.csv"),
                        help="M2's out-of-fold token predictions from train_token_models.py (same folds)")
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "models", "perspective_cv"))
    args = parser.parse_args(argv)
    if args.quick:
        args.max_folds, args.epochs = 2, 1
    os.makedirs(args.out, exist_ok=True)

    loader = MBICDataLoader()
    annotations = loader.load_annotations()
    targets = build_sentence_targets(annotations)
    groups = build_group_targets(annotations)
    logger.info(f"groups: {summarize_groups(groups)}")
    dt = loader.load_token_dataset(drop_missing_tfidf=False).reset_index(drop=True)
    paper = align_paper_words(targets, dt).set_index("dt_index").loc[dt.index].reset_index()
    found = ((paper.sentence_id >= 0) & (paper.word_index >= 0)).values
    fold_of = make_folds(len(targets), args.folds, args.seed)

    m2 = pd.read_csv(args.m2_tokens) if os.path.exists(args.m2_tokens) else None
    m2_pred = ({(s, w): p for s, w, p in zip(m2.sentence_id, m2.word_index, m2.m2)} if m2 is not None else {})

    rows, fold_rows = [], []
    oof_all, flags = np.full(len(paper), np.nan), np.full(len(paper), -1)
    n_run = args.max_folds or args.folds
    for k in range(n_run):
        started = time.time()
        test_ids = np.where(fold_of == k)[0]
        train_ids, val_ids = split_validation(np.where(fold_of != k)[0], 0.1, args.seed + k)
        model = GroupTokenModel(TrainConfig(model_name=args.model_name, epochs=args.epochs, seed=args.seed))
        model.fit([targets[i] for i in train_ids], [groups[i] for i in train_ids])
        val_pred = dict(zip(val_ids, model.predict([targets[i] for i in val_ids])))
        test_pred = dict(zip(test_ids, model.predict([targets[i] for i in test_ids])))
        model.release()

        # The "all" output on the paper's words, with a threshold from the validation sentences.
        in_set = lambda ids: np.where(np.isin(paper.sentence_id, ids) & found)[0]
        val_rows, test_rows = in_set(val_ids), in_set(test_ids)
        first = lambda preds: {s: p[:, 0] for s, p in preds.items()}
        threshold = best_f1_threshold(paper.label4.values[val_rows], word_scores(paper, val_rows, first(val_pred)))
        scores = word_scores(paper, test_rows, first(test_pred))
        oof_all[test_rows], flags[test_rows] = scores, (scores >= threshold).astype(int)

        # Group-level scores on every token of the test sentences that both groups judged.
        losses = {"m3_liberal": [], "m3_conservative": [], "m3_all_for_liberal": [], "m3_all_for_conservative": [],
                  "m2_for_liberal": [], "m2_for_conservative": []}
        for s in test_ids:
            g, p = groups[s], test_pred[s]
            if g.judges[1] == 0 or g.judges[2] == 0 or np.isnan(p).any():
                continue
            m2_words = np.array([m2_pred.get((s, w), np.nan) for w in range(len(g.votes))])
            for w in range(len(g.votes)):
                rows.append({"fold": k, "sentence_id": int(s), "word_index": w,
                             "votes_all": g.votes[w, 0], "judges_all": g.judges[0],
                             "votes_liberal": g.votes[w, 1], "judges_liberal": g.judges[1],
                             "votes_conservative": g.votes[w, 2], "judges_conservative": g.judges[2],
                             "m3_all": p[w, 0], "m3_liberal": p[w, 1], "m3_conservative": p[w, 2],
                             "m2": m2_words[w]})
            losses["m3_liberal"].append(group_log_loss(g.votes[:, 1], g.judges[1], p[:, 1]))
            losses["m3_conservative"].append(group_log_loss(g.votes[:, 2], g.judges[2], p[:, 2]))
            losses["m3_all_for_liberal"].append(group_log_loss(g.votes[:, 1], g.judges[1], p[:, 0]))
            losses["m3_all_for_conservative"].append(group_log_loss(g.votes[:, 2], g.judges[2], p[:, 0]))
            if not np.isnan(m2_words).any():
                losses["m2_for_liberal"].append(group_log_loss(g.votes[:, 1], g.judges[1], m2_words))
                losses["m2_for_conservative"].append(group_log_loss(g.votes[:, 2], g.judges[2], m2_words))
        fold_rows.append({"fold": k, **{name: float(np.mean(v)) if v else None for name, v in losses.items()},
                          "f1_all": detection_metrics(paper.label4.values[test_rows], scores,
                                                      flags[test_rows])["f1"]})
        logger.info(f"fold {k + 1}/{n_run} ({time.time() - started:.0f} s): " + ", ".join(
            f"{name} {value:.4f}" for name, value in fold_rows[-1].items() if name != "fold" and value is not None))

    tokens = pd.DataFrame(rows)
    tokens.to_csv(os.path.join(args.out, "tokens_oof.csv"), index=False)
    per_fold = pd.DataFrame(fold_rows)
    per_fold.to_csv(os.path.join(args.out, "per_fold.csv"), index=False)

    share_l = tokens.votes_liberal / tokens.judges_liberal
    share_c = tokens.votes_conservative / tokens.judges_conservative
    observed_gap = share_c - share_l
    predicted_gap = tokens.m3_conservative - tokens.m3_liberal
    flagged_by_someone = (tokens.votes_all > 0).values
    gap = {"tokens": int(len(tokens)),
           "spearman_gap_all_tokens": float(spearmanr(predicted_gap, observed_gap).correlation),
           "spearman_gap_flagged_tokens": float(spearmanr(predicted_gap[flagged_by_someone],
                                                          observed_gap[flagged_by_someone]).correlation),
           "flagged_tokens": int(flagged_by_someone.sum())}
    paired = {}
    if "m2_for_liberal" in per_fold and per_fold.m2_for_liberal.notna().all() and len(per_fold) > 1:
        for group in ("liberal", "conservative"):
            a, b = per_fold[f"m2_for_{group}"], per_fold[f"m3_{group}"]
            paired[group] = {"m2_log_loss": float(a.mean()), "m3_log_loss": float(b.mean()),
                             "m3_better_in_folds": int((b < a).sum()), "folds": int(len(a)),
                             "t_test_p": float(ttest_rel(b, a).pvalue)}
    in_run = (paper.sentence_id >= 0).values & (fold_of[paper.sentence_id.clip(lower=0)] < n_run)
    scored = np.where(found & in_run)[0]
    detection = detection_metrics(paper.label4.values[scored], oof_all[scored], flags[scored])

    # The words where M3 predicts the largest gaps, averaged over their occurrences.
    words = tokens.assign(word=[targets[s].words[w].lower() for s, w in zip(tokens.sentence_id, tokens.word_index)],
                          predicted_gap=predicted_gap, observed_gap=observed_gap)
    by_word = (words.groupby("word").agg(n=("predicted_gap", "size"), predicted_gap=("predicted_gap", "mean"),
                                         observed_gap=("observed_gap", "mean"))
               .query("n >= 5").sort_values("predicted_gap"))
    extremes = pd.concat([by_word.head(10), by_word.tail(10)]).round(3)
    extremes.to_csv(os.path.join(args.out, "largest_predicted_gaps.csv"))

    report = {"settings": vars(args), "groups": summarize_groups(groups), "gap": gap, "paired_log_loss": paired,
              "m3_all_on_paper_words": detection, "per_fold_mean": per_fold.mean(numeric_only=True).to_dict()}
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps({k: report[k] for k in ("gap", "paired_log_loss", "m3_all_on_paper_words")}, indent=2, default=float))
    print("\nWords with the largest predicted gaps (conservative minus liberal share):")
    print(extremes.to_string())


if __name__ == "__main__":
    main()
