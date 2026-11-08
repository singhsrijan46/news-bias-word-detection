#!/usr/bin/env python
"""
Do the models really agree more with liberal annotators, or is that a side effect of group size?

A sentence has on average 4.7 liberal and 2.7 conservative annotators (the paper's grouping:
liberal below -3, conservative above 3). Shares from smaller groups are noisier, and any model
agrees less with a noisier target. This check gives both groups the same size: for every
sentence it draws as many annotators from each group as the smaller group has, recomputes both
groups' shares, and compares the models' cross-validated predictions with them. It repeats the
draw many times.

Usage:
    python scripts/audit_group_balance.py             # 50 draws, about two minutes
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.annotations import annotator_ideology, ideology_groups, rebuild_sentence_annotations  # noqa: E402
from src.data.dataset_loader import MBICDataLoader  # noqa: E402

MODELS = ["teacher", "s_distil", "s_distil_int8"]


def group_votes(annotations: pd.DataFrame):
    """Per sentence: the liberal and the conservative annotators' highlighted positions."""
    groups = ideology_groups(annotator_ideology(annotations), centre_band=3)
    out = []
    for sentence in rebuild_sentence_annotations(annotations):
        lib = [j.positions for j in sentence.judgements if groups.get(j.annotator_id) == "liberal"]
        con = [j.positions for j in sentence.judgements if groups.get(j.annotator_id) == "conservative"]
        out.append((lib, con))
    return out


def matched_shares(votes, keys, rng):
    """Both groups' shares for the listed (sentence, word) keys, from equal-sized random draws."""
    lib_share, con_share = np.full(len(keys), np.nan), np.full(len(keys), np.nan)
    drawn = {}
    for i, (sid, word) in enumerate(keys):
        if sid not in drawn:
            lib, con = votes[sid]
            k = min(len(lib), len(con))
            drawn[sid] = None if k == 0 else (
                [lib[j] for j in rng.choice(len(lib), k, replace=False)],
                [con[j] for j in rng.choice(len(con), k, replace=False)])
        pick = drawn[sid]
        if pick is not None:
            lib_share[i] = np.mean([word in p for p in pick[0]])
            con_share[i] = np.mean([word in p for p in pick[1]])
    return lib_share, con_share


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cv-dir", default=os.path.join(ROOT, "outputs", "student", "cv"))
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "audit"))
    args = parser.parse_args(argv)

    oof = pd.read_csv(os.path.join(args.cv_dir, "paper_words_oof.csv"))
    votes = group_votes(MBICDataLoader().load_annotations())
    keys = list(zip(oof.sentence_id, oof.word_index))
    rng = np.random.default_rng(args.seed)
    rows = []
    for draw in range(args.draws):
        lib, con = matched_shares(votes, keys, rng)
        ok = ~np.isnan(lib) & ~np.isnan(con)
        rows.append({"draw": draw, "model": "groups against each other",
                     "spearman_liberal": spearmanr(lib[ok], con[ok]).correlation})
        for m in MODELS:
            p = oof[m].values[ok]
            rows.append({"draw": draw, "model": m,
                         "spearman_liberal": spearmanr(p, lib[ok]).correlation,
                         "spearman_conservative": spearmanr(p, con[ok]).correlation,
                         "abs_error_liberal": float(np.mean(np.abs(p - lib[ok]))),
                         "abs_error_conservative": float(np.mean(np.abs(p - con[ok])))})
    draws = pd.DataFrame(rows)
    draws["spearman_gap"] = draws.spearman_liberal - draws.spearman_conservative
    summary = []
    for m in MODELS:
        d = draws[draws.model == m]
        summary.append({"model": m, "words": int(ok.sum()),
                        "spearman_liberal": d.spearman_liberal.mean(), "spearman_conservative": d.spearman_conservative.mean(),
                        "gap_mean": d.spearman_gap.mean(), "gap_sd": d.spearman_gap.std(ddof=1),
                        "draws_liberal_higher": int((d.spearman_gap > 0).sum()), "draws": int(len(d)),
                        "abs_error_liberal": d.abs_error_liberal.mean(),
                        "abs_error_conservative": d.abs_error_conservative.mean()})
    between = draws[draws.model == "groups against each other"].spearman_liberal
    table = pd.DataFrame(summary)
    os.makedirs(args.out, exist_ok=True)
    draws.to_csv(os.path.join(args.out, "group_balance_draws.csv"), index=False)
    with open(os.path.join(args.out, "group_balance.json"), "w", encoding="utf-8") as f:
        json.dump({"draws": args.draws, "groups_against_each_other_spearman": float(between.mean()),
                   "models": summary}, f, indent=2, default=float)
    print(f"Equal-sized groups ({args.draws} draws); liberal and conservative shares agree with each other at "
          f"Spearman {between.mean():.3f}\n")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
