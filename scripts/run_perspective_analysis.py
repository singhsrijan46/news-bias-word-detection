#!/usr/bin/env python
"""
Perspective analysis of MBIC's raw per-annotator annotations.

Reproduces the preliminary evidence in the "Whose Bias?" project proposal:
  1. Rebuild every annotator's word highlights and check them against the
     released 4-vote labels (biased_words4).
  2. How contested word labels are: the share of annotators behind each
     highlighted word.
  3. How much the binary label depends on the vote cutoff (3, 4 or 5 votes).
  4. Whether annotator ideology predicts which words get highlighted
     (permutation tests, continuous ideology and the paper's groups).
  5. Words that liberal and conservative annotators flag at different rates.

The seeds are fixed, so the full run reproduces the proposal's numbers exactly.

Usage:
    python scripts/run_perspective_analysis.py           # full run, a few minutes
    python scripts/run_perspective_analysis.py --quick   # 20 shuffles per test, smoke test only
"""

import argparse
import json
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.perspective import (
    contestedness_distribution,
    group_majority_tests,
    ideology_word_choice_test,
    word_rate_differences,
)
from src.data.annotations import (
    annotator_ideology,
    compare_with_published,
    ideology_groups,
    rebuild_sentence_annotations,
)
from src.data.dataset_loader import MBICDataLoader
from src.utils.helpers import setup_logger

logger = setup_logger("perspective_runner")

CONTINUOUS_SEED = 3  # seeds used for the proposal's numbers
GROUP_SEED = 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--continuous-permutations", type=int, default=300)
    parser.add_argument("--group-permutations", type=int, default=200)
    parser.add_argument("--quick", action="store_true", help="20 shuffles per test; p-values are too coarse to report")
    args = parser.parse_args(argv)
    if args.quick:
        args.continuous_permutations = args.group_permutations = 20

    loader = MBICDataLoader()
    output_dir = os.path.normpath(loader._resolve_path(loader.paths.get("perspective_dir", "outputs/perspective")))
    os.makedirs(output_dir, exist_ok=True)

    annotations = loader.load_annotations()
    sentences = rebuild_sentence_annotations(annotations)
    ideology = annotator_ideology(annotations)
    logger.info(f"Rebuilt {len(sentences)} sentences from {len(annotations)} annotator judgements "
                f"by {len(ideology)} survey records")

    summary = {"settings": {"continuous_permutations": args.continuous_permutations,
                            "group_permutations": args.group_permutations,
                            "continuous_seed": CONTINUOUS_SEED, "group_seed": GROUP_SEED}}

    logger.info("1/5 Checking rebuilt labels against biased_words4")
    summary["reconstruction"] = compare_with_published(sentences, loader.load_raw_sentences())

    logger.info("2/5 Measuring how contested word labels are")
    summary["contestedness"] = contestedness_distribution(sentences)

    logger.info("3/5 Counting biased words at vote cutoffs 3, 4 and 5")
    tokens = loader.load_token_dataset(drop_missing_tfidf=False)
    summary["cutoff_sensitivity"] = {
        "words": int(len(tokens)),
        **{f"biased_at_{k}_votes": int(tokens[f"label{k}"].sum()) for k in (3, 4, 5)},
    }

    paper_groups = ideology_groups(ideology, centre_band=3)
    shares = {g: sum(v == g for v in paper_groups.values()) / len(paper_groups) for g in ("liberal", "centre", "conservative")}
    summary["paper_grouping_shares"] = shares

    logger.info(f"4/5 Ideology tests ({args.continuous_permutations} and {args.group_permutations} shuffles)")
    summary["ideology_word_choice"] = ideology_word_choice_test(
        sentences, ideology, n_permutations=args.continuous_permutations, seed=CONTINUOUS_SEED)
    summary["group_majorities_paper_grouping"] = group_majority_tests(
        sentences, paper_groups, n_permutations=args.group_permutations, seed=GROUP_SEED)

    logger.info("5/5 Per-word liberal vs conservative highlight rates (simple left/right split)")
    words = word_rate_differences(sentences, ideology_groups(ideology, centre_band=0))
    words_path = os.path.join(output_dir, "word_rate_differences.csv")
    words.to_csv(words_path, index=False)
    summary["word_rates"] = {
        "words_tested": int(len(words)),
        "significant_uncorrected_p_below_0.01": int((words["p"] < 0.01).sum()),
        "significant_after_bonferroni": words.loc[words["p_bonferroni"] < 0.05, "word"].tolist(),
    }

    summary_path = os.path.join(output_dir, "perspective_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_report(summary, words)
    print(f"\nSaved: {summary_path}\n       {words_path}")


def print_report(summary: dict, words) -> None:
    rec, con, cut = summary["reconstruction"], summary["contestedness"], summary["cutoff_sensitivity"]
    cont, grp = summary["ideology_word_choice"], summary["group_majorities_paper_grouping"]
    print("\n" + "=" * 72)
    print("MBIC PERSPECTIVE ANALYSIS")
    print("=" * 72)
    print(f"Rebuilt vs published 4-vote labels: Jaccard {rec['mean_jaccard']:.3f}, precision {rec['precision']:.3f}, "
          f"recall {rec['recall']:.3f}, exact {rec['exact_matches']}/{rec['sentences']} sentences")
    print(f"\nHighlighted tokens: {con['highlighted_tokens']:,} of {con['tokens']:,}. Share of annotators behind each:")
    for b in con["bins"]:
        print(f"  {b['band']:>9}: {b['share']:6.1%}  ({b['tokens']:,} tokens)")
    print(f"\nBiased words at >=3 / >=4 / >=5 votes: {cut['biased_at_3_votes']:,} / {cut['biased_at_4_votes']:,} / "
          f"{cut['biased_at_5_votes']:,} of {cut['words']:,}")
    print("\nIdeology vs which words are highlighted (continuous ideology, all annotators):")
    for name, label in (("raw", "raw highlights"), ("which_words", "overall rate removed")):
        r = cont[name]
        print(f"  {label:<22} mean r^2 {r['mean_r2']:.4f} vs {r['null_mean_r2']:.4f} +/- {r['null_sd_r2']:.4f} "
              f"| z = {r['z']:.1f}, p = {r['p']:.3f} | signed r {r['mean_signed_r']:+.4f}, p = {r['p_signed']:.3f}")
    print("\nPaper's grouping (liberal < -3 < centre < 3 < conservative):")
    for name, label in (("split_words", "split words"), ("opposite_verdicts", "opposite sentence verdicts"),
                        ("centre_outlet_extra_words", "centre outlets: C minus L words")):
        r = grp[name]
        print(f"  {label:<32} {r['observed']:8.2f} vs {r['null_mean']:8.2f} +/- {r['null_sd']:.2f} | z = {r['z']:.1f}, p = {r['p']:.3f}")
    print(f"\nPer-word rates: {summary['word_rates']['words_tested']} words tested; "
          f"Bonferroni-significant: {', '.join(summary['word_rates']['significant_after_bonferroni']) or 'none'}")
    top = words[words["p"] < 0.01].sort_values("difference")
    print(top[["word", "liberal_rate", "conservative_rate", "p", "p_bonferroni"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
