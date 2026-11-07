"""
Perspective analyses of MBIC's word-level crowd annotations.

Built on `src.data.annotations`. These tests ask whether bias perception in MBIC is
contested, and whether annotators' political ideology shapes which words they flag.
Every p-value comes from a permutation test that shuffles ideology across
annotators, which keeps each annotator's own highlights intact.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from src.data.annotations import SentenceAnnotations

SIDES = ("liberal", "conservative")


def contestedness_distribution(
    sentences: List[SentenceAnnotations],
    edges: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> Dict[str, object]:
    """
    How contested are word labels? Among tokens that at least one annotator
    highlighted, the share falling in each band of "share of the sentence's
    annotators who highlighted it". The last band includes 100%.
    """
    shares = np.array([v for s in sentences for v in s.vote_shares()])
    highlighted = shares[shares > 0]
    bins = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (highlighted >= lo) & ((highlighted <= hi) if i == len(edges) - 2 else (highlighted < hi))
        bins.append({"band": f"{lo:.0%}-{hi:.0%}", "tokens": int(in_bin.sum()), "share": float(in_bin.mean())})
    return {"tokens": int(len(shares)), "highlighted_tokens": int(len(highlighted)), "bins": bins}


def _highlight_matrices(
    sentences: List[SentenceAnnotations], annotator_index: Dict[str, int]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Per sentence: annotator indices and a 0/1 annotators x tokens matrix of
    the tokens that some, but not all, annotators highlighted."""
    matrices = []
    for s in sentences:
        highlights = np.zeros((s.n_annotators, len(s.tokens)))
        for row, judgement in enumerate(s.judgements):
            if judgement.positions:
                highlights[row, sorted(judgement.positions)] = 1
        votes = highlights.sum(0)
        disputed = (votes > 0) & (votes < s.n_annotators)
        if disputed.any():
            index = np.array([annotator_index[j.annotator_id] for j in s.judgements])
            matrices.append((index, highlights[:, disputed]))
    return matrices


def _ideology_correlation(
    matrices: List[Tuple[np.ndarray, np.ndarray]], ideology: np.ndarray, remove_annotator_rate: bool
) -> Tuple[float, float]:
    """Mean r^2 and mean signed r between ideology and highlights, over disputed tokens."""
    r_squared, r_signed = [], []
    for index, highlights in matrices:
        x = ideology[index]
        if x.std() == 0:
            continue
        y = highlights - highlights.mean(1, keepdims=True) if remove_annotator_rate else highlights
        xz = (x - x.mean()) / x.std()
        y_centred = y - y.mean(0)
        sd = y_centred.std(0)
        varies = sd > 0
        r = (xz[:, None] * y_centred[:, varies]).mean(0) / sd[varies]
        r_squared += list(r ** 2)
        r_signed += list(r)
    return float(np.mean(r_squared)), float(np.mean(r_signed))


def ideology_word_choice_test(
    sentences: List[SentenceAnnotations],
    ideology: pd.Series,
    n_permutations: int = 300,
    seed: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Does an annotator's ideology (continuous, -10..10) predict which words they highlight?

    For every token some but not all of a sentence's annotators highlighted, correlate
    ideology with the 0/1 highlight and average r^2 over tokens; compare with the same
    statistic after shuffling ideology across annotators. `which_words` first removes
    each annotator's overall highlight rate in the sentence, so it tests which words
    people flag rather than how many. `p_signed` tests for an overall direction
    (more conservative -> more highlighting).

    `ideology` must be indexed by survey record id; its order and `seed` fix the shuffles.
    """
    annotator_index = {annotator: i for i, annotator in enumerate(ideology.index)}
    values = ideology.to_numpy(dtype=float)
    matrices = _highlight_matrices(sentences, annotator_index)
    rng = np.random.default_rng(seed)
    results = {}
    for name, remove_rate in (("raw", False), ("which_words", True)):
        observed_r2, observed_r = _ideology_correlation(matrices, values, remove_rate)
        null = np.array(
            [_ideology_correlation(matrices, rng.permutation(values), remove_rate) for _ in range(n_permutations)]
        )
        results[name] = {
            "mean_r2": observed_r2,
            "null_mean_r2": float(null[:, 0].mean()),
            "null_sd_r2": float(null[:, 0].std()),
            "z": float((observed_r2 - null[:, 0].mean()) / null[:, 0].std()),
            "p": float((np.sum(null[:, 0] >= observed_r2) + 1) / (n_permutations + 1)),
            "mean_signed_r": observed_r,
            "p_signed": float((np.sum(np.abs(null[:, 1]) >= abs(observed_r)) + 1) / (n_permutations + 1)),
        }
    results["disputed_tokens"] = int(sum(h.shape[1] for _, h in matrices))
    return results


def _group_statistics(
    sentences: List[SentenceAnnotations], groups: Dict[str, str], min_per_group: int
) -> np.ndarray:
    """[split words, sentences with opposite verdicts, conservative minus liberal
    highlighted words per sentence in centre-outlet sentences]."""
    split_words = opposite_verdicts = 0
    centre_outlet_words = {side: [] for side in SIDES}
    for s in sentences:
        liberal = [j for j in s.judgements if groups[j.annotator_id] == "liberal"]
        conservative = [j for j in s.judgements if groups[j.annotator_id] == "conservative"]
        if s.outlet_type == "center":
            for j in s.judgements:
                if groups[j.annotator_id] in centre_outlet_words:
                    centre_outlet_words[groups[j.annotator_id]].append(len(j.positions))
        if len(liberal) < min_per_group or len(conservative) < min_per_group:
            continue
        votes_l = Counter(p for j in liberal for p in j.positions)
        votes_c = Counter(p for j in conservative for p in j.positions)
        for position in set(votes_l) | set(votes_c):
            rate_l, rate_c = votes_l[position] / len(liberal), votes_c[position] / len(conservative)
            split_words += max(rate_l, rate_c) >= 0.5 and abs(rate_l - rate_c) >= 0.5
        biased_l = np.mean([j.says_biased for j in liberal])
        biased_c = np.mean([j.says_biased for j in conservative])
        opposite_verdicts += (biased_l > 0.5 and biased_c < 0.5) or (biased_l < 0.5 and biased_c > 0.5)
    centre_gap = np.mean(centre_outlet_words["conservative"]) - np.mean(centre_outlet_words["liberal"])
    return np.array([split_words, opposite_verdicts, centre_gap])


def group_majority_tests(
    sentences: List[SentenceAnnotations],
    groups: Dict[str, str],
    n_permutations: int = 200,
    seed: int = 2,
    min_per_group: int = 2,
) -> Dict[str, Dict[str, float]]:
    """
    Liberal vs conservative annotators, compared within sentences.

    - `split_words`: tokens one side highlights at >= 50% and the other at least
      50 points less (sentences with at least `min_per_group` annotators per side).
    - `opposite_verdicts`: sentences whose liberal and conservative majorities give
      opposite biased / non-biased labels.
    - `centre_outlet_extra_words`: conservative minus liberal mean number of
      highlighted words per sentence, in sentences from centre outlets.

    Two-sided permutation p-values; shuffles group labels across annotators.
    `groups` order and `seed` fix the shuffles.
    """
    annotators = list(groups)
    labels = np.array([groups[a] for a in annotators])
    rng = np.random.default_rng(seed)
    observed = _group_statistics(sentences, groups, min_per_group)
    null = np.array(
        [
            _group_statistics(sentences, dict(zip(annotators, rng.permutation(labels))), min_per_group)
            for _ in range(n_permutations)
        ]
    )
    results = {}
    for column, name in enumerate(("split_words", "opposite_verdicts", "centre_outlet_extra_words")):
        mean, sd = null[:, column].mean(), null[:, column].std()
        results[name] = {
            "observed": float(observed[column]),
            "null_mean": float(mean),
            "null_sd": float(sd),
            "z": float((observed[column] - mean) / sd),
            "p": float((np.sum(np.abs(null[:, column] - mean) >= abs(observed[column] - mean)) + 1) / (n_permutations + 1)),
        }
    return results


def word_rate_differences(
    sentences: List[SentenceAnnotations],
    groups: Dict[str, str],
    min_seen: int = 25,
    min_highlights: int = 8,
    min_length: int = 3,
) -> pd.DataFrame:
    """
    Per word: how often liberal and conservative annotators highlighted it, out of
    the times they saw it, with a two-sided Fisher exact test and a Bonferroni
    correction over all words tested. Exploratory: most words are seen too rarely
    for a per-word test to have power.
    """
    counts = defaultdict(lambda: np.zeros(4))  # liberal hits, liberal seen, conservative hits, conservative seen
    for s in sentences:
        for j in s.judgements:
            side = groups[j.annotator_id]
            if side not in SIDES:
                continue
            column = 0 if side == "liberal" else 2
            for position, word in enumerate(s.tokens):
                counts[word][column + 1] += 1
                counts[word][column] += position in j.positions
    rows = []
    for word, (lib_hits, lib_seen, con_hits, con_seen) in counts.items():
        if lib_seen >= min_seen and con_seen >= min_seen and lib_hits + con_hits >= min_highlights and len(word) >= min_length:
            table = [[int(lib_hits), int(lib_seen - lib_hits)], [int(con_hits), int(con_seen - con_hits)]]
            rows.append((word, lib_hits / lib_seen, con_hits / con_seen, int(lib_seen), int(con_seen), fisher_exact(table)[1]))
    table = pd.DataFrame(rows, columns=["word", "liberal_rate", "conservative_rate", "liberal_seen", "conservative_seen", "p"])
    table["difference"] = table["liberal_rate"] - table["conservative_rate"]
    table["p_bonferroni"] = (table["p"] * len(table)).clip(upper=1.0)
    return table.sort_values("p").reset_index(drop=True)
