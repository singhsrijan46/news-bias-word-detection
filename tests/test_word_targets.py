import numpy as np
import pandas as pd
import pytest

from src.data.word_targets import align_paper_words, build_sentence_targets, first_subword_positions
from src.evaluation.word_metrics import (
    best_f1_threshold, calibration_metrics, expected_calibration_error, paired_comparison, per_fold_metrics,
)


def toy_annotations() -> pd.DataFrame:
    return pd.DataFrame({
        "text": ["They swarm the border."] * 3,
        "type": ["right"] * 3,
        "survey_record_id": ["a", "b", "c"],
        "political_ideology": [-5, 0, 6],
        "words": ["swarm", "swarm, border", None],
        "label": ["Biased", "Biased", "Non-biased"],
    })


def test_targets_count_votes_and_keep_character_spans():
    (sentence,) = build_sentence_targets(toy_annotations())
    assert sentence.words == ["They", "swarm", "the", "border"]
    assert sentence.spans[1] == (5, 10)
    assert list(sentence.votes) == [0, 2, 0, 1]
    assert sentence.shares == pytest.approx([0, 2 / 3, 0, 1 / 3])
    assert list(sentence.labels(min_votes=2)) == [0, 1, 0, 0]


def test_first_subword_skips_special_tokens_and_marks_truncated_words():
    words = [(0, 4), (5, 13), (14, 20)]  # "They", "swarming", "border"
    offsets = [(0, 0), (0, 4), (5, 10), (10, 13), (0, 0)]  # <s> They swarm ing </s>: "border" was cut off
    assert list(first_subword_positions(words, offsets)) == [1, 2, -1]


def test_first_subword_handles_words_split_by_punctuation():
    words = [(0, 1), (2, 3)]  # "U" and "S" in "U.S."
    offsets = [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (0, 0)]
    assert list(first_subword_positions(words, offsets)) == [1, 3]


def test_align_paper_words_matches_spacy_pieces_in_order():
    text = "The year’s pro-life ruling hit the ruling party"
    annotations = pd.DataFrame({"text": [text], "type": ["left"], "survey_record_id": ["a"],
                                "political_ideology": [0], "words": [None], "label": ["Non-biased"]})
    targets = build_sentence_targets(annotations)  # words: The year’s pro-life ruling hit the ruling party
    dt = pd.DataFrame({
        "sentence": [text] * 6,
        "text_low": ["year", "pro", "life", "ruling", "ruling", "party"],
        "order": [1, 3, 5, 6, 9, 10],
        "label4": [0, 0, 1, 0, 0, 0],
    })
    aligned = align_paper_words(targets, dt)
    assert list(aligned.word_index) == [1, 2, 2, 3, 6, 7]
    assert list(aligned.label4) == [0, 0, 1, 0, 0, 0]


def test_best_f1_threshold_separates_clean_scores():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert 0.3 < best_f1_threshold(y, scores) <= 0.8


def test_calibration_is_zero_when_predictions_equal_shares():
    shares = np.array([0.0, 0.1, 0.4, 0.4, 0.9])
    assert expected_calibration_error(shares, shares) == pytest.approx(0.0)
    metrics = calibration_metrics((shares >= 0.4).astype(int), shares, shares)
    assert metrics["share_mse"] == pytest.approx(0.0)
    assert metrics["spearman_vs_share"] == pytest.approx(1.0)


def test_paired_comparison_prefers_the_model_that_tracks_vote_shares():
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(4):
        shares = rng.uniform(0, 1, 200)
        labels = (shares >= 0.4).astype(int)
        noisy = np.clip(shares + rng.normal(0, 0.3, 200), 0, 1)
        rows.append(pd.DataFrame({"fold": fold, "label4": labels, "share": shares,
                                  "a": noisy, "a_flag": (noisy >= 0.4).astype(int),
                                  "b": shares, "b_flag": labels}))
    per_fold = per_fold_metrics(pd.concat(rows), ["a", "b"])
    table = paired_comparison(per_fold, "a", "b").set_index("metric")
    assert table.loc["f1", "difference"] > 0
    assert table.loc["share_ece", "difference"] < 0
    assert table.loc["spearman_vs_share", "folds_b_higher"] == 4
