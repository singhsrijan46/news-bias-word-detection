import pandas as pd
import pytest

from src.analysis.perspective import contestedness_distribution, word_rate_differences
from src.data.annotations import (
    annotator_ideology,
    compare_with_published,
    ideology_groups,
    match_highlights,
    rebuild_sentence_annotations,
    tokenize,
)
from src.data.dataset_loader import MBICDataLoader


def toy_annotations() -> pd.DataFrame:
    """Two sentences: three annotators on the first, two on the second."""
    return pd.DataFrame(
        {
            "text": ["They swarm the border."] * 3 + ["Rates rose."] * 2,
            "type": ["right"] * 3 + ["center"] * 2,
            "survey_record_id": ["a", "b", "c", "a", "b"],
            "political_ideology": [-5, 0, 6, -5, 0],
            "words": ["swarm", "swarm, border", None, None, "rose"],
            "label": ["Biased", "Biased", "Non-biased", "Non-biased", "Biased"],
        }
    )


def test_tokenize_keeps_inner_apostrophes_and_hyphens():
    assert tokenize("Pro-life judges' ruling isn't final.") == ["pro-life", "judges", "ruling", "isn't", "final"]


def test_match_highlights_prefers_a_consecutive_phrase():
    tokens = tokenize("the radical left and the radical right")
    assert match_highlights(tokens, "radical right") == {5, 6}


def test_match_highlights_falls_back_to_single_words():
    tokens = tokenize("a shameful and corrupt deal")
    assert match_highlights(tokens, "shameful corrupt") == {1, 3}


def test_match_highlights_splits_phrases_and_handles_missing_values():
    tokens = tokenize("a shameful and corrupt deal")
    assert match_highlights(tokens, "shameful , corrupt,") == {1, 3}
    assert match_highlights(tokens, float("nan")) == set()


def test_rebuild_counts_votes_per_token():
    first, second = rebuild_sentence_annotations(toy_annotations())
    assert first.tokens == ["they", "swarm", "the", "border"]
    assert first.n_annotators == 3
    assert first.vote_counts()[1] == 2
    assert first.vote_shares() == pytest.approx([0, 2 / 3, 0, 1 / 3])
    assert first.biased_words(min_votes=2) == {"swarm"}
    assert second.outlet_type == "center"
    assert [j.says_biased for j in second.judgements] == [False, True]


def test_ideology_groups_follow_the_papers_banding():
    groups = ideology_groups(pd.Series({"a": -4.0, "b": -3.0, "c": 3.0, "d": 4.0}), centre_band=3)
    assert groups == {"a": "liberal", "b": "centre", "c": "centre", "d": "conservative"}


def test_annotator_ideology_keeps_first_appearance_order():
    ideology = annotator_ideology(toy_annotations())
    assert list(ideology.index) == ["a", "b", "c"]
    assert list(ideology) == [-5.0, 0.0, 6.0]


def test_contestedness_bands_cover_every_highlighted_token():
    result = contestedness_distribution(rebuild_sentence_annotations(toy_annotations()))
    assert result["highlighted_tokens"] == 3  # swarm (2/3), border (1/3), rose (1/2)
    assert sum(b["tokens"] for b in result["bins"]) == 3
    assert sum(b["share"] for b in result["bins"]) == pytest.approx(1.0)


def test_word_rate_differences_reports_rates_per_side():
    sentences = rebuild_sentence_annotations(toy_annotations())
    groups = ideology_groups(annotator_ideology(toy_annotations()), centre_band=0)
    table = word_rate_differences(sentences, groups, min_seen=1, min_highlights=1, min_length=3)
    swarm = table.set_index("word").loc["swarm"]
    assert swarm["liberal_rate"] == 1.0 and swarm["conservative_rate"] == 0.0


def test_rebuilt_labels_reproduce_the_published_labels():
    loader = MBICDataLoader()
    sentences = rebuild_sentence_annotations(loader.load_annotations())
    result = compare_with_published(sentences, loader.load_raw_sentences())
    assert result["sentences"] == 1700
    assert result["mean_jaccard"] > 0.95
    assert result["precision"] > 0.95 and result["recall"] > 0.95
