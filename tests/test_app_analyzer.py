import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "bias-lens", "python"))

from analyzer import AGREED_SHARE, GAP_SHOWN, Analyzer, Perspective, band, heat_strip, sentence_spans  # noqa: E402
from src.models.onnx_token_model import word_spans  # noqa: E402

SHARES = {"slammed": 0.72, "reckless": 0.45, "shameful": 0.9}


class FakeModel:
    """Scores words from a fixed table, like OnnxTokenModel.score."""

    def score(self, text):
        spans = word_spans(text)
        return spans, np.array([SHARES.get(text[a:b].lower(), 0.02) for a, b in spans], dtype=np.float32)


def test_sentence_spans_split_lines_and_sentences_and_trim_spaces():
    text = "Headline here\n  First one.  Second one? \"Third.\" Fourth\n\n"
    pieces = [text[a:b] for a, b in sentence_spans(text)]
    assert pieces == ["Headline here", "First one.", "Second one?", '"Third."', "Fourth"]


def test_bands_follow_the_threshold_and_the_agreed_share():
    assert band(0.1, threshold=0.3) == "none"
    assert band(0.3, threshold=0.3) == "contested"
    assert band(AGREED_SHARE, threshold=0.3) == "agreed"


def test_heat_strip_fills_13_columns_even_for_short_texts():
    strip = heat_strip([0.9, 0.1], threshold=0.3)
    assert len(strip) == 13
    assert strip[0] == 8 and strip[-1] == 0  # first sentence fully flagged, second not at all
    assert heat_strip([], threshold=0.3) == [0] * 13
    assert heat_strip([0.3] * 26, threshold=0.3) == [3] * 13  # 0.3 of 8 rows, rounded up


def test_analyze_marks_words_at_their_place_in_the_text():
    analyzer = Analyzer(FakeModel(), threshold=0.3, lexicon={"reckless"})
    text = "Critics slammed the plan.\nIt was reckless and shameful."
    result = analyzer.analyze(text)
    flagged = [w for s in result["sentences"] for w in s["words"] if w["band"] != "none"]
    assert [text[w["start"]:w["end"]] for w in flagged] == ["slammed", "reckless", "shameful"]
    assert [w["band"] for w in flagged] == ["agreed", "contested", "agreed"]
    assert [w["lexicon"] for w in flagged] == [False, True, False]
    summary = result["summary"]
    assert (summary["sentences"], summary["words"], summary["flagged"]) == (2, 9, 3)
    assert summary["top"][0] == {"text": "shameful", "share": pytest.approx(0.9)}
    assert len(summary["strip"]) == 13


def test_analyze_leaves_unflagged_low_scores_out_of_the_response():
    result = Analyzer(FakeModel(), threshold=0.3).analyze("Rates rose today.")
    assert result["sentences"][0]["words"] == []
    assert result["summary"]["flagged"] == 0


# all annotators, liberal ones, conservative ones
GROUP_SHARES = {"slammed": (0.7, 0.8, 0.5), "reckless": (0.45, 0.2, 0.6), "shameful": (0.9, 0.9, 0.85)}


class FakeGroupModel:
    """Scores words from a fixed table, like OnnxTokenModel.score_heads with M3's three outputs."""

    def score_heads(self, text):
        spans = word_spans(text)
        rows = [GROUP_SHARES.get(text[a:b].lower(), (0.02, 0.02, 0.02)) for a, b in spans]
        return spans, np.array(rows, dtype=np.float32)


def test_perspective_lists_the_words_each_group_would_flag_more():
    view = Perspective("unused", threshold=0.3, model=FakeGroupModel())
    assert view.available
    text = "Critics slammed the plan.\nIt was reckless and shameful."
    result = view.analyze(text)
    assert [text[w["start"]:w["end"]] for w in result["words"]] == ["slammed", "reckless", "shameful"]
    assert [w["text"] for w in result["liberal_more"]] == ["slammed"]  # 0.8 against 0.5
    assert [w["text"] for w in result["conservative_more"]] == ["reckless"]  # 0.6 against 0.2
    assert abs(0.9 - 0.85) < GAP_SHOWN  # so "shameful" is in neither list
    assert len(result["strip"]) == 13


def test_perspective_is_unavailable_without_its_model(tmp_path):
    assert not Perspective(str(tmp_path), threshold=0.3).available
