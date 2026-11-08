import importlib.util
import os

import numpy as np

EDGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "edge")
spec = importlib.util.spec_from_file_location("check_int8_on_board", os.path.join(EDGE, "check_int8_on_board.py"))
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_scores_are_flattened_with_missing_words_as_nan():
    flat = check.flatten([[0.1, None], [0.5]])
    assert np.isnan(flat[1]) and flat[0] == 0.1 and flat[2] == 0.5


def test_bands_follow_the_app():
    assert check.band(np.array([0.1, 0.3, 0.7]), 0.267).tolist() == [0, 1, 2]


def test_compare_counts_flags_and_bands_that_change():
    a = np.array([0.1, 0.26, 0.5, 0.61, np.nan])
    b = np.array([0.1, 0.27, 0.5, 0.59, 0.4])
    result = check.compare(a, b, 0.267)
    assert result["words"] == 4  # the NaN word is left out
    assert result["identical_share"] == 0.5
    assert result["flag_changes"] == 1  # 0.26 -> 0.27 crosses the threshold
    assert result["band_changes"] == 2  # and 0.61 -> 0.59 leaves "most readers"


def test_the_board_script_scores_every_word():
    from score_sentences import score_all

    class Model:
        def score(self, text):
            return None, np.array([0.2, np.nan], dtype=np.float32)

    assert score_all(Model(), ["a b"]) == [[np.float32(0.2).item(), None]]
