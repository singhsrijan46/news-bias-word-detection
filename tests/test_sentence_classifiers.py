import importlib.util
import os

import pandas as pd

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
spec = importlib.util.spec_from_file_location("audit_sentence_classifiers",
                                              os.path.join(SCRIPTS, "audit_sentence_classifiers.py"))
classifiers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classifiers)


def test_the_biased_output_is_found_by_its_label_or_defaults_to_the_second():
    assert classifiers.biased_index({0: "neutral", 1: "biased"}) == 1
    assert classifiers.biased_index({"0": "Biased", "1": "Non-biased"}) == 0
    assert classifiers.biased_index({0: "LABEL_0", 1: "LABEL_1"}) == 1
    assert classifiers.biased_index(None) == 1


def test_each_sentence_gets_its_pair_or_several():
    assert classifiers.pair_name("Biden spoke to reporters.") == "biden / trump"
    assert classifiers.pair_name("Biden and Pelosi spoke.") == "several"


def test_swap_pairs_puts_the_left_coded_version_first():
    swaps = classifiers.swap_pairs(["Democrats won.", "Trump spoke.", "Nothing here."])
    assert swaps.original_side.tolist() == ["left", "right"]
    assert swaps.left_text.tolist() == ["Democrats won.", "Biden spoke."]
    assert swaps.right_text.tolist() == ["Republicans won.", "Trump spoke."]


def test_originals_seen_in_babe_are_marked():
    swaps = classifiers.swap_pairs(["Democrats won.", "Trump spoke."])
    marked = classifiers.mark_seen_originals(swaps, {"trump spoke."})
    assert marked.original_in_babe.tolist() == [False, True]


def test_swap_summary_counts_one_sided_verdicts():
    swaps = pd.DataFrame({"m_left": [0.4, 0.6, 0.3], "m_right": [0.6, 0.6, 0.3]})
    row = classifiers.swap_summary(swaps, ["m"], {"m": 0.5}).iloc[0]
    assert abs(row.right_minus_left - 0.2 / 3) < 1e-9
    assert abs(row.decision_flips - 1 / 3) < 1e-9
    assert abs(row.flagged_right_only - 1 / 3) < 1e-9 and row.flagged_left_only == 0
