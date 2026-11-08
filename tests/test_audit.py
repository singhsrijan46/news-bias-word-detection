import importlib.util
import os

import numpy as np

from src.models.onnx_token_model import word_spans

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
spec = importlib.util.spec_from_file_location("audit_models", os.path.join(SCRIPTS, "audit_models.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_counterfactual_swaps_one_side_and_keeps_case():
    assert audit.counterfactual("Democrats blocked the bill.") == ("Republicans blocked the bill.", "left")
    assert audit.counterfactual("Republican senators praised Trump's pro-life stance.") == (
        "Democratic senators praised Biden's pro-choice stance.", "right")
    assert audit.counterfactual("LIBERAL critics left.") == ("CONSERVATIVE critics left.", "left")


def test_counterfactual_skips_sentences_with_both_sides_or_neither():
    assert audit.counterfactual("Trump and Biden debated.") is None
    assert audit.counterfactual("The weather was fine.") is None


class TableModel:
    """Scores words from a table; anything else gets 0.1."""

    def __init__(self, table):
        self.table = table

    def score(self, text):
        spans = word_spans(text)
        return spans, np.array([self.table.get(text[a:b].lower(), 0.1) for a, b in spans], dtype=np.float32)


def test_swap_audit_separates_pair_words_from_other_words():
    model = TableModel({"republicans": 0.5, "democrats": 0.3})
    pairs, shifts = audit.swap_audit({"m": model}, ["Democrats blocked the bill."], {"m": 0.4})
    assert list(pairs.left_word) == ["Democrats"] and list(pairs.right_word) == ["Republicans"]
    assert pairs.share_right.iloc[0] - pairs.share_left.iloc[0] > 0.19
    assert shifts.abs_shift.iloc[0] == 0.0 and shifts.flag_flips.iloc[0] == 0
