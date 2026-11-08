import importlib.util
import os

import numpy as np
import pandas as pd

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
spec = importlib.util.spec_from_file_location("evaluate_babe", os.path.join(SCRIPTS, "evaluate_babe.py"))
babe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(babe)


def test_word_lists_are_parsed_and_missing_ones_stay_missing():
    assert babe.parse_word_list("['radical', 'far-left']") == ["radical", "far-left"]
    assert babe.parse_word_list("[]") == []
    assert babe.parse_word_list(None) is None
    assert babe.parse_word_list(np.array(["mob"])) == ["mob"]


def test_topic_and_outlet_spellings_are_merged():
    assert babe.normalize_topic("Black Lives Matter") == babe.normalize_topic("blm") == "blm"
    assert babe.normalize_topic("gun control") == "gun-control"
    assert babe.normalize_topic("vaccine") == "vaccines"
    assert babe.normalize_outlet("The Federalist") == babe.normalize_outlet("Federalist")
    assert babe.normalize_outlet("HuffPost") == babe.normalize_outlet("Huffpost")


def test_expert_words_are_matched_to_the_sentence_words():
    text = "The radical far-left mob attacked the mayor."
    assert babe.expert_labels(text, ["radical", "far-left"]).tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert babe.expert_labels("What the hell is this", ["what", "the", "hell"]).tolist() == [1, 1, 1, 0, 0]
    assert babe.expert_labels("Nothing here.", []).tolist() == [0, 0]


def test_sentence_score_is_the_highest_word_score_ignoring_truncated_words():
    assert babe.sentence_score(np.array([0.1, np.nan, 0.7], dtype=np.float32)) == np.float32(0.7)
    assert np.isnan(babe.sentence_score(np.array([np.nan])))


def test_mbic_sentences_are_split_off_despite_quotes_case_and_spaces():
    mbic = pd.DataFrame({"sentence": ["Trump’s “wall” is  rising.",
                                      "A long sentence that BABE copied from MBIC but then edited slightly at the end."]})
    data = pd.DataFrame({"text": ["trump's \"wall\" is rising.",
                                  "A long sentence that BABE copied from MBIC but then edited slightly in the end.",
                                  "A new sentence.", "A new sentence."]})
    overlap, new, near = babe.split_by_mbic(data, mbic)
    assert overlap.text.tolist() == ["trump's \"wall\" is rising."]
    assert new.text.tolist() == ["A new sentence."]  # the repeat is dropped
    assert near == 1


def test_metrics_table_reports_the_threshold_and_the_best_f1():
    table = pd.DataFrame({"label": [0, 0, 1, 1, 0, 1], "m": [0.1, 0.5, 0.6, 0.9, 0.2, 0.3]})
    row = babe.metrics_table(table, ["m"], {"m": 0.55}).iloc[0]
    assert row.n == 6 and row.positive_share == 0.5
    assert row.precision == 1.0 and abs(row.recall - 2 / 3) < 1e-9
    assert row.best_f1 >= row.f1
