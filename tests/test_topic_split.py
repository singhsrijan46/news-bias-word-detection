import importlib.util
import os

import pandas as pd

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
spec = importlib.util.spec_from_file_location("train_token_models", os.path.join(SCRIPTS, "train_token_models.py"))
token_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(token_models)


def test_topic_folds_hold_out_one_topic_each_in_sentence_order():
    # One row per annotator judgement; sentences are numbered by first appearance, as in build_sentence_targets.
    annotations = pd.DataFrame({"text": ["b", "a", "b", "c", "a"],
                                "topic": ["Sport", "vaccines", "Sport", "sport ", "vaccines"]})
    folds, names = token_models.topic_folds(annotations)
    assert names == ["sport", "vaccines"]
    assert folds.tolist() == [0, 1, 0]  # sentences b, a, c


def test_random_folds_are_balanced_and_seeded():
    a = token_models.make_folds(25, 10, seed=0)
    assert sorted(set(a)) == list(range(10)) and max(list(a).count(k) for k in range(10)) == 3
    assert (a == token_models.make_folds(25, 10, seed=0)).all()
