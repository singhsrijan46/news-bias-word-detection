import importlib.util
import os

import numpy as np
import pandas as pd

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
spec = importlib.util.spec_from_file_location("coverage_curves", os.path.join(SCRIPTS, "coverage_curves.py"))
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


def words(scores, flags, labels, shares, fold=0):
    return pd.DataFrame({"fold": fold, "m": scores, "m_flag": flags, "label4": labels, "share": shares})


def test_each_folds_threshold_is_recovered_from_its_flags():
    table = pd.concat([words([0.1, 0.3, 0.6, 0.8], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0.1, 0.5, 0.9], fold=0),
                       words([0.2, 0.25, 0.4], [0, 1, 1], [0, 1, 1], [0, 0.3, 0.7], fold=1)])
    t = coverage.fold_thresholds(table, "m")
    assert abs(t[0] - 0.45) < 1e-9 and abs(t[1] - 0.225) < 1e-9


def test_abstaining_drops_the_words_nearest_the_threshold():
    conf = np.array([0.05, 0.4, 0.01, 0.3])
    assert coverage.keep_most_certain(conf, 0.5).tolist() == [False, True, False, True]
    assert coverage.keep_most_certain(conf, 1.0).all()


def test_contested_means_two_to_five_votes_in_ten():
    assert coverage.is_contested(np.array([0.1, 0.2, 0.5, 0.6])).tolist() == [False, True, True, False]


def test_full_coverage_is_the_ordinary_f1_and_abstaining_removes_errors():
    # Two errors, both close to the threshold of 0.5: abstaining on them leaves a perfect model.
    table = words([0.05, 0.1, 0.45, 0.55, 0.9, 0.95], [0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1],
                  [0, 0, 0.4, 0.3, 0.8, 0.9])
    curve = coverage.curve(table, ["m"]).set_index("coverage")
    assert abs(curve.loc[1.0, "f1"] - 2 / 3) < 1e-9
    assert curve.loc[0.7, "f1"] == 1.0  # keeps 4 of the 6 words
    assert curve.loc[0.7, "contested_share_of_abstained"] == 1.0
