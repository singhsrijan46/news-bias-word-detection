import numpy as np
import pandas as pd
import pytest

from src.models.perspective_model import build_group_targets, group_log_loss, summarize_groups


def toy_annotations() -> pd.DataFrame:
    """One sentence judged by a liberal (-5), a centrist (0) and a conservative (6)."""
    return pd.DataFrame({
        "text": ["They swarm the border."] * 3,
        "type": ["right"] * 3,
        "survey_record_id": ["a", "b", "c"],
        "political_ideology": [-5, 0, 6],
        "words": ["swarm", "swarm, border", None],
        "label": ["Biased", "Biased", "Non-biased"],
    })


def test_group_targets_count_votes_per_ideology_group():
    (group,) = build_group_targets(toy_annotations())
    assert list(group.judges) == [3, 1, 1]  # all, liberal, conservative
    assert list(group.votes[1]) == [2, 1, 0]  # "swarm": two of all, the liberal, not the conservative
    assert group.shares()[1] == pytest.approx([2 / 3, 1.0, 0.0])
    assert summarize_groups([group])["sentences_with_both_groups"] == 1


def test_group_log_loss_rewards_matching_probabilities():
    votes = np.array([3.0, 0.0])
    good = group_log_loss(votes, 4, np.array([0.75, 0.01]))
    bad = group_log_loss(votes, 4, np.array([0.1, 0.9]))
    assert good < bad
