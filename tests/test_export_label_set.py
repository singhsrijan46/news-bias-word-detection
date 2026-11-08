import importlib.util
import os

import pandas as pd

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
spec = importlib.util.spec_from_file_location("export_label_set", os.path.join(SCRIPTS, "export_label_set.py"))
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_votes_are_counted_overall_and_by_ideology_group():
    annotations = pd.DataFrame({
        "text": ["Radical mob attacked.", "Radical mob attacked.", "Radical mob attacked.", "Calm day."],
        "survey_record_id": ["a", "b", "c", "a"],
        "political_ideology": [-8, 0, 7, -8],  # liberal, centre, conservative
        "words": ["radical, mob", "mob", float("nan"), float("nan")],
        "label": ["Biased", "Biased", "Non-biased", "Non-biased"],
        "type": ["left", "left", "left", "center"],
        "topic": ["gun-control", "gun-control", "gun-control", "sport"],
        "outlet": ["alternet", "alternet", "alternet", "reuters"],
    })
    sentences, words = export.label_set(annotations)
    first = sentences.iloc[0]
    assert (first.annotators, first.annotators_liberal, first.annotators_centre, first.annotators_conservative) == (3, 1, 1, 1)
    assert first.said_biased == 2 and first.outlet == "alternet" and first.topic == "gun-control"
    s0 = words[words.sentence_id == 0].set_index("word")
    assert s0.loc["Radical", ["votes", "votes_liberal", "votes_centre", "votes_conservative"]].tolist() == [1, 1, 0, 0]
    assert s0.loc["mob", ["votes", "votes_liberal", "votes_centre", "votes_conservative"]].tolist() == [2, 1, 1, 0]
    assert s0.loc["attacked", "votes"] == 0
    assert (s0.loc["mob", "start"], s0.loc["mob", "end"]) == (8, 11)
    assert words[words.sentence_id == 1].votes.sum() == 0
