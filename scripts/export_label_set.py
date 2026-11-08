#!/usr/bin/env python
"""
Export the project's label set: MBIC's word-level vote counts, overall and by annotator ideology.

The paper turns ten crowd judgements per sentence into one yes/no label per word. This keeps the
counts behind that label, so that a word's vote share, and each ideology group's share, can be
computed directly. It is the proposal's first deliverable, described in docs/DATASHEET.md.

Two tables, in data/derived/:
  mbic_sentences.csv       one row per sentence: its text, outlet, the outlet's leaning, topic,
                           and how many annotators of each group judged it
  mbic_word_votes.csv.gz   one row per word: its position, and how many annotators of each
                           group highlighted it

Groups follow the paper: liberal for a self-reported ideology from -10 to -4, centre from -3 to 3,
conservative from 4 to 10. Only counts are released, never individual annotators' judgements.

Usage:
    python scripts/export_label_set.py
"""

import argparse
import os
import sys
from typing import Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.annotations import TOKEN_PATTERN, annotator_ideology, ideology_groups, rebuild_sentence_annotations  # noqa: E402
from src.data.dataset_loader import MBICDataLoader  # noqa: E402

GROUPS = ("liberal", "centre", "conservative")


def label_set(annotations: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The sentence table and the word table, from MBIC's raw per-annotator rows."""
    groups = ideology_groups(annotator_ideology(annotations), centre_band=3)
    first = annotations.groupby("text", sort=False).first()
    biased = annotations.assign(biased=annotations.label == "Biased").groupby("text", sort=False).biased.sum()
    sentence_rows, word_rows = [], []
    for sid, sentence in enumerate(rebuild_sentence_annotations(annotations)):
        meta = first.loc[sentence.text]
        judges = {g: [j for j in sentence.judgements if groups.get(j.annotator_id) == g] for g in GROUPS}
        sentence_rows.append({"sentence_id": sid, "text": sentence.text, "outlet": meta["outlet"],
                              "outlet_leaning": sentence.outlet_type, "topic": meta["topic"],
                              "annotators": sentence.n_annotators,
                              **{f"annotators_{g}": len(judges[g]) for g in GROUPS},
                              "said_biased": int(biased.loc[sentence.text])})
        for w, match in enumerate(TOKEN_PATTERN.finditer(sentence.text)):
            word_rows.append({"sentence_id": sid, "word_index": w, "word": match.group(0),
                              "start": match.start(), "end": match.end(),
                              "votes": sum(w in j.positions for j in sentence.judgements),
                              **{f"votes_{g}": sum(w in j.positions for j in judges[g]) for g in GROUPS}})
    return pd.DataFrame(sentence_rows), pd.DataFrame(word_rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(ROOT, "data", "derived"))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    sentences, words = label_set(MBICDataLoader().load_annotations())
    sentences.to_csv(os.path.join(args.out, "mbic_sentences.csv"), index=False)
    words.to_csv(os.path.join(args.out, "mbic_word_votes.csv.gz"), index=False,
                 compression={"method": "gzip", "mtime": 0})  # mtime 0: the same data gives the same file
    shares = words.votes / words.sentence_id.map(sentences.set_index("sentence_id").annotators)
    print(f"{len(sentences)} sentences and {len(words)} words; {int((words.votes >= 4).sum())} words with at least "
          f"4 votes ({(words.votes >= 4).mean():.1%}); {int((words.votes > 0).sum())} highlighted by anyone; "
          f"mean share {shares.mean():.3f}")


if __name__ == "__main__":
    main()
