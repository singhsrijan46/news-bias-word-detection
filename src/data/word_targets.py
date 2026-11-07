"""
Per-word targets for token models, rebuilt from MBIC's per-annotator annotations.

For every sentence this keeps the words (the same tokens as
src.data.annotations, with their character spans in the original text), how
many annotators highlighted each word, and how many annotators judged the
sentence. Two training targets follow: the paper's hard label (at least 4 votes)
and the vote share (the soft label).

It also lines up the paper's word-level dataset (dt_final: 31,794 non-stop
words) with these words, so token models can be scored on exactly the paper's
words and labels.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.annotations import TOKEN_PATTERN, rebuild_sentence_annotations


@dataclass
class SentenceTargets:
    """One sentence's words and how many of its annotators highlighted each."""

    sentence_id: int
    text: str
    outlet_type: str
    spans: List[Tuple[int, int]]  # character span of each word in `text`
    votes: np.ndarray  # annotators who highlighted each word
    n_annotators: int

    @property
    def words(self) -> List[str]:
        return [self.text[a:b] for a, b in self.spans]

    @property
    def shares(self) -> np.ndarray:
        """Vote share of each word: the soft label."""
        return self.votes / self.n_annotators

    def labels(self, min_votes: int = 4) -> np.ndarray:
        """The paper's hard label: 1 if at least `min_votes` annotators highlighted the word."""
        return (self.votes >= min_votes).astype(np.int64)


def build_sentence_targets(annotations: pd.DataFrame) -> List[SentenceTargets]:
    """Targets for every sentence, in order of first appearance in the annotations."""
    targets = []
    for i, sentence in enumerate(rebuild_sentence_annotations(annotations)):
        spans = [m.span() for m in TOKEN_PATTERN.finditer(sentence.text)]
        counts = sentence.vote_counts()
        votes = np.array([counts.get(k, 0) for k in range(len(spans))], dtype=np.int64)
        targets.append(SentenceTargets(i, sentence.text, sentence.outlet_type, spans, votes, sentence.n_annotators))
    return targets


def first_subword_positions(word_spans: Sequence[Tuple[int, int]], token_offsets: Sequence[Tuple[int, int]]) -> np.ndarray:
    """
    Index of the first sub-word token that overlaps each word, or -1 if truncation removed it.

    `token_offsets` are a fast tokenizer's character offsets; special tokens have
    empty offsets such as (0, 0) and are skipped.
    """
    positions = np.full(len(word_spans), -1, dtype=np.int64)
    t, n = 0, len(token_offsets)
    for w, (start, end) in enumerate(word_spans):
        while t < n and (token_offsets[t][1] <= start or token_offsets[t][0] == token_offsets[t][1]):
            t += 1
        if t < n and token_offsets[t][0] < end:
            positions[w] = t
    return positions


def align_paper_words(targets: List[SentenceTargets], dt_final: pd.DataFrame) -> pd.DataFrame:
    """
    Find each dt_final word among its sentence's words.

    dt_final was tokenized by spaCy, which splits some words differently: "year’s"
    becomes "year" and "’s", and "pro-life" becomes "pro", "-" and "life". Words
    are matched in sentence order. An exact match moves past the word; a word that
    is only part of a longer token ("year" in "year’s") can share that token with
    the next piece.

    Returns one row per dt_final row: its index, sentence_id, word_index (-1 if not
    found) and the paper's label4.
    """
    by_text = {t.text: t for t in targets}
    records = []
    for sentence, group in dt_final.groupby("sentence", sort=False):
        target = by_text.get(sentence)
        words = [w.lower() for w in target.words] if target is not None else []
        cursor = 0
        for index, row in group.sort_values("order").iterrows():
            word = str(row["text_low"]).lower()
            hit = -1
            for j in range(cursor, len(words)):
                if words[j] == word:
                    hit, cursor = j, j + 1
                    break
                if len(word) > 1 and word in words[j]:
                    hit, cursor = j, j
                    break
            records.append({
                "dt_index": index,
                "sentence_id": target.sentence_id if target is not None else -1,
                "word_index": hit,
                "label4": int(row["label4"]),
            })
    return pd.DataFrame(records)
