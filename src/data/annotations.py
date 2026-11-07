"""
Rebuild per-annotator, word-level bias highlights from MBIC's raw annotations.

The released labels (`biased_words4`) keep only the words that at least four of a
sentence's ~10 annotators highlighted. The raw file (`data/raw/annotations.xlsx`)
has one row per annotator per sentence, with that annotator's highlights as free
text in `words` (comma-separated words or phrases). This module maps those
highlights back onto token positions, so every annotator's judgement can be used:
as soft labels (vote shares), per ideology group, or per annotator.
"""

import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set

import pandas as pd

# Words, keeping inner apostrophes and hyphens ("isn't", "pro-life").
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
# Annotators separated highlighted words and phrases with commas (sometimes ; or newlines).
PHRASE_SEPARATORS = re.compile(r"[,;\n]")


def tokenize(text) -> List[str]:
    """Lower-cased word tokens of a sentence or a highlighted phrase."""
    return [m.group(0).lower() for m in TOKEN_PATTERN.finditer(str(text))]


def _find_run(tokens: List[str], run: List[str]) -> int:
    """Start index of the first occurrence of `run` as consecutive tokens, or -1."""
    for i in range(len(tokens) - len(run) + 1):
        if tokens[i:i + len(run)] == run:
            return i
    return -1


def match_highlights(tokens: List[str], words) -> Set[int]:
    """
    Map one annotator's free-text highlights onto token positions of a sentence.

    Each phrase is matched as a run of consecutive tokens (first occurrence). If it
    does not occur as a run, each of its tokens is matched on its own (first
    occurrence). Missing highlights (NaN) give an empty set.
    """
    positions: Set[int] = set()
    if not isinstance(words, str):
        return positions
    for phrase in PHRASE_SEPARATORS.split(words):
        phrase_tokens = tokenize(phrase)
        if not phrase_tokens:
            continue
        start = _find_run(tokens, phrase_tokens)
        if start >= 0:
            positions.update(range(start, start + len(phrase_tokens)))
        else:
            positions.update(tokens.index(w) for w in phrase_tokens if w in tokens)
    return positions


@dataclass(frozen=True)
class Judgement:
    """One annotator's judgement of one sentence."""

    annotator_id: str
    ideology: float  # -10 (liberal) to +10 (conservative)
    positions: FrozenSet[int]  # token positions the annotator highlighted
    says_biased: bool  # the annotator's sentence-level label


@dataclass
class SentenceAnnotations:
    """All annotators' judgements of one MBIC sentence."""

    text: str
    tokens: List[str]
    outlet_type: str  # left / center / right, as rated by AllSides
    judgements: List[Judgement] = field(default_factory=list)

    @property
    def n_annotators(self) -> int:
        return len(self.judgements)

    def vote_counts(self) -> Counter:
        """Number of annotators who highlighted each token position."""
        return Counter(p for j in self.judgements for p in j.positions)

    def vote_shares(self) -> List[float]:
        """Share of this sentence's annotators who highlighted each token (a soft label)."""
        counts = self.vote_counts()
        return [counts.get(i, 0) / self.n_annotators for i in range(len(self.tokens))]

    def biased_words(self, min_votes: int = 4) -> Set[str]:
        """Words highlighted by at least `min_votes` annotators (the paper uses 4)."""
        return {self.tokens[p] for p, k in self.vote_counts().items() if k >= min_votes}


def rebuild_sentence_annotations(annotations: pd.DataFrame) -> List[SentenceAnnotations]:
    """Group the raw per-annotator rows by sentence, in order of first appearance."""
    sentences = []
    for text, group in annotations.groupby("text", sort=False):
        tokens = tokenize(text)
        sentence = SentenceAnnotations(text=text, tokens=tokens, outlet_type=group["type"].iloc[0])
        for row in group.itertuples(index=False):
            sentence.judgements.append(
                Judgement(
                    annotator_id=row.survey_record_id,
                    ideology=float(row.political_ideology),
                    positions=frozenset(match_highlights(tokens, row.words)),
                    says_biased=row.label == "Biased",
                )
            )
        sentences.append(sentence)
    return sentences


def compare_with_published(
    sentences: List[SentenceAnnotations], labeled: pd.DataFrame, min_votes: int = 4
) -> Dict[str, float]:
    """
    Compare rebuilt labels with the released `biased_words4` lists of `labeled_dataset`.

    Multi-word entries in the released lists are skipped: the rebuilt labels are per token.
    """
    published = {
        row.sentence: {w.lower() for w in ast.literal_eval(row.biased_words4)}
        for row in labeled.itertuples()
    }
    exact, jaccard, tp, fp, fn = 0, 0.0, 0, 0, 0
    for s in sentences:
        rebuilt = s.biased_words(min_votes)
        reference = {w for w in published[s.text] if " " not in w}
        union = rebuilt | reference
        exact += rebuilt == reference
        jaccard += len(rebuilt & reference) / len(union) if union else 1.0
        tp += len(rebuilt & reference)
        fp += len(rebuilt - reference)
        fn += len(reference - rebuilt)
    return {
        "sentences": len(sentences),
        "exact_matches": exact,
        "mean_jaccard": jaccard / len(sentences),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
    }


def annotator_ideology(annotations: pd.DataFrame) -> pd.Series:
    """Each annotator's ideology score, indexed by survey record, in order of first appearance."""
    first_rows = annotations.drop_duplicates("survey_record_id").set_index("survey_record_id")
    return first_rows["political_ideology"].astype(float)


def ideology_groups(ideology: pd.Series, centre_band: int = 3) -> Dict[str, str]:
    """
    Label annotators liberal, centre or conservative from the -10..10 scale.

    Negative scores are liberal. `centre_band=3` (centre = -3..3) reproduces the
    paper's split of 44.3% liberal, 29.1% centre and 26.7% conservative;
    `centre_band=0` puts everyone except exact zeros on a side.
    """
    return {
        annotator: "liberal" if score < -centre_band else "conservative" if score > centre_band else "centre"
        for annotator, score in ideology.items()
    }
