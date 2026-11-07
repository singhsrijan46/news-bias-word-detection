"""
Unlabeled sentences from MBIC's full articles, for distilling a teacher into a small student.

MBIC's 1,700 labeled sentences come from 918 full news articles, which hold about
31,000 more sentences that nobody annotated. A teacher model can score those, and a
small student can learn from the teacher's scores as well as from the annotators'
own vote shares.
"""

import re
from typing import Dict, List, Tuple

import pandas as pd

# A sentence ends at . ! or ?, optionally followed by a closing quote or bracket.
SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)]))\s+")


def normalize(text) -> str:
    """Collapse runs of whitespace, so the same sentence always compares equal."""
    return " ".join(str(text).split())


def split_sentences(text: str) -> List[str]:
    """Split article text into sentences: first by line, then at sentence-ending punctuation."""
    sentences: List[str] = []
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            sentences.extend(s.strip() for s in SENTENCE_END.split(line) if s.strip())
    return sentences


def article_ids(labeled: pd.DataFrame) -> Dict[str, int]:
    """A number for every distinct full article in labeled_dataset."""
    articles = labeled["article"].dropna().astype(str).map(normalize).drop_duplicates()
    return {text: i for i, text in enumerate(articles)}


def sentence_articles(labeled: pd.DataFrame) -> Dict[str, int]:
    """The article each labeled sentence came from (-1 when its article is missing)."""
    ids = article_ids(labeled)
    return {normalize(row.sentence): ids.get(normalize(row.article), -1) if isinstance(row.article, str) else -1
            for row in labeled.itertuples()}


def article_pool(labeled: pd.DataFrame, min_words: int = 6, max_words: int = 80) -> pd.DataFrame:
    """
    One row per unlabeled article sentence: article_id and sentence.

    Sentences identical to a labeled MBIC sentence are left out, as are repeats and
    lines too short or too long to be ordinary sentences (headlines, captions, lists).
    """
    labeled_texts = {normalize(s) for s in labeled["sentence"]}
    rows, seen = [], set()
    for text, article_id in article_ids(labeled).items():
        for sentence in split_sentences(text):
            key = normalize(sentence)
            if key in labeled_texts or key in seen or not min_words <= len(key.split()) <= max_words:
                continue
            seen.add(key)
            rows.append({"article_id": article_id, "sentence": key})
    return pd.DataFrame(rows)
