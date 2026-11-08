"""
Bias Lens analysis: split an article into sentences, score every word with the
on-board model, and summarise the result for the web page and the LED matrix.

A word's score is the model's estimate of the share of readers who would flag it
as biased. It was learned from MBIC's annotators: crowd workers in the US, about
ten per sentence, rating news sentences from 2017 to 2020. The score describes
how readers react to a word, not whether the word is right or wrong.
"""

import os
import re
import threading
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)]))\s+")
MAX_CHARS = 20000  # longest text accepted: about 3,000 words
MIN_SHARE = 0.05  # words scored below this are left out of the response
AGREED_SHARE = 0.6  # from here up, most readers would flag the word
STRIP_COLUMNS, STRIP_ROWS = 13, 8  # the UNO Q's LED matrix
GAP_SHOWN = 0.1  # the perspective view lists words whose two groups differ by at least this
GAPS_LISTED = 6  # words listed for each group
CPU = threading.Lock()  # one analysis at a time, by either model, so they do not compete for the CPU


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Character spans of the sentences: split by line, then at sentence-ending punctuation."""
    spans = []
    for line in re.finditer(r"[^\n]+", text):
        cut = [line.start()]
        cut += [line.start() + m.end() for m in SENTENCE_END.finditer(line.group(0))]
        cut.append(line.end())
        for a, b in zip(cut, cut[1:]):
            piece = text[a:b]
            left, right = len(piece) - len(piece.lstrip()), len(piece.rstrip())
            if right > left:
                spans.append((a + left, a + right))
    return spans


def band(share: float, threshold: float) -> str:
    """'none' below the flag threshold, 'contested' while readers split, 'agreed' once most would flag it."""
    if share < threshold:
        return "none"
    return "contested" if share < AGREED_SHARE else "agreed"


def heat_strip(sentence_peaks: Sequence[float], threshold: float,
               columns: int = STRIP_COLUMNS, rows: int = STRIP_ROWS) -> List[int]:
    """
    The article as `columns` stretches of sentences, each as high as its most-flagged word.

    0 means nothing in that stretch is flagged; otherwise the height is the peak share
    in eighths, rounded up (so 1.0 fills all 8 rows). Short texts repeat sentences across
    columns, so the strip always fills the matrix.
    """
    n = len(sentence_peaks)
    if n == 0:
        return [0] * columns
    strip = []
    for c in range(columns):
        lo = c * n // columns
        hi = max(lo + 1, (c + 1) * n // columns)
        peak = max(sentence_peaks[lo:hi])
        strip.append(0 if peak < threshold else min(rows, int(np.ceil(peak * rows))))
    return strip


def load_lexicon(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


class Analyzer:
    """Scores texts with a word model; one analysis at a time, so they do not compete for the CPU."""

    def __init__(self, model, threshold: float, lexicon: Set[str] = frozenset(), info: Optional[Dict] = None):
        self.model = model  # anything with .score(sentence) -> (word spans, shares)
        self.threshold = threshold
        self.lexicon = lexicon
        self.info = info or {}
        self._lock = CPU

    @classmethod
    def from_dir(cls, model_dir: str, precision: str = "int8", threads: int = 4) -> "Analyzer":
        from onnx_token_model import OnnxTokenModel

        model = OnnxTokenModel(model_dir, precision, threads)
        threshold = float(model.info.get("flag_threshold") or 0.3)
        return cls(model, threshold, load_lexicon(os.path.join(model_dir, "lexicon.txt")),
                   {**model.info, "precision": precision, "threads": threads})

    def analyze(self, text: str) -> Dict:
        text = text[:MAX_CHARS]
        started = time.perf_counter()
        sentences = []
        with self._lock:
            for s0, s1 in sentence_spans(text):
                spans, shares = self.model.score(text[s0:s1])
                words = []
                for (a, b), share in zip(spans, shares):
                    if np.isnan(share) or share < MIN_SHARE:
                        continue
                    word = text[s0 + a:s0 + b]
                    words.append({"start": s0 + a, "end": s0 + b, "text": word, "share": round(float(share), 3),
                                  "band": band(float(share), self.threshold),
                                  "lexicon": word.lower() in self.lexicon})
                peak = max((w["share"] for w in words), default=0.0)
                sentences.append({"start": s0, "end": s1, "peak": peak, "words": words,
                                  "word_count": len(spans)})
        seconds = time.perf_counter() - started
        flagged = [w for s in sentences for w in s["words"] if w["band"] != "none"]
        total_words = sum(s["word_count"] for s in sentences)
        top = sorted(flagged, key=lambda w: -w["share"])[:8]
        summary = {
            "sentences": len(sentences),
            "words": total_words,
            "flagged": len(flagged),
            "contested": sum(w["band"] == "contested" for w in flagged),
            "agreed": sum(w["band"] == "agreed" for w in flagged),
            "flagged_sentences": sum(any(w["band"] != "none" for w in s["words"]) for s in sentences),
            "in_lexicon": sum(w["lexicon"] for w in flagged),
            "strip": heat_strip([s["peak"] for s in sentences], self.threshold),
            "top": [{"text": w["text"], "share": w["share"]} for w in top],
            "seconds": round(seconds, 3),
            "words_per_second": round(total_words / seconds, 1) if seconds > 0 else None,
        }
        return {"text": text, "sentences": sentences, "summary": summary, "threshold": self.threshold}

    def describe(self) -> Dict:
        keys = ("model_name", "teacher", "precision", "threads", "training_sentences", "pool_sentences",
                "flag_threshold", "cv_f1", "cv_roc_auc", "cv_share_ece")
        return {"threshold": self.threshold, "agreed_share": AGREED_SHARE, "max_chars": MAX_CHARS,
                "lexicon_words": len(self.lexicon), **{k: self.info.get(k) for k in keys}}


class Perspective:
    """
    On request: how liberal and how conservative readers would react to each word.

    This is M3, the perspective model (docs/AUDIT.md, section 3). It estimates three shares
    per word: of all MBIC annotators, of liberal ones (ideology below -3) and of conservative
    ones (above 3). It fits conservative annotators better than the main model's pooled share,
    but its gap between the two groups follows the observed gap only weakly, so single words'
    differences are uncertain. It is a RoBERTa-base model, about 8 seconds per article on
    the board, so it is loaded on first use.
    """

    def __init__(self, model_dir: str, threshold: float, precision: str = "int8", threads: int = 4, model=None):
        self.model_dir, self.threshold = model_dir, threshold
        self.precision, self.threads = precision, threads
        self._model = model  # anything with .score_heads(sentence) -> (word spans, shares per head)
        self._loading = threading.Lock()

    @property
    def available(self) -> bool:
        return self._model is not None or os.path.exists(os.path.join(self.model_dir, f"model_{self.precision}.onnx"))

    def model(self):
        with self._loading:
            if self._model is None:
                from onnx_token_model import OnnxTokenModel

                self._model = OnnxTokenModel(self.model_dir, self.precision, self.threads)
        return self._model

    def analyze(self, text: str) -> Dict:
        """Each group's share for the words either group might flag, and the words they differ on most."""
        text = text[:MAX_CHARS]
        model = self.model()
        started = time.perf_counter()
        words, peaks = [], []
        with CPU:
            for s0, s1 in sentence_spans(text):
                spans, shares = model.score_heads(text[s0:s1])
                peak = 0.0
                for (a, b), (everyone, liberal, conservative) in zip(spans, shares):
                    if np.isnan(everyone) or max(liberal, conservative) < MIN_SHARE:
                        continue
                    peak = max(peak, float(everyone))
                    words.append({"start": s0 + a, "end": s0 + b, "text": text[s0 + a:s0 + b],
                                  "liberal": round(float(liberal), 3), "conservative": round(float(conservative), 3)})
                peaks.append(peak)
        candidates = [w for w in words if max(w["liberal"], w["conservative"]) >= self.threshold]
        gap = lambda w: w["liberal"] - w["conservative"]  # noqa: E731
        return {"words": words, "threshold": self.threshold, "gap_shown": GAP_SHOWN,
                "liberal_more": sorted((w for w in candidates if gap(w) >= GAP_SHOWN), key=lambda w: -gap(w))[:GAPS_LISTED],
                "conservative_more": sorted((w for w in candidates if -gap(w) >= GAP_SHOWN), key=gap)[:GAPS_LISTED],
                "strip": heat_strip(peaks, self.threshold), "seconds": round(time.perf_counter() - started, 3)}
