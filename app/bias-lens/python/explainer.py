"""
On request: a short explanation of one flagged word, from a small language model on the board.

The model is Qwen2.5-0.5B-Instruct (Apache 2.0) in ONNX with 4-bit weights, run with ONNX
Runtime like the bias-word model, and loaded on first use. Models this small are often wrong
in a confident voice, so the explanation is built to be checked:

  - The facts come from the board's own measurements: how many readers would flag the word,
    its band, and whether the paper's bias lexicon lists it.
  - The model only finishes sentences that this code starts ("... because it"). A reason is kept
    only if it is short and in the model's own words: long reasons, and reasons that copy the
    sentence, were mostly confused in our tests (docs/EXPLANATIONS.md).
  - A suggested replacement word is offered only by models whose suggestions held up in testing
    ("suggest_alternatives" in llm_info.json). Even then it is tested with the bias-word model: it
    must be a different word, not a political counterpart, not in the lexicon, and lower the estimate.
  - The page labels the model's text as experimental, and says how often it was sensible.

The model stays loaded for a few minutes after its last use, then is unloaded to free memory.
"""

import json
import os
import re
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

IDLE_UNLOAD_SECONDS = 300  # free the model's memory after this long without a request
MAX_REASON_TOKENS, MAX_ALTERNATIVE_TOKENS = 40, 8
MAX_REASON_WORDS = 20  # longer reasons were mostly confused
MAX_COPIED_WORDS = 3  # a reason may repeat at most this many words of the sentence in a row
REPETITION_PENALTY = 1.1
REQUIRED_DROP = 0.15  # a replacement must lower the estimate by this much, or fall below the flag threshold
SYSTEM = "You are a careful, neutral news editor."
# Left- and right-coded terms: replacing one with its counterpart moves the bias, it does not remove it.
COUNTERPARTS = [("democrat", "republican"), ("democrats", "republicans"), ("democratic", "republican"),
                ("liberal", "conservative"), ("liberals", "conservatives"), ("left-wing", "right-wing"),
                ("progressive", "conservative"), ("pro-choice", "pro-life"), ("biden", "trump"),
                ("pelosi", "mcconnell"), ("left", "right")]


class LocalLM:
    """Greedy text completion with an ONNX decoder and its key/value cache: one prompt at a time."""

    def __init__(self, session, tokenizer, eos_ids: Sequence[int]):
        self.session, self.tokenizer, self.eos = session, tokenizer, set(eos_ids)
        self.past_inputs = [i for i in session.get_inputs() if i.name.startswith("past_key_values")]
        self.present_names = [o.name for o in session.get_outputs()][1:]
        shape = self.past_inputs[0].shape
        self.kv_heads, self.head_dim = int(shape[1]), int(shape[3])

    @classmethod
    def from_dir(cls, model_dir: str, threads: int = 4) -> "LocalLM":
        import onnxruntime as ort
        from tokenizers import Tokenizer

        with open(os.path.join(model_dir, "llm_info.json"), encoding="utf-8") as f:
            info = json.load(f)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), options,
                                       providers=["CPUExecutionProvider"])
        lm = cls(session, Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json")), info["eos_token_ids"])
        lm.info = info
        return lm

    def complete(self, prompt: str, stops: Sequence[str], max_new_tokens: int) -> str:
        """The text the model writes after `prompt`, up to the first stop string or end of turn."""
        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        past = {i.name: np.zeros((1, self.kv_heads, 0, self.head_dim), np.float32) for i in self.past_inputs}
        feed, total, out = ids, len(ids), []
        for _ in range(max_new_tokens):
            outputs = self.session.run(None, {
                "input_ids": np.array([feed], np.int64),
                "attention_mask": np.ones((1, total), np.int64),
                "position_ids": np.arange(total - len(feed), total, dtype=np.int64)[None],
                **past})
            logits = outputs[0][0, -1].astype(np.float64)
            if out:  # discourage repeating itself, as in the model's own generation settings
                seen = np.array(sorted(set(out)))
                logits[seen] = np.where(logits[seen] > 0, logits[seen] / REPETITION_PENALTY,
                                        logits[seen] * REPETITION_PENALTY)
            token = int(np.argmax(logits))
            if token in self.eos:
                break
            out.append(token)
            if any(stop in self.tokenizer.decode(out) for stop in stops):
                break
            past = {name.replace("present", "past_key_values"): value
                    for name, value in zip(self.present_names, outputs[1:])}
            feed, total = [token], total + 1
        text = self.tokenizer.decode(out)
        for stop in stops:
            text = text.split(stop)[0]
        return text.strip()


def turn(role: str, text: str) -> str:
    """One message in the chat format of Qwen's instruct models."""
    return f"<|im_start|>{role}\n{text}<|im_end|>\n"


def reason_prompt(sentence: str, word: str) -> str:
    return (turn("system", SYSTEM)
            + turn("user", f'Sentence: "{sentence}"\nWhy might some readers find the word "{word}" one-sided '
                           f"in this sentence? Answer in one sentence.")
            + f'<|im_start|>assistant\nSome readers might find "{word}" one-sided here because it')


def alternative_prompt(sentence: str, word: str) -> str:
    return (turn("system", SYSTEM)
            + turn("user", f'Sentence: "{sentence}"\nSuggest one more neutral word or short phrase to use '
                           f'instead of "{word}" in this sentence.')
            + f'<|im_start|>assistant\nA more neutral choice than "{word}" is "')


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def longest_copied_run(reason: str, sentence: str) -> int:
    """The longest run of consecutive words that the reason repeats from the sentence."""
    r, s = _words(reason), _words(sentence)
    best = 0
    for i in range(len(r)):
        for j in range(len(s)):
            k = 0
            while i + k < len(r) and j + k < len(s) and r[i + k] == s[j + k]:
                k += 1
            best = max(best, k)
    return best


def clean_reason(text: str, sentence: str) -> Optional[str]:
    """The model's reason as a short clause in its own words, or None."""
    text = re.sub(r"\s+", " ", text).strip().rstrip(".,;:")
    words = text.split()
    if not 3 <= len(words) <= MAX_REASON_WORDS or text.lower().startswith(("sentence", "word")):
        return None
    if longest_copied_run(text, sentence) > MAX_COPIED_WORDS:
        return None
    return text


def clean_alternative(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip("\"'“”‘’.,;:!?").strip()


def counterpart(a: str, b: str) -> bool:
    pair = {a.lower(), b.lower()}
    return any(pair == {left, right} for left, right in COUNTERPARTS)


def acceptable_alternative(word: str, alternative: str, lexicon: Set[str]) -> Tuple[bool, str]:
    """Whether a suggested replacement can be tested at all, and why not."""
    if not alternative or len(alternative.split()) > 3:
        return False, "no usable suggestion"
    low, original = alternative.lower(), word.lower()
    if low == original or original in low.split() or low.rstrip("s") == original.rstrip("s"):
        return False, "the suggestion repeats the word"
    if counterpart(original, low) or any(counterpart(original, w) for w in low.split()):
        return False, "the suggestion swaps political sides"
    if low in lexicon or any(w in lexicon for w in low.split()):
        return False, "the suggestion is in the bias lexicon"
    return True, ""


def score_replacement(score: Callable, sentence: str, start: int, end: int, alternative: str) -> float:
    """The bias-word model's highest share among the replacement's words, in the rewritten sentence."""
    rewritten = sentence[:start] + alternative + sentence[end:]
    spans, shares = score(rewritten)
    inside = [s for (a, b), s in zip(spans, shares) if a >= start and b <= start + len(alternative) and not np.isnan(s)]
    return float(max(inside)) if inside else 0.0


class Explainer:
    """Loads the language model on first use, and unloads it when it has been idle for a while."""

    def __init__(self, model_dir: str, threads: int = 4, lm: Optional[LocalLM] = None):
        self.model_dir, self.threads = model_dir, threads
        self._lm, self._last_used = lm, 0.0
        self._fixed = lm is not None  # a model passed in (tests) is never unloaded
        self._lock = threading.Lock()
        if not self._fixed:
            threading.Thread(target=self._unload_when_idle, daemon=True).start()

    @property
    def available(self) -> bool:
        return self._lm is not None or os.path.exists(os.path.join(self.model_dir, "model.onnx"))

    def describe(self) -> Dict:
        if self._fixed:
            return dict(getattr(self._lm, "info", {}))
        path = os.path.join(self.model_dir, "llm_info.json")
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
        return {k: info.get(k) for k in ("name", "licence", "repo", "revision", "suggest_alternatives",
                                         "sensible_reasons")}

    def _model(self) -> LocalLM:
        if self._lm is None:
            self._lm = LocalLM.from_dir(self.model_dir, self.threads)
        self._last_used = time.time()
        return self._lm

    def _unload_when_idle(self) -> None:
        while True:
            time.sleep(30)
            with self._lock:
                if self._lm is not None and time.time() - self._last_used > IDLE_UNLOAD_SECONDS:
                    self._lm = None

    def explain(self, sentence: str, start: int, end: int, share: float, band: str, in_lexicon: bool,
                score: Callable, threshold: float, lexicon: Set[str]) -> Dict:
        """Facts about the word at sentence[start:end], the model's reason, and a checked replacement."""
        word = sentence[start:end]
        suggest = bool(self.describe().get("suggest_alternatives"))
        started = time.perf_counter()
        with self._lock:
            lm = self._model()
            # Stop at the end of the sentence, but not inside an abbreviation such as "U.S.".
            reason = clean_reason(lm.complete(reason_prompt(sentence, word), [". ", "\n"], MAX_REASON_TOKENS),
                                  sentence)
            alternative = clean_alternative(lm.complete(alternative_prompt(sentence, word), ['"', "\n"],
                                                        MAX_ALTERNATIVE_TOKENS)) if suggest else ""
            self._last_used = time.time()
        ok, rejected, replacement_share = False, None, None
        if suggest:
            ok, rejected = acceptable_alternative(word, alternative, lexicon)
            if ok:
                replacement_share = score_replacement(score, sentence, start, end, alternative)
                if not (replacement_share < threshold or replacement_share <= share - REQUIRED_DROP):
                    ok, rejected = False, "the bias model did not rate the suggestion as more neutral"
        return {
            "word": word, "share": round(share, 3), "band": band, "lexicon": in_lexicon,
            "reason": f'Some readers might find "{word}" one-sided here because it {reason}.' if reason else None,
            "suggestions": suggest,
            "alternative": alternative if ok else None,
            "alternative_share": round(replacement_share, 3) if ok else None,
            "alternative_rejected": None if ok or not suggest else (rejected or "no usable suggestion"),
            "seconds": round(time.perf_counter() - started, 1),
        }
