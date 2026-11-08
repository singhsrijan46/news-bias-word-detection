"""
On request: a short explanation of one flagged word, from a small language model on the board.

The model is Qwen2.5-1.5B-Instruct (Apache 2.0) with 4-bit weights, run by llama.cpp's server
inside the app's container and started on first use. llm_info.json names the runtime: the smaller
Qwen2.5-0.5B-Instruct in ONNX, run with ONNX Runtime like the bias-word model, also works
(scripts/app/fetch_llm.py --model qwen2.5-0.5b-instruct). Models this small are often wrong in a
confident voice, so the explanation is built to be checked:

  - The facts come from the board's own measurements: how many readers would flag the word,
    its band, and whether the paper's bias lexicon lists it.
  - The model only finishes sentences that this code starts ("... because it"). A reason is kept
    only if it is short and in the model's own words: long reasons, and reasons that copy the
    sentence, were mostly confused in our tests (docs/EXPLANATIONS.md).
  - A suggested replacement word is offered only by models whose suggestions held up in testing
    ("suggest_alternatives" in llm_info.json). Even then it is tested with the bias-word model: it
    must be a different word, not a political counterpart, not in the lexicon, and lower the estimate.
  - The page labels the model's text as experimental, and says how often it was sensible.

llama.cpp's server listens on a Unix socket in a private folder, not on the network. It runs
offline, with its web page and slot listing turned off. Its log is read and thrown away; it is
printed only if the server fails to start, before it has seen any text.

The model stays loaded for a few minutes after its last use, then is unloaded to free memory.
"""

import atexit
import collections
import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

IDLE_UNLOAD_SECONDS = 300  # free the model's memory after this long without a request
MAX_REASON_TOKENS, MAX_ALTERNATIVE_TOKENS = 40, 8
MAX_REASON_WORDS = 20  # longer reasons were mostly confused
MAX_COPIED_WORDS = 3  # a reason may repeat at most this many words of the sentence in a row
MAX_SENTENCE_CHARS = 600  # a longer sentence is cut to this much around the word, in the prompts only
REPETITION_PENALTY = 1.1
REQUIRED_DROP = 0.15  # a replacement must lower the estimate by this much, or fall below the flag threshold
SYSTEM = "You are a careful, neutral news editor."
# Left- and right-coded terms: replacing one with its counterpart moves the bias, it does not remove it.
COUNTERPARTS = [("democrat", "republican"), ("democrats", "republicans"), ("democratic", "republican"),
                ("liberal", "conservative"), ("liberals", "conservatives"), ("left-wing", "right-wing"),
                ("progressive", "conservative"), ("pro-choice", "pro-life"), ("biden", "trump"),
                ("pelosi", "mcconnell"), ("left", "right")]
# llama.cpp's server
CONTEXT_TOKENS = 1024  # the prompts take 100 to 250 tokens; the model's own 32,768 would reserve 900 MB
STARTUP_SECONDS = 180  # loading the model from the board's storage, the first time after a reboot
REQUEST_SECONDS = 300


def read_info(model_dir: str) -> Dict:
    with open(os.path.join(model_dir, "llm_info.json"), encoding="utf-8") as f:
        return json.load(f)


def model_files(info: Dict) -> List[str]:
    """The files a model needs besides llm_info.json, as listed by scripts/app/fetch_llm.py."""
    return info.get("files") or ["model.onnx", "tokenizer.json"]  # older ONNX models' info lists none


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
        penalty = getattr(self, "info", {}).get("repeat_penalty", REPETITION_PENALTY)
        for _ in range(max_new_tokens):
            outputs = self.session.run(None, {
                "input_ids": np.array([feed], np.int64),
                "attention_mask": np.ones((1, total), np.int64),
                "position_ids": np.arange(total - len(feed), total, dtype=np.int64)[None],
                **past})
            logits = outputs[0][0, -1].astype(np.float64)
            if out and penalty != 1.0:  # discourage repeating itself
                seen = np.array(sorted(set(out)))
                logits[seen] = np.where(logits[seen] > 0, logits[seen] / penalty, logits[seen] * penalty)
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

    def alive(self) -> bool:
        return True

    def close(self) -> None:
        pass


def server_args(model_file: str, address: str, threads: int) -> List[str]:
    """How llama.cpp's server is run: one request at a time, a small context, no prompt cache in RAM
    (by default it may take 8 GB), offline, and without its web page or slot listing."""
    if address.endswith(".sock"):
        where = ["--host", address]
    else:  # "127.0.0.1:8090", for the evaluation on the board (scripts/app/evaluate_explainer.py)
        host, port = address.rsplit(":", 1)
        where = ["--host", host, "--port", port]
    return ["-m", model_file, *where, "-t", str(threads), "-tb", str(threads), "-c", str(CONTEXT_TOKENS),
            "-np", "1", "--cache-ram", "0", "--offline", "--no-webui", "--no-slots"]


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


class LlamaServer:
    """Greedy text completion by llama.cpp's server, one prompt at a time.

    `start` runs the server from the model folder and waits until it has loaded the model; `close`
    stops it. With no process, the class talks to a server started elsewhere.
    """

    def __init__(self, address: str, info: Dict, process: Optional[subprocess.Popen] = None,
                 folder: Optional[str] = None):
        self.address, self.info, self.process, self.folder = address, info, process, folder
        self._ready = False
        self._startup_log = collections.deque(maxlen=20)
        self._log_reader = threading.Thread(target=self._read_log, daemon=True)
        if process is not None:
            self._log_reader.start()

    @classmethod
    def start(cls, model_dir: str, threads: int = 4) -> "LlamaServer":
        info = read_info(model_dir)
        folder = tempfile.mkdtemp(prefix="bias-lens-llm-")  # only this user can reach the socket
        address = os.path.join(folder, "llama.sock")
        command = [os.path.join(model_dir, "llama", "llama-server"),
                   *server_args(os.path.join(model_dir, info["model_file"]), address, threads)]
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_")}  # only the settings above apply
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, env=env,
                                   stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        server = cls(address, info, process, folder)
        try:
            server.wait_until_ready()
        except Exception:
            server.close()
            raise
        return server

    def _read_log(self) -> None:
        for line in self.process.stderr:  # read it all, or the server stalls once the pipe is full
            if not self._ready:
                self._startup_log.append(line.rstrip())

    def _request(self, method: str, path: str, body: Optional[Dict] = None,
                 timeout: float = REQUEST_SECONDS) -> Tuple[int, Dict]:
        if self.address.endswith(".sock"):
            connection = _UnixConnection(self.address, timeout)
        else:
            host, port = self.address.rsplit(":", 1)
            connection = http.client.HTTPConnection(host, int(port), timeout=timeout)
        try:
            payload = None if body is None else json.dumps(body)
            connection.request(method, path, body=payload,
                               headers={"Content-Type": "application/json"} if payload else {})
            response = connection.getresponse()
            data = response.read()
        finally:
            connection.close()
        try:
            return response.status, json.loads(data) if data else {}
        except ValueError:
            return response.status, {}

    def wait_until_ready(self, timeout: float = STARTUP_SECONDS) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                self._log_reader.join(2)  # its last lines say why
                raise RuntimeError("llama.cpp's server stopped while starting:\n" + "\n".join(self._startup_log))
            try:
                if self._request("GET", "/health", timeout=5)[0] == 200:
                    self._ready = True
                    return
            except (OSError, http.client.HTTPException):
                pass  # not listening yet
            time.sleep(0.25)
        raise RuntimeError("llama.cpp's server did not start in time")

    def complete(self, prompt: str, stops: Sequence[str], max_new_tokens: int) -> str:
        """The text the model writes after `prompt`, up to the first stop string or end of turn."""
        status, data = self._request("POST", "/completion", {
            "prompt": prompt, "n_predict": max_new_tokens, "stop": list(stops),
            "temperature": 0.0, "top_k": 1,  # greedy: the same word in the same sentence gets the same answer
            # Discourages repeating ("repeat_penalty" in llm_info.json; 1.0 turns it off). Unlike
            # LocalLM's penalty, llama.cpp's also counts the prompt's last 64 tokens.
            "repeat_penalty": self.info.get("repeat_penalty", REPETITION_PENALTY), "repeat_last_n": 64,
            "cache_prompt": True})  # the reason and the replacement prompts share their start
        if status != 200:
            raise RuntimeError(f"llama.cpp's server answered {status}")
        text = data.get("content", "")
        for stop in stops:
            text = text.split(stop)[0]
        return text.strip()

    def alive(self) -> bool:
        return self.process is None or self.process.poll() is None

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.folder:
            shutil.rmtree(self.folder, ignore_errors=True)


def load_model(model_dir: str, threads: int = 4):
    """The model in `model_dir`, with the runtime its llm_info.json names."""
    if read_info(model_dir).get("runtime") == "llama.cpp":
        return LlamaServer.start(model_dir, threads)
    return LocalLM.from_dir(model_dir, threads)


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


def sentence_window(sentence: str, start: int, end: int, limit: int = MAX_SENTENCE_CHARS) -> str:
    """The sentence, or for a long one about `limit` characters around the word, cut between words."""
    if len(sentence) <= limit:
        return sentence
    room = max(limit - (end - start), 0)
    a = max(0, start - room // 2)
    b = min(len(sentence), end + room - (start - a))
    a = max(0, min(a, b - limit))  # use the room that the end of the sentence leaves
    if a > 0:
        space = sentence.find(" ", a, start)
        a = space + 1 if space != -1 else a
    if b < len(sentence):
        space = sentence.rfind(" ", end, b)
        b = space if space != -1 else b
    return ("…" if a > 0 else "") + sentence[a:b] + ("…" if b < len(sentence) else "")


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


def full_reason(word: str, reason: str) -> str:
    """The reason as the sentence the page shows; "'s used" joins "it" as "it's used"."""
    joiner = "" if reason.startswith("'") else " "
    return f'Some readers might find "{word}" one-sided here because it{joiner}{reason}.'


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

    INFO_KEYS = ("name", "licence", "repo", "revision", "runtime", "suggest_alternatives", "sensible_reasons",
                 "typical_seconds")

    def __init__(self, model_dir: str, threads: int = 4, lm=None):
        self.model_dir, self.threads = model_dir, threads
        self._lm, self._last_used = lm, 0.0
        self._fixed = lm is not None  # a model passed in (tests, the evaluation) is never unloaded
        self._lock = threading.Lock()
        if not self._fixed:
            threading.Thread(target=self._unload_when_idle, daemon=True).start()
            atexit.register(self._stop_at_exit)

    @property
    def available(self) -> bool:
        if self._lm is not None:
            return True
        try:
            info = read_info(self.model_dir)
        except (OSError, ValueError):
            return False
        return all(os.path.exists(os.path.join(self.model_dir, name)) for name in model_files(info))

    def describe(self) -> Dict:
        if self._fixed:
            info = getattr(self._lm, "info", {})
        else:
            try:
                info = read_info(self.model_dir)
            except (OSError, ValueError):
                return {}
        return {k: info[k] for k in self.INFO_KEYS if k in info}

    def _model(self):
        if self._lm is not None and not self._fixed and not self._lm.alive():
            self._unload()  # the server stopped: start a new one
        if self._lm is None:
            self._lm = load_model(self.model_dir, self.threads)
        self._last_used = time.time()
        return self._lm

    def _unload(self) -> None:
        if self._lm is not None and not self._fixed:
            self._lm.close()
            self._lm = None

    def close(self) -> None:
        with self._lock:
            self._unload()

    def _stop_at_exit(self) -> None:
        lm = self._lm  # without the lock: a request still running ends with the app
        if lm is not None and not self._fixed:
            lm.close()

    def _unload_when_idle(self) -> None:
        while True:
            time.sleep(30)
            with self._lock:
                if self._lm is not None and time.time() - self._last_used > IDLE_UNLOAD_SECONDS:
                    self._unload()

    def explain(self, sentence: str, start: int, end: int, share: float, band: str, in_lexicon: bool,
                score: Callable, threshold: float, lexicon: Set[str]) -> Dict:
        """Facts about the word at sentence[start:end], the model's reason, and a checked replacement."""
        word = sentence[start:end]
        suggest = bool(self.describe().get("suggest_alternatives"))
        shown = sentence_window(sentence, start, end)  # what the language model sees
        started = time.perf_counter()
        with self._lock:
            lm = self._model()
            try:
                # Stop at the end of the sentence, but not inside an abbreviation such as "U.S.".
                reason = clean_reason(lm.complete(reason_prompt(shown, word), [". ", "\n"], MAX_REASON_TOKENS),
                                      shown)
                alternative = clean_alternative(lm.complete(alternative_prompt(shown, word), ['"', "\n"],
                                                            MAX_ALTERNATIVE_TOKENS)) if suggest else ""
            except Exception:
                self._unload()  # start afresh on the next request
                raise
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
            "reason": full_reason(word, reason) if reason else None,
            "suggestions": suggest,
            "alternative": alternative if ok else None,
            "alternative_share": round(replacement_share, 3) if ok else None,
            "alternative_rejected": None if ok or not suggest else (rejected or "no usable suggestion"),
            "seconds": round(time.perf_counter() - started, 1),
        }
