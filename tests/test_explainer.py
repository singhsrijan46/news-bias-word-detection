import http.server
import json
import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "bias-lens", "python"))

import explainer  # noqa: E402
from explainer import (Explainer, LlamaServer, LocalLM, acceptable_alternative, clean_alternative,  # noqa: E402
                       clean_reason, full_reason, score_replacement, sentence_window, server_args)
from src.models.onnx_token_model import word_spans  # noqa: E402

SHARES = {"slammed": 0.5, "criticized": 0.1, "attacked": 0.45, "liberal": 0.3, "conservative": 0.15}


def score(text):
    """The bias-word model, faked: shares from a table, 0.02 for other words."""
    spans = word_spans(text)
    return spans, np.array([SHARES.get(text[a:b].lower(), 0.02) for a, b in spans], dtype=np.float32)


SENTENCE = "Critics slammed the costly bus plan for the northern suburbs."


def test_a_reason_is_kept_only_as_a_short_clause_in_the_models_own_words():
    assert clean_reason("  implies a harsh, emotional reaction. ", SENTENCE) == "implies a harsh, emotional reaction"
    assert clean_reason("is bad", SENTENCE) is None  # too short to say anything
    assert clean_reason("Sentence: it judges the plan", SENTENCE) is None  # the model repeating the prompt
    assert clean_reason(" ".join(["word"] * 21), SENTENCE) is None  # longer than 20 words: mostly confused
    assert clean_reason("judges the costly plan instead of describing it", SENTENCE) == \
        "judges the costly plan instead of describing it"  # repeats 2 words in a row: "the costly"
    assert clean_reason("judges the costly bus plan instead of describing it", SENTENCE) is None  # repeats 4


def test_suggestions_that_repeat_swap_sides_or_are_loaded_are_not_tested():
    lexicon = {"reckless"}
    assert acceptable_alternative("slammed", "criticized", lexicon) == (True, "")
    assert not acceptable_alternative("slammed", "Slammed", lexicon)[0]
    assert not acceptable_alternative("aliens", "alien", lexicon)[0]
    assert acceptable_alternative("liberal", "conservative", lexicon) == (False, "the suggestion swaps political sides")
    assert acceptable_alternative("scheme", "reckless plan", lexicon) == (False, "the suggestion is in the bias lexicon")
    assert not acceptable_alternative("plan", "a very long replacement phrase", lexicon)[0]
    assert clean_alternative(' "criticized." ') == "criticized"


def test_a_replacement_is_scored_on_its_own_words_in_the_rewritten_sentence():
    sentence = "Critics slammed the plan."
    assert abs(score_replacement(score, sentence, 8, 15, "criticized") - 0.1) < 1e-6
    assert abs(score_replacement(score, sentence, 8, 15, "attacked") - 0.45) < 1e-6


class FakeLM:
    def __init__(self, reason, alternative, suggest=True):
        self.reason, self.alternative = reason, alternative
        self.info = {"name": "fake", "suggest_alternatives": suggest}
        self.prompts = []

    def complete(self, prompt, stops, max_new_tokens):
        self.prompts.append(prompt)
        return self.reason if "Why might" in prompt else self.alternative

    def alive(self):
        return True

    def close(self):
        pass


def explain(lm, word="slammed", sentence="Critics slammed the plan.", share=0.5):
    start = sentence.index(word)
    return Explainer("unused", lm=lm).explain(sentence, start, start + len(word), share, "contested", False,
                                               score, threshold=0.27, lexicon=set())


def test_a_suggestion_is_shown_only_when_the_bias_model_rates_it_more_neutral():
    kept = explain(FakeLM("implies a harsh reaction rather than a description", "criticized"))
    assert kept["reason"] == 'Some readers might find "slammed" one-sided here because it implies a harsh reaction rather than a description.'
    assert kept["alternative"] == "criticized" and abs(kept["alternative_share"] - 0.1) < 1e-6
    dropped = explain(FakeLM("is", "attacked"))  # a reason too short to keep, and a suggestion as loaded as the word
    assert dropped["reason"] is None and dropped["alternative"] is None
    assert dropped["alternative_rejected"] == "the bias model did not rate the suggestion as more neutral"
    swapped = explain(FakeLM("names one side", "conservative"), word="liberal", sentence="The liberal media lied.", share=0.3)
    assert swapped["alternative"] is None and swapped["alternative_rejected"] == "the suggestion swaps political sides"


def test_models_without_reliable_suggestions_are_not_asked_for_one():
    lm = FakeLM("implies a harsh reaction rather than a description", "criticized", suggest=False)
    result = explain(lm)
    assert result["reason"] is not None and result["suggestions"] is False
    assert result["alternative"] is None and result["alternative_rejected"] is None
    assert len(lm.prompts) == 1  # only the reason was generated


def test_a_reason_that_fails_the_checks_comes_back_labelled_not_hidden():
    passed = explain(FakeLM("implies a harsh reaction rather than a description", "criticized", suggest=False))
    assert passed["unchecked_reason"] is None and passed["reason_problem"] is None
    long_reason = "implies " + " ".join(["a very harsh and one-sided reaction"] * 4)
    failed = explain(FakeLM(long_reason, "criticized", suggest=False))
    assert failed["reason"] is None and failed["reason_problem"] == "too long"
    assert failed["unchecked_reason"].startswith('Some readers might find "slammed" one-sided here because it implies')
    copied = explain(FakeLM("says critics slammed the plan, as they did", "", suggest=False))
    assert copied["reason_problem"] == "it repeats the sentence"
    empty = explain(FakeLM("", "", suggest=False))
    assert empty["unchecked_reason"] is None and empty["reason_problem"] == "no answer"


class Named:
    def __init__(self, name, shape=None):
        self.name, self.shape = name, shape


class FakeTokenizer:
    VOCAB = ["<prompt>", "crit", "icized", '"', " later"]

    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=False):
        return self.Encoding([0, 0, 0])

    def decode(self, ids):
        return "".join(self.VOCAB[i] for i in ids)


class FakeSession:
    """A decoder that writes the tokens 1, 2, 3, 4 in turn, with a growing key/value cache."""

    def __init__(self):
        self.step, self.lengths = 0, []

    def get_inputs(self):
        return [Named("input_ids"), Named("attention_mask"), Named("position_ids"),
                Named("past_key_values.0.key", [1, 2, "past", 4]), Named("past_key_values.0.value", [1, 2, "past", 4])]

    def get_outputs(self):
        return [Named("logits"), Named("present.0.key"), Named("present.0.value")]

    def run(self, _, feeds):
        total = feeds["attention_mask"].shape[1]
        self.lengths.append((feeds["input_ids"].shape[1], feeds["past_key_values.0.key"].shape[2], total))
        self.step += 1
        logits = np.zeros((1, feeds["input_ids"].shape[1], 5), np.float32)
        logits[0, -1, min(self.step, 4)] = 10.0
        cache = np.zeros((1, 2, total, 4), np.float32)
        return [logits, cache, cache]


def test_local_lm_feeds_the_prompt_once_then_one_token_at_a_time_and_stops_at_a_stop_string():
    session = FakeSession()
    lm = LocalLM(session, FakeTokenizer(), eos_ids=[])
    assert lm.complete("ignored", stops=['"'], max_new_tokens=8) == "criticized"
    # The prompt (3 tokens) with an empty cache, then single tokens with the cache growing by one.
    assert session.lengths == [(3, 0, 3), (1, 3, 4), (1, 4, 5)]


def test_long_sentences_are_cut_around_the_word_for_the_prompt():
    short = "Critics slammed the plan."
    assert sentence_window(short, 8, 15) == short
    sentence = " ".join(f"w{i}" for i in range(300)) + " slammed " + " ".join(f"v{i}" for i in range(300))
    start = sentence.index("slammed")
    window = sentence_window(sentence, start, start + 7, limit=100)
    assert "slammed" in window and window.startswith("…") and window.endswith("…")
    left, right = window.index("slammed"), len(window) - window.index("slammed") - 7
    assert 90 <= len(window) <= 102 and abs(left - right) <= 6  # about 100 characters, centred on the word
    assert set(window.strip("…").split(" ")) <= set(sentence.split(" "))  # cut between words
    near_end = sentence_window(sentence, len(sentence) - 4, len(sentence), limit=100)
    assert near_end.endswith("v299") and not near_end.endswith("…") and len(near_end) >= 90


def test_llama_server_runs_offline_without_its_web_page_and_with_a_small_context():
    args = server_args("/m/model.gguf", "/tmp/x/llama.sock", 4)
    assert args[args.index("--host") + 1] == "/tmp/x/llama.sock" and "--port" not in args
    assert {"--offline", "--no-webui", "--no-slots"} <= set(args)
    assert args[args.index("-c") + 1] == "1024" and args[args.index("--cache-ram") + 1] == "0"
    assert args[args.index("-np") + 1] == "1" and args[args.index("-t") + 1] == "4"
    tcp = server_args("/m/model.gguf", "127.0.0.1:8090", 4)
    assert tcp[tcp.index("--host") + 1] == "127.0.0.1" and tcp[tcp.index("--port") + 1] == "8090"


class FakeLlamaServer(http.server.BaseHTTPRequestHandler):
    """Answers /health (loading at first) and /completion like llama.cpp's server."""
    health_calls, bodies = 0, []

    def do_GET(self):
        FakeLlamaServer.health_calls += 1
        self._send(200 if FakeLlamaServer.health_calls > 1 else 503, {"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLlamaServer.bodies.append(body)
        self._send(200, {"content": " implies a harsh reaction. It also", "stop_type": "word"})

    def _send(self, status, data):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_llama_server_client_waits_for_the_model_then_decodes_greedily():
    server = http.server.HTTPServer(("127.0.0.1", 0), FakeLlamaServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        lm = LlamaServer(f"127.0.0.1:{server.server_address[1]}", {"name": "fake"})
        lm.wait_until_ready(timeout=10)  # the first /health says it is still loading
        assert FakeLlamaServer.health_calls == 2
        assert lm.complete("prompt", [". ", "\n"], 40) == "implies a harsh reaction"
        body = FakeLlamaServer.bodies[-1]
        assert body["prompt"] == "prompt" and body["n_predict"] == 40 and body["stop"] == [". ", "\n"]
        assert body["temperature"] == 0.0 and body["top_k"] == 1 and body["repeat_penalty"] == 1.1
        assert lm.alive()  # a server started elsewhere is not watched
        lm.info = {"repeat_penalty": 1.0}  # a model's llm_info.json can turn the penalty off
        lm.complete("prompt", ["\n"], 8)
        assert FakeLlamaServer.bodies[-1]["repeat_penalty"] == 1.0
    finally:
        server.shutdown()


def test_the_shown_reason_is_one_sentence_even_when_the_model_starts_with_a_contraction():
    assert full_reason("vile", "condemns rather than describes") == \
        'Some readers might find "vile" one-sided here because it condemns rather than describes.'
    assert full_reason("lied", "'s used three times") == \
        'Some readers might find "lied" one-sided here because it\'s used three times.'


def test_the_runtime_comes_from_the_models_info_and_missing_files_mean_no_explainer(tmp_path, monkeypatch):
    info = {"name": "Qwen", "runtime": "llama.cpp", "model_file": "model.gguf", "suggest_alternatives": True,
            "files": ["model.gguf", "llama/llama-server"], "sha256": {"model.gguf": "..."}}
    (tmp_path / "llm_info.json").write_text(json.dumps(info))
    (tmp_path / "model.gguf").write_text("")
    explainer_ = Explainer(str(tmp_path))
    assert not explainer_.available  # llama/llama-server is missing
    (tmp_path / "llama").mkdir()
    (tmp_path / "llama" / "llama-server").write_text("")
    assert explainer_.available
    assert explainer_.describe() == {"name": "Qwen", "runtime": "llama.cpp", "suggest_alternatives": True}
    started = []
    monkeypatch.setattr(LlamaServer, "start", classmethod(lambda cls, d, t=4: started.append(d) or "server"))
    assert explainer.load_model(str(tmp_path)) == "server" and started == [str(tmp_path)]
    assert not Explainer(str(tmp_path / "nowhere")).available


class DyingLM(FakeLM):
    def __init__(self, *args):
        super().__init__(*args)
        self.closed = False

    def alive(self):
        return False

    def close(self):
        self.closed = True


def test_a_server_that_stopped_is_started_again_on_the_next_request(monkeypatch):
    dead = DyingLM("implies a harsh reaction rather than a description", "criticized")
    fresh = FakeLM("implies a harsh reaction rather than a description", "criticized", suggest=False)
    monkeypatch.setattr(explainer, "load_model", lambda model_dir, threads=4: fresh)
    monkeypatch.setattr(Explainer, "describe", lambda self: {"suggest_alternatives": False})
    explainer_ = Explainer("unused")
    explainer_._lm = dead
    result = explainer_.explain("Critics slammed the plan.", 8, 15, 0.5, "contested", False, score, 0.27, set())
    assert dead.closed and fresh.prompts and result["reason"] is not None


class ExitedProcess:
    """A llama-server that failed to load its model."""

    def __init__(self, command, **kwargs):
        self.command, self.env = command, kwargs["env"]
        self.stderr = iter(["llama_model_load: error loading model\n", "main: exiting\n"])

    def poll(self):
        return 1


def test_a_server_that_fails_to_start_reports_its_log_and_leaves_nothing_behind(tmp_path, monkeypatch):
    (tmp_path / "llm_info.json").write_text(json.dumps({"runtime": "llama.cpp", "model_file": "model.gguf"}))
    started = []

    def popen(command, **kwargs):
        started.append(ExitedProcess(command, **kwargs))
        return started[-1]

    monkeypatch.setattr(explainer.subprocess, "Popen", popen)
    monkeypatch.setenv("LLAMA_ARG_HOST", "0.0.0.0")
    with pytest.raises(RuntimeError, match="error loading model"):
        LlamaServer.start(str(tmp_path))
    process = started[0]
    assert process.command[0] == os.path.join(str(tmp_path), "llama", "llama-server")
    assert process.command[process.command.index("-m") + 1] == os.path.join(str(tmp_path), "model.gguf")
    socket_path = process.command[process.command.index("--host") + 1]
    assert socket_path.endswith("llama.sock") and not os.path.exists(os.path.dirname(socket_path))  # removed
    assert not any(k.startswith("LLAMA_") for k in process.env)  # the environment cannot change its settings
