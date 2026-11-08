import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "bias-lens", "python"))

from explainer import (Explainer, LocalLM, acceptable_alternative, clean_alternative, clean_reason,  # noqa: E402
                       score_replacement)
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
