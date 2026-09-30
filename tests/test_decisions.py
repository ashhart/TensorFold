"""Decision prompts and probabilities follow SGLang prompt format version 1, without a model loaded."""

import json

from tensorfold.server.decisions import DecisionError, build_response, prepare
from tensorfold.server.errors import RequestError
from tensorfold.server.http import make_handler
from tensorfold.server.scheduler import Scheduler
from tests.http_fakes import post


class _Tokenizer:
    """One code point per token, so a one-character label is one token and ``yes`` is not."""

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(chr(int(token)) for token in ids)

    def apply_chat_template(self, messages, **kwargs):
        text = messages[0]["content"] + "\n"
        if kwargs.get("tokenize", True) is False:
            return text
        return self.encode(text)


def _choice():
    return {
        "input": "The integration keeps failing.",
        "questions": [{
            "id": "team",
            "type": "choice",
            "question": "Which team should handle this ticket?",
            "options": [
                {"name": "billing", "description": "Payment issues"},
                {"name": "technical"},
            ],
        }],
    }


def test_choice_prompt_matches_sglang_wording():
    prepared = prepare(_Tokenizer(), _choice())
    text = _Tokenizer().decode(prepared[0].prompt_ids)
    assert "The integration keeps failing.\n\nQuestion: Which team should handle this ticket?" in text
    assert "A: billing - Payment issues" in text
    assert "B: technical" in text
    assert text.endswith("Answer with the letter of one option only.\n")
    assert prepared[0].label_ids == [ord("A"), ord("B")]


def test_label_that_is_not_one_token_is_refused():
    body = {
        "input": "The integration keeps failing.",
        "questions": [{"id": "urgent", "type": "yes_no", "question": "The customer needs an answer today."}],
    }
    try:
        prepare(_Tokenizer(), body)
    except DecisionError as exc:
        assert "yes" in str(exc)
    else:
        raise AssertionError("expected a one-token refusal")


def test_probabilities_use_temperature_and_label_mass_does_not():
    prepared = prepare(_Tokenizer(), _choice())
    cool = build_response(_choice(), prepared, [([0.0, 2.0], 2.0)])
    hot = build_response({**_choice(), "temperature": 2}, prepared, [([0.0, 2.0], 2.0)])
    assert abs(sum(cool["answers"]["team"]["probabilities"].values()) - 1) < 1e-9
    assert cool["answers"]["team"]["choice"] == "technical"
    assert cool["answers"]["team"]["probabilities"]["technical"] > hot["answers"]["team"]["probabilities"]["technical"]
    assert cool["answers"]["team"]["label_mass"] == hot["answers"]["team"]["label_mass"]
    assert cool["usage"]["completion_tokens"] == 0
    assert cool["prompt_format_version"] == 1


def test_http_decisions_returns_the_scored_body():
    class App:
        served_name = "qwen"
        model_ids = ("qwen",)
        max_batch_size = 1

        def decisions(self, body):
            if body.get("input") == "":
                raise RequestError("input must not be blank")
            return {"object": "decisions", "answers": {"team": {"choice": "technical"}}}

    status, raw = post(App(), _choice(), path="/v1/decisions")
    assert status == 200
    assert json.loads(raw)["answers"]["team"]["choice"] == "technical"
    status, raw = post(App(), {"input": "", "questions": []}, path="/v1/decisions")
    assert status == 400
    assert "blank" in json.loads(raw)["error"]["message"]


def test_scheduler_scores_on_the_engine_thread():
    class Engine:
        active_count = 0
        prefill_chunks = 0

        def score_labels(self, prompt, labels):
            return [float(labels[0]), 0.0], 1.0

    scheduler = Scheduler(Engine(), lanes=1, eos_ids=frozenset())
    scheduler.start()
    try:
        logits, logsumexp = scheduler.on_engine(lambda engine: engine.score_labels([7], [4, 5]))
    finally:
        scheduler.stop()
    assert logits == [4.0, 0.0]
    assert logsumexp == 1.0


def test_handler_without_decisions_is_not_found():
    class App:
        served_name = "qwen"
        model_ids = ("qwen",)
        max_batch_size = 1

    status, _ = post(App(), _choice(), path="/v1/decisions")
    assert status == 404
    make_handler(App())  # the factory still builds for servers that never score
