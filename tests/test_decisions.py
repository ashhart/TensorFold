"""Decision prompts and probabilities follow SGLang prompt format version 1, without a model loaded."""

import json
import math

import pytest

from tensorfold.server.decisions import DecisionError, build_response, prepare, prompts_for, reduce_vocab_shards
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


class _YesNoTokenizer(_Tokenizer):
    """``yes`` and ``no`` are one token when they are the text being added after the prompt."""

    def encode(self, text, add_special_tokens=False):
        if text.endswith("yes"):
            return [ord(char) for char in text[:-3]] + [1000]
        if text.endswith("no"):
            return [ord(char) for char in text[:-2]] + [1001]
        return [ord(char) for char in text]


def _score():
    return {
        "input": "The integration keeps failing.",
        "questions": [{
            "id": "frustration",
            "type": "score",
            "question": "How frustrated is the customer?",
            "levels": ["calm", "upset"],
        }],
    }


def _yes_no():
    return {
        "input": "The integration keeps failing.",
        "questions": [{
            "id": "urgent",
            "type": "yes_no",
            "question": "The customer needs an answer today.",
            "yes": "Needs a reply today",
            "no": "Can wait",
        }],
    }


def _render(content):
    return content + "\n"


def test_score_and_yes_no_prompts_match_sglang_wording():
    scored = prepare(_Tokenizer(), _score())
    text = _Tokenizer().decode(scored[0].prompt_ids)
    assert "Question: How frustrated is the customer?" in text
    assert "0: calm" in text
    assert "1: upset" in text
    assert text.endswith("Answer with the number of one level only.\n")
    assert scored[0].names == ["0", "1"]
    assert scored[0].label_ids == [ord("0"), ord("1")]

    prepared = prepare(_YesNoTokenizer(), _yes_no())
    text = _YesNoTokenizer().decode(prepared[0].prompt_ids)
    assert "Is the following true? The customer needs an answer today." in text
    assert "yes: Needs a reply today" in text
    assert "no: Can wait" in text
    assert text.endswith("Answer with yes or no only.\n")
    assert prepared[0].label_ids == [1000, 1001]
    assert prepared[0].names == ["yes", "no"]


def test_rendered_prompts_match_the_tokenizer_path():
    tokenizer = _Tokenizer()
    rendered = prompts_for(_choice(), _render, tokenizer.encode)
    assert rendered[0].prompt_ids == prepare(tokenizer, _choice())[0].prompt_ids
    assert rendered[0].label_ids == [ord("A"), ord("B")]


def test_score_is_the_expected_level_and_ids_come_back_when_asked():
    prepared = prepare(_Tokenizer(), _score())
    body = {**_score(), "return_prompt_token_ids": True, "temperature": 1}
    answer = build_response(body, prepared, [([0.0, math.log(3)], math.log(1 + 3 + 1))])["answers"]["frustration"]
    assert answer["score"] == pytest.approx(0.75)
    assert answer["prompt_token_ids"] == prepared[0].prompt_ids
    assert answer["label_token_ids"] == prepared[0].label_ids
    assert "choice" not in answer


def test_yes_no_has_probabilities_without_a_score():
    prepared = prepare(_YesNoTokenizer(), _yes_no())
    answer = build_response(_yes_no(), prepared, [([math.log(3), 0.0], math.log(3 + 1 + 1))])["answers"]["urgent"]
    assert answer["probabilities"]["yes"] == pytest.approx(0.75)
    assert set(answer) == {"type", "probabilities", "label_mass"}
    assert answer["label_mass"] == pytest.approx((3 + 1) / (3 + 1 + 1))


@pytest.mark.parametrize(("body", "fragment"), [
    ({"input": "   ", "questions": _choice()["questions"]}, "blank"),
    ({"input": "ticket", "questions": []}, "at least one"),
    ({"input": "ticket", "questions": [_choice()["questions"][0], _choice()["questions"][0]]}, "repeated"),
    ({"input": "ticket", "questions": _choice()["questions"], "temperature": 0}, "above 0"),
    ({"input": "ticket", "questions": _choice()["questions"], "temperature": False}, "above 0"),
    ({"input": "ticket", "questions": _choice()["questions"], "stream": False}, "unknown field"),
    ({"input": "ticket", "questions": _choice()["questions"], "chat_template_kwargs": {"enable_thinking": True}}, "enable_thinking"),
    ({"input": "ticket", "questions": [{"id": "q", "type": "choice", "question": "Which?",
                                        "options": [{"name": "billing"}, {"name": "a\nb"}]}]}, "line breaks"),
    ({"input": "ticket", "questions": [{"id": "q", "type": "choice", "question": "Which?",
                                        "options": [{"name": "Same"}, {"name": " same "}]}]}, "repeats"),
    ({"input": "ticket", "questions": [{"id": "q", "type": "score", "question": "How?", "levels": ["only"]}]}, "2 to 10"),
    ({"input": "ticket", "questions": [{"id": "q", "type": "yes_no", "question": "  "}]}, "blank"),
    ({"input": "ticket", "questions": [{"id": "q", "type": "maybe", "question": "Which?"}]}, "unknown question type"),
])
def test_invalid_requests_are_refused(body, fragment):
    with pytest.raises(DecisionError, match=fragment):
        prepare(_Tokenizer(), body)


def test_a_prompt_past_the_context_window_is_refused():
    with pytest.raises(DecisionError, match="context length"):
        prepare(_Tokenizer(), _choice(), context_len=8)


def test_two_vocabulary_shards_rebuild_the_full_logsumexp():
    rows = [[1.0, 3.0], [0.0, 2.0]]
    flat = [value for row in rows for value in row]
    peak = max(flat)
    expected = peak + math.log(math.fsum(math.exp(value - peak) for value in flat))
    logits, logsumexp = reduce_vocab_shards(rows, [0, 3], 2)
    assert logits == [1.0, 2.0]
    assert logsumexp == pytest.approx(expected)
    cool = build_response(_choice(), prepare(_Tokenizer(), _choice()), [(logits, logsumexp)])
    hot = build_response({**_choice(), "temperature": 4}, prepare(_Tokenizer(), _choice()), [(logits, logsumexp)])
    assert cool["answers"]["team"]["label_mass"] == pytest.approx(hot["answers"]["team"]["label_mass"])
    with pytest.raises(ValueError, match="outside the vocabulary"):
        reduce_vocab_shards(rows, [4], 2)


def test_cuda_decisions_scores_through_the_template():
    pytest.importorskip("tokenizers")
    from tensorfold.cuda.http import make_handler as cuda_handler
    from tensorfold.cuda.server import App
    from tensorfold.cuda.turns import Turns

    class Template:
        def render(self, messages, *, tools, enable_thinking, extra=None):
            assert enable_thinking is False
            assert tools is None
            return messages[0]["content"] + "\n"

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return type("Encoded", (), {"ids": [ord(char) for char in text]})()

    class Engine:
        def score_labels(self, prompt, labels):
            self.seen = (list(prompt), list(labels))
            return [0.0, 2.0], 2.0

    app = object.__new__(App)
    app.template = Template()
    app.tok = Tok()
    app.engine = Engine()
    app.context_window = 0
    app.turns = Turns()
    payload = app.decisions(_choice())
    assert payload["answers"]["team"]["choice"] == "technical"
    assert payload["usage"]["completion_tokens"] == 0
    assert app.engine.seen[1] == [ord("A"), ord("B")]

    missing = object.__new__(App)
    missing.engine = object()
    status, raw = _cuda_post(cuda_handler, missing, _choice())
    assert status == 400
    assert "does not score" in raw


def _cuda_post(factory, app, body, path="/v1/decisions"):
    from io import BytesIO

    payload = json.dumps(body).encode()
    incoming = (f"POST {path} HTTP/1.0\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\n\r\n").encode() + payload

    class Connection:
        def __init__(self):
            self.output = bytearray()

        def makefile(self, *args):
            return BytesIO(incoming)

        def sendall(self, data):
            self.output.extend(data)

    connection = Connection()
    factory(app)(connection, ("127.0.0.1", 0), None)
    headers, response = bytes(connection.output).split(b"\r\n\r\n", 1)
    return int(headers.split()[1]), response.decode()
