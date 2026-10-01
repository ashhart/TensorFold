"""Decision prompts and probabilities follow SGLang prompt format version 1, without a model loaded."""

import json
import math

import pytest

from tensorfold.server.decisions import DecisionError, build_response, prepare, prompts_for, reduce_vocab_shards
from tensorfold.server.errors import RequestError
from tensorfold.server.http import make_handler
from tensorfold.server.scheduler import ChatJob, Scheduler
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


def test_probabilities_at_tiny_temperature_remain_finite():
    body = {**_choice(), "temperature": 1e-320}
    prepared = prepare(_Tokenizer(), body)
    for logits, expected in (([1.0, 2.0], [0.0, 1.0]), ([2.0, 2.0], [0.5, 0.5])):
        answer = build_response(body, prepared, [(logits, 3.0)])["answers"]["team"]
        assert list(answer["probabilities"].values()) == expected
        assert math.isfinite(answer["label_mass"])


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


def test_http_decision_error_is_400_and_other_failures_are_500():
    class Client:
        def decisions(self, body):
            raise DecisionError("yes is not one token")

    class Broken:
        def decisions(self, body):
            raise RuntimeError("engine broke")

    status, raw = post(Client(), _choice(), path="/v1/decisions")
    assert status == 400
    assert json.loads(raw)["error"]["message"] == "yes is not one token"
    status, raw = post(Broken(), _choice(), path="/v1/decisions")
    assert status == 500
    assert json.loads(raw)["error"]["message"] == "engine broke"


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


def test_a_decision_fills_beside_a_live_stream():
    class Engine:
        active_count = 1
        prefill_chunks = 0
        streams: list = []
        finished_caches: dict = {}
        round_stats: list = []
        seen = None

        def prompt_chunks(self, ids):
            return self

        def floor(self, n):
            return 0

        def begin_stream(self, stream, **kwargs):
            self.seen = (self.active_count, tuple(stream.label_ids))
            stream.finished = True
            stream.scored = ([4.0, 0.0], 1.0)
            stream.cached_tokens = 0
            stream.emitted = []
            return iter(())

        def step(self):
            return {}

        def discard_stream(self, stream):
            pass

    engine = Engine()
    scheduler = Scheduler(engine, lanes=2, eos_ids=frozenset())
    scheduler.start()
    job = ChatJob(job_id="decision-1", prompt_ids=[7, 8], max_tokens=1, temperature=0.0,
                  drafts=False, label_ids=(4, 5))
    try:
        scheduler.submit(job)
        assert job.done.wait(2.0)
    finally:
        scheduler.stop()
    assert job.error is None
    assert job.scored == ([4.0, 0.0], 1.0)
    assert engine.seen == (1, (4, 5))


def test_a_decision_prefill_reads_the_last_row_and_draws_nothing():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    from tensorfold.engine.family_prefill import FamilyPrefill, drain
    from tensorfold.engine.lane_engine import LaneStream
    from tensorfold.engine.prefill_plan import PromptChunks

    class Engine(FamilyPrefill):
        streams: list = []

        def prompt_chunks(self, ids):
            return PromptChunks(None, len(ids), step=max(len(ids), 1))

        def _family_start(self, cache, cached_tokens, chunks):
            return [], 0

        def _family_feed_steps(self, tokens, cache, chunks, *args, **kwargs):
            yield from ()
            return mx.array([[1.0, 3.0, 0.0]])

        def _family_first(self, *args, **kwargs):
            raise AssertionError("a decision draws no token")

        def copy_single_cache(self, cache):
            return cache

    engine = Engine()
    engine.model = type("Model", (), {"head": staticmethod(lambda hidden: hidden)})()
    stream = LaneStream(stream_id="d", prompt_ids=[7, 8], max_new_tokens=1)
    stream.label_ids = (0, 1)
    drain(engine._family_prefill_steps(stream, cache=None, cached_tokens=0, checkpoints_at=()))
    assert stream.finished
    assert stream.finish_reason == "decision"
    assert stream.emitted == []
    assert stream.scored[0] == pytest.approx([1.0, 3.0])
    assert stream.scored[1] == pytest.approx(3.0 + math.log(math.exp(-2.0) + 1.0 + math.exp(-3.0)))


class _KeepTokenizer:
    """One character a token, with a generation suffix a history boundary can sit in front of."""

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(chr(int(token)) for token in ids)

    def apply_chat_template(self, messages, **kwargs):
        body = "\n".join(str(message.get("content", "")) for message in messages)
        text = f"<u>{body}</u>"
        if kwargs.get("add_generation_prompt", True):
            text += "<a>"
        if kwargs.get("tokenize", True) is False:
            return text
        return self.encode(text)


def _long_choice(question: str) -> dict:
    return {
        "input": "a" * 600,
        "questions": [{
            "id": "q",
            "type": "choice",
            "question": question,
            "options": [{"name": "one"}, {"name": "two"}],
        }],
        "return_prompt_token_ids": True,
    }


def test_a_decision_keeps_a_shared_input_and_a_later_chat_resumes_it():
    pytest.importorskip("mlx.core")
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tests.lane_fakes import FakeEngine
    from tests.test_lane_server import make_app

    class GridEngine(FakeEngine):
        def __init__(self, model=None, **kwargs):
            super().__init__(model, **kwargs)
            self.prefill_plan = PrefillPlan(128)

    def open_app():
        return make_app(engine_factory=GridEngine, checkpoint_slots=8, lanes=1, tokenizer=_KeepTokenizer())

    fresh = open_app()
    resume = open_app()
    chat_fresh = open_app()
    try:
        cold = fresh.decisions(_long_choice("North stair."))
        cold_ids = cold["answers"]["q"]["prompt_token_ids"]
        assert fresh.engine.prefill_calls[-1][1] == 0
        other = resume.decisions(_long_choice("South stair."))
        other_ids = other["answers"]["q"]["prompt_token_ids"]
        warm = resume.decisions(_long_choice("North stair."))
        warm_ids = warm["answers"]["q"]["prompt_token_ids"]
        cached = resume.engine.prefill_calls[-1][1]
        split = next(i for i, (left, right) in enumerate(zip(other_ids, warm_ids)) if left != right)
        assert warm_ids == cold_ids
        assert 0 < cached < split
        assert split - cached <= 128
        assert warm["answers"]["q"]["probabilities"] == cold["answers"]["q"]["probabilities"]
        assert warm["answers"]["q"]["label_mass"] == cold["answers"]["q"]["label_mass"]
        pinned = [len(entry.tokens) for entry in resume.checkpoints._entries if entry.pinned]
        assert cached in pinned
        messages = [{"role": "user", "content": ("a" * 600) + "\n\nSay ready."}]
        resumed_chat = resume.chat(messages, max_tokens=4, sampling={"draft": False, "enable_thinking": False})
        fresh_chat = chat_fresh.chat(messages, max_tokens=4, sampling={"draft": False, "enable_thinking": False})
        assert resumed_chat["cached_tokens"] == cached
        assert resumed_chat["runtime"]["token_sha"] == fresh_chat["runtime"]["token_sha"]
        assert fresh_chat["cached_tokens"] == 0
    finally:
        fresh.close()
        resume.close()
        chat_fresh.close()


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
    ({"input": "ticket", "questions": _choice()["questions"],
      "chat_template_kwargs": {"enable_thinking": True}}, "enable_thinking"),
    ({**_choice(), "chat_template_kwargs": []}, "must be an object"),
    ({**_choice(), "chat_template_kwargs": False}, "must be an object"),
    ({**_choice(), "chat_template_kwargs": ""}, "must be an object"),
    ({**_choice(), "chat_template_kwargs": {"reasoning_effort": "high"}}, "unknown field"),
    ({"input": "ticket", "questions": [{"id": "q", "type": "choice", "question": "Which?",
                                        "options": [{"name": "billing"}, {"name": "a\nb"}]}]}, "line breaks"),
    ({"input": "ticket", "questions": [{"id": "q", "type": "choice", "question": "Which?",
                                        "options": [{"name": "Same"}, {"name": " same "}]}]}, "repeats"),
    ({"input": "ticket", "questions": [{"id": "q", "type": "score", "question": "How?",
                                        "levels": ["only"]}]}, "2 to 10"),
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


def test_cuda_decision_error_is_400_and_other_failures_are_500():
    pytest.importorskip("tokenizers")
    from tensorfold.cuda.http import make_handler as cuda_handler

    class Client:
        def decisions(self, body):
            raise DecisionError("yes is not one token")

    class Broken:
        def decisions(self, body):
            raise RuntimeError("engine broke")

    status, raw = _cuda_post(cuda_handler, Client(), _choice())
    assert status == 400
    assert json.loads(raw)["error"]["message"] == "yes is not one token"
    status, raw = _cuda_post(cuda_handler, Broken(), _choice())
    assert status == 500
    assert json.loads(raw)["error"]["message"] == "engine broke"


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
