"""Decision models with a joint schema head (Cloudflare/clef): prompt encoding, request translation and routes."""

from __future__ import annotations

import json

import pytest

from tensorfold.server.decisions import DecisionError
from tensorfold.server.joint_schema import JointSchema, encode, question_options

torch = pytest.importorskip("torch")


def chars(text: str) -> list[int]:
    return [ord(char) for char in text]


def text(ids) -> str:
    return "".join(chr(i) for i in ids)


QUESTIONS = {
    "status": {"type": "choice", "instructions": "What is the invoice status?",
               "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": None}},
    "urgency": {"type": "score", "criteria": ["Can wait", "Today"]},
    "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
}


def test_spans_mark_each_instruction_and_option_in_the_release_wording():
    encoded = encode(chars, {"total": 1250, "a": 1}, QUESTIONS)
    prompt = text(encoded.ids)
    assert prompt.startswith("<|im_start|>system\nRead the complete state")
    assert 'STATE:\n{"a":1,"total":1250}\n\nSCHEMA FIELDS:\n\nFIELD 1\nID: status\nTYPE: choice' in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    status, urgency, large = encoded.questions
    assert text(encoded.ids[slice(*status.span)]) == "What is the invoice status?"
    assert text(encoded.ids[slice(*urgency.span)]) == "urgency"          # no instructions: the id stands in
    assert status.option_ids == ("draft", "overdue", "paid")              # choices sorted by id
    assert [text(encoded.ids[a:b]) for a, b in status.option_spans] == [
        '{"option_id":"draft"}', '{"description":"Invoice is past due.","option_id":"overdue"}',
        '{"description":"Invoice is paid.","option_id":"paid"}']
    assert urgency.option_ids == ("0", "1") and large.option_ids == ("true", "false")
    assert [q.type for q in encoded.questions] == [1, 2, 0]


def test_noul_descriptions_default_and_can_be_replaced():
    assert question_options({"type": "noul"})[0] == ("true", "The proposition is true or the answer is yes.")
    assert question_options({"type": "noul", "criteria": {"false": "No."}})[1] == ("false", "No.")


def test_the_state_is_cut_to_fit_and_an_oversized_schema_is_refused():
    full = encode(chars, "x" * 50, QUESTIONS)
    cut = encode(chars, "x" * 50, QUESTIONS, max_length=len(full.ids) - 20)
    assert len(cut.ids) == len(full.ids) - 20
    assert text(cut.ids[slice(*cut.questions[0].span)]) == "What is the invoice status?"
    with pytest.raises(DecisionError, match="before the state"):
        encode(chars, "x", QUESTIONS, max_length=100)


def test_the_release_head_scores_every_option_of_every_question():
    from tensorfold.server.joint_schema import _head_class

    torch.manual_seed(0)
    head = _head_class()(hidden_size=16, width=8, routing_layers=1, layers=1, heads=2, feedforward=16).eval()
    encoded = encode(chars, "state", QUESTIONS)
    hidden = torch.randn(len(encoded.ids), 16)
    table = torch.randn(256, 16)
    with torch.inference_mode():
        logits = head(hidden, torch.tensor(encoded.ids), encoded.questions, lambda ids: table[ids % 256])
    assert [len(row) for row in logits] == [3, 2, 2]
    assert all(torch.isfinite(row).all() for row in logits)


class _Engine:
    def hidden_rows(self, prompt):
        self.prompt = list(prompt)
        return torch.zeros((len(prompt), 4))

    def head_rows(self, ids):
        return torch.zeros((len(ids), 4))


def _ranked_head(hidden, ids, questions, rows):
    """Option logits 0, 1, 2 ... in the release's option order: its last option always wins."""

    return [torch.arange(len(q.option_ids), dtype=torch.float32) for q in questions]


def _joint() -> JointSchema:
    return JointSchema(_ranked_head, _Engine(), chars)


def _decisions_body() -> dict:
    return {"model": "clef", "input": "Checkout is down.", "questions": [
        {"id": "team", "type": "choice", "question": "Who handles it?",
         "options": [{"name": "zeta", "description": "Last by name"}, {"name": "alpha"}]},
        {"id": "urgency", "type": "score", "question": "How urgent?", "levels": ["low", "mid", "high"]},
        {"id": "down", "type": "yes_no", "question": "Is a service down?", "yes": "It is down."}]}


def test_decisions_answers_in_the_request_order_from_the_head():
    payload = _joint().decisions(_decisions_body())
    team, urgency, down = (payload["answers"][k] for k in ("team", "urgency", "down"))
    assert list(team["probabilities"]) == ["zeta", "alpha"]              # the request's order, not the release's
    assert team["choice"] == "zeta"                                       # the release sorts it last: logit 1
    assert team["label_mass"] is None
    assert urgency["score"] == pytest.approx(sum(i * p for i, p in enumerate(urgency["probabilities"].values())))
    assert urgency["probabilities"]["2"] > urgency["probabilities"]["0"]
    assert down["probabilities"]["no"] > down["probabilities"]["yes"]     # false is the release's second option
    assert payload["usage"]["completion_tokens"] == 0


def test_decisions_carry_question_wording_and_descriptions_to_the_prompt():
    joint = _joint()
    joint.decisions(_decisions_body())
    prompt = text(joint.engine.prompt)
    assert "STATE:\nCheckout is down.\n" in prompt
    assert 'INSTRUCTION: Who handles it?\nALLOWED OPTIONS:\nOPTION 1: {"option_id":"alpha"}' in prompt
    assert '{"description":"It is down.","option_id":"true"}' in prompt
    assert '{"description":"The proposition is false or the answer is no.","option_id":"false"}' in prompt


def test_temperature_flattens_the_probabilities():
    cool = _joint().decisions(_decisions_body())["answers"]["team"]["probabilities"]["zeta"]
    hot = _joint().decisions({**_decisions_body(), "temperature": 10})["answers"]["team"]["probabilities"]["zeta"]
    assert 0.5 < hot < cool


def test_systemone_answers_in_the_release_shape():
    body = {"model": "clef", "state": "Checkout is down.", "questions": QUESTIONS}
    payload = _joint().systemone(body)
    status, urgency, large = (payload["answers"][k] for k in ("status", "urgency", "large"))
    assert status["choice"] == "paid" and list(status["probabilities"]) == ["paid", "overdue", "draft"]
    assert urgency["legend"] == {"0": "Can wait", "1": "Today"} and urgency["score"] == pytest.approx(0.7311)
    assert large == {"type": "noul", "noul": pytest.approx(0.2689)}
    assert payload["usage"]["output_tokens"] == 0


@pytest.mark.parametrize("body, fragment", [
    ({"state": "s", "questions": QUESTIONS}, "model and state"),
    ({"model": "m", "state": "s", "questions": {}}, "at least one question"),
    ({"model": "m", "state": "s", "questions": {"q": {"type": "maybe"}}}, "type must be"),
    ({"model": "m", "state": "s", "questions": {"q": {"type": "choice", "criteria": {}}}}, "non-empty object"),
    ({"model": "m", "state": "s", "questions": {"q": {"type": "score", "criteria": "abc"}}}, "non-empty list"),
    ({"model": "m", "state": "s", "questions": {"q": {"type": "noul", "criteria": {"maybe": "x"}}}}, "true and false"),
    ({"model": "m", "state": "s", "images": ["x"], "questions": QUESTIONS}, "not served yet"),
])
def test_systemone_refuses_malformed_bodies(body, fragment):
    with pytest.raises(DecisionError, match=fragment):
        _joint().systemone(body)


def test_an_engine_without_hidden_rows_cannot_serve_a_head():
    with pytest.raises(ValueError, match="cannot run"):
        JointSchema(_ranked_head, object(), chars)


def test_cuda_routes_answer_through_the_head():
    pytest.importorskip("tokenizers")
    from tensorfold.cuda.http import make_handler
    from tensorfold.cuda.server import App
    from tensorfold.cuda.turns import Turns
    from tests.test_decisions import _cuda_post

    app = object.__new__(App)
    app.engine, app.joint, app.turns = _Engine(), _joint(), Turns()
    status, raw = _cuda_post(make_handler, app, _decisions_body())
    assert status == 200 and json.loads(raw)["answers"]["team"]["choice"] == "zeta"
    status, raw = _cuda_post(make_handler, app, {"model": "clef", "state": "s", "questions": QUESTIONS},
                             path="/v1/systemone")
    assert status == 200 and json.loads(raw)["answers"]["status"]["choice"] == "paid"
    status, raw = _cuda_post(make_handler, app, {"model": "clef", "state": "s", "questions": {}}, path="/v1/systemone")
    assert status == 400 and "at least one question" in raw

    plain = object.__new__(App)
    plain.engine, plain.turns = _Engine(), Turns()
    status, raw = _cuda_post(make_handler, plain, {"model": "m", "state": "s", "questions": QUESTIONS},
                             path="/v1/systemone")
    assert status == 400 and "joint schema head" in raw


def test_unquantized_weights_need_a_decision_head(tmp_path):
    from tensorfold.families.qwen3_5 import cuda_engine

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {}}))
    with pytest.raises(ValueError, match="only for a decision model"):
        cuda_engine(tmp_path, no_drafts=True)
