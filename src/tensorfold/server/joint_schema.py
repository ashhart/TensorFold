"""Decision models with a joint schema head (Cloudflare's Clef): every option of every question scored in one prefill.

The head reads the backbone's final normed state at every prompt row, not the next-token logits, so these models
answer ``/v1/decisions`` through it instead of label tokens. ``/v1/systemone`` takes the model's own request body.
Prompt wording, option order and the head follow ``joint_schema_model.py`` in Cloudflare/clef (Apache-2.0).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorfold.server.decisions import (_CHOICE_FIELDS, _SCORE_FIELDS, _YES_NO_FIELDS, DecisionError, _choice,
                                         _render_text, _score, _softmax, _unknown, _validate_body)

HEAD_FILE = "joint_head.safetensors"
HEAD_CONFIG = "joint_head_config.json"
MAX_LENGTH = 16384                     # the release's encode_record default
SYSTEM_PROMPT = ("Read the complete state and schema. Decide every field jointly. Each answer "
                 "must be exactly one of that field's allowed options.")
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
NOUL_DEFAULTS = {"true": "The proposition is true or the answer is yes.",
                 "false": "The proposition is false or the answer is no."}


def has_head(model_dir: Path) -> bool:
    return (Path(model_dir) / HEAD_FILE).exists()


def render(value: Any) -> str:
    """The release's state and option text: strings as they are, anything else as sorted compact JSON."""

    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    """(option id, description) in the release's order: true then false, choices sorted by id, levels by index."""

    kind = str(question["type"])
    if kind == "noul":
        criteria = {**NOUL_DEFAULTS, **(question.get("criteria") or {})}
        return [(key, criteria[key]) for key in ("true", "false")]
    if kind == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


@dataclass(frozen=True)
class Question:
    id: str
    type: int
    span: tuple[int, int]                         # the instruction's rows
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class Encoded:
    ids: list[int]
    questions: tuple[Question, ...]


def encode(encode_text: Callable[[str], list[int]], state: Any, questions: dict[str, dict[str, Any]],
           max_length: int = MAX_LENGTH) -> Encoded:
    """The release's ``encode_record`` for text: each piece encoded alone, the state cut to fit ``max_length``."""

    schema = encode_text("\n\nSCHEMA FIELDS:\n")
    found = []
    for index, (question_id, question) in enumerate(questions.items()):
        schema += encode_text(f"\nFIELD {index + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: ")
        start = len(schema)
        instructions = question.get("instructions")
        schema += encode_text(render(str(question_id) if instructions is None or instructions == "" else instructions))
        span = (start, len(schema))
        schema += encode_text("\nALLOWED OPTIONS:\n")
        spans, option_ids = [], []
        for option_index, (option_id, description) in enumerate(question_options(question)):
            schema += encode_text(f"OPTION {option_index + 1}: ")
            start = len(schema)
            semantics = {"option_id": option_id} if description is None else \
                {"option_id": option_id, "description": description}
            schema += encode_text(render(semantics))
            spans.append((start, len(schema)))
            option_ids.append(option_id)
            schema += encode_text("\n")
        schema += encode_text("END FIELD\n")
        found.append(Question(str(question_id), QUESTION_TYPES[str(question["type"])], span, tuple(spans),
                              tuple(option_ids)))
    prefix = encode_text(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
    suffix = encode_text("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    fixed = len(prefix) + len(schema) + len(suffix)
    if fixed > max_length:
        raise DecisionError(f"the schema needs {fixed} tokens before the state; the limit is {max_length}")
    state_ids = encode_text(render(state))[:max_length - fixed]
    shift = len(prefix) + len(state_ids)
    moved = tuple(Question(q.id, q.type, (q.span[0] + shift, q.span[1] + shift),
                           tuple((a + shift, b + shift) for a, b in q.option_spans), q.option_ids) for q in found)
    return Encoded(prefix + state_ids + schema + suffix, moved)


def _head_class():
    import torch
    from torch.nn import functional

    class EvidenceRoutingLayer(torch.nn.Module):
        def __init__(self, width: int, heads: int, feedforward: int) -> None:
            super().__init__()
            self.query_norm = torch.nn.LayerNorm(width)
            self.memory_norm = torch.nn.LayerNorm(width)
            self.attention = torch.nn.MultiheadAttention(width, heads, batch_first=True)
            self.feedforward_norm = torch.nn.LayerNorm(width)
            self.feedforward = torch.nn.Sequential(torch.nn.Linear(width, feedforward), torch.nn.GELU(),
                                                   torch.nn.Dropout(0.0), torch.nn.Linear(feedforward, width),
                                                   torch.nn.Dropout(0.0))

        def forward(self, queries, memory):
            memory = self.memory_norm(memory)
            routed, _ = self.attention(self.query_norm(queries), memory, memory, need_weights=False)
            queries = queries + routed
            return queries + self.feedforward(self.feedforward_norm(queries))

    class JointSchemaHead(torch.nn.Module):
        """The release's head for one record; parameter names match ``joint_head.safetensors``."""

        def __init__(self, hidden_size: int, width: int, routing_layers: int, layers: int, heads: int,
                     feedforward: int) -> None:
            super().__init__()
            self.hidden_norm = torch.nn.LayerNorm(hidden_size)
            for name in ("memory", "question", "option_question", "global", "option_context", "option_lexical"):
                setattr(self, f"{name}_projection", torch.nn.Linear(hidden_size, width, bias=False))
            self.type_embedding = torch.nn.Embedding(3, width)
            self.evidence_layers = torch.nn.ModuleList(
                [EvidenceRoutingLayer(width, heads, feedforward) for _ in range(routing_layers)])
            self.option_summary_norm = torch.nn.LayerNorm(width)
            self.layers = torch.nn.ModuleList([torch.nn.TransformerDecoderLayer(
                d_model=width, nhead=heads, dim_feedforward=feedforward, dropout=0.0, activation="gelu",
                batch_first=True, norm_first=True) for _ in range(layers)])
            self.field_norm = torch.nn.LayerNorm(width)
            self.option_norm = torch.nn.LayerNorm(width)
            self.residual_scorer = torch.nn.Sequential(torch.nn.Linear(width * 4, width), torch.nn.GELU(),
                                                       torch.nn.Dropout(0.0), torch.nn.Linear(width, 1))
            self.prior_logit_scale = torch.nn.Parameter(torch.zeros(()))
            self.joint_logit_scale = torch.nn.Parameter(torch.zeros(()))
            self.residual_gate = torch.nn.Parameter(torch.zeros(()))

        def forward(self, hidden, ids, questions: tuple[Question, ...], output_rows: Callable) -> list:
            """``hidden``: (rows, hidden_size) final normed states; ``output_rows(ids)``: the LM head's rows."""

            hidden = self.hidden_norm(hidden)
            memory = self.memory_projection(hidden).unsqueeze(0)
            last = hidden[-1]
            mean = lambda span: hidden[span[0]:span[1]].mean(dim=0)
            vectors = torch.stack([mean(q.span) for q in questions])
            types = torch.tensor([q.type for q in questions], device=hidden.device)
            contexts = [torch.stack([mean(span) for span in q.option_spans]) for q in questions]
            lexicals = [torch.stack([output_rows(ids[a:b]).mean(dim=0) for a, b in q.option_spans])
                        for q in questions]
            counts = [len(q.option_spans) for q in questions]
            routed = torch.cat([self.option_context_projection(c) + self.option_lexical_projection(x)
                                + self.option_question_projection(vectors[i]).unsqueeze(0)
                                for i, (c, x) in enumerate(zip(contexts, lexicals))], dim=0).unsqueeze(0)
            for layer in self.evidence_layers:
                routed = layer(routed, memory)
            options_by_field = list(torch.split(routed[0], counts, dim=0))
            base = self.question_projection(vectors)
            summaries = []
            for field, options in zip(base, options_by_field):
                weights = torch.softmax(torch.matmul(options, field) / math.sqrt(options.shape[-1]), dim=0)
                summaries.append(torch.sum(weights.unsqueeze(-1) * options, dim=0))
            fields = (base + self.option_summary_norm(torch.stack(summaries))
                      + self.global_projection(last).unsqueeze(0) + self.type_embedding(types)).unsqueeze(0)
            for layer in self.layers:
                fields = layer(fields, memory)
            fields = self.field_norm(fields[0])
            prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
            joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
            logits = []
            for index, (field, lexical, options) in enumerate(zip(fields, lexicals, options_by_field)):
                anchor = functional.normalize(vectors[index] + last, dim=-1)
                prior = prior_scale * torch.matmul(functional.normalize(lexical, dim=-1), anchor)
                options = self.option_norm(options)
                field = field.unsqueeze(0).expand_as(options)
                cosine = functional.cosine_similarity(field, options, dim=-1)
                features = torch.cat([field, options, field * options, torch.abs(field - options)], dim=-1)
                joint = joint_scale * cosine + self.residual_scorer(features).squeeze(-1)
                logits.append(prior + torch.sigmoid(self.residual_gate) * joint)
            return logits

    return JointSchemaHead


class JointSchema:
    """A head over an engine with ``hidden_rows`` (every prompt row's final normed state) and ``head_rows``."""

    def __init__(self, head: Any, engine: Any, encode_text: Callable[[str], list[int]]) -> None:
        if not hasattr(engine, "hidden_rows"):
            raise ValueError("this checkpoint has a joint schema head, which this model's CUDA engine cannot run")
        self.head, self.engine, self.encode_text = head, engine, encode_text

    @classmethod
    def load(cls, model_dir: Path, engine: Any, encode_text: Callable[[str], list[int]]) -> JointSchema:
        """The release's head in bf16 on the GPU, warmed by one small decision so no request pays the kernel builds."""

        import torch
        from safetensors.torch import load_file

        model_dir = Path(model_dir)
        head = _head_class()(**json.loads((model_dir / HEAD_CONFIG).read_text()))
        head.load_state_dict(load_file(str(model_dir / HEAD_FILE)), strict=True)
        joint = cls(head.to(device="cuda", dtype=torch.bfloat16).eval(), engine, encode_text)
        joint.logits("warm-up", {"ready": {"type": "noul", "instructions": "Is the server ready?"}})
        return joint

    def logits(self, state: Any, questions: dict[str, dict[str, Any]]) -> tuple[Encoded, list[list[float]]]:
        """One prefill, then every question's option logits in the release's option order."""

        import torch

        encoded = encode(self.encode_text, state, questions)
        hidden = self.engine.hidden_rows(encoded.ids)
        ids = torch.tensor(encoded.ids, dtype=torch.int64, device=hidden.device)
        with torch.inference_mode():
            out = self.head(hidden, ids, encoded.questions, self.engine.head_rows)
        return encoded, [row.float().tolist() for row in out]

    # -- /v1/decisions: SGLang's request and response, answered by the head ------------------

    def decisions(self, body: dict[str, Any]) -> dict[str, Any]:
        _validate_body(body)
        state = body.get("input")
        if not _render_text(state).strip():
            raise DecisionError("input must not be blank")
        questions, names = {}, {}
        kinds = {item["id"]: item.get("type") for item in body["questions"]}
        for item in body["questions"]:
            try:
                questions[item["id"]], names[item["id"]] = _as_question(item)
            except DecisionError as exc:
                raise DecisionError(f"question {item['id']!r}: {exc}") from exc
        encoded, logits = self.logits(state, questions)
        temperature = float(body.get("temperature") or 1.0)
        answers = {}
        for question, row in zip(encoded.questions, logits):
            kind = kinds[question.id]
            by_id = dict(zip(question.option_ids, _softmax(row, temperature)))
            ordered = names[question.id]                     # (response name, option id) in the request's order
            probabilities = {name: by_id[option] for name, option in ordered}
            answer: dict[str, Any] = {"type": kind, "probabilities": probabilities, "label_mass": None}
            if kind == "choice":
                answer["choice"] = max(probabilities, key=probabilities.__getitem__)
            elif kind == "score":
                answer["score"] = math.fsum(index * p for index, p in enumerate(probabilities.values()))
            if body.get("return_prompt_token_ids"):
                answer["prompt_token_ids"] = encoded.ids
            answers[question.id] = answer
        tokens = len(encoded.ids)
        return {"object": "decisions", "model": body.get("model") or "default", "prompt_format_version": None,
                "answers": answers,
                "usage": {"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens}}

    # -- /v1/systemone: the release's own request and response -------------------------------

    def systemone(self, body: dict[str, Any]) -> dict[str, Any]:
        questions = body.get("questions")
        if not isinstance(body.get("model"), str) or "state" not in body:
            raise DecisionError("model and state are required")
        if body.get("images") or body.get("videos"):
            raise DecisionError("image and video states are not served yet")
        if not isinstance(questions, dict) or not questions:
            raise DecisionError("at least one question is required")
        for question_id, question in questions.items():
            if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
                raise DecisionError(f"{question_id}: type must be noul, choice, or score")
            criteria = question.get("criteria")
            if question["type"] == "choice" and (not isinstance(criteria, dict) or not criteria):
                raise DecisionError(f"{question_id}: criteria must be a non-empty object of options")
            if question["type"] == "score" and (not isinstance(criteria, list) or not criteria):
                raise DecisionError(f"{question_id}: criteria must be a non-empty list of levels")
            if question["type"] == "noul" and criteria is not None and (
                    not isinstance(criteria, dict) or set(criteria) - {"true", "false"}):
                raise DecisionError(f"{question_id}: noul criteria may only describe true and false")
        encoded, logits = self.logits(body["state"], questions)
        answers = {q.id: systemone_answer(questions[q.id], dict(zip(q.option_ids, _softmax(row, 1.0))))
                   for q, row in zip(encoded.questions, logits)}
        return {"model": body["model"], "answers": answers,
                "usage": {"input_tokens": len(encoded.ids), "output_tokens": 0}}


def _as_question(item: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """A ``/v1/decisions`` question as the release's, with (response name, option id) pairs in request order."""

    kind = item.get("type")
    if kind == "choice":
        _unknown(item, _CHOICE_FIELDS)
        names, details, _ = _choice(item)
        criteria = {name: detail or None for name, detail in zip(names, details)}
        question = {"type": "choice", "criteria": criteria}
        pairs = [(name, name) for name in names]
    elif kind == "score":
        _unknown(item, _SCORE_FIELDS)
        names, details, _ = _score(item)
        question = {"type": "score", "criteria": details}
        pairs = [(name, name) for name in names]
    elif kind == "yes_no":
        _unknown(item, _YES_NO_FIELDS)
        criteria = {key: _render_text(item[field]) for key, field in (("true", "yes"), ("false", "no"))
                    if item.get(field) is not None}
        question = {"type": "noul", **({"criteria": criteria} if criteria else {})}
        pairs = [("yes", "true"), ("no", "false")]
    else:
        raise DecisionError(f"unknown question type {kind!r}")
    text = _render_text(item.get("question"))
    if not text.strip():
        raise DecisionError("a question must not be blank")
    question["instructions"] = item["question"]            # the release renders objects as sorted JSON
    return question, pairs


def systemone_answer(question: dict[str, Any], probabilities: dict[str, float]) -> dict[str, Any]:
    """The release's SystemOne answer for one question from its per-option probabilities."""

    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(option) for option in question["criteria"]]
        choice = max(options, key=probabilities.__getitem__)
        return {"type": "choice", "choice": choice, "confidence": round(probabilities[choice], 4),
                "probabilities": {option: round(probabilities[option], 4) for option in options}}
    levels = [str(index) for index in range(len(question["criteria"]))]
    return {"type": "score", "score": round(sum(index * probabilities[level] for index, level in enumerate(levels)), 4),
            "confidence": round(max(probabilities[level] for level in levels), 4),
            "legend": dict(zip(levels, question["criteria"])),
            "probabilities": {level: round(probabilities[level], 4) for level in levels}}
