"""Qwen3.8-27B's JSON-schema replies on CUDA: drafted equals serial, concurrent equals solo, and the replies validate.

Needs ``TENSORFOLD_MLX_MODEL=<Vontra/Qwen3.8-27B-MLX-4bit dir>``, ``TENSORFOLD_QWEN27_DRAFTER=<z-lab/Qwen3.8-27B-DFlash2
dir>`` and xgrammar (``pip install 'tensorfold[grammar]'``); skipped otherwise. About 22 GB of GPU memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
pytest.importorskip("xgrammar")

from tensorfold.engine import grammar  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

MODEL = os.environ.get("TENSORFOLD_MLX_MODEL", "")
DRAFTER = os.environ.get("TENSORFOLD_QWEN27_DRAFTER", "")
CONTEXT = 8192
SAMPLINGS = {"greedy": None, "seed7": Sampling(7, 1.0, 20, 0.95)}

# public, made-up schemas and prompts
PERSON = {"type": "object", "additionalProperties": False, "required": ["name", "born", "fields"],
          "properties": {"name": {"type": "string"}, "born": {"type": "integer", "minimum": 1000, "maximum": 2100},
                         "fields": {"type": "array", "maxItems": 3,
                                    "items": {"type": "string", "enum": ["mathematics", "computing", "physics",
                                                                         "poetry", "music"]}}}}
WEATHER = {"type": "object", "additionalProperties": False, "required": ["city", "celsius", "sky", "wind_kph"],
           "properties": {"city": {"type": "string"}, "celsius": {"type": "number"},
                          "sky": {"type": "string", "enum": ["sunny", "cloudy", "rain", "snow"]},
                          "wind_kph": {"type": "integer", "minimum": 0, "maximum": 200}}}
RECIPE = {"type": "object", "additionalProperties": False, "required": ["title", "servings", "ingredients"],
          "properties": {"title": {"type": "string"}, "servings": {"type": "integer", "minimum": 1, "maximum": 12},
                         "vegetarian": {"type": "boolean"},
                         "ingredients": {"type": "array", "maxItems": 5, "items": {
                             "type": "object", "additionalProperties": False, "required": ["item", "grams"],
                             "properties": {"item": {"type": "string"},
                                            "grams": {"type": "integer", "minimum": 1, "maximum": 2000}}}}}}
CASES = {
    "person": (PERSON, "Describe Ada Lovelace as a short JSON record."),
    "weather": (WEATHER, "Make up a plausible weather report for Lisbon in spring, as JSON."),
    "recipe": (RECIPE, "Give a simple pancake recipe as JSON."),
}


def valid(value, schema: dict) -> bool:
    """The JSON-schema subset the schemas above use."""

    kind = schema.get("type")
    if "enum" in schema and value not in schema["enum"]:
        return False
    if kind == "object":
        props = schema.get("properties", {})
        return (isinstance(value, dict) and all(k in value for k in schema.get("required", []))
                and (schema.get("additionalProperties", True) or set(value) <= set(props))
                and all(valid(v, props[k]) for k, v in value.items() if k in props))
    if kind == "array":
        return (isinstance(value, list) and len(value) <= schema.get("maxItems", len(value))
                and all(valid(v, schema["items"]) for v in value))
    if kind in ("integer", "number"):
        number = isinstance(value, int) if kind == "integer" else isinstance(value, (int, float))
        return (number and not isinstance(value, bool)
                and schema.get("minimum", value) <= value <= schema.get("maximum", value))
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    return False


def _sha(ids: list[int]) -> str:
    return hashlib.sha256(",".join(str(int(t)) for t in ids).encode()).hexdigest()


@pytest.fixture(scope="module")
def engine():
    if not (MODEL and Path(MODEL).is_dir() and DRAFTER and Path(DRAFTER).is_dir()):
        pytest.skip("needs TENSORFOLD_MLX_MODEL and TENSORFOLD_QWEN27_DRAFTER")
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    return Qwen27Engine(Path(MODEL), Path(DRAFTER), max_rows=12, context=CONTEXT, context_explicit=True)


@pytest.fixture(scope="module")
def chat(engine):
    from tokenizers import Tokenizer

    from tensorfold.cuda.server import ChatTemplate

    tok = Tokenizer.from_file(str(Path(MODEL) / "tokenizer.json"))
    template = ChatTemplate(Path(MODEL))
    grammars = grammar.for_model(MODEL, grammar.vocab_size(MODEL), engine.eos)
    compiled = {name: grammars.compile(grammar.Spec("json_schema", json.dumps(schema)))
                for name, (schema, _) in CASES.items()}

    def prompt(text: str, thinking: bool = False) -> list[int]:
        rendered = template.render([{"role": "user", "content": text}], tools=None, enable_thinking=thinking)
        return tok.encode(rendered, add_special_tokens=False).ids

    def fresh(name: str, thinking: bool = False):
        return grammars.constraint(compiled[name], think_end=tok.token_to_id("</think>") if thinking else None)

    return type("Chat", (), {"prompt": staticmethod(prompt), "fresh": staticmethod(fresh), "tok": tok})


def _run(engine, prompt, count, sampling, draft, constraint=None):
    out: list[int] = []
    extra = {} if constraint is None else {"constraint": constraint}
    stats = engine.generate(list(prompt), count, sampling, lambda new: out.extend(new) and False, draft=draft, **extra)
    return out, stats


def _answer(chat, ids: list[int], engine) -> str:
    text = chat.tok.decode([t for t in ids if t not in engine.eos], skip_special_tokens=False)
    return text.split("</think>")[-1]


@pytest.mark.parametrize("sampling", list(SAMPLINGS), ids=list(SAMPLINGS))
@pytest.mark.parametrize("name", list(CASES))
def test_constrained_drafted_replies_equal_serial_and_validate(engine, chat, name, sampling):
    schema, text = CASES[name]
    prompt = chat.prompt(text)
    serial, s_stats = _run(engine, prompt, 320, SAMPLINGS[sampling], False, chat.fresh(name))
    drafted, d_stats = _run(engine, prompt, 320, SAMPLINGS[sampling], True, chat.fresh(name))
    assert _sha(drafted) == _sha(serial), (name, sampling)
    assert serial[-1] in engine.eos, "the JSON value completed within the reply limit"
    assert valid(json.loads(_answer(chat, serial, engine)), schema), _answer(chat, serial, engine)
    assert d_stats["rounds"] < s_stats["rounds"]                     # drafting kept tokens under the grammar


@pytest.mark.parametrize("sampling", list(SAMPLINGS), ids=list(SAMPLINGS))
def test_unconstrained_drafted_replies_still_equal_serial(engine, chat, sampling):
    prompt = chat.prompt(CASES["weather"][1])
    serial, _ = _run(engine, prompt, 160, SAMPLINGS[sampling], False)
    drafted, _ = _run(engine, prompt, 160, SAMPLINGS[sampling], True)
    assert _sha(drafted) == _sha(serial)


def test_with_thinking_the_schema_holds_after_think_end(engine, chat):
    prompt = chat.prompt("Answer in a few words of thought, then as JSON: " + CASES["weather"][1], thinking=True)
    sampling = SAMPLINGS["seed7"]
    serial, _ = _run(engine, prompt, 1600, sampling, False, chat.fresh("weather", thinking=True))
    drafted, _ = _run(engine, prompt, 1600, sampling, True, chat.fresh("weather", thinking=True))
    assert _sha(drafted) == _sha(serial)
    think_end = chat.tok.token_to_id("</think>")
    if think_end in serial and serial[-1] in engine.eos:          # the model finished thinking within the limit
        assert valid(json.loads(_answer(chat, serial, engine)), WEATHER)


def test_concurrent_constrained_replies_equal_their_solo_runs(engine, chat):
    """Constrained and plain requests share rounds (the engine's --parallel 4 decoder on the same weights); each reply
    equals its solo run through the same decoder and its serial reply on the one-stream engine."""

    from tensorfold.cuda.scheduler import Scheduler
    from tensorfold.families.qwen3_5.cuda.engine import KEEP
    from tensorfold.families.qwen3_5.cuda.multi import MultiDecoder

    multi = MultiDecoder(engine.w, engine.draft, allow_copy=True, context=4096, keep=KEEP, points=engine.points)
    multi.calibrate(4)
    scheduler = Scheduler(multi, max_streams=4)
    requests = []
    for name in CASES:
        for sampling in SAMPLINGS:
            requests.append((name, sampling, True))
    requests += [(None, "greedy", True), (None, "seed7", True), ("weather", "seed7", False)]

    def submit(request):
        name, sampling, draft = request
        text = CASES[name or "recipe"][1]
        out: list[int] = []
        scheduler.submit(chat.prompt(text), 240, SAMPLINGS[sampling], draft, lambda new: out.extend(new) and False,
                         constraint=chat.fresh(name) if name else None)
        return out

    together: list = [None] * len(requests)

    def worker(i):
        together[i] = submit(requests[i])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(requests))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for request, got in zip(requests, together):
        name, sampling, _ = request
        solo = submit(request)
        assert _sha(got) == _sha(solo), request
        text = CASES[name or "recipe"][1]
        serial, _ = _run(engine, chat.prompt(text), 240, SAMPLINGS[sampling], False,
                         chat.fresh(name) if name else None)
        assert _sha(solo) == _sha(serial), request
        if name is not None and got[-1] in engine.eos:
            assert valid(json.loads(_answer(chat, got, engine)), CASES[name][0]), request
