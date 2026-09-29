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

from tensorfold.cuda import grammar  # noqa: E402
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
        return grammars.constraint(compiled[name], after_think=thinking)

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


def test_a_constrained_prompt_filled_between_rounds_equals_its_solo_and_serial_runs(engine, chat):
    """A prompt longer than a fill step prefills a step at a time between the decoding streams' rounds, and its first
    token is chosen at the last step, under its grammar; beside it a plain stream ignores end tokens. Each reply equals
    its solo run and its serial reply."""

    import time

    from tensorfold.cuda.scheduler import Scheduler
    from tensorfold.families.qwen3_5.cuda.engine import KEEP
    from tensorfold.families.qwen3_5.cuda.multi import STEP, MultiDecoder

    multi = MultiDecoder(engine.w, engine.draft, allow_copy=True, context=6144, keep=KEEP, points=engine.points)
    multi.calibrate(4)
    scheduler = Scheduler(multi, max_streams=4)
    log = " ".join(f"Day {d}: Lisbon {12 + d % 9} C, wind {5 + d % 17} km/h, {('sunny', 'cloudy', 'rain')[d % 3]}."
                   for d in range(1, 181))
    long_text = f"Here is a made-up weather log.\n{log}\nSummarise the last day as a JSON weather report."
    assert len(chat.prompt(long_text)) > 2 * STEP
    requests = [("person", "greedy", True, True, CASES["person"][1]),
                (None, "seed7", True, False, CASES["recipe"][1]),              # plain, past its end tokens
                ("weather", "seed7", True, True, long_text)]

    def submit(request, started=None):
        name, sampling, draft, stop_eos, text = request
        out: list[int] = []

        def emit(new):
            out.extend(new)
            if started is not None:
                started.set()
            return False

        scheduler.submit(chat.prompt(text), 200, SAMPLINGS[sampling], draft, emit, stop_eos,
                         constraint=chat.fresh(name) if name else None)
        return out

    together: list = [None] * len(requests)
    decoding = threading.Event()

    def worker(i):
        together[i] = submit(requests[i], decoding if i == 0 else None)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    assert decoding.wait(120)
    time.sleep(0.2)                              # the first two decode while the long prompt fills
    threads.append(threading.Thread(target=worker, args=(2,)))
    threads[-1].start()
    for t in threads:
        t.join()
    assert len(together[1]) == 200                # ignore_eos: to its count
    for request, got in zip(requests, together):
        name, sampling, _, stop_eos, text = request
        solo = submit(request)
        assert _sha(got) == _sha(solo), request[:4]
        out: list[int] = []
        engine.generate(chat.prompt(text), 200, SAMPLINGS[sampling], lambda new: out.extend(new) and False,
                        draft=False, stop_eos=stop_eos, **({"constraint": chat.fresh(name)} if name else {}))
        assert _sha(solo) == _sha(out), request[:4]
        if name is not None and got[-1] in engine.eos:
            assert valid(json.loads(_answer(chat, got, engine)), CASES[name][0]), request[:4]


class _Failing:
    """A grammar that fails as xgrammar might: at a round's window after ``after`` of them, or at the first token."""

    def __init__(self, inner, after: int = 1 << 30, first: bool = False):
        self.inner, self.after, self.first, self.calls = inner, after, first, 0

    def window(self, tokens, parents):
        self.calls += 1
        if self.calls > self.after:
            raise grammar.GrammarError("the reply's grammar failed: simulated")
        return self.inner.window(tokens, parents)

    def mask(self, logits, window=None):
        if window is None and self.first:
            raise grammar.GrammarError("the reply's grammar failed at the first token: simulated")
        return self.inner.mask(logits, window)

    def advance(self, tokens):
        self.inner.advance(tokens)


def test_a_failed_grammar_ends_only_its_own_stream(engine, chat):
    """Under --parallel, a grammar failing at a round's window or at the first token ends that request with its error;
    the requests beside it and the one after equal their serial replies, and the scheduler goes on."""

    from tensorfold.cuda.scheduler import Scheduler
    from tensorfold.families.qwen3_5.cuda.engine import KEEP
    from tensorfold.families.qwen3_5.cuda.multi import MultiDecoder

    multi = MultiDecoder(engine.w, engine.draft, allow_copy=True, context=4096, keep=KEEP, points=engine.points)
    multi.calibrate(4)
    scheduler = Scheduler(multi, max_streams=4)
    results: dict = {}

    def go(key, name, constraint, sampling):
        out: list[int] = []
        try:
            results[key] = (out, scheduler.submit(chat.prompt(CASES[name][1]), 240, SAMPLINGS[sampling], True,
                                                  lambda new: out.extend(new) and False, constraint=constraint))
        except Exception as exc:                        # noqa: BLE001
            results[key] = (out, exc)

    jobs = [("person", "person", chat.fresh("person"), "greedy"), ("weather", "weather", chat.fresh("weather"), "seed7"),
            ("mid", "recipe", _Failing(chat.fresh("recipe"), after=2), "greedy"),
            ("first", "person", _Failing(chat.fresh("person"), first=True), "seed7")]
    threads = [threading.Thread(target=go, args=job, daemon=True) for job in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)
    out, err = results["mid"]
    assert isinstance(err, grammar.GrammarError) and 1 <= len(out) < 240, err
    out, err = results["first"]
    assert isinstance(err, grammar.GrammarError) and out == [], err
    for key, name, _, sampling in jobs[:2]:
        out, stats = results[key]
        serial, _ = _run(engine, chat.prompt(CASES[name][1]), 240, SAMPLINGS[sampling], False, chat.fresh(name))
        assert isinstance(stats, dict) and _sha(out) == _sha(serial), key
    go("after", "recipe", chat.fresh("recipe"), "greedy")
    serial, _ = _run(engine, chat.prompt(CASES["recipe"][1]), 240, None, False, chat.fresh("recipe"))
    assert _sha(results["after"][0]) == _sha(serial) and scheduler.thread.is_alive()
    assert not multi.streams and not multi.filling
