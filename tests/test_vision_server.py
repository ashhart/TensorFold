"""Image prompt and cache boundaries exercised without accelerator imports or model weights."""

from __future__ import annotations

import base64
import io
import sys
import threading
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from tensorfold.cuda.server import App
from tensorfold.cuda.streams import PrefixCache
from tensorfold.engine.family_prefill import FamilyPrefill
from tensorfold.engine.lane_engine import LaneStream
from tensorfold.engine.prefill_plan import PrefillPlan
from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine
from tensorfold.server.errors import RequestError
from tensorfold.server.messages import normalize_messages
from tensorfold.server.prompts import prepare_images, prepare_prompt
from tensorfold.server.scheduler import ChatJob, Scheduler


@pytest.fixture(autouse=True)
def block_accelerators(monkeypatch):
    for name in ("mlx", "mlx.core", "mlx.nn", "mlx_lm", "mlx_vlm", "torch", "triton"):
        monkeypatch.setitem(sys.modules, name, None)


def image_messages(color="red"):
    image = pytest.importorskip("PIL.Image")
    output = io.BytesIO()
    image.new("RGB", (2, 2), color).save(output, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    return [{"role": "user", "content": [
        {"type": "text", "text": "before"}, {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "after"},
    ]}]


class Frontend:
    def __init__(self, tokens=(10, 11, 12, 13)):
        self.tokens = tokens
        self.calls = []

    def prepare(self, rendered, images, *, max_prompt_tokens):
        self.calls.append((rendered, images, max_prompt_tokens))
        if max_prompt_tokens is not None and len(self.tokens) > max_prompt_tokens:
            raise ValueError("expanded image prompt exceeds the context limit")
        return NS(token_ids=self.tokens, image_hashes=tuple(image.content_hash for image in images))


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered with image markers"

    def encode(self, text, **kwargs):
        return [21, 22] if not kwargs else NS(ids=[21, 22])

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


def prompt_app(frontend):
    return NS(tokenizer=Tokenizer(), tokenizer_lock=threading.Lock(), vision=frontend, late_system="user",
              context_window=32, reasoning_effort="medium", render=lambda *args, **kwargs: ([1, 2, 3], 2))


def cuda_app(frontend):
    app = App.__new__(App)
    app.tok, app.vision = Tokenizer(), frontend
    app.context_window, app.native_context_window = 32, 64
    app.max_tokens, app.default_thinking = 8, False
    app.lock, app.served = threading.Lock(), "test-model"
    app.template_calls = []
    app.engine_calls = []

    def render(messages, **kwargs):
        normalized = normalize_messages(messages, allow_images=kwargs.get("allow_images", False))
        app.template_calls.append((normalized, kwargs))
        return "rendered prompt"

    def generate(prompt, max_tokens, sampling, on_tokens, draft=True, **kwargs):
        app.engine_calls.append((prompt, max_tokens, sampling, draft, kwargs))
        on_tokens([65])
        return {"cached": 0}

    app.template = NS(render=render)
    app.engine = NS(generate=generate, eos=(0,), context_window=32)
    app.sampling_for = lambda *args: None
    return app


def test_normalize_images_preserves_parts_and_instruction_order():
    messages = [{"role": "developer", "content": "first"}, {"role": "system", "content": "second"},
                *image_messages(), {"role": "developer", "content": "later"}]
    result = normalize_messages(messages, allow_images=True, late_system="user")
    assert result[0] == {"role": "system", "content": "first\n\nsecond"}
    assert result[1] == messages[2] and isinstance(result[1]["content"], list)
    assert result[2] == {"role": "user", "content": "later"}
    assert messages[0]["role"] == "developer"


@pytest.mark.parametrize("role", ["system", "developer", "assistant", "tool"])
def test_normalize_rejects_images_outside_user_role(role):
    messages = image_messages()
    messages[0]["role"] = role
    with pytest.raises(RequestError, match="only in user"):
        normalize_messages(messages, allow_images=True)


def test_normalize_images_are_opt_in_and_audio_remains_unsupported():
    with pytest.raises(RequestError, match="text parts only"):
        normalize_messages(image_messages())
    messages = image_messages()
    messages[0]["content"].append({"type": "input_audio", "input_audio": {"data": "x"}})
    with pytest.raises(RequestError, match="text and image_url"):
        normalize_messages(messages, allow_images=True)
    assert normalize_messages([{"role": "user", "content": [{"type": "text", "text": "a"},
                                                               {"type": "text", "text": "b"}]}],
                              allow_images=True)[0]["content"] == "ab"


def test_prepare_images_decodes_cpu_and_passes_expanded_tokens():
    frontend, templates = Frontend(), []
    prepared = prepare_images(frontend, image_messages(), lambda value: templates.append(value) or "rendered",
                              context_limit=8)
    assert prepared.tokens == [10, 11, 12, 13] and prepared.history_len == 0
    assert prepared.vision.image_hashes == (frontend.calls[0][1][0].content_hash,)
    assert templates[0][0]["content"][1] == {"type": "image", "detail": "auto"}
    assert frontend.calls[0][2] == 8 and frontend.calls[0][1][0].pixels == bytes([255, 0, 0]) * 4


def test_prepare_images_preserves_zero_capacity():
    with pytest.raises(RequestError, match="context limit"):
        prepare_images(Frontend(), image_messages(), str, context_limit=0)


def test_prepare_images_errors_are_request_refusals():
    with pytest.raises(RequestError, match="--vision"):
        prepare_images(None, image_messages(), str)
    messages = image_messages()
    messages[0]["content"][1]["image_url"]["url"] = "data:image/png;base64,aW52YWxpZA=="
    with pytest.raises(RequestError, match="invalid or unsupported"):
        prepare_images(Frontend(), messages, str)
    with pytest.raises(RequestError, match="context limit"):
        prepare_images(Frontend(), image_messages(), str, context_limit=3)


def test_prepare_prompt_preserves_text_render_and_direct_prompt_paths():
    app = prompt_app(None)
    prepared = prepare_prompt(app, [{"role": "user", "content": "text"}], [], False, None, {})
    assert (prepared.tokens, prepared.history_len, prepared.vision) == ([1, 2, 3], 2, None)
    assert prepare_prompt(app, None, [], False, "direct", {}).tokens == [21, 22]
    assert prepare_prompt(app, None, [], False, [3, 4], {}).tokens == [3, 4]
    assert not app.tokenizer.calls


def test_prepare_prompt_passes_tools_thinking_and_normalized_arguments():
    app = prompt_app(Frontend())
    calls = [{"function": {"name": "lookup", "arguments": '{"query":"x"}'}}]
    messages = [{"role": "assistant", "content": None, "tool_calls": calls}, *image_messages()]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    prepared = prepare_prompt(app, messages, tools, True, None, {"reasoning_effort": "high"})
    template, kwargs = app.tokenizer.calls[0]
    assert prepared.vision is not None and prepared.history_len == 0
    assert kwargs == dict(add_generation_prompt=True, tokenize=False, enable_thinking=True, tools=tools,
                          reasoning_effort="high")
    assert template[0]["tool_calls"][0]["function"]["arguments"] == {"query": "x"}
    assert isinstance(calls[0]["function"]["arguments"], str)


def test_cuda_prepare_forwards_images_and_checks_expanded_context():
    app = cuda_app(Frontend())
    body = {"messages": image_messages(), "max_tokens": 2, "chat_template_kwargs": {"enable_thinking": True}}
    prepared = app.prepare(body, True)
    assert prepared.prompt == [10, 11, 12, 13] and prepared.vision is not None and prepared.thinking
    assert app.template_calls[0][1]["allow_images"] is True
    app.engine.context_window = 5
    with pytest.raises(RequestError, match="4 tokens.*2 reply tokens"):
        app.prepare(body, True)
    assert not app.engine_calls


def test_cuda_prepare_rejects_expanded_prompt_before_engine_submission():
    app = cuda_app(Frontend(tuple(range(40))))
    with pytest.raises(RequestError, match="expanded image prompt"):
        app.prepare({"messages": image_messages()}, True)
    assert not app.engine_calls


@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("draft", [False, True])
def test_cuda_run_forwards_vision_without_changing_text_call_options(image, draft):
    app = cuda_app(Frontend())
    messages = image_messages() if image else [{"role": "user", "content": "text"}]
    body = {"messages": messages, "draft": draft, "max_tokens": 2}
    prepared = app.prepare(body, True)
    deltas = []
    result = app.run(body, True, lambda chunk: deltas.append(chunk) or True, prepared=prepared)
    prompt, count, sampling, received_draft, options = app.engine_calls[0]
    assert prompt == prepared.prompt and count == 2 and sampling is None and received_draft is draft
    assert options == ({"vision": prepared.vision} if image else {})
    assert result["content"] == "A" and deltas == [{"content": "A"}]


def fake_decode(monkeypatch):
    module = ModuleType("tensorfold.families.qwen3_5.cuda.decode")
    calls = []

    def prefill(weights, prompt, sampling, drafter, *, state=None, vision=None, keep_at=None, **_):
        calls.append((list(prompt), state, vision))
        st = NS(pos=len(prompt))
        return (st, 65) if keep_at is None else (st, 65, (NS(pos=keep_at), None))

    module.prefill = prefill
    module.draft_decode = lambda *args, **kwargs: NS(seconds=0, rounds=1, widths=[2])
    monkeypatch.setitem(sys.modules, module.__name__, module)
    engine = Qwen27Engine.__new__(Qwen27Engine)
    engine.context_window, engine.scheduler, engine.tp = 32, None, 1
    engine.w, engine.draft, engine.max_rows, engine.allow_copy = NS(), None, 4, True
    engine.cache, engine.points = PrefixCache(8), None
    engine.cache.add([1, 2], NS(pos=2), None)
    engine.vision = NS(encode=lambda prepared, prompt: NS(identity=prepared, tokens=tuple(prompt)))
    return engine, calls


def test_cuda_different_images_always_prefill_fresh_and_preserve_text_cache(monkeypatch):
    engine, calls = fake_decode(monkeypatch)
    old_cache = engine.cache.entries[:]
    first, second = object(), object()
    one = engine.generate([1, 2, 3], 2, None, lambda ids: False, vision=first)
    two = engine.generate([1, 2, 3], 2, None, lambda ids: False, vision=second)
    assert calls[0][1] is None and calls[1][1] is None
    assert calls[0][2].identity is first and calls[1][2].identity is second
    assert one["cached"] == two["cached"] == 0 and engine.cache.entries == old_cache
    text = engine.generate([1, 2, 3, 4], 2, None, lambda ids: False)
    assert calls[2][1] is old_cache[0][1] and calls[2][2] is None and text["cached"] == 2
    assert engine.cache.entries[-1][0] == [1, 2, 3]            # a text prompt's entry ends one token early


def test_cuda_concurrent_submission_receives_prepared_vision(monkeypatch):
    engine, _ = fake_decode(monkeypatch)
    submissions = []
    engine.scheduler = NS(submit=lambda *args, **kwargs: submissions.append((args, kwargs)))
    vision = object()
    engine.generate([1, 2, 3], 2, None, lambda ids: False, vision=vision)
    engine.generate([1, 2, 3], 2, None, lambda ids: False)
    assert submissions[0][1] == {"stop_eos": True, "vision": vision} and submissions[1][1] == {"stop_eos": True}


class Checkpoints:
    def __init__(self):
        self.calls, self.cache = [], [object()]

    def peek(self, *args, **kwargs):
        self.calls.append("peek")
        return NS(cache=self.cache)

    def match(self, prompt, **kwargs):
        self.calls.append("match")
        return 2, self.cache, prompt[:2]

    def insert(self, *args, **kwargs):
        self.calls.append("insert")


def scheduler_fixture(memory=None):
    calls, checkpoints = [], Checkpoints()
    engine = NS(prefill_guard=None, finished_caches={}, streams=[], prompt_chunks=PrefillPlan(2).chunks)
    engine.model = NS(vision=NS(estimate_workspace_bytes=lambda prepared: 4096))

    def add_stream(stream, **kwargs):
        calls.append((stream, kwargs))
        stream.cached_tokens = kwargs["cached_tokens"]
        stream.history_checkpoints = [(stream.prompt_ids[:2], [object()])]
        engine.streams.append(stream)

    engine.add_stream = add_stream
    scheduler = Scheduler(engine, lanes=4, eos_ids=frozenset(), checkpoints=checkpoints, prompt_memory=memory)
    scheduler._read_disk_block = lambda *args: checkpoints.calls.append("disk")
    return scheduler, calls, checkpoints


def test_scheduler_image_jobs_bypass_checkpoint_reads_and_writes():
    scheduler, calls, checkpoints = scheduler_fixture()
    for index in range(2):
        job = ChatJob(str(index), [1, 2, 3, 4], 2, 0, history_len=2, shared_prefix_lens=(2,), vision=object())
        scheduler._start_job(job)
        stream, options = calls[-1]
        assert job.error is None and stream.prompt_data is job.vision and not stream.retain
        assert options == dict(cache=None, cached_tokens=0, checkpoints_at=[])
        scheduler.engine.finished_caches[job.job_id] = ([1, 2, 3, 4, 5], [object()])
        scheduler._retire(job)
        assert job.done.is_set() and job.cached_tokens == 0
    assert not checkpoints.calls


def test_scheduler_text_jobs_keep_checkpoint_reuse_and_storage():
    scheduler, calls, checkpoints = scheduler_fixture()
    job = ChatJob("text", [1, 2, 3, 4], 2, 0, history_len=2, shared_prefix_lens=(2,))
    scheduler._start_job(job)
    stream, options = calls[-1]
    assert job.error is None and stream.prompt_data is None and stream.retain
    assert options["cache"] is checkpoints.cache and job.cached_tokens == 2
    assert checkpoints.calls == ["disk", "peek", "match", "insert"]
    scheduler.engine.finished_caches[job.job_id] = ([1, 2, 3, 4, 5], [object()])
    scheduler._retire(job)
    assert checkpoints.calls[-1] == "insert" and checkpoints.calls.count("insert") == 2


def test_scheduler_admits_image_memory_before_starting_prefill():
    admitted, workspace = [], []
    memory = NS(begin=lambda *args, admit: admitted.append(admit), end=lambda: None,
                require_workspace=lambda count: workspace.append(count))
    scheduler, calls, _ = scheduler_fixture(memory)
    scheduler._start_job(ChatJob("image", [1, 2, 3, 4], 2, 0, vision=object()))
    assert calls and admitted == [True] and workspace == [4096]


def test_scheduler_workspace_refusal_never_starts_image_prefill():
    def refuse(count):
        raise RequestError("image workspace does not fit")

    ended = []
    memory = NS(begin=lambda *args, **kwargs: None, end=lambda: ended.append(True), require_workspace=refuse)
    scheduler, calls, checkpoints = scheduler_fixture(memory)
    job = ChatJob("image", [1, 2, 3, 4], 2, 0, vision=object())
    scheduler._start_job(job)
    assert isinstance(job.error, RequestError) and "workspace" in str(job.error)
    assert job.done.is_set() and ended == [True] and not calls and not checkpoints.calls


@pytest.fixture
def numpy_mlx(monkeypatch):
    core = ModuleType("mlx.core")
    core.array, core.uint32, core.eval = np.array, np.uint32, lambda *args: None
    mlx = ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    import tensorfold.engine.family_prefill as family_prefill
    monkeypatch.setattr(family_prefill, "drop_spares", lambda cache: cache)


def family_prefill_fixture():
    calls = []

    class Model:
        def make_cache(self):
            return [NS(state=None)]

        def encode_vision(self, prepared, cache):
            encoded = NS(prepared=prepared)
            calls.append(("encode", encoded))
            return encoded

        def hidden(self, tokens, cache):
            calls.append(("text", tokens.copy()))
            return tokens.astype(np.float32)[..., None]

        def prefill_vision(self, tokens, cache, prepared, begin, end):
            calls.append(("vision", prepared, begin, end, tokens.copy()))
            return tokens.astype(np.float32)[..., None]

    class Engine(FamilyPrefill):
        model, prefill_guard, prefill_chunks = Model(), None, 0
        prefill_plan = PrefillPlan(2)
        prompt_chunks = staticmethod(prefill_plan.chunks)
        copy_single_cache = staticmethod(lambda cache: list(cache))

        def _family_first(self, stream, cache, hidden, cached_tokens, row):
            stream.cached_tokens = cached_tokens
            return 65

    return Engine(), calls


def test_family_prefill_encodes_once_and_passes_each_image_chunk(numpy_mlx):
    engine, calls = family_prefill_fixture()
    prepared = object()
    stream = LaneStream("image", [1, 2, 3, 4, 5], 2, prompt_data=prepared, retain=False)
    engine._family_prefill(stream, cache=None, cached_tokens=0, checkpoints_at=[2, 4])
    assert [call[0] for call in calls] == ["encode", "vision", "vision", "vision"]
    assert all(call[1] is calls[0][1] for call in calls[1:])
    assert [(call[2], call[3]) for call in calls[1:]] == [(0, 2), (2, 4), (4, 5)]
    assert stream.history_checkpoints == [] and stream.emitted == [65] and stream.cached_tokens == 0


def test_an_image_prompt_skips_the_message_cuts_a_text_prompt_keeps(numpy_mlx):
    engine, calls = family_prefill_fixture()
    engine.prefill_plan = PrefillPlan(4, openers=(7,), min_chunk=2)
    engine.prompt_chunks = engine.prefill_plan.chunks
    prompt = [7, 1, 2, 7, 3, 4]
    assert engine.prompt_chunks(prompt).between(0, 6) == [(0, 3), (3, 6)]     # text: cut at the second message
    engine._family_prefill(LaneStream("image", prompt, 2, prompt_data=object(), retain=False),
                           cache=None, cached_tokens=0, checkpoints_at=[3])
    assert [(call[2], call[3]) for call in calls if call[0] == "vision"] == [(0, 4), (4, 6)]      # image: the grid


def test_family_prefill_refuses_any_cached_image_state(numpy_mlx):
    engine, calls = family_prefill_fixture()
    with pytest.raises(ValueError, match="fresh cache"):
        engine._family_prefill(LaneStream("image", [1, 2, 3, 4], 2, prompt_data=object()),
                               cache=engine.model.make_cache(), cached_tokens=2, checkpoints_at=[])
    assert not calls


def test_family_prefill_text_retains_checkpoint_path(numpy_mlx):
    engine, calls = family_prefill_fixture()
    stream = LaneStream("text", [1, 2, 3, 4, 5], 2)
    engine._family_prefill(stream, cache=None, cached_tokens=0, checkpoints_at=[2])
    assert [call[0] for call in calls] == ["text", "text", "text"]
    assert [len(tokens) for tokens, cache in stream.history_checkpoints] == [2]


def test_failed_image_prefill_does_not_keep_partial_checkpoint(numpy_mlx):
    engine, calls = family_prefill_fixture()
    original = engine.model.prefill_vision

    def fail_second(tokens, cache, prepared, begin, end):
        if begin > 0:
            raise RuntimeError("interrupted image prefill")
        return original(tokens, cache, prepared, begin, end)

    engine.model.prefill_vision = fail_second
    stream = LaneStream("image", [1, 2, 3, 4, 5], 2, prompt_data=object(), retain=False)
    with pytest.raises(RuntimeError, match="interrupted"):
        engine._family_prefill(stream, cache=None, cached_tokens=0, checkpoints_at=[2])
    assert stream.history_checkpoints == [] and len(calls) == 2


def test_prepare_images_fetches_urls_only_when_the_frontend_allows(monkeypatch):
    from tensorfold.vision import images

    messages = image_messages()
    part = messages[0]["content"][1]
    data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
    part["image_url"] = {"url": "https://example.com/image.png"}
    monkeypatch.setattr(images, "fetch_image", lambda *args, **kwargs: (data, "image/png"))
    with pytest.raises(RequestError, match="--vision-urls"):
        prepare_images(Frontend(), messages, str)
    frontend = Frontend()
    frontend.allow_urls = True
    assert prepare_images(frontend, messages, str).tokens == [10, 11, 12, 13]


def test_image_preparation_is_bounded_and_refuses_with_capacity_errors(monkeypatch):
    import threading

    from tensorfold.server import prompts
    from tensorfold.server.errors import CapacityError

    monkeypatch.setattr(prompts, "IMAGE_SLOTS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(prompts, "IMAGE_WAITERS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(prompts, "IMAGE_WAIT_S", 0.01)
    held = prompts.image_slot()
    with pytest.raises(CapacityError, match="busy"):          # a slot never frees within the wait
        prompts.image_slot()
    assert prompts.IMAGE_WAITERS.acquire(blocking=False)      # the waiter it took was given back
    with pytest.raises(CapacityError, match="queue is full"):  # every waiter place taken: refused at once
        prompts.image_slot()
    prompts.IMAGE_WAITERS.release()
    held.release()
    prompts.image_slot().release()


def test_request_log_redacts_every_image_part():
    from tensorfold.server.http import redact_images

    body = {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}}]}]}
    logged = redact_images(body)
    assert logged["messages"][0]["content"][1] == {"type": "image_url", "image_url": {"url": "<redacted>"}}
    assert logged["messages"][0]["content"][0] == body["messages"][0]["content"][0] and logged["model"] == "m"
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:")    # the request is untouched


def test_cuda_required_call_continuation_keeps_the_images(monkeypatch):
    from tensorfold.vision import qwen_processing

    app = cuda_app(Frontend())
    app.vision.frontend = NS(config={"image_token_id": 7})
    grown = []
    monkeypatch.setattr(qwen_processing, "continued",
                        lambda prepared, ids, config: grown.append((prepared, list(ids), config)) or NS(ids=list(ids)))
    cuts = iter([(0, [1000])])
    app._call_gate = lambda prompt, tools: NS(cut=lambda new: next(cuts, None), observe=lambda token: None)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
    body = {"messages": image_messages(), "max_tokens": 4, "tools": tools, "tool_choice": "required"}
    prepared = app.prepare(body, True)
    app.run(body, True, lambda chunk: True, prepared=prepared)
    first, second = app.engine_calls[0], app.engine_calls[1]
    assert first[0] == prepared.prompt and first[4] == {"vision": prepared.vision}
    assert second[0] == [*prepared.prompt, 1000] and second[4] == {"vision": NS(ids=second[0])}
    assert grown == [(prepared.vision, second[0], {"image_token_id": 7})]
