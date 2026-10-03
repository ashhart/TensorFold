"""The prompt pass hook: a prompt that fills alone takes several whole plan chunks a forward through a family's
``hidden_pass``, and everything it leaves is one chunk a forward's: replies, drafted == serial, checkpoints at resume
points, resumed == fresh, the draft head's context."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, str(Path(__file__).parent))

from lane_fakes import FakeEngine, FakeFamily  # noqa: E402
from tensorfold.engine.family_common import cache_arrays  # noqa: E402
from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402
from tensorfold.engine.prefill_plan import PrefillPlan  # noqa: E402
from tensorfold.server.cancellation import Cancellation, PrefillGuard  # noqa: E402


class PassFamily(FakeFamily):
    """The fake target with a prompt pass that records each forward's chunk sizes and each draft absorb."""

    fused_rows = 16

    def __init__(self) -> None:
        super().__init__()
        self.passes: list[tuple[int, ...]] = []
        self.absorbs: list[tuple[int, list[int]]] = []

    def hidden_pass(self, inputs, cache, sizes):
        self.passes.append(tuple(int(n) for n in sizes))
        return self.hidden(inputs, cache)


def _cache_limit():
    old = mx.set_cache_limit(0)
    mx.set_cache_limit(old)
    return old


class LimitPassFamily(PassFamily):
    """Records MLX's cache limit inside each pass."""

    def __init__(self) -> None:
        super().__init__()
        self.limits: list[int] = []

    def hidden_pass(self, inputs, cache, sizes):
        self.limits.append(_cache_limit())
        return super().hidden_pass(inputs, cache, sizes)


class DraftingPassFamily(PassFamily):
    mtp = "head"

    def absorb_draft_context(self, hidden, next_tokens, cache, start=0):
        self.absorbs.append((int(start), [int(t) for t in np.array(next_tokens).reshape(-1)]))

    def speculate(self, cache, token, position, sampling, start=0):
        return token

    def settle(self, *args):
        return None


def _fill(engine, prompt, drafts=True, **kw):
    stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=6, drafts=drafts)
    engine.add_stream(stream, **kw)
    while engine.active_count:
        engine.step()
    return stream


def _engine(family, width, step=32):
    engine = FakeEngine(family, prefill_pass=width)
    engine.prefill_plan = PrefillPlan(step)                    # chunk starts every 32 tokens
    return engine


def test_a_prompt_alone_takes_passes_of_whole_plan_chunks():
    prompt = [(5 * i + 3) % 90 + 1 for i in range(150)]
    one = _fill(_engine(PassFamily(), 1), prompt)
    family = PassFamily()
    wide = _fill(_engine(family, 4), prompt)
    assert family.passes == [(32, 32, 32, 32)]                  # then the 22-row chunk alone
    assert wide.prefill_widths == [4, 1] and one.prefill_widths == [1] * 5
    assert wide.emitted == one.emitted


def test_a_family_out_of_the_pass_fills_a_chunk_a_forward():
    prompt = [(5 * i + 3) % 90 + 1 for i in range(150)]
    family = PassFamily()
    family.prompt_pass = False
    out = _fill(_engine(family, 8), prompt)
    assert family.passes == [] and out.prefill_widths == [1] * 5
    assert out.emitted == _fill(_engine(PassFamily(), 1), prompt).emitted


def test_a_chunk_the_fused_rows_take_runs_alone():
    family = PassFamily()
    prompt = [(3 * i + 1) % 90 + 1 for i in range(136)]
    wide = _fill(_engine(family, 8), prompt)
    assert family.passes == [(32, 32, 32, 32)]                  # the 8-row tail is its own forward
    assert wide.emitted == _fill(_engine(PassFamily(), 1), prompt).emitted


def test_checkpoints_end_passes_and_hold_one_chunk_a_forwards_state():
    prompt = [(7 * i + 2) % 90 + 1 for i in range(200)]
    family = PassFamily()
    wide = _fill(_engine(family, 8), prompt, checkpoints_at=(70,))
    one = _fill(_engine(PassFamily(), 1), prompt, checkpoints_at=(70,))
    assert family.passes[0] == (32, 32)                         # the pass ends at the checkpoint (70 -> 64)
    assert [t for t, _ in wide.history_checkpoints] == [t for t, _ in one.history_checkpoints] == [prompt[:64]]
    assert wide.history_checkpoints[0][1][0].rows == one.history_checkpoints[0][1][0].rows
    assert wide.emitted == one.emitted


def test_a_live_stream_keeps_a_new_prompt_to_a_chunk_a_step():
    family = PassFamily()
    engine = _engine(family, 8)
    first = LaneStream(stream_id="a", prompt_ids=[1, 2, 3], max_new_tokens=400)
    engine.add_stream(first)
    engine.step()
    assert engine.active_count == 1
    engine.add_stream(LaneStream(stream_id="b", prompt_ids=[(i % 90) + 1 for i in range(200)], max_new_tokens=4))
    assert family.passes == []


def test_the_guard_sets_the_width():
    prompt = [(11 * i + 5) % 90 + 1 for i in range(200)]
    family = PassFamily()
    engine = _engine(family, 8)
    engine.prefill_guard = PrefillGuard(Cancellation(), None, wide=False)
    _fill(engine, prompt)
    assert family.passes == []
    family = PassFamily()
    engine = _engine(family, 8)
    memory = SimpleNamespace(pass_width=lambda cache, sizes: 2, pass_room=lambda cache, sizes, extra: True,
                             before_chunk=lambda *a: None, after_chunk=lambda *a: None)
    engine.prefill_guard = PrefillGuard(Cancellation(), memory)
    _fill(engine, prompt)
    assert family.passes == [(32, 32), (32, 32), (32, 32)]


def test_the_draft_head_takes_each_chunk_as_one_chunk_a_forward_gives_it():
    prompt = [(13 * i + 7) % 90 + 1 for i in range(100)]
    family = DraftingPassFamily()
    one = DraftingPassFamily()
    _fill(_engine(family, 4), prompt, drafts=False)            # no draft rounds: the fake head only records
    _fill(_engine(one, 1), prompt, drafts=False)
    assert family.passes == [(32, 32, 32)]
    # a pass hands each chunk's rows at its offset in the pass; the tokens after each chunk are the same
    assert [t for _, t in family.absorbs] == [t for _, t in one.absorbs]
    assert [s for s, _ in family.absorbs] == [0, 32, 64, 0] and [s for s, _ in one.absorbs] == [0, 0, 0, 0]


def test_a_pass_keeps_its_buffers_in_a_larger_cache_where_the_budget_has_room():
    prompt = [(17 * i + 3) % 90 + 1 for i in range(200)]
    before = mx.set_cache_limit(8 * 1024**3)
    try:
        family = LimitPassFamily()
        stream = _fill(_engine(family, 8), prompt)              # in process: no budget to ask
        assert family.limits == [16 * 1024**3] and _cache_limit() == 8 * 1024**3
        assert stream.prefill_widths == [6, 1] and stream.prefill_raised == [True, False]
        for room in (True, False):
            family = LimitPassFamily()
            engine = _engine(family, 8)
            memory = SimpleNamespace(pass_width=lambda cache, sizes: len(sizes), before_chunk=lambda *a: None,
                                     after_chunk=lambda *a: None, pass_room=lambda cache, sizes, extra: room)
            engine.prefill_guard = PrefillGuard(Cancellation(), memory)
            _fill(engine, prompt)
            assert family.limits == [16 * 1024**3 if room else 8 * 1024**3] and _cache_limit() == 8 * 1024**3
    finally:
        mx.set_cache_limit(before)


def test_a_request_after_a_raised_pass_is_admitted_as_after_any_other():
    """A pass's raised cache leaves freed buffers behind: the next request's admission counts them free and takes
    them back before refusing, so it fits exactly when it would have with an empty cache."""

    from test_prompt_memory import Runtime, controller, populated

    runtime = Runtime(resident=1000)
    memory = controller(budget=10**9, runtime=runtime)
    memory.observe_cache(populated())
    memory.begin(3000, 64, admit=False)
    memory.budget = memory.projected(3000) + 100
    memory.end()
    assert memory.would_fit(3000, 64) and not memory.would_fit(3000 + 1024, 64)
    runtime.cache = 5000                                        # what a 16 GiB pass cache kept after the pass
    assert memory.would_fit(3000, 64) and not memory.would_fit(3000 + 1024, 64)
    memory.begin(3000, 64)                                      # admission frees them instead of refusing
    assert runtime.cache == 0


def test_a_pass_is_sized_by_its_routed_experts_where_the_model_has_them():
    from test_prompt_memory import Runtime, controller

    from tensorfold.server.prompt_memory import pass_row_bytes

    moe = SimpleNamespace(args=SimpleNamespace(num_experts_per_tok=8, hidden_size=4096, moe_intermediate_size=2048))
    assert pass_row_bytes(moe) == 3 * 8 * (6 * 4096 + 8 * 2048) // 2
    assert pass_row_bytes(SimpleNamespace(args=SimpleNamespace(hidden_size=5120))) == 0      # dense
    memory = controller(runtime=Runtime())
    memory.observed_work = 1000
    assert memory.pass_bytes([2048, 2048, 1000]) == 2000         # no experts: a chunk's workspace a chunk past one
    memory.pass_row_bytes = 10
    assert memory.pass_bytes([2048, 2048, 1000]) == 30480        # the rows past the largest chunk's


def test_prompt_memory_shrinks_a_pass_to_its_budget():
    from test_prompt_memory import Runtime, controller, populated

    runtime = Runtime(resident=1000)
    cache = populated()
    runtime.caches.append(cache)
    memory = controller(budget=10**9, runtime=runtime)
    memory.observe_cache(cache)
    memory.observed_work = 1000                                 # a full chunk's workspace
    memory.begin(256, 0, admit=False)
    memory.budget = memory.projected(256, current_cache=cache) + 3500
    assert memory.pass_width(cache, [256] * 8) == 4             # three chunks' workspace more fit, not four
    runtime.cache = 1000                                        # MLX's freed buffers don't count against it
    assert memory.pass_width(cache, [256] * 8) == 4
    assert memory.pass_room(cache, [256] * 4, 500) and not memory.pass_room(cache, [256] * 4, 501)
    memory.observed_work = 10**6
    assert memory.pass_width(cache, [256] * 8) == 1


# -- real families: tiny checkpoints through the lane engine, one chunk a forward against passes -----------------

@pytest.fixture(params=["cpu", "gpu"])
def device(request):
    if request.param == "gpu" and not mx.metal.is_available():
        pytest.skip("needs Metal")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu if request.param == "gpu" else mx.cpu)
    yield request.param
    mx.set_default_device(previous)


@pytest.fixture(scope="module")
def glm_checkpoint(tmp_path_factory):
    from glm5_fakes import write_checkpoint

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return write_checkpoint(tmp_path_factory.mktemp("glm5-pass"))
    finally:
        mx.set_default_device(previous)


def _glm(checkpoint, drafts):
    from tensorfold.families.glm5_next import mtp as glm_mtp
    from tensorfold.families.glm5_next import weights
    from tensorfold.families.glm5_next.runtime import GLMFlash

    model = weights.load_backbone(checkpoint)
    return GLMFlash(model, glm_mtp.load(model) if drafts else None, drafts=drafts)


def _flash_next():
    from test_qwen4_exp_prefill import tiny_model

    from tensorfold.families.qwen4_exp.runtime import FlashNext

    return FlashNext(tiny_model(seed=4), None, drafts=0)


def _nemotron():
    from test_nemotron_pass import _tiny

    from tensorfold.families.nemotron_h.model import NemotronH

    # without the fused decode kernels (tuned to the checkpoint's widths): the prompt path and plain decode
    return NemotronH(_tiny(), fused=False, mtp_path=None, drafts=0, tokenizer=SimpleNamespace(encode=lambda t: [1]))


def _run(runtime, prompt, width, *, grid=32, drafts=True, **kw):
    calls = []
    together = runtime.hidden_pass

    def counted(inputs, cache, sizes):
        calls.append(tuple(sizes))
        return together(inputs, cache, sizes)

    runtime.hidden_pass = counted
    try:
        engine = LaneEngine(runtime, prefill_pass=width, max_rows=getattr(runtime, "exact_width", 1),
                            max_draft=max(0, getattr(runtime, "exact_width", 1) - 1))
        engine.prefill_plan = PrefillPlan(grid)
        stream = LaneStream(stream_id="x", prompt_ids=list(prompt), max_new_tokens=6, drafts=drafts)
        engine.add_stream(stream, **kw)
        while engine.active_count:
            engine.step()
    finally:
        del runtime.hidden_pass
    return stream, calls


def _same_checkpoints(a, b):
    assert [t for t, _ in a.history_checkpoints] == [t for t, _ in b.history_checkpoints]
    for (_, x), (_, y) in zip(a.history_checkpoints, b.history_checkpoints):
        xs, ys = cache_arrays(x), cache_arrays(y)
        assert len(xs) == len(ys) and all(u.shape == v.shape and bool(mx.array_equal(u, v).item())
                                          for u, v in zip(xs, ys))


@pytest.mark.parametrize("family", ["glm", "glm-drafts", "flash-next", "nemotron"])
def test_passes_leave_one_chunk_a_forwards_replies_and_checkpoints(device, glm_checkpoint, monkeypatch, family):
    if family == "flash-next":
        pytest.skip("the tiny Flash Next prompts on Metal only and decodes on the GPU only at the checkpoint's widths "
                    "(qmv_rows needs K % 512); tests/test_prompt_pass.py runs its pass, the served runs its engine")
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm as PM

    monkeypatch.setattr(PM, "gpu_tensor_units", lambda: False)         # the passes, on an M5 too
    make = {"glm": lambda: _glm(glm_checkpoint, 0), "glm-drafts": lambda: _glm(glm_checkpoint, 2),
            "flash-next": _flash_next, "nemotron": _nemotron}[family]
    runtime = make()
    vocab = 90
    prompt = [int(t) for t in np.random.default_rng(21).integers(6, vocab, size=180)]
    one, none = _run(runtime, prompt, 1, checkpoints_at=(100,))
    wide, calls = _run(runtime, prompt, 4, checkpoints_at=(100,))
    assert none == [] and calls and all(len(c) > 1 for c in calls)
    assert wide.emitted == one.emitted
    _same_checkpoints(wide, one)
    # the next turn resumes from the pass's checkpoint like a fresh prefill of one chunk a forward
    follow = [*prompt, *one.emitted, 7, 8, *prompt[:40]]
    fresh, _ = _run(runtime, follow, 1)
    resumed, _ = _run(runtime, follow, 4, cache=LaneEngine.copy_single_cache(wide.history_checkpoints[0][1]),
                      cached_tokens=96)
    assert resumed.emitted == fresh.emitted
    if family == "glm-drafts":                                  # drafted == "draft": false, through passes
        serial, _ = _run(runtime, prompt, 4, drafts=False)
        assert serial.emitted == wide.emitted


@pytest.mark.parametrize("family", ["glm", "nemotron"])
def test_tensor_unit_chips_fill_a_chunk_a_forward(device, glm_checkpoint, monkeypatch, family):
    """With tensor units (M5) GLM-5.3 and Nemotron fill a prompt a chunk a forward, as 0.5.0 does, until a run there
    measures their passes; on M1-M4 the same runtime takes passes and gives the same reply."""

    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm as PM

    def chip(units):
        monkeypatch.setattr(PM, "gpu_tensor_units", lambda: units)

    chip(False)
    runtime = _glm(glm_checkpoint, 0) if family == "glm" else _nemotron()
    prompt = [int(t) for t in np.random.default_rng(21).integers(6, 90, size=180)]
    wide, calls = _run(runtime, prompt, 4)
    assert runtime.prompt_pass and calls and max(wide.prefill_widths) > 1
    chip(True)
    alone, calls = _run(runtime, prompt, 4)
    assert not runtime.prompt_pass and calls == [] and set(alone.prefill_widths) == {1}
    assert alone.emitted == wide.emitted
