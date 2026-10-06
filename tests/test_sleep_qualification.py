"""The qualification report must reject unexercised drafting and surviving device allocations."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def qualification(tmp_path, monkeypatch):
    from tensorfold import families
    from tensorfold.cuda import precision, prompt_precision, server, sleep

    spec = importlib.util.spec_from_file_location("qualify_sleep", Path(__file__).parents[1] / "tools/qualify_sleep.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    state = SimpleNamespace(kind="qwen3_5", generation=0, failed_draft_generation=None, missing_draft_stats=False,
                            allocated=0, reserved=0, closed=False, loaded=0)

    class Engine:
        def generate(self, prompt, count, sampling, received, *, draft, stop_eos):
            received(list(range(count)))
            stats = {"cached": len(prompt) - 1 if draft else 0}
            if not state.missing_draft_stats:
                stats["drafted"] = 2 if draft and state.generation != state.failed_draft_generation else 0
            return stats

    def factory(*args, **kwargs):
        state.loaded += 1
        return Engine()

    runtime = SimpleNamespace(cache=lambda engine: SimpleNamespace(entries=[]))
    package = SimpleNamespace(cuda_engine=factory, cuda_sleep=lambda: runtime)
    monkeypatch.setattr(families, "detect", lambda path: SimpleNamespace(model_type=state.kind, package=package))
    monkeypatch.setattr(server, "App", lambda engine, *a, **kw: SimpleNamespace(engine=engine, tok=object(),
                                                                             template=object()))

    class Adapter:
        def __init__(self, app, *args, **kwargs):
            self.app = app
            self.identity = SimpleNamespace(paths=[tmp_path], manifest={})

        def prepare(self):
            pass

        def release(self):
            self.app.engine = None

        def restore(self):
            state.generation += 1
            self.app.engine = factory()

        cleanup = release

        def memory_snapshot(self):
            if self.app.engine is not None:
                return {"allocated_bytes": 1000, "reserved_bytes": 2000}
            return {"allocated_bytes": state.allocated, "reserved_bytes": state.reserved}

        def cache_snapshot(self):
            return {"loaded_prefixes": 1, "load_failures": 0}

        def close(self):
            state.closed = True

    monkeypatch.setattr(sleep, "CudaSleep", Adapter)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None, mem_get_info=lambda: (1000, 2000),
                             get_device_name=lambda: "test device"),
        __version__="test", version=SimpleNamespace(cuda="test")))
    args = SimpleNamespace(draft=tmp_path / "draft", no_drafts=False, parallel=1, context=128, preserve_cache=False,
                           precision="checkpoint", prefill_fp8=False, prompt_tokens=8, tokens=16, cycles=2,
                           seed=None, synthetic=False, output=tmp_path / "report.json")
    previous = (precision.mode(), precision.asked(), prompt_precision.fp8())
    yield tool, args, state, tmp_path
    precision.set_mode(previous[0], asked=previous[1])
    prompt_precision.set_fp8(previous[2])


@pytest.mark.parametrize("generation", [0, 1, 2])
@pytest.mark.parametrize("kind", ["qwen3_5", "nemotron_h"])
def test_qualification_refuses_zero_draft_proposals_before_and_after_wake(qualification, generation, kind):
    tool, args, state, model = qualification
    state.kind, state.failed_draft_generation = kind, generation
    if kind == "nemotron_h":
        args.draft = None
    with pytest.raises(AssertionError, match="draft.*proposals"):
        tool.qualify(args, model)
    assert not args.output.exists() and state.closed


def test_qualification_refuses_missing_draft_counters(qualification):
    tool, args, state, model = qualification
    state.missing_draft_stats = True
    with pytest.raises(AssertionError, match="draft.*proposals"):
        tool.qualify(args, model)
    assert not args.output.exists() and state.closed


@pytest.mark.parametrize("allocated,reserved", [(400, 700), (0, 700)])
@pytest.mark.parametrize("preserve", [False, True])
def test_qualification_requires_zero_allocator_memory_in_every_mode(qualification, allocated, reserved, preserve):
    tool, args, state, model = qualification
    state.allocated, state.reserved, args.preserve_cache = allocated, reserved, preserve
    with pytest.raises(AssertionError, match="CUDA.*survived sleep"):
        tool.qualify(args, model)
    assert not args.output.exists() and state.closed


@pytest.mark.parametrize("draft,no_drafts,preserve", [(True, False, False), (False, False, False),
                                                     (True, True, False), (False, True, True)])
def test_qualification_passes_exercised_drafts_and_explicit_serial_modes(qualification, draft, no_drafts, preserve):
    tool, args, state, model = qualification
    args.draft = args.draft if draft else None
    args.no_drafts, args.preserve_cache = no_drafts, preserve
    if no_drafts or not draft:
        state.missing_draft_stats = True
    tool.qualify(args, model)
    report = json.loads(args.output.read_text())
    assert report["passed"] and len(report["cycles"]) == 2 and state.closed


def test_one_token_draft_qualification_is_refused_before_loading(qualification):
    tool, args, state, model = qualification
    args.tokens = 1
    with pytest.raises(ValueError, match="at least two.*tokens"):
        tool.qualify(args, model)
    assert state.loaded == 0 and not args.output.exists()
