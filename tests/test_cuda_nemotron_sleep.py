"""Nemotron runtime ownership and lazy prefix reuse across the shared lifecycle."""

import sys
import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from test_cuda_sleep import Buffer, checkpoint, cuda  # noqa: F401

from tensorfold.server.lifecycle import Lifecycle, LifecycleError


@dataclass
class Config:
    hidden: int = 16


def setup(checkpoint, cuda, monkeypatch, *, cache_dir=None, mismatch=False, native_context=4096,  # noqa: F811
          served_context=4096, admitted_context=None):
    from tensorfold.cuda import sleep
    from tensorfold.families.nemotron_h import cuda_sleep
    from tensorfold.families.nemotron_h.cuda.app import NemotronEngine

    calls, refs = [], []

    def load(path, **options):
        if admitted_context is not None and options.get("context", 0) > admitted_context:
            raise ValueError("requested context exceeds native checkpoint limit")
        obj = NemotronEngine.__new__(NemotronEngine)
        obj.e = Buffer()
        obj.e.w, obj.e.c, obj.e.device = Buffer(), Config(), "cpu"
        obj.e.max_rows = 16
        obj.max_len = native_context + 16
        if admitted_context is not None:
            obj.capacity_plan = {"context_window": admitted_context}
        obj.tp, obj.rank, obj.drafts, obj.confidence = 1, 0, 3, .3
        obj.draft_ids = (1, 2)
        obj.mtp, obj.serial = Buffer(), Buffer()
        obj.cache, obj.sleep_cache = [], None
        obj._make = lambda: obj.e.w
        if calls and mismatch:
            obj.confidence = .5
        calls.append(options)
        refs.extend(weakref.ref(x) for x in (obj, obj.e, obj.e.w, obj.mtp, obj.serial))
        return obj

    monkeypatch.setattr(sleep, "clear_tensor_caches", lambda: cuda.append("globals"))
    app = SimpleNamespace(engine=load(checkpoint), effective_context_window=served_context, vision=None, tok=Buffer())
    adapter = sleep.CudaSleep(app, load, checkpoint, {}, runtime=cuda_sleep(), cache_dir=cache_dir)
    app.lifecycle = Lifecycle(release=adapter.release, restore=adapter.restore, cleanup=adapter.cleanup,
                              preflight=adapter.prepare)
    return app, adapter, calls, refs


def test_sleep_releases_weights_serial_twin_and_factory_closure(checkpoint, cuda, monkeypatch):  # noqa: F811
    app, _adapter, calls, refs = setup(checkpoint, cuda, monkeypatch)
    tok = app.tok
    app.lifecycle.sleep()
    assert app.engine is None and all(ref() is None for ref in refs)
    app.lifecycle.wake_up()
    assert app.tok is tok and app.engine.context_window == 4096
    assert calls[-1] == {"context": 4096, "context_explicit": True}


def test_changed_draft_settings_refuse_publication(checkpoint, cuda, monkeypatch):  # noqa: F811
    app, _adapter, _calls, refs = setup(checkpoint, cuda, monkeypatch, mismatch=True)
    app.lifecycle.sleep()
    with pytest.raises(LifecycleError, match="settings"):
        app.lifecycle.wake_up()
    assert app.engine is None and all(ref() is None for ref in refs)


def test_served_window_stays_pinned_when_native_capacity_rounds_up(checkpoint, cuda, monkeypatch):  # noqa: F811
    app, _adapter, calls, _refs = setup(checkpoint, cuda, monkeypatch, native_context=4592)
    app.lifecycle.sleep()
    app.lifecycle.wake_up()
    assert app.context_window == 4096
    assert app.engine.context_window == 4592
    assert calls[-1] == {"context": 4096, "context_explicit": True}


def test_rounded_capacity_does_not_exceed_native_limit_on_reload(checkpoint, cuda, monkeypatch):  # noqa: F811
    app, _adapter, calls, _refs = setup(checkpoint, cuda, monkeypatch, native_context=4592,
                                      served_context=4592, admitted_context=4096)
    app.lifecycle.sleep()
    app.lifecycle.wake_up()
    assert app.context_window == app.engine.context_window == 4592
    assert calls[-1] == {"context": 4096, "context_explicit": True}


def test_disk_prefixes_restore_lazily_and_repeat_generations(checkpoint, tmp_path, cuda, monkeypatch):  # noqa: F811
    from test_cuda_prefix_store import Codec

    from tensorfold.families.nemotron_h.cuda import sleep as runtime

    codec = Codec()
    monkeypatch.setattr(runtime, "prefix_codec", lambda: codec)
    monkeypatch.setattr(runtime, "prefix_room", lambda size, prompt: True)
    monkeypatch.setattr(sys.modules["torch"].cuda, "get_device_capability", lambda: (9, 0), raising=False)
    monkeypatch.setattr(sys.modules["torch"], "__version__", "test", raising=False)
    monkeypatch.setattr(sys.modules["torch"], "version", SimpleNamespace(cuda="test"), raising=False)
    app, adapter, _calls, refs = setup(checkpoint, cuda, monkeypatch, cache_dir=tmp_path / "cache")
    app.engine.cache = [([1, 2], SimpleNamespace(value=3)), ([4, 5], SimpleNamespace(value=9))]
    app.lifecycle.sleep()
    assert all(ref() is None for ref in refs)
    app.lifecycle.wake_up()
    assert app.engine.cache == [] and not codec.loads
    assert app.engine._resume([1, 2, 3])[0] == [1, 2]
    assert codec.loads == [[1, 2]]
    app.lifecycle.sleep()
    app.lifecycle.wake_up()
    assert app.engine._resume([4, 5, 6])[0] == [4, 5]
    adapter.close()


def test_disk_prefix_denied_by_memory_uses_resident_hit(checkpoint, tmp_path, cuda, monkeypatch):  # noqa: F811
    from tensorfold.families.nemotron_h.cuda import sleep as runtime

    app, _adapter, _calls, _refs = setup(checkpoint, cuda, monkeypatch)
    monkeypatch.setattr(runtime, "prefix_room", lambda size, prompt: False)
    from test_cuda_prefix_store import Codec, cache

    from tensorfold.cuda.prefix_store import PrefixStore

    with PrefixStore(tmp_path, "identity", Codec()) as store:
        store.save(cache([1, 2]))
        runtime.attach(app.engine, store)
        resident = ([1], {"resident": True})
        app.engine.cache = [resident]
        assert app.engine._resume([1, 2, 3]) == resident
        assert store.snapshot()["memory_misses"] == 1


@pytest.mark.parametrize("draft_head", [False, True])
def test_failed_weight_read_closes_staging(checkpoint, cuda, monkeypatch, draft_head):  # noqa: F811
    import importlib.util
    from pathlib import Path

    from tensorfold.families.nemotron_h import cuda as package

    monkeypatch.setitem(sys.modules, "tensorfold.cuda.experts", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tensorfold.families.qwen3_5.cuda.qmm_fast", SimpleNamespace(tile=None))
    monkeypatch.setitem(sys.modules, "tensorfold.families.qwen3_5.cuda.weights", SimpleNamespace(QLinear=None))
    spec = importlib.util.spec_from_file_location("_nemotron_weight_cleanup", Path(package.__file__).parent / "weights.py")
    weights = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, weights)
    spec.loader.exec_module(weights)
    closed = []

    class Reader:
        def get(self, name):
            raise OSError("injected read failure")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(weights, "_Reader", lambda *a: Reader())
    monkeypatch.setattr(weights.Config, "read", lambda p: SimpleNamespace(pattern="M"))
    with pytest.raises(OSError, match="injected read failure"):
        if draft_head:
            weights.load_mtp(checkpoint, Config())
        else:
            weights.load(checkpoint)
    assert closed == [True]
