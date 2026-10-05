"""Host ownership and reload checks; these do not measure GPU reclamation."""

import gc
import hashlib
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cuda import precision, prompt_precision
from tensorfold.server.lifecycle import Lifecycle, LifecycleError


class Buffer:
    pass


class Engine:
    def __init__(self, events, *, context=4096):
        self.events = events
        self.context_window = context
        self.w = Buffer()
        self.w.precision, self.w.quant, self.w.fast_prefill = "full", "mlx", True
        self.draft, self.vision, self.cache, self.room, self.multi = (Buffer() for _ in range(5))
        self.multi.target, self.multi.draft, self.multi.vision = self.w, self.draft, self.vision
        self.multi.cycle = self.multi
        self.scheduler = SimpleNamespace(max_streams=2)
        self.tp, self.rank, self.concurrent, self.allow_copy = 1, 0, True, True
        self.max_rows, self.tree_rows = 12, None

    def close(self):
        self.events.append("close")
        self.scheduler = None


@pytest.fixture
def checkpoint(tmp_path):
    tmp_path = tmp_path / "target"
    tmp_path.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors", "chat_template.jinja"):
        (tmp_path / name).write_bytes(b"checkpoint content")
    return tmp_path


@pytest.fixture
def cuda(monkeypatch):
    events = []
    device = SimpleNamespace(synchronize=lambda: events.append("sync"), empty_cache=lambda: events.append("empty"),
                             memory_allocated=lambda: 10, memory_reserved=lambda: 20)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=device))
    previous = (precision.mode(), precision.asked(), prompt_precision.fp8())
    yield events
    precision.set_mode(previous[0], asked=previous[1])
    prompt_precision.set_fp8(previous[2])


def setup(checkpoint, cuda, monkeypatch, factory=None, drafter=None, cache_dir=None):
    from tensorfold.cuda import sleep
    from tensorfold.families.qwen3_5 import cuda_sleep

    monkeypatch.setattr(sleep, "clear_tensor_caches", lambda: cuda.append("globals"))
    engine = Engine(cuda)
    refs = [weakref.ref(getattr(engine, key)) for key in ("w", "draft", "vision", "cache", "room", "multi")]
    refs.append(weakref.ref(engine))
    frontend = Buffer()
    app = SimpleNamespace(engine=engine, vision=engine.vision, tok=frontend, template=frontend,
                          context_window=8192, effective_context_window=4096)
    calls = []

    def load(path, **options):
        calls.append((path, options, precision.mode(), prompt_precision.fp8()))
        return factory() if factory else Engine(cuda)

    adapter = sleep.CudaSleep(app, load, checkpoint, {"drafter": str(drafter or checkpoint), "parallel": 2,
                                                   "vision": True, "no_drafts": False}, runtime=cuda_sleep(), cache_dir=cache_dir)
    app.lifecycle = Lifecycle(release=adapter.release, restore=adapter.restore, cleanup=adapter.cleanup,
                              preflight=adapter.prepare)
    return app, adapter, refs, calls


def cache_setup(checkpoint, tmp_path, cuda, monkeypatch):
    from test_cuda_prefix_store import Codec, cache

    from tensorfold.families.qwen3_5 import cuda as package

    codec = Codec()
    monkeypatch.setitem(sys.modules, "tensorfold.families.qwen3_5.cuda.prefix_snapshot", codec)
    monkeypatch.setattr(package, "prefix_snapshot", codec, raising=False)
    monkeypatch.setattr(sys.modules["torch"].cuda, "get_device_capability", lambda: (9, 0), raising=False)
    monkeypatch.setattr(sys.modules["torch"], "__version__", "test", raising=False)
    monkeypatch.setattr(sys.modules["torch"], "version", SimpleNamespace(cuda="test"), raising=False)

    def load():
        engine = Engine(cuda)
        engine.w.norm = SimpleNamespace(device="cpu")
        engine.multi.cache = cache()
        engine.multi.memory_gate = None
        return engine

    app, adapter, refs, _calls = setup(checkpoint, cuda, monkeypatch, factory=load,
                                     cache_dir=tmp_path / "snapshots")
    app.engine.multi.cache = cache([1, 2], [3, 4])
    return app, adapter, codec, refs


def test_snapshot_failure_refuses_sleep_without_stopping_runtime(checkpoint, tmp_path, cuda, monkeypatch):
    app, adapter, codec, refs = cache_setup(checkpoint, tmp_path, cuda, monkeypatch)
    codec.fail_save = True
    with pytest.raises(LifecycleError, match="disk full"):
        app.lifecycle.sleep()
    assert app.lifecycle.snapshot()["state"] == "awake"
    assert app.engine is refs[-1]() and "close" not in cuda
    assert app.engine.multi.cache.longest([1, 2, 8])[0] == [1, 2]
    adapter.close()


def test_saved_cache_survives_teardown_and_is_loaded_only_on_matching_request(checkpoint, tmp_path, cuda, monkeypatch):
    app, adapter, codec, refs = cache_setup(checkpoint, tmp_path, cuda, monkeypatch)
    app.lifecycle.sleep()
    assert all(ref() is None for ref in refs)
    assert adapter.cache_snapshot()["saved_prefixes"] == 2
    app.lifecycle.wake_up()
    assert not codec.loads and not app.engine.multi.cache.entries
    assert app.engine.multi.cache.longest([1, 2, 7])[0] == [1, 2]
    assert codec.loads == [[1, 2]]
    app.lifecycle.sleep()
    app.lifecycle.wake_up()
    assert app.engine.multi.cache.longest([3, 4, 5])[0] == [3, 4]
    adapter.close()


def test_bad_snapshot_keeps_wake_retryable_until_bytes_restored(checkpoint, tmp_path, cuda, monkeypatch):
    app, adapter, _codec, _refs = cache_setup(checkpoint, tmp_path, cuda, monkeypatch)
    app.lifecycle.sleep()
    path = next((tmp_path / "snapshots").rglob("*.safetensors"))
    original = path.read_bytes()
    path.write_bytes(b"bad")
    with pytest.raises(LifecycleError, match="invalid snapshot"):
        app.lifecycle.wake_up()
    assert app.lifecycle.snapshot()["state"] == "sleeping" and app.engine is None
    path.write_bytes(original)
    app.lifecycle.wake_up()
    assert app.engine.multi.cache.longest([1, 2, 7])[0] == [1, 2]
    adapter.close()


def test_sleep_drops_all_runtime_owners_and_reloads_same_app(checkpoint, cuda, monkeypatch):
    with precision.using("full", asked=True), prompt_precision.using(True):
        app, adapter, refs, calls = setup(checkpoint, cuda, monkeypatch)
    frontend = app.tok
    assert app.context_window == 4096
    assert adapter.memory_snapshot() == {"allocated_bytes": 10, "reserved_bytes": 20}
    app.lifecycle.sleep()
    assert app.engine is app.vision is None
    assert all(ref() is None for ref in refs)
    assert cuda == ["close", "sync", "globals", "empty"]
    app.lifecycle.wake_up()
    assert app.tok is frontend and app.template is frontend
    assert app.engine.vision is app.vision
    assert calls[0] == (checkpoint.resolve(), {"drafter": str(checkpoint.resolve()), "parallel": 2,
                        "vision": True, "no_drafts": False, "context": 4096, "context_explicit": True}, "full", True)
    new = weakref.ref(app.engine)
    app.lifecycle.sleep()
    assert new() is None


@pytest.mark.parametrize("change", ["content", "remove", "new"])
def test_checkpoint_change_refuses_sleep_before_teardown(checkpoint, cuda, monkeypatch, change):
    app, adapter, refs, calls = setup(checkpoint, cuda, monkeypatch)
    path = checkpoint / "model.safetensors"
    if change == "content":
        old = path.stat()
        path.write_bytes(b"checkpoint changed")
        import os
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    elif change == "remove":
        path.unlink()
    else:
        (checkpoint / "added.safetensors").write_bytes(b"new weights")
    with pytest.raises(LifecycleError, match="checkpoint"):
        app.lifecycle.sleep()
    assert app.lifecycle.snapshot()["state"] == "awake"
    assert app.engine is refs[-1]() and not cuda


def test_changed_draft_blocks_wake_and_original_content_allows_retry(checkpoint, tmp_path, cuda, monkeypatch):
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_bytes(b"original draft config")
    (draft / "model.safetensors").write_bytes(b"draft weights")
    app, adapter, refs, calls = setup(checkpoint, cuda, monkeypatch, drafter=draft)
    app.lifecycle.sleep()
    path = draft / "config.json"
    original = path.read_bytes()
    path.write_bytes(b"different config")
    with pytest.raises(LifecycleError, match="checkpoint"):
        app.lifecycle.wake_up()
    assert not calls and app.engine is None
    path.write_bytes(original)
    app.lifecycle.wake_up()
    assert app.lifecycle.snapshot()["ready"]


def test_checkpoint_identity_hashes_complete_shards_and_distinguishes_target_and_draft(checkpoint, tmp_path):
    from tensorfold.cuda.sleep import CheckpointIdentity

    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_bytes(b"draft config")
    (draft / "model.safetensors").write_bytes(b"draft weights")
    payload = bytes(range(256)) * 8192 + b"partial final block"
    shard = checkpoint / "model-00002.safetensors"
    shard.write_bytes(payload)
    expected = {(str(root.resolve()), path.name): hashlib.sha256(path.read_bytes()).hexdigest()
                for root in (checkpoint, draft) for path in root.iterdir()}
    (checkpoint / ".cache").mkdir()
    (checkpoint / ".cache" / "ignored.json").write_bytes(b"ignored")
    (checkpoint / "ignored.md").write_bytes(b"ignored")
    identity = CheckpointIdentity(checkpoint, str(draft))
    assert identity.manifest == expected
    identity.verify()
    shard.write_bytes(payload[:-1] + b"X")
    with pytest.raises(ValueError, match="content changed"):
        identity.verify()


def test_checkpoint_read_failure_refuses_sleep_without_releasing_runtime(checkpoint, cuda, monkeypatch):
    app, _adapter, refs, _calls = setup(checkpoint, cuda, monkeypatch)
    original_open = Path.open

    def unreadable(path, *args, **kwargs):
        if path == checkpoint / "model.safetensors":
            raise PermissionError("checkpoint read refused")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unreadable)
    with pytest.raises(LifecycleError, match="missing or unreadable"):
        app.lifecycle.sleep()
    assert app.lifecycle.snapshot()["state"] == "awake"
    assert app.engine is refs[-1]() and not cuda


def test_checkpoint_changed_during_reload_is_not_published_and_can_retry(checkpoint, cuda, monkeypatch):
    import os

    path = checkpoint / "model.safetensors"
    original = path.read_bytes()
    attempts, reloaded = [], []

    def load():
        engine = Engine(cuda)
        attempts.append(1)
        reloaded.append(weakref.ref(engine))
        if len(attempts) == 1:
            before = path.stat()
            path.write_bytes(b"X" * len(original))
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return engine

    app, _adapter, _refs, _calls = setup(checkpoint, cuda, monkeypatch, load)
    app.lifecycle.sleep()
    with pytest.raises(LifecycleError, match="content changed"):
        app.lifecycle.wake_up()
    assert app.lifecycle.snapshot()["state"] == "sleeping"
    assert app.engine is None and reloaded[0]() is None
    path.write_bytes(original)
    app.lifecycle.wake_up()
    assert app.lifecycle.snapshot()["ready"] and app.engine is reloaded[1]()


def test_failed_constructor_drops_traceback_allocations_before_cleanup(checkpoint, cuda, monkeypatch):
    partial = []
    attempts = []

    def load():
        attempts.append(1)
        if len(attempts) == 1:
            allocation = Buffer()
            partial.append(weakref.ref(allocation))
            raise RuntimeError("load failed after allocation")
        return Engine(cuda)

    app, adapter, refs, calls = setup(checkpoint, cuda, monkeypatch, load)
    app.lifecycle.sleep()
    import tensorfold.cuda.sleep as sleep

    def clear():
        gc.collect()
        assert partial[0]() is None

    monkeypatch.setattr(sleep, "clear_tensor_caches", clear)
    with pytest.raises(LifecycleError, match="load failed"):
        app.lifecycle.wake_up()
    assert app.engine is app.vision is None
    assert app.lifecycle.snapshot()["state"] == "sleeping"
    app.lifecycle.wake_up()
    assert app.lifecycle.snapshot()["ready"]


@pytest.mark.parametrize("field,value", [("context_window", 2048), ("allow_copy", False),
                                         ("w.precision", "checkpoint"), ("scheduler.max_streams", 1),
                                         ("draft.taps", (1, 5)), ("draft.windows", [4095, 32768]),
                                         ("draft.fast", True)])
def test_mismatched_reload_closes_partial_worker_and_permits_retry(checkpoint, cuda, monkeypatch, field, value):
    attempts, partial = [], []

    def load():
        engine = Engine(cuda)
        attempts.append(1)
        if len(attempts) == 1:
            parent, separator, name = field.partition(".")
            setattr(getattr(engine, parent) if separator else engine, name if separator else field, value)
            partial.append(weakref.ref(engine))
        return engine

    app, adapter, refs, calls = setup(checkpoint, cuda, monkeypatch, load)
    app.lifecycle.sleep()
    with pytest.raises(LifecycleError, match="settings"):
        app.lifecycle.wake_up()
    assert app.engine is app.vision is None and partial[0]() is None
    assert cuda.count("close") == 2
    app.lifecycle.wake_up()
    assert app.lifecycle.snapshot()["ready"]


def test_failed_restore_allocator_cleanup_keeps_admission_closed(checkpoint, cuda, monkeypatch):
    def fail():
        raise RuntimeError("allocation failed")

    app, adapter, refs, calls = setup(checkpoint, cuda, monkeypatch, fail)
    app.lifecycle.sleep()

    def cleanup_fail():
        raise RuntimeError("allocator cleanup failed")

    monkeypatch.setattr(sys.modules["torch"].cuda, "empty_cache", cleanup_fail)
    with pytest.raises(LifecycleError, match="cleanup failed"):
        app.lifecycle.wake_up()
    assert app.lifecycle.snapshot()["state"] == "error"
    with pytest.raises(LifecycleError):
        with app.lifecycle.admit():
            pytest.fail("fatal cleanup reopened admission")


def test_missing_checkpoint_refused_before_adapter_setup(checkpoint, cuda, monkeypatch):
    (checkpoint / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="checkpoint"):
        setup(checkpoint, cuda, monkeypatch)


def test_global_tensor_caches_release_without_importing_unused_modules(monkeypatch):
    from tensorfold.cuda.sleep import clear_tensor_caches
    from functools import lru_cache

    @lru_cache(None)
    def base(index):
        return Buffer()

    one = Buffer()
    refs = [weakref.ref(base(0)), weakref.ref(one)]
    monkeypatch.setitem(sys.modules, "tensorfold.cuda.kernels.attention", SimpleNamespace(clear_tensor_cache=base.cache_clear))
    ones = {"cuda:0": one}
    monkeypatch.setitem(sys.modules, "tensorfold.cuda.nvfp4.linear", SimpleNamespace(clear_tensor_cache=ones.clear))
    del one
    clear_tensor_caches()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("kind", ["affine", "nvfp4", "exl3"])
def test_readers_close_pinned_staging_after_failed_load(checkpoint, cuda, monkeypatch, kind):
    import importlib.util
    import tensorfold.families.qwen3_5.cuda as package

    modules = {}
    for name in ("weights", "exl3_load", "nvfp4_load"):
        fullname = f"tensorfold.families.qwen3_5.cuda.{name}"
        spec = importlib.util.spec_from_file_location(fullname, Path(package.__file__).parent / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, fullname, module)
        monkeypatch.setattr(package, name, module, raising=False)
        spec.loader.exec_module(module)
        modules[name] = module
    weights, exl3, nvfp4 = (modules[name] for name in ("weights", "exl3_load", "nvfp4_load"))
    monkeypatch.setattr(weights.Config, "read", lambda path: SimpleNamespace(layers=0))
    (checkpoint / "config.json").write_text("{}")
    closed = []

    def fail(*args, **kwargs):
        raise RuntimeError("injected read failure")

    class Reader:
        def __iter__(self):
            fail()

        def close(self):
            closed.append(True)

    monkeypatch.setattr(weights, "_Tensors", lambda *a, **k: Reader())
    monkeypatch.setattr(exl3, "quant_config", lambda *a: None)
    monkeypatch.setattr(nvfp4, "quantized", lambda *a: False)
    if kind == "affine":
        load = weights.load
    elif kind == "nvfp4":
        monkeypatch.setitem(sys.modules, "tensorfold.cuda.capacity", SimpleNamespace(headers=lambda *a: {}))
        monkeypatch.setitem(sys.modules, "tensorfold.cuda.nvfp4.linear",
                            SimpleNamespace(Fp4Linear=None, Fp8Linear=None, Staging=fail))
        load = nvfp4.load_nvfp4
    else:
        from tensorfold.cuda.exl3 import format as fmt

        monkeypatch.setattr(fmt, "scan", lambda *a, **k: SimpleNamespace(bad={}, groups={"model.language_model.x": 1},
                                                                        plain=[]))
        monkeypatch.setitem(sys.modules, "tensorfold.cuda.exl3.prefill", SimpleNamespace(Workspace=lambda: None))
        monkeypatch.setattr(exl3, "_where", lambda *a: {})
        monkeypatch.setattr(exl3, "_files", lambda *a: Reader())
        monkeypatch.setattr(exl3, "_read", fail)
        load = exl3.load_exl3
    with pytest.raises(RuntimeError, match="injected read failure"):
        load(checkpoint)
    assert closed == [True]
