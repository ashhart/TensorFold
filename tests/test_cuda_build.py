"""CUDA extensions build for the GPU present, whatever architecture list the container sets (#56), and say what a
silent start does: a first build, or a lock a killed build left."""

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cuda import build

NGC_LIST = "8.0 8.6 9.0 10.0 11.0 12.0+PTX"
HINT = "if no other build is running, a killed build left it: stop this start, delete the lock and start again"
SM121 = "-gencode=arch=compute_121,code=sm_121"


def _gpu(monkeypatch, capability, name="GPU"):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: name)


def test_every_extension_builds_through_the_helper():
    src = Path(__file__).resolve().parents[1] / "src" / "tensorfold"
    direct = [p.relative_to(src) for p in src.rglob("*.py")
              if p.name != "build.py" and "from torch.utils.cpp_extension import load" in p.read_text()]
    assert direct == []


@pytest.mark.torch
def test_the_flags_name_only_this_gpu(monkeypatch):
    _gpu(monkeypatch, (12, 1))
    assert build.arch_flags() == ["-gencode=arch=compute_121,code=sm_121"]


@pytest.mark.torch
def test_an_older_gpu_is_refused_by_name(monkeypatch):
    _gpu(monkeypatch, (8, 6), "NVIDIA GeForce RTX 3090")
    with pytest.raises(RuntimeError, match=r"capability 9\.0 or newer.*RTX 3090.*is 8\.6"):
        build.arch_flags()


@pytest.mark.torch
def test_the_container_list_adds_nothing(monkeypatch, tmp_path):
    import torch.utils.cpp_extension as ext

    _gpu(monkeypatch, (12, 1))
    seen = {}
    monkeypatch.setattr(ext, "load", lambda **kw: seen.update(kw) or "module")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", NGC_LIST)
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))     # load looks up the build directory
    assert build.load(name="x", sources=[], extra_cuda_cflags=["-O3"]) == "module"
    assert seen["extra_cuda_cflags"] == ["-O3", "-gencode=arch=compute_121,code=sm_121"]
    assert ext._get_cuda_arch_flags(seen["extra_cuda_cflags"]) == []


@pytest.fixture
def ext(tmp_path, monkeypatch):
    """Extension ``tf_test`` with its build directory under ``tmp_path`` and torch's build replaced by a recorder."""

    import torch.utils.cpp_extension as cpp_extension

    _gpu(monkeypatch, (12, 1))
    sources = []
    for name in ("ext.cpp", "ext.cu"):
        (tmp_path / name).write_text("// source\n")
        sources.append(str(tmp_path / name))
    calls, said = [], []

    def directory(name, verbose):
        (tmp_path / name).mkdir(exist_ok=True)
        return str(tmp_path / name)

    monkeypatch.setattr(cpp_extension, "_get_build_directory", directory)
    monkeypatch.setattr(cpp_extension, "load", lambda *a, **k: calls.append((a, k)) or "module")
    monkeypatch.setattr(build, "_say", said.append)
    return SimpleNamespace(build=build, torch=cpp_extension, dir=Path(directory("tf_test", False)), sources=sources,
                           calls=calls, said=said)


def _built(ext, mtime=None):
    module = ext.dir / f"tf_test{getattr(ext.torch, 'LIB_EXT', '.so')}"
    module.write_bytes(b"")
    stamp = time.time() + 60 if mtime is None else mtime
    os.utime(module, (stamp, stamp))


@pytest.mark.torch
def test_every_other_argument_reaches_torch_unchanged(ext):
    flags = {"extra_cuda_cflags": ["-O3", "--fmad=false"], "extra_include_paths": ["inc"], "verbose": False}
    assert ext.build.load(name="tf_test", sources=ext.sources, **flags) == "module"
    assert ext.calls == [((), {"name": "tf_test", "sources": ext.sources, **flags,
                               "extra_cuda_cflags": ["-O3", "--fmad=false", SM121]})]


@pytest.mark.torch
def test_a_first_build_says_so(ext):
    ext.build.load(name="tf_test", sources=ext.sources, verbose=False)
    assert ext.said == ["building CUDA extension tf_test (first start after an install or update; "
                        "later starts reuse it)"]
    assert len(ext.calls) == 1


@pytest.mark.torch
def test_a_built_extension_starts_quietly(ext):
    _built(ext)
    ext.build.load(name="tf_test", sources=ext.sources, verbose=False)
    assert ext.said == []
    assert len(ext.calls) == 1


@pytest.mark.torch
def test_a_source_newer_than_the_built_module_is_a_build(ext):
    _built(ext, mtime=1.0)
    ext.build.load(name="tf_test", sources=ext.sources, verbose=False)
    assert [line.split(" (")[0] for line in ext.said] == ["building CUDA extension tf_test"]


@pytest.mark.torch
def test_a_lock_is_named_with_the_fix(ext):
    lock = ext.dir / "lock"
    lock.write_text("")
    ext.build.load(name="tf_test", sources=ext.sources, verbose=False)
    assert len(ext.said) == 1
    assert str(lock) in ext.said[0] and HINT in ext.said[0]
    assert "building" not in ext.said[0]
    assert len(ext.calls) == 1                       # torch still decides: it waits, then loads


@pytest.mark.torch
def test_a_build_directory_argument_is_the_one_checked(ext, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "lock").write_text("")
    ext.build.load(name="tf_test", sources=ext.sources, build_directory=str(other), verbose=False)
    assert str(other / "lock") in ext.said[0]
    assert ext.calls[0][1]["build_directory"] == str(other)


@pytest.mark.torch
def test_an_older_gpu_is_refused_before_any_line(ext, monkeypatch):
    _gpu(monkeypatch, (8, 6), "NVIDIA GeForce RTX 3090")
    with pytest.raises(RuntimeError, match="RTX 3090"):
        ext.build.load(name="tf_test", sources=ext.sources, verbose=False)
    assert ext.said == [] and ext.calls == []


@pytest.mark.torch
def test_a_wait_past_the_delay_repeats_the_hint(ext, monkeypatch):
    lock = ext.dir / "lock"
    lock.write_text("")
    repeated = threading.Event()

    def say(text):
        ext.said.append(text)
        if text.startswith("still waiting"):
            repeated.set()

    monkeypatch.setattr(ext.build, "_say", say)
    monkeypatch.setattr(ext.build, "LOCK_WAIT_SECONDS", 0.01)
    monkeypatch.setattr(ext.torch, "load", lambda **k: repeated.wait(5))
    assert ext.build.load(name="tf_test", sources=ext.sources, verbose=False) is True
    assert len(ext.said) == 2
    assert str(lock) in ext.said[1] and HINT in ext.said[1]


@pytest.mark.torch
def test_a_finished_load_does_not_repeat_the_hint(ext, monkeypatch):
    (ext.dir / "lock").write_text("")
    monkeypatch.setattr(ext.build, "LOCK_WAIT_SECONDS", 0.05)
    ext.build.load(name="tf_test", sources=ext.sources, verbose=False)
    time.sleep(0.3)
    assert len(ext.said) == 1


@pytest.mark.torch
def test_no_repeat_once_the_lock_is_gone_or_replaced(ext):
    lock = ext.dir / "lock"
    lock.write_text("")
    os.utime(lock, (1.0, 1.0))
    seen = os.stat(lock)
    identity = (seen.st_ino, seen.st_mtime_ns)
    ext.build._still_waiting(str(lock), identity)
    assert len(ext.said) == 1                        # the same file still there: repeated
    lock.unlink()
    ext.build._still_waiting(str(lock), identity)
    lock.write_text("")                              # a new lock: this start took it and is building
    ext.build._still_waiting(str(lock), identity)
    assert len(ext.said) == 1


@pytest.mark.torch
@pytest.mark.parametrize("lookup", ["missing", "raises"])
def test_without_torchs_directory_lookup_the_build_goes_ahead_quietly(ext, monkeypatch, lookup):
    if lookup == "missing":
        monkeypatch.delattr(ext.torch, "_get_build_directory")
    else:
        monkeypatch.setattr(ext.torch, "_get_build_directory", lambda *a, **k: 1 / 0)
    (ext.dir / "lock").write_text("")
    assert ext.build.load(name="tf_test", sources=ext.sources, verbose=False) == "module"
    assert ext.said == []


@pytest.mark.torch
def test_the_lock_named_is_the_one_torch_waits_on(tmp_path, monkeypatch):
    """Real torch, nothing compiled: ``load`` waits on the lock the line names until it is deleted."""

    _gpu(monkeypatch, (12, 1))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "extensions"))
    lock = tmp_path / "extensions" / "tf_lock_probe" / "lock"      # TORCH_EXTENSIONS_DIR/<name>/lock
    lock.parent.mkdir(parents=True)
    lock.write_text("")
    source = tmp_path / "tf_lock_probe.cpp"
    source.write_text("int tf_lock_probe() { return 0; }\n")
    said, ended = [], []
    monkeypatch.setattr(build, "_say", said.append)

    def start():
        try:
            ended.append(build.load(name="tf_lock_probe", sources=[str(source)], verbose=False))
        except Exception as error:  # noqa: BLE001 - nothing was built, so the import after the wait fails
            ended.append(error)

    thread = threading.Thread(target=start, daemon=True)
    thread.start()
    thread.join(0.5)
    assert thread.is_alive() and not ended           # torch is waiting on that file
    assert len(said) == 1 and str(lock) in said[0]
    lock.unlink()
    thread.join(10)
    assert not thread.is_alive() and len(ended) == 1
