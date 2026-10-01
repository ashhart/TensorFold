"""The measured HC layout does not require Gluon on other architectures or hosts."""

import builtins

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
from tensorfold.families.qwen4_exp.cuda import hc_check

_hc_fuser = hc_check.fuser


@pytest.fixture(autouse=True)
def clear_dispatch_cache(monkeypatch):
    import sys
    from types import ModuleType

    for name, entry in (("hc_fused", "write_norm"), ("hc_upmix", "prefill_upmix")):
        module = ModuleType(f"{hc_check.__package__}.{name}")
        setattr(module, entry, lambda *args, **kwargs: None)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    hc_check._checked.clear()
    yield
    hc_check._checked.clear()


def test_cpu_does_not_query_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: pytest.fail("CPU queried CUDA"))
    assert _hc_fuser(torch.device("cpu")) is None


@pytest.mark.parametrize("capability", [(8, 9), (9, 0), (12, 0)])
def test_other_architectures_do_not_import_the_new_kernel(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "hc_fused":
            pytest.fail("an unqualified device imported the new kernel")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert _hc_fuser(torch.device("cuda", 0)) is None


def test_an_older_triton_can_keep_the_released_kernel_on_gb10(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 1))
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == "hc_fused":
            raise ImportError("Gluon is unavailable in this installed Triton")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    assert _hc_fuser(torch.device("cuda", 0)) is None


@pytest.mark.parametrize("equal", [True, False])
def test_check_is_once_per_device_and_caches_both_outcomes(monkeypatch, capsys, equal):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 1))
    calls = []
    def check(candidate, device):
        calls.append(device.index)
        return equal
    monkeypatch.setattr(hc_check, "_check", check)
    for index in (0, 1, 0, 1):
        assert (_hc_fuser(torch.device("cuda", index)) is not None) == equal
    assert calls == [0, 1]
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2 and all("HC fusion self-check" in line for line in lines)
    assert all(("enabled" if equal else "using released kernels") in line for line in lines)


def test_a_check_exception_disables_fusion_for_the_process(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 1))
    def broken(*args):
        raise RuntimeError("compiler or launch failure")
    monkeypatch.setattr(hc_check, "_check", broken)
    assert _hc_fuser(torch.device("cuda", 0)) is None
    monkeypatch.setattr(hc_check, "_check", lambda *args: pytest.fail("failed check was retried"))
    assert _hc_fuser(torch.device("cuda", 0)) is None
    assert capsys.readouterr().out.count("failed (RuntimeError); using released kernels") == 1


def test_concurrent_first_callers_wait_for_the_same_result(monkeypatch, capsys):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 1))
    entered, release, calls = Event(), Event(), []
    def check(candidate, device):
        calls.append(device.index)
        entered.set()
        assert release.wait(5)
        return True
    monkeypatch.setattr(hc_check, "_check", check)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_hc_fuser, torch.device("cuda", 0)) for _ in range(4)]
        assert entered.wait(5)
        release.set()
        results = [f.result(timeout=5) for f in futures]
    assert calls == [0] and results[0] is not None and all(r is results[0] for r in results)
    assert capsys.readouterr().out.count("self-check") == 1


def test_upmix_and_normalization_have_independent_cached_results(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 1))
    calls = []
    monkeypatch.setattr(hc_check, "_check", lambda *args: (calls.append("norm"), True)[1])
    monkeypatch.setattr(hc_check, "_check_upmix", lambda *args: (calls.append("upmix"), False)[1])
    device = torch.device("cuda", 0)
    assert _hc_fuser(device) is not None
    assert _hc_fuser(device, upmix=True) is None
    assert _hc_fuser(device) is not None and _hc_fuser(device, upmix=True) is None
    assert calls == ["norm", "upmix"]
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2 and "HC upmix" in lines[1] and "using released kernels" in lines[1]
