"""The server wires the weights it holds after loading, within its budget and the system ceiling, and releases them."""

from types import SimpleNamespace

from tensorfold.server import residency

MIB = 1024**2


class FakeMX:
    def __init__(self, active: int, recommended: int, refuse: bool = False) -> None:
        self.active, self.recommended, self.refuse = active, recommended, refuse
        self.limits: list[int] = []
        self.calls: list[str] = []

    def synchronize(self) -> None:
        self.calls.append("sync")

    def clear_cache(self) -> None:
        self.calls.append("clear")

    def get_active_memory(self) -> int:
        return self.active

    def device_info(self) -> dict:
        return {"max_recommended_working_set_size": self.recommended}

    def set_wired_limit(self, limit: int) -> int:
        if self.refuse or limit > self.recommended:
            raise ValueError("over the recommended working set")
        self.limits.append(limit)
        return 0


def test_it_wires_what_mlx_holds_once_the_freed_buffers_are_gone(monkeypatch):
    monkeypatch.setattr(residency, "system_ceiling", lambda mx: 100)
    mx = FakeMX(active=40, recommended=100)
    assert residency.wire_resident(mx, budget_bytes=80) == 40
    assert mx.limits == [40] and mx.calls == ["sync", "clear"]


def test_the_budget_and_the_system_ceiling_cap_it(monkeypatch):
    monkeypatch.setattr(residency, "system_ceiling", lambda mx: 30)
    assert residency.wire_resident(FakeMX(active=40, recommended=100), budget_bytes=80) == 30
    assert residency.wire_resident(FakeMX(active=40, recommended=100), budget_bytes=20) == 20


def test_a_limit_mlx_refuses_leaves_the_weights_unwired(monkeypatch):
    monkeypatch.setattr(residency, "system_ceiling", lambda mx: 100)
    mx = FakeMX(active=40, recommended=100, refuse=True)
    assert residency.wire_resident(mx, budget_bytes=80) == 0 and mx.limits == []


def test_the_ceiling_is_the_kernel_limit_when_it_is_set_lower(monkeypatch):
    for out, want in (("1\n", 1 * MIB), ("0\n", 10 * MIB), ("", 10 * MIB), ("64\n", 10 * MIB)):
        monkeypatch.setattr(residency.subprocess, "run", lambda *a, out=out, **k: SimpleNamespace(stdout=out))
        assert residency.system_ceiling(FakeMX(0, 10 * MIB)) == want


def test_unwire_releases_everything():
    mx = FakeMX(active=40, recommended=100)
    residency.unwire(mx)
    assert mx.limits == [0] and mx.calls == ["sync"]


def test_wiring_changes_no_bits():
    import pytest

    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("needs a Metal GPU")
    import mlx.nn as nn

    mx.random.seed(7)
    layer = nn.QuantizedLinear(512, 1024, group_size=64, bits=4)
    x = mx.random.normal((64, 512)).astype(mx.bfloat16)
    before = layer(x)
    mx.eval(before)
    try:
        assert residency.wire_resident(mx, 1 << 40) > 0
        after = layer(x)
        mx.eval(after)
    finally:
        residency.unwire(mx)
    assert bool(mx.array_equal(before, after).item())
