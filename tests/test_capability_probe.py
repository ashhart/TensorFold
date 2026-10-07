"""The custom-kernel capability probe: mocked-probe gate logic, cache behavior, and a real result."""

from __future__ import annotations

import functools

import pytest

from tensorfold.kernels import capability

pytest.importorskip("mlx.core")


def test_custom_kernels_follows_the_probe(monkeypatch):
    monkeypatch.setattr(capability, "_probe", lambda: False)
    assert capability.custom_kernels() is False
    monkeypatch.setattr(capability, "_probe", lambda: True)
    assert capability.custom_kernels() is True


def test_the_probe_runs_once(monkeypatch):
    calls = []

    @functools.cache
    def counted() -> bool:
        calls.append(1)
        return True

    monkeypatch.setattr(capability, "_probe", counted)
    assert capability.custom_kernels() is True
    assert capability.custom_kernels() is True
    assert len(calls) == 1


def test_gates_ask_the_probe(monkeypatch):
    from tensorfold.kernels.glm.flash.v1 import fused
    from tensorfold.kernels.qwen.dense.v1 import lane_gdn

    monkeypatch.setattr(capability, "custom_kernels", lambda: False)
    monkeypatch.setattr(lane_gdn, "_STEP_KERNEL", None)
    assert fused.metal() is False
    assert lane_gdn._step_kernel() is None

    seen = []
    monkeypatch.setattr(capability, "custom_kernels", lambda: seen.append(1) or True)
    assert fused.metal() is False  # the GPU-default-device check stays: a probe result alone opens no gate
    assert lane_gdn._step_kernel() is not None
    assert seen


def test_the_real_probe_returns_a_bool():
    assert isinstance(capability.custom_kernels(), bool)
