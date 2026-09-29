"""Prompt kernels that don't build or don't match MLX's bits say so loudly, or stop with TF_REQUIRE_KERNELS=1 (#88)."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm


def _broken() -> bool:
    raise RuntimeError("[metal::Device] Unable to build metal library from source\n"
                       "utils.h:5484:3: error: no matching function for call to 'tf_gather_qmm_rhs_impl'")


@pytest.fixture
def check(monkeypatch):
    """A fresh process's first check on an M1-M4 GPU; the test picks what the self-check does."""

    monkeypatch.setattr(prefill_mm, "_tiles", [])
    monkeypatch.setattr(prefill_mm, "_tensor_units", lambda: False)
    monkeypatch.delenv("TF_REQUIRE_KERNELS", raising=False)
    return lambda outcome: monkeypatch.setattr(prefill_mm, "_self_check", outcome)


def test_a_kernel_that_does_not_build_warns_loudly_and_prompts_use_mlx(check, capsys):
    check(_broken)
    assert prefill_mm.tiles() is False
    err = capsys.readouterr().err
    assert "WARNING" in err and "error: no matching function" in err and f"MLX {mx.__version__}" in err
    assert "replies are the same" in err and "TF_REQUIRE_KERNELS=1" in err


def test_kernels_with_other_bits_warn_too(check, capsys):
    check(lambda: False)
    assert prefill_mm.tiles() is False
    assert "gave other bits than MLX's" in capsys.readouterr().err


def test_require_kernels_stops_instead_of_falling_back(check, monkeypatch):
    check(_broken)
    monkeypatch.setenv("TF_REQUIRE_KERNELS", "1")
    with pytest.raises(RuntimeError, match="prompt kernels did not build"):
        prefill_mm.tiles()


def test_kernels_that_match_say_nothing(check, capsys):
    check(lambda: True)
    assert prefill_mm.tiles() is True
    assert capsys.readouterr().err == ""


def test_glm_checks_its_expert_kernels_at_startup(monkeypatch):
    from tensorfold.families.glm5_next.runtime import GLMFlash

    calls = []
    monkeypatch.setattr(prefill_mm, "fast_prefill", lambda: True)
    monkeypatch.setattr(prefill_mm, "tiles", lambda: calls.append("tiles") or True)
    GLMFlash.resolve_prefill_identity(object())
    assert calls == ["tiles"]
