"""G0 gate: discovery and backend-option tests for the DeepSeek V4 serial CUDA port.

Covers the discovery half of gate G0 (R1-R3 of the serial-contract plan):
  * CPU family discovery imports neither the torch backend nor MLX;
  * the existing MLX selection remains valid;
  * an unsupported backend (cuda, when undeclared) is refused before any load;
  * and the *intended-failing* serial-contract tests: tp != 1, --parallel > 1,
    incompatible drafters and unknown formats must fail before loading.

The serial-contract tests are xfail-strict because the serial CUDA engine is not
implemented yet (T16). strict=True means they can never be silently marked
passed. Their rejection behaviour is pinned against the qwen3_5_moe / qwen3_5
cuda-engine contract, the independent oracle for this gate. No GPU test here is
marked passed; the engine tests are recorded as intended-failing (RED) pending
T16, and the engine itself is gated on T19 (kernel evidence).
"""
from __future__ import annotations

import sys

import pytest

from tensorfold import families

DS_FAMILY = families.families()["deepseek_v4"]
# a gguf storage config: refused by discovery for any declared backend today
GGUF = {"quantization_config": {"quant_method": "gguf"}, "bits": 2.5625}


def _heavy(modules):
    return {m for m in modules
            if m == "torch" or m.startswith(("torch.", "mlx", "transformers", "transformers."))}


# ---------------------------------------------------------------------------
# GREEN: CPU family discovery imports neither the torch backend nor MLX.
# ---------------------------------------------------------------------------


def test_importing_family_discovery_pulls_no_torch_or_mlx():
    before = _heavy(sys.modules)

    import tensorfold.families.deepseek_v4 as dsv4  # discovery must not drag heavy backends in

    added = _heavy(sys.modules) - before
    assert added == set(), f"family discovery imported a heavy backend: {sorted(added)}"
    # the family resolves its model_type without a backend import
    assert dsv4.MODEL_TYPES == ("deepseek_v4",)
    assert not hasattr(dsv4, "torch") and not hasattr(dsv4, "mlx")


def test_mlx_selection_remains_valid():
    # G0: the existing MLX family selection stays valid (it must survive the CUDA port).
    assert "mlx" in families.backends_of(DS_FAMILY)
    assert "mlx" in families.readable_quants(DS_FAMILY, "mlx")


def test_unsupported_backend_rejected_before_load():
    # G0/R3: --backend cuda is refused at discovery time (no CUDA backend
    # declared), before any weight is loaded.
    assert "cuda" not in families.backends_of(DS_FAMILY)
    with pytest.raises(ValueError, match="does not read"):
        families.require_readable(DS_FAMILY, GGUF, "cuda")  # refused before any load
    # gguf storage is also refused for the declared mlx backend, before any load
    with pytest.raises(ValueError, match="gguf"):
        families.require_readable(DS_FAMILY, GGUF, "mlx")


# ---------------------------------------------------------------------------
# RED / intended-failing: serial-contract rejection (pending T16).
# ---------------------------------------------------------------------------


def _serial_engine(**opts):  # forwards to the future factory; absent today
    import tensorfold.families.deepseek_v4 as dsv4

    factory = getattr(dsv4, "cuda_engine", None)
    if factory is None:
        raise AssertionError("deepseek_v4.cuda_engine not implemented yet (T16)")
    return factory(DS_FAMILY, **opts)


@pytest.mark.xfail(reason="pending deepseek_v4.cuda_engine serial CUDA factory (T16)", strict=True)
def test_serial_engine_rejects_tp_neq_1_before_load():
    # Serial contract: tp != 1 (e.g. --tp 2) must be refused before any load.
    # Oracle: qwen3_5_moe refuses tp != 1 ("runs on one GPU: drop --tp").
    with pytest.raises(ValueError, match=r"tp.*1|tp.*one GPU|tp.*serial"):
        _serial_engine(tp=2)


@pytest.mark.xfail(reason="pending deepseek_v4.cuda_engine serial CUDA factory (T16)", strict=True)
def test_serial_engine_rejects_parallel_gt_1_before_load():
    # Serial contract: --parallel > 1 is refused before any load (one request).
    with pytest.raises(ValueError, match=r"parallel|concurrent|one request"):
        _serial_engine(parallel=8)


@pytest.mark.xfail(reason="pending deepseek_v4.cuda_engine serial CUDA factory (T16)", strict=True)
def test_serial_engine_rejects_incompatible_drafter_before_load():
    # Serial contract: no drafting; an incompatible/external drafter is refused.
    with pytest.raises(ValueError, match=r"drafter|draft|no drafting"):
        _serial_engine(drafter="dflash")


@pytest.mark.xfail(reason="pending deepseek_v4.cuda_engine serial CUDA factory (T16)", strict=True)
def test_serial_engine_rejects_unknown_format_before_load():
    # Serial contract: an unknown/unsupported storage format is refused before
    # any load, mirroring require_readable(mlx) refusing gguf today.
    with pytest.raises(ValueError, match=r"format|storage|unsupported"):
        _serial_engine(format="gguf")
