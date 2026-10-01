"""CPU discovery, CUDA registration and rejection before loading."""

from __future__ import annotations

import sys

import pytest

from tensorfold import families

DS_FAMILY = families.families()["deepseek_v4"]
# a gguf storage config: the supported serial storage for this family
GGUF = {"quantization_config": {"quant_method": "gguf"}, "bits": 2.5625}


def _heavy(modules):
    return {m for m in modules if m == "torch" or m.startswith(("torch.", "mlx", "transformers", "transformers."))}


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
    # G0: the existing MLX selection stays valid (it must survive the CUDA port).
    assert "mlx" in families.backends_of(DS_FAMILY)
    assert "mlx" in families.readable_quants(DS_FAMILY, "mlx")


def test_unsupported_storage_rejected_before_load():
    families.require_readable(DS_FAMILY, GGUF, "cuda")
    with pytest.raises(ValueError, match="does not read"):
        families.require_readable(DS_FAMILY, {"quantization_config": {"quant_method": "unknown"}}, "cuda")


def _serial_engine(**opts):
    # This path cannot be read: bad options must fail before filesystem/model access.
    return DS_FAMILY.package.cuda_engine("/missing-model-no-load-sentinel", **opts)


def test_serial_engine_rejects_tp_neq_1_before_load():
    # No-load sentinel: rejection must happen inside the factory call itself --
    # a tp != 1 config must never return an engine or load any weight.
    with pytest.raises(ValueError, match=r"tp.*1|tp.*one GPU|tp.*serial"):
        _serial_engine(tp=2)


def test_serial_engine_rejects_parallel_gt_1_before_load():
    # Serial contract: the first CUDA release is one-request serial, so a
    # --parallel > 1 config must be refused before any load.
    with pytest.raises(ValueError, match=r"parallel|concurrent|one request"):
        _serial_engine(parallel=8)


def test_serial_engine_rejects_incompatible_drafter_before_load():
    # No-load sentinel: an incompatible draft model must be refused before load.
    with pytest.raises(ValueError, match=r"drafter|draft|no drafting"):
        _serial_engine(drafter="dflash")


def test_serial_engine_rejects_unknown_format_before_load():
    # GGUF is the intended supported storage for this family, so the unknown-
    # format case must use a genuinely unsupported value, never gguf.
    with pytest.raises(ValueError, match=r"format|storage|unsupported"):
        _serial_engine(format="onnx")


def test_serial_engine_signature_pins_cuda_family_contract():
    """Pins the future cuda_engine factory signature to adding-a-cuda-family.md.

    RED until the factory exists; when it lands with this signature, this test
    must pass (GREEN) and the xfail mark removed. Contract:

        def cuda_engine(model_dir, *, drafter="", tp=1, rank=0, master="",
                        master_port=29551, no_drafts=False, mtp_drafts=None,
                        **options):
    """
    import inspect

    import tensorfold.families.deepseek_v4 as dsv4

    factory = getattr(dsv4, "cuda_engine", None)
    if factory is None:
        pytest.fail("deepseek_v4.cuda_engine not implemented yet (T16) — pending RED")

    sig = inspect.signature(factory)
    p = sig.parameters
    # keyword-only serial params with the documented defaults
    assert p["drafter"].default == ""
    assert p["tp"].default == 1
    assert p["rank"].default == 0
    assert p["master"].default == ""
    assert p["master_port"].default == 29551
    assert p["no_drafts"].default is False
    assert p["mtp_drafts"].default is None
    # every param except model_dir is keyword-only (drafter/tp/rank/master/...)
    assert all(v.kind == inspect.Parameter.KEYWORD_ONLY for k, v in p.items() if k not in ("model_dir", "options"))
    # a catch-all **options must remain for parallel/format/... rejections
    assert any(v.kind == inspect.Parameter.VAR_KEYWORD for v in p.values())
