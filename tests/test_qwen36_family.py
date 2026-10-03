"""Qwen3.6 MoE's CUDA family: found by model_type, refuses settings it cannot serve before any GPU work."""

import json

import pytest

from tensorfold import families
from tensorfold.families import qwen3_5_moe


def _config(tmp_path, bits=4, group=64):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe", "quantization": {"bits": bits, "group_size": group, "mode": "affine"},
        "text_config": {"model_type": "qwen3_5_moe_text"}}))
    return tmp_path


def test_the_family_is_found_by_model_type(tmp_path):
    assert families.detect(_config(tmp_path)).module == "tensorfold.families.qwen3_5_moe"


def test_cuda_reads_only_4_bit_groups_of_64(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    qwen3_5_moe.check(_config(tmp_path))
    with pytest.raises(ValueError, match="groups of 64"):
        qwen3_5_moe.check(_config(tmp_path, bits=8))


def test_a_mac_reads_other_affine_widths(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")      # the row decoder reads every MLX affine width
    qwen3_5_moe.check(_config(tmp_path, bits=8))


@pytest.mark.parametrize("options, message", [({"tp": 2}, "one GPU"), ({"parallel": 4, "mtp_drafts": 16}, "0 to 15"),
                                              ({"drafter": "some/drafter", "tp": 2}, "one GPU")])
def test_settings_it_cannot_serve_are_refused_first(tmp_path, options, message):
    with pytest.raises(ValueError, match=message):
        qwen3_5_moe.cuda_engine(_config(tmp_path), **options)


@pytest.mark.parametrize("options, streams, depth", [({}, 1, 3), ({"parallel": 8}, 8, 3),
                                                     ({"parallel": 4, "no_drafts": True}, 4, 0)])
def test_parallel_reaches_the_engine(tmp_path, monkeypatch, options, streams, depth):
    from tensorfold.families.qwen3_5_moe.cuda import engine

    made = {}
    monkeypatch.setattr(engine, "Qwen36Engine", lambda path, **kw: made.update(kw) or "engine")
    assert qwen3_5_moe.cuda_engine(_config(tmp_path), **options) == "engine"
    assert made["streams"] == streams and made["depth"] == depth


def test_a_dflash_drafter_runs_on_the_dense_engine(tmp_path, monkeypatch):
    from tensorfold.families.qwen3_5.cuda import engine

    made = {}
    monkeypatch.setattr(engine, "Qwen27Engine", lambda path, draft, **kw: made.update(kw, draft=draft) or "engine")
    assert qwen3_5_moe.cuda_engine(_config(tmp_path), drafter="/d/ornith-dflash", parallel=2) == "engine"
    assert str(made["draft"]) == "/d/ornith-dflash" and made["streams"] == 2 and made["loader"] is not None
    assert made["max_rows"] == qwen3_5_moe.DFLASH_ROWS


def test_dflash1_is_chosen_by_architecture(tmp_path):
    import json

    pytest.importorskip("torch")
    pytest.importorskip("triton")

    from tensorfold.families.qwen3_5.cuda.dflash1 import DFlash1, drafter_class
    from tensorfold.families.qwen3_5.cuda.dflash2 import DFlash2

    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["DFlashDraftModel"]}))
    assert drafter_class(tmp_path) is DFlash1
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["DFlash2DraftModel"]}))
    assert drafter_class(tmp_path) is DFlash2
    from tensorfold.families.qwen3_5.cuda.dspark import DSpark

    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3DSparkModel"],
                                                      "speculators_model_type": "dspark"}))
    assert drafter_class(tmp_path) is DSpark
