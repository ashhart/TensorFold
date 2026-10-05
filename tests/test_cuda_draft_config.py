"""Drafter compatibility is checked from metadata before device initialization."""

import json
import sys

import pytest
from cuda_27b_headers import _write

from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine


def checkpoints(tmp_path, *, taps=(1, 5, 9, 13, 17, 21, 25, 29), changes=None, shape=(2560, 20480)):
    target = _write(tmp_path / "target", {"text_config": {
        "hidden_size": 2560, "num_hidden_layers": 32, "vocab_size": 248320}}, [])
    cfg = {"architectures": ["DFlashDraftModel"], "hidden_size": 2560, "num_target_layers": 32,
           "vocab_size": 248320, "dflash_config": {"target_layer_ids": list(taps), "mask_token_id": 248077}}
    cfg.update(changes or {})
    draft = _write(tmp_path / "draft", cfg, [("fc.weight", "BF16", list(shape))])
    return target, draft


def test_eight_taps_and_declared_order_are_accepted(tmp_path):
    from tensorfold.families.qwen3_5.cuda.draft_config import validate

    target, draft = checkpoints(tmp_path, taps=(29, 25, 21, 17, 13, 9, 5, 1))
    assert validate(target, draft) == (29, 25, 21, 17, 13, 9, 5, 1)


@pytest.mark.parametrize("taps", [(), (-1,), (32,), (1, 1), (True,), (1.5,), ("1",)])
def test_invalid_taps_fail_before_torch_import(tmp_path, monkeypatch, taps):
    target, draft = checkpoints(tmp_path, taps=taps)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="target_layer_ids"):
        Qwen27Engine(target, draft)


@pytest.mark.parametrize("changes,shape,match", [
    ({"hidden_size": 5120}, (5120, 20480), "hidden_size"),
    ({"num_target_layers": 64}, (2560, 20480), "num_target_layers"),
    ({"vocab_size": 100}, (2560, 20480), "vocab_size"),
    ({}, (2560, 12800), "fc.weight"),
    ({"architectures": ["UnknownDraftModel"]}, (2560, 20480), "architecture"),
    ({"dflash_config": {"target_layer_ids": [1], "mask_token_id": 248320}}, (2560, 2560), "mask_token_id"),
    ({"dflash_config": {}}, (2560, 12800), "target_layer_ids"),
])
def test_incompatible_pairing_fails_before_torch_import(tmp_path, monkeypatch, changes, shape, match):
    target, draft = checkpoints(tmp_path, changes=changes, shape=shape)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match=match):
        Qwen27Engine(target, draft)


def test_legacy_dflash2_keeps_its_trained_default_layers(tmp_path):
    from tensorfold.families.qwen3_5.cuda.draft_config import validate

    target, draft = checkpoints(tmp_path, changes={"architectures": ["DFlash2DraftModel"],
                                                 "num_target_layers": 64, "dflash_config": {}},
                                shape=(2560, 12800))
    cfg = json.loads((target / "config.json").read_text())
    cfg["text_config"]["num_hidden_layers"] = 64
    (target / "config.json").write_text(json.dumps(cfg))
    assert validate(target, draft) == (5, 19, 33, 47, 61)


@pytest.mark.parametrize("declared", [True, False])
def test_full_attention_draft_cache_is_not_estimated_as_sliding(declared):
    from tensorfold.cuda.geometry import draft_geometry

    cfg = {"architectures": ["DFlashDraftModel"], "hidden_size": 2560, "intermediate_size": 9216,
           "num_hidden_layers": 6, "num_key_value_heads": 8, "head_dim": 128, "sliding_window": 4096}
    if declared:
        cfg["layer_types"] = ["sliding_attention"] * 5 + ["full_attention"]
    plan = draft_geometry(cfg, 1, 12, bounded=True, streams=2, kept=3)
    # Seven live/retained copies, K and V, 8 heads, 128 columns, two bytes, 4096 more rows.
    assert plan.bytes_at(8192) - plan.bytes_at(4096) == 117440512 * (1 if declared else 6)
