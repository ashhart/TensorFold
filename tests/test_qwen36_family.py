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


def test_only_4_bit_groups_of_64_are_read(tmp_path):
    qwen3_5_moe.check(_config(tmp_path))
    with pytest.raises(ValueError, match="groups of 64"):
        qwen3_5_moe.check(_config(tmp_path, bits=8))


@pytest.mark.parametrize("options, message", [({"tp": 2}, "one GPU"), ({"parallel": 4, "mtp_drafts": 16}, "0 to 15"),
                                              ({"drafter": "some/drafter"}, "its own MTP layer")])
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
