"""Flash Next on sm_70 at four ranks: the startup estimate gives every rank the same expert share, as loaded."""

import pytest

pytest.importorskip("torch")

from tensorfold.cuda.geometry import indexed_weights  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.ranks import share  # noqa: E402

TEXT = {"text_config": {"moe_intermediate_size": 640, "num_key_value_heads": 2}}
TENSORS = {
    "model.language_model.layers.0.mlp.experts.7.gate_proj.weight": {"dtype": "U8", "shape": [640, 1280]},
    "model.language_model.layers.0.mlp.experts.7.gate_proj.weight_scale": {"dtype": "F8_E4M3", "shape": [640, 160]},
    "model.language_model.layers.0.mlp.experts.7.down_proj.weight": {"dtype": "U8", "shape": [2560, 320]},
    "model.language_model.layers.0.mlp.experts.7.down_proj.weight_scale": {"dtype": "F8_E4M3", "shape": [2560, 40]},
    "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": {"dtype": "BF16", "shape": [640, 2560]},
    "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": {"dtype": "BF16", "shape": [2560, 640]},
}


def _rank_bytes(rank: int, world: int = 4) -> dict:
    t = indexed_weights(world, False, text=TEXT, rank=rank, host_embedding=True)
    return {name: t(name, info)[0] for name, info in TENSORS.items()}


def test_four_ranks_hold_equal_expert_shares():
    ranks = [_rank_bytes(r) for r in range(4)]
    assert all(r == ranks[0] for r in ranks[1:])
    got = ranks[0]
    assert got["model.language_model.layers.0.mlp.experts.7.gate_proj.weight"] == 160 * 1280
    assert got["model.language_model.layers.0.mlp.experts.7.down_proj.weight"] == 2560 * 80
    assert got["model.language_model.layers.0.mlp.experts.7.down_proj.weight_scale"] == 2560 * 10
    # the 16-bit shared down keeps whole 64-input groups: 160 inputs stored as 192
    assert got["model.language_model.layers.0.mlp.shared_expert.down_proj.weight"] == 2560 * 192 * 2
    assert [share(640, r, 4) for r in range(4)] == [(160 * r, 160 * (r + 1)) for r in range(4)]


def test_two_ranks_are_unchanged():
    ranks = [_rank_bytes(r, 2) for r in range(2)]
    assert ranks[0] == ranks[1]
    assert ranks[0]["model.language_model.layers.0.mlp.shared_expert.down_proj.weight"] == 2560 * 320 * 2
