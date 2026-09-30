"""Full GLM-5.3 over four ranks on real weights: every rank's share of a linear or of the routed experts computes its
part of the full-width result (column splits concatenate, row splits and experts sum). Needs TF_GLM53_CKPT."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

CKPT = os.environ.get("TF_GLM53_CKPT", "")
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
              pytest.mark.skipif(not CKPT, reason="set TF_GLM53_CKPT to a GLM-5.3 EXL3 checkpoint")]

WORLD, LAYER = 4, 3


def _cfg():
    from tensorfold.families.glm_moe_dsa.config import Config
    return Config.from_dict(json.loads((Path(CKPT) / "config.json").read_text()))


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("prefix,kind", [("self_attn.q_b_proj", "cols"), ("self_attn.o_proj", "rows"),
                                         ("mlp.shared_experts.gate_proj", "cols"),
                                         ("mlp.shared_experts.down_proj", "rows")])
def test_linear_shares_compose_the_full_linear(prefix, kind):
    from tensorfold.families.glm_moe_dsa.cuda.weights import RankReader, load_linear

    name = f"model.layers.{LAYER}.{prefix}"
    full = load_linear(RankReader(CKPT, 0, 1), name)
    x = torch.randn(3, full.k, device="cuda", dtype=torch.bfloat16)
    y = full(x).float()
    parts = [load_linear(RankReader(CKPT, r, WORLD), name) for r in range(WORLD)]
    if kind == "cols":
        got = torch.cat([p(x).float() for p in parts], dim=1)
    else:
        k = full.k // WORLD
        got = sum(p(x[:, r * k:(r + 1) * k].contiguous()).float() for r, p in enumerate(parts))
    assert got.shape == y.shape
    assert _rel(got, y) < 5e-3, _rel(got, y)


def test_expert_shares_sum_to_the_full_experts():
    from tensorfold.cuda.exl3 import experts as x3
    from tensorfold.families.glm_moe_dsa.cuda.weights import RankReader, load_experts

    cfg, ids = _cfg(), list(range(32))
    full = load_experts(RankReader(CKPT, 0, 1), cfg, LAYER, ids=ids)
    rows, slots = 3, 8
    x = torch.randn(rows, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    pick = torch.randint(0, len(ids), (rows, slots), device="cuda", dtype=torch.int32)
    wts = torch.rand(rows, slots, device="cuda", dtype=torch.float32)

    def run(ex):
        s = x3.Scratch(ex, rows, slots)
        return x3.routed(x, pick, wts, ex, s, None, rows).clone()

    y = run(full)
    got = sum(run(load_experts(RankReader(CKPT, r, WORLD), cfg, LAYER, ids=ids)) for r in range(WORLD))
    assert _rel(got, y) < 5e-3, _rel(got, y)
