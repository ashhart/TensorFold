"""Qwen3.6 MoE on CUDA, host side: an 8-bit lm_head keeps its format in the MTP draft head's vocab subset (tiled
for the lane matmul at its own width), and a shared expert stored in a different format from the routed experts is
refused by name."""

from types import SimpleNamespace

import numpy as np
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen3_5.cuda.qmm_fast import untile  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import QLinear  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import mtp, weights  # noqa: E402


def _affine(n: int, k: int, bits: int, gs: int = 64) -> QLinear:
    g = torch.Generator().manual_seed(n * 31 + k + bits)
    words = torch.randint(-2**31, 2**31 - 1, (n, k * bits // 32), dtype=torch.int32, generator=g)
    scales = torch.rand((n, k // gs), generator=g).to(torch.bfloat16)
    biases = torch.rand((n, k // gs), generator=g).to(torch.bfloat16)
    return QLinear(words, scales, biases, gs=gs, bits=bits)


def test_an_8bit_head_keeps_its_bits_in_the_draft_vocab_subset():
    full = _affine(256, 128, bits=8)
    w = SimpleNamespace(head=full, norm=torch.zeros(1))
    ids = np.array([3, 17, 200, 255])
    head = mtp.Head(w, m=None, ids=ids).head
    assert (head.layout, head.bits, head.gs, head.n) == ("tiled", 8, 64, len(ids))
    back = untile(head)
    assert torch.equal(back.weight, full.weight[ids]) and torch.equal(back.scales, full.scales[ids])


def _get(experts: int, shared_bits: int):
    t = {}
    for name, bits, n in (("gate", 4, experts), ("shared_expert_gate", 4, 1)):
        q = _affine(n, 128, bits)
        t.update({f"m.{name}.weight": q.weight, f"m.{name}.scales": q.scales, f"m.{name}.biases": q.biases})
    for proj in ("gate_proj", "up_proj", "down_proj"):
        mine = [_affine(64, 128, 4) for _ in range(experts)]
        shared = _affine(64, 128, shared_bits)
        for s in ("weight", "scales", "biases"):
            t[f"m.switch_mlp.{proj}.{s}"] = torch.stack([getattr(q, s) for q in mine])
            t[f"m.shared_expert.{proj}.{s}"] = getattr(shared, s)
    return t.__getitem__


def test_a_shared_expert_in_another_format_is_refused_by_name():
    with pytest.raises(ValueError, match="shared_expert.gate_proj is stored in a different format"):
        weights.routed("m.", _get(4, shared_bits=8), top_k=2)
