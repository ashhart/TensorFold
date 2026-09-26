"""CUDA tree attention: same path gives the same bits alone or in a window."""

import math

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda.attention import attention


def _inputs(w: int, p: int, *, h: int = 24, hk: int = 4, d: int = 256):
    gen = torch.Generator(device="cuda").manual_seed(910 + w + p)
    shape = (w, h, d)
    kv_shape = (w, hk, d)
    cache_shape = (p, hk, d)
    q = torch.randn(shape, generator=gen, device="cuda").bfloat16()
    kn = torch.randn(kv_shape, generator=gen, device="cuda").bfloat16()
    vn = torch.randn(kv_shape, generator=gen, device="cuda").bfloat16()
    kc = torch.randn(cache_shape, generator=gen, device="cuda").bfloat16()
    vc = torch.randn(cache_shape, generator=gen, device="cuda").bfloat16()
    return q, kn, vn, kc, vc


def _path(parents: list[int], node: int) -> list[int]:
    rows = []
    while node >= 0:
        rows.append(node)
        node = parents[node]
    return rows[::-1]


def _serial(inputs, parents, node):
    q, kn, vn, kc, vc = inputs
    rows = _path(parents, node)
    keys = torch.cat((kc, kn[rows]), 0)
    values = torch.cat((vc, vn[rows]), 0)
    root = torch.tensor([-1], dtype=torch.int32, device="cuda")
    return attention(q[node:node + 1], kn[node:node + 1], vn[node:node + 1],
                     keys[:-1].contiguous(), values[:-1].contiguous(), root,
                     scale=1 / math.sqrt(q.shape[-1]))[0]


@pytest.mark.parametrize("w,p", [(1, 0), (9, 13), (16, 511), (32, 512), (32, 1003), (128, 513)])
def test_branch_nodes_match_serial_bits(w, p):
    inputs = _inputs(w, p)
    parents = [-1] + [(i - 1) // 2 for i in range(1, w)]
    p_dev = torch.tensor(parents, dtype=torch.int32, device="cuda")
    out = attention(*inputs, p_dev, scale=1 / 16)
    for node in range(w):
        assert torch.equal(out[node], _serial(inputs, parents, node)), f"node {node} differs"


def test_128_chain_matches_serial_bits_at_chunk_boundaries():
    w, p = 128, 499
    inputs = _inputs(w, p, h=12, hk=2, d=128)
    parents = [-1] + list(range(w - 1))
    out = attention(*inputs, torch.tensor(parents, dtype=torch.int32, device="cuda"),
                    scale=1 / math.sqrt(128))
    for node in (0, 1, 12, 13, 31, 63, 127):
        assert torch.equal(out[node], _serial(inputs, parents, node)), f"node {node} differs"


def test_long_cache_matches_serial_bits():
    w, p = 128, 20501
    inputs = _inputs(w, p)
    parents = [-1] + [(i - 1) // 2 for i in range(1, w)]
    out = attention(*inputs, torch.tensor(parents, dtype=torch.int32, device="cuda"), scale=1 / 16)
    for node in (0, 3, 8, 63, 127):
        assert torch.equal(out[node], _serial(inputs, parents, node)), f"node {node} differs"


def test_attention_matches_torch_reference():
    w, p = 7, 73
    inputs = _inputs(w, p)
    q, kn, vn, kc, vc = inputs
    parents = [-1, 0, 0, 1, 1, 2, 3]
    out = attention(*inputs, torch.tensor(parents, dtype=torch.int32, device="cuda"), scale=1 / 16)
    for node in range(w):
        rows = _path(parents, node)
        keys = torch.cat((kc, kn[rows]), 0).float().repeat_interleave(6, 1)
        values = torch.cat((vc, vn[rows]), 0).float().repeat_interleave(6, 1)
        scores = torch.einsum("hd,thd->ht", q[node].float(), keys) / 16
        ref = torch.einsum("ht,thd->hd", scores.softmax(-1), values).bfloat16()
        assert (out[node].float() - ref.float()).abs().max() < 0.035
