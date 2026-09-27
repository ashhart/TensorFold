"""A tiny random Nemotron-H (every block kind, a MoE with the folded shared expert, an MTP head) for tests."""

from __future__ import annotations

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.families.nemotron_h.cuda.weights import MTP, Attention, Block, Config, Mamba, MoE, Weights
from tensorfold.families.qwen3_5.cuda.qmm_fast import tile
from tensorfold.families.qwen3_5.cuda.weights import QLinear


def tiny_config(pattern: str = "ME*EME", heads: int = 16, kv_heads: int = 1) -> Config:
    return Config(hidden=256, vocab=512, pattern=pattern, heads=heads, kv_heads=kv_heads, head_dim=128, m_heads=4,
                  m_head_dim=64, m_groups=2, m_state=128, conv_kernel=4, experts=8, top_k=6, moe_width=128,
                  shared_width=256, scaling=2.5, norm_topk=True, eps=1e-5, eos=(0,))


def random_q(n: int, k: int, g: torch.Generator, *, scale: float = 0.004, lead: tuple = ()):
    words = torch.randint(-(2 ** 31), 2 ** 31 - 1, (*lead, n, k // 8), generator=g, device="cuda",
                          dtype=torch.int64).to(torch.int32)
    s = torch.rand((*lead, n, k // 64), generator=g, device="cuda") * scale + scale / 4
    b = -s * 7.5 + (torch.rand((*lead, n, k // 64), generator=g, device="cuda") - 0.5) * scale
    return words, s.to(torch.bfloat16), b.to(torch.bfloat16)


def tiny_weights(seed: int = 0, *, pattern: str = "ME*EME", mtp: bool = True, heads: int = 16,
                 kv_heads: int = 1) -> Weights:
    c = tiny_config(pattern, heads, kv_heads)
    g = torch.Generator(device="cuda").manual_seed(seed)
    D = c.hidden

    def randn(*shape, scale=1.0):
        return torch.randn(shape, generator=g, device="cuda") * scale

    def qlin(n, k, scale=0.004):
        return tile(QLinear(*random_q(n, k, g, scale=scale)))

    def norm(n):
        return (1.0 + randn(n, scale=0.1)).to(torch.bfloat16)

    def attn():
        return Attention(qkv=qlin(c.qkv_dim, D), o=qlin(D, c.heads * c.head_dim))

    def moe():
        e2 = c.experts + 2                    # routed, then the shared expert's two halves
        ex = grouped.make([random_q(c.moe_width, D, g, scale=0.01, lead=(e2,))],
                          random_q(D, c.moe_width, g, scale=0.01, lead=(e2,)), 64)
        return MoE(router=randn(c.experts, D, scale=0.1).to(torch.bfloat16), bias=randn(c.experts, scale=0.02),
                   experts=ex)

    blocks = []
    for kind in c.pattern:
        if kind == "M":
            m = Mamba(in_proj=qlin(c.proj_dim, D), out_proj=qlin(D, c.xd),
                      conv_w=randn(4, c.conv_dim, scale=0.4).to(torch.bfloat16).float(),
                      conv_b=randn(c.conv_dim, scale=0.1).to(torch.bfloat16).float(),
                      a=-torch.exp(torch.rand(c.m_heads, generator=g, device="cuda") * 2),
                      d=(1 + randn(c.m_heads, scale=0.1)).to(torch.bfloat16).float(),
                      dt_bias=randn(c.m_heads, scale=0.5) - 1.0, gnorm=norm(c.xd))
            blocks.append(Block("M", norm(D), mamba=m))
        elif kind == "*":
            blocks.append(Block("*", norm(D), attn=attn()))
        else:
            blocks.append(Block("E", norm(D), moe=moe()))
    ew, es, eb = random_q(c.vocab, D, g, scale=0.05)
    w = Weights(config=c, embed=QLinear(ew, es, eb), blocks=blocks, norm_f=norm(D), head=qlin(c.vocab, D, 0.02))
    if mtp:
        w.mtp = MTP(enorm=norm(D), hnorm=norm(D), eh_proj=qlin(D, 2 * D), attn_norm=norm(D), attn=attn(),
                    moe_norm=norm(D), moe=moe(), final_norm=norm(D))
    return w
