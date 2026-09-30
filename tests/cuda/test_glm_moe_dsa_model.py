"""Full GLM-5.3 milestone M2 on real weights (TF_GLM53_CKPT), one GPU:

1. four ranks (threads) give the same hidden states as one rank of the same code over layers 0-3 (sharding + rank-
   order reductions are right);
2. the absorbed-latent attention equals transformers' reference GlmMoeDsaAttention (expanded K/V) given the same
   dequantized weights (RoPE layout, absorb/expand and scaling are right).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

CKPT = os.environ.get("TF_GLM53_CKPT", "")
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
              pytest.mark.skipif(not CKPT, reason="set TF_GLM53_CKPT to a GLM-5.3 EXL3 checkpoint")]

from threadcomm import run_ranks  # noqa: E402

LAYERS = (0, 1, 2, 3)


def _cfg():
    from tensorfold.families.glm_moe_dsa.config import Config
    return Config.from_dict(json.loads((Path(CKPT) / "config.json").read_text()))


def _model(rank, world, comm, layers=LAYERS):
    from tensorfold.families.glm_moe_dsa.cuda.model import RankModel
    from tensorfold.families.glm_moe_dsa.cuda.weights import RankReader, load_layer

    cfg = _cfg()
    r = RankReader(CKPT, rank, world)
    return RankModel(cfg, rank, world, comm, embed=r.get("model.embed_tokens.weight", "cuda"),
                     final_norm=r.get("model.norm.weight", "cuda"), lm_head=r.get("lm_head.weight", "cuda"),
                     layers=[load_layer(r, cfg, i) for i in layers])


class _Solo:
    world, rank = 1, 0

    def all_gather(self, send, recv):
        recv.view(-1).copy_(send.reshape(-1))


def _run(model, T=24):
    g = torch.Generator().manual_seed(5)
    tokens = torch.randint(0, 150000, (T,), generator=g).cuda()
    pos = torch.arange(T, device="cuda")
    caches = [torch.zeros((T, model.cfg.latent_width), dtype=torch.bfloat16, device="cuda") for _ in model.layers]
    return model.forward(tokens, pos, caches).float()


def test_four_ranks_equal_one_rank():
    solo = _run(_model(0, 1, _Solo()))
    torch.cuda.empty_cache()
    outs = run_ranks(lambda r, c: _run(_model(r, 4, c)), 4)
    for r in range(4):
        assert torch.equal(outs[r], outs[0]), "ranks disagree"
    rel = float((outs[0] - solo).norm() / solo.norm())
    assert rel < 2e-2, rel          # different kernel splits and summation order; same model


def _dense(lin):
    """W [out, in] of an Exl3Linear, by applying it to the identity in pieces."""
    k = lin.k
    eye = torch.eye(k, device="cuda", dtype=torch.bfloat16)
    cols = [lin(eye[i:i + 128].contiguous(), out_dtype=torch.float32) for i in range(0, k, 128)]
    return torch.cat(cols).t().contiguous()        # [out, in]


def test_absorbed_attention_matches_transformers_reference():
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as ref
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    m = _model(0, 1, _Solo(), layers=(0,))
    L, c = m.layers[0], m.cfg
    raw = json.loads((Path(CKPT) / "config.json").read_text())
    raw.pop("quantization_config", None)
    hc = GlmMoeDsaConfig(**raw)
    hc._attn_implementation = "eager"
    attn = ref.GlmMoeDsaAttention(hc, 0).cuda().float()
    attn.indexer = None                                # T <= index_topk: every key is selected
    with torch.no_grad():
        attn.q_a_proj.weight.copy_(_dense(L.q_a))
        attn.q_a_layernorm.weight.copy_(L.q_a_norm.float())
        attn.q_b_proj.weight.copy_(_dense(L.q_b))
        attn.kv_a_proj_with_mqa.weight.copy_(_dense(L.kv_a)[:c.latent_width])
        attn.kv_a_layernorm.weight.copy_(L.kv_a_norm.float())
        attn.kv_b_proj.weight.copy_(L.kv_b.float())
        attn.o_proj.weight.copy_(_dense(L.o_proj))
    T = 16
    h = torch.randn(1, T, c.hidden_size, device="cuda") * 0.5
    pos = torch.arange(T, device="cuda")
    rot = ref.GlmMoeDsaRotaryEmbedding(hc).cuda()
    cos, sin = rot(h, pos[None])
    topk = torch.arange(T, device="cuda", dtype=torch.int32)[None, None, :].expand(1, T, T)
    with torch.no_grad():
        want = attn(h, (cos, sin), None, position_ids=pos[None], prev_topk_indices=topk)[0][0]
        cache = torch.zeros((T, c.latent_width), dtype=torch.bfloat16, device="cuda")
        got = m.attention(L, h[0].bfloat16(), pos, cache)[0]
    rel = float((got - want).norm() / want.norm())
    assert rel < 3e-2, rel          # bf16 activations vs the fp32 reference


def test_indexer_topk_matches_transformers_reference():
    """M5: past index_topk, our token-level indexer selects the keys transformers' GlmMoeDsaIndexer selects."""
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as ref
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    m = _model(0, 1, _Solo(), layers=(0,))
    L, c = m.layers[0], m.cfg
    raw = json.loads((Path(CKPT) / "config.json").read_text())
    raw.pop("quantization_config", None)
    hc = GlmMoeDsaConfig(**raw)
    idx = ref.GlmMoeDsaIndexer(hc, 0).cuda().float()
    ix = L.indexer
    with torch.no_grad():
        idx.wq_b.weight.copy_(_dense(ix["wq_b"]))
        idx.wk.weight.copy_(ix["wk"].float())
        idx.k_norm.weight.copy_(ix["k_norm"][0].float())
        idx.k_norm.bias.copy_(ix["k_norm"][1].float())
        idx.weights_proj.weight.copy_(ix["weights_proj"].float())
    T = c.index_topk + 952
    g = torch.Generator(device="cpu").manual_seed(3)
    h = (torch.randn(T, c.hidden_size, generator=g) * 0.5).cuda()
    qa = (torch.randn(T, c.q_lora_rank, generator=g) * 0.5).cuda()
    pos = torch.arange(T, device="cuda")
    rot = ref.GlmMoeDsaRotaryEmbedding(hc).cuda()
    cos, sin = rot(h[None], pos[None])
    rows = torch.arange(T - 16, T, device="cuda")                  # the last rows see more keys than index_topk
    with torch.no_grad():
        want = idx(h.bfloat16().float()[None], qa.bfloat16().float()[None], (cos, sin), None, pos[None])[0]
        icache = torch.zeros((T, c.index_head_dim), dtype=torch.bfloat16, device="cuda")
        m.indexer(L, h[:T - 16].bfloat16(), qa[:T - 16].bfloat16(), pos[:T - 16], icache)     # write earlier keys
        got = m.indexer(L, h[T - 16:].bfloat16(), qa[T - 16:].bfloat16(), rows, icache)
    for r in range(16):
        a = set(got[r].tolist())
        b = set(want[T - 16 + r].tolist())
        assert len(a & b) / len(b) > 0.97, (r, len(a & b) / len(b))    # bf16 keys vs the fp32 reference
