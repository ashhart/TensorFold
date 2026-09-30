"""DeepSeek-V4.1 RMSNorm, RoPE and MQA attention kernels against the reference's PyTorch math."""

import json
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.deepseek_v41 import reference as R
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import kernels as K

CFG = Config.from_dict(json.loads((Path(__file__).parents[1] / "fixtures" / "deepseek_v41" / "config.json").read_text()))


def test_rmsnorm():
    x = torch.randn((5, 1280), device="cuda") * 4
    w = (torch.rand((1280,), device="cuda") + 0.5).to(torch.bfloat16)
    ref = R.rms(x, w, CFG.rms_norm_eps).to(torch.bfloat16)
    assert (K.rmsnorm(x, w, CFG.rms_norm_eps).float() - ref.float()).abs().max() <= 0.02


@pytest.mark.parametrize("ratio", [0, 2])
def test_rope_and_inverse(ratio):
    freqs = R.inv_freq(CFG, ratio, "cuda")
    cos, sin = K.rope_tables(freqs, 4096)
    x = torch.randn((7, 32, 512), device="cuda")
    pos = torch.tensor([0, 1, 5, 127, 128, 900, 4095], device="cuda")
    ref = R.rope(x, pos, freqs)
    got = K.rope(x, pos, cos, sin, out_dtype=torch.float32)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    back = K.rope(got, pos, cos, sin, inverse=True, out_dtype=torch.float32)
    torch.testing.assert_close(back, x, rtol=1e-4, atol=1e-4)


def _ref_attention(q, comp, idx, ring, pos, sink, W):
    outs = []
    for r in range(q.shape[0]):
        p = int(pos[r])
        keys = [ring[[s % ring.shape[0] for s in range(max(0, p - W + 1), p + 1)]].float()]
        if idx is not None:
            sel = [int(i) for i in idx[r] if int(i) >= 0]
            keys.insert(0, comp[sel].float())
        k = torch.cat(keys)
        s = torch.einsum("hd,sd->hs", q[r].float(), k) * 512 ** -0.5
        full = torch.cat([s, sink[:, None]], dim=1)
        outs.append(torch.softmax(full, -1)[:, :-1] @ k)
    return torch.stack(outs)


@pytest.mark.parametrize("with_comp,positions", [(False, [0, 5, 200]), (True, [0, 1, 63, 300, 511, 900])])
def test_mqa_matches_the_masked_softmax(with_comp, positions):
    g = torch.Generator(device="cuda").manual_seed(len(positions))
    W, H, n_sel = 128, 32, 512
    ring = torch.randn((4096, 512), generator=g, device="cuda").to(torch.bfloat16)
    comp = torch.randn((1025, 512), generator=g, device="cuda").to(torch.bfloat16)
    pos = torch.tensor(positions, device="cuda")
    idx = None
    if with_comp:                                  # a random subset of the visible entries, ascending, -1 padded
        rows = []
        for p in positions:
            vis = torch.randperm(p + 1, generator=torch.Generator().manual_seed(p))[:n_sel].sort().values
            rows.append(torch.cat([vis, torch.full((n_sel - len(vis),), -1)]))
        idx = torch.stack(rows).int().cuda()
    q = (torch.randn((len(positions), H, 512), generator=g, device="cuda") * 0.2).to(torch.bfloat16)
    sink = torch.randn((H,), generator=g, device="cuda")
    buf = K.AttnBuffers(8, H, 512, n_sel + W)
    got = K.mqa(q, comp if with_comp else None, idx, ring, pos, sink, W, buf, 512 ** -0.5)
    ref = _ref_attention(q, comp, idx, ring, pos, sink, W)
    assert (got - ref).abs().max() <= 0.03 * ref.abs().max()


def test_index_select_takes_all_visible_then_top_k():
    g = torch.Generator(device="cuda").manual_seed(4)
    keys = torch.randn((2048, 128), generator=g, device="cuda").to(torch.bfloat16)
    iq = torch.randn((3, 32, 128), generator=g, device="cuda").to(torch.bfloat16)
    wts = torch.randn((3, 32), generator=g, device="cuda")
    pos = torch.tensor([10, 400, 1999], device="cuda")
    idx = K.index_select(iq, wts, keys, pos, 1, 512)
    assert idx[0, :11].tolist() == list(range(11)) and (idx[0, 11:] == -1).all()
    assert idx[1, :401].tolist() == list(range(401)) and (idx[1, 401:] == -1).all()
    full = (wts[2, :, None] * torch.relu(iq[2].float() @ keys[:2000].float().T)).sum(0)
    want = torch.topk(full, 512).indices.sort().values
    overlap = len(set(want.tolist()) & set(idx[2].tolist()))
    assert (idx[2] >= 0).all() and overlap >= 505          # tensor-core vs fp32 rounding may swap near-ties
    for r in range(3):                                      # row-invariant
        assert torch.equal(K.index_select(iq[r:r + 1], wts[r:r + 1], keys, pos[r:r + 1], 1, 512)[0], idx[r])


def test_route_matches_topk_on_sqrt_softplus():
    g = torch.Generator(device="cuda").manual_seed(11)
    logits = torch.randn((9, 384), generator=g, device="cuda") * 3
    bias = torch.randn((384,), generator=g, device="cuda") * 0.1
    pick, wts = K.route(logits, bias, 6, 1.5)
    sc = torch.sqrt(torch.nn.functional.softplus(logits))
    ref = torch.topk(sc + bias, 6, dim=-1).indices
    assert torch.equal(pick.long().sort(-1).values, ref.sort(-1).values)
    w = sc.gather(1, pick.long())
    torch.testing.assert_close(wts, w / w.sum(-1, keepdim=True) * 1.5, rtol=1e-5, atol=1e-6)


def test_router_logits_are_row_invariant_and_accurate():
    g = torch.Generator(device="cuda").manual_seed(5)
    x = (torch.randn((7, 5120), generator=g, device="cuda")).to(torch.bfloat16)
    w = (torch.randn((384, 5120), generator=g, device="cuda") * 0.02).half()
    got = K.router_logits(x, w)
    ref = x.double() @ w.double().T
    assert (got.double() - ref).abs().max() <= 1e-3 * ref.abs().max()
    for r in range(7):
        assert torch.equal(K.router_logits(x[r:r + 1], w)[0], got[r])


def test_engram_gate_matches_the_reference_and_is_row_invariant():
    g = torch.Generator(device="cuda").manual_seed(9)
    D = CFG.hidden_size
    X = torch.randn((5, 4, D), generator=g, device="cuda").to(torch.bfloat16)
    kv = torch.randn((5, 5 * D), generator=g, device="cuda").to(torch.bfloat16)
    qw, kw = torch.rand((4, D), generator=g, device="cuda"), torch.rand((4, D), generator=g, device="cuda")
    got = K.engram_gate(X, kv, qw, kw, CFG.rms_norm_eps)
    h, key, val = X.float(), kv[:, :4 * D].view(5, 4, D).float(), kv[:, 4 * D:].float()
    dot = (h * qw * kw * key).sum(-1) * torch.rsqrt(h.pow(2).mean(-1) + CFG.rms_norm_eps) \
        * torch.rsqrt(key.pow(2).mean(-1) + CFG.rms_norm_eps) / D ** 0.5
    gate = torch.sigmoid(torch.sign(dot) * torch.sqrt(dot.abs().clamp(min=1e-6)))
    ref = (h + gate[:, :, None] * val[:, None, :]).to(torch.bfloat16)
    assert (got.float() - ref.float()).abs().max() <= 0.02 * ref.float().abs().max()
    for r in range(5):
        assert torch.equal(K.engram_gate(X[r:r + 1], kv[r:r + 1], qw, kw, CFG.rms_norm_eps)[0], got[r])


def test_candidate_blocks_pin_the_newest_and_mask_the_rest():
    scores = torch.full((1, 64), float("-inf"), device="cuda")
    scores[0, :50] = torch.arange(50, device="cuda", dtype=torch.float32) * 0.0
    scores[0, 3] = 5.0                                  # block 0 has the best entry
    scores[0, 20] = 4.0                                 # block 2 next
    pos = torch.tensor([49], device="cuda")             # 50 visible entries (ratio 1): newest block is 6
    blocks = K.candidate_blocks(scores, pos, 1, 8, 3)
    assert sorted(blocks[0].tolist()) == [0, 2, 6]
    masked = K.mask_to_blocks(scores, blocks, 8)
    kept = torch.isfinite(masked[0]).nonzero().flatten().tolist()
    assert kept == list(range(8)) + list(range(16, 24)) + list(range(48, 50))


def test_mqa_merge_applies_the_inverse_rope():
    g = torch.Generator(device="cuda").manual_seed(12)
    W, H = 128, 32
    ring = torch.randn((4096, 512), generator=g, device="cuda").to(torch.bfloat16)
    pos = torch.tensor([3, 77, 1000], device="cuda")
    q = (torch.randn((3, H, 512), generator=g, device="cuda") * 0.2).to(torch.bfloat16)
    sink = torch.randn((H,), generator=g, device="cuda")
    buf = K.AttnBuffers(8, H, 512, W)
    cos, sin = K.rope_tables(R.inv_freq(CFG, 0, "cuda"), 4096)
    plain = K.mqa(q, None, None, ring, pos, sink, W, buf, 512 ** -0.5)
    fused = K.mqa(q, None, None, ring, pos, sink, W, buf, 512 ** -0.5, cos, sin)
    want = K.rope(plain, pos, cos, sin, inverse=True, out_dtype=torch.bfloat16)
    assert fused.dtype == torch.bfloat16
    assert (fused.float() - want.float()).abs().max() <= 0.02 * want.float().abs().max()
