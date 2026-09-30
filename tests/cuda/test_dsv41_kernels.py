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


def _ref_attention(q, comp, ratio, swa, pos, sink, W):
    outs = []
    for r in range(q.shape[0]):
        p = int(pos[r])
        keys = [swa[max(0, p - W + 1):p + 1].float()]
        if ratio:
            keys.insert(0, comp[:(p + 1) // ratio].float())
        k = torch.cat(keys)
        s = torch.einsum("hd,sd->hs", q[r].float(), k) * 512 ** -0.5
        full = torch.cat([s, sink[:, None]], dim=1)
        outs.append(torch.softmax(full, -1)[:, :-1] @ k)
    return torch.stack(outs)


@pytest.mark.parametrize("ratio,positions", [(0, [0, 5, 200]), (2, [0, 1, 2, 63, 300, 511]), (1, [0, 7, 640])])
def test_mqa_matches_the_masked_softmax(ratio, positions):
    g = torch.Generator(device="cuda").manual_seed(ratio)
    cap, W, H = 1024, 128, 32
    swa = torch.randn((cap, 512), generator=g, device="cuda").to(torch.bfloat16)
    comp = torch.randn((cap // max(ratio, 1) + 1, 512), generator=g, device="cuda").to(torch.bfloat16).float()
    pos = torch.tensor(positions, device="cuda")
    q = (torch.randn((len(positions), H, 512), generator=g, device="cuda") * 0.2).to(torch.bfloat16)
    sink = torch.randn((H,), generator=g, device="cuda")
    buf = K.AttnBuffers(8, H, 512, comp.shape[0] + W)
    n_buf = comp.shape[0] if ratio else 0
    got = K.mqa(q, comp if ratio else None, n_buf, ratio, swa, pos, sink, W, buf, 512 ** -0.5)
    ref = _ref_attention(q, comp, ratio, swa, pos, sink, W)
    assert (got - ref).abs().max() <= 0.03 * ref.abs().max()


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
