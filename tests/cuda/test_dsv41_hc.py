"""DeepSeek-V4.1 fused hyper-connection kernels against the reference's PyTorch math."""

import json
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.deepseek_v41 import reference as R
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import hc

CFG = Config.from_dict(json.loads((Path(__file__).parents[1] / "fixtures" / "deepseek_v41" / "config.json").read_text()))


class _Ck:
    def __init__(self, t):
        self.t = t

    def get(self, name, dtype=None):
        x = self.t[name.rsplit(".", 1)[-1]]
        return x if dtype is None else x.to(dtype)


@pytest.mark.parametrize("rows", [1, 3, 17])
def test_pre_and_post_match_the_reference(rows):
    g = torch.Generator(device="cuda").manual_seed(rows)
    D = CFG.hidden_size
    X = (torch.randn((rows, 4, D), generator=g, device="cuda") * 3).to(torch.bfloat16)
    fn = torch.randn((24, 4 * D), generator=g, device="cuda") * 0.01
    base = torch.randn((24,), generator=g, device="cuda")
    scale = torch.rand((3,), generator=g, device="cuda") + 0.5
    norm_w = (torch.rand((D,), generator=g, device="cuda") + 0.5).to(torch.bfloat16)
    pre_in = torch.rand((rows, 4), generator=g, device="cuda")
    ref = R.HCMix(_Ck({"hc_fn": fn, "hc_base": base, "hc_scale": scale}), "layers.0.hc")
    post_r, comb_r, x_r, pre_r = ref(X, pre_in, norm_w, CFG)
    buf = hc.HCBuffers(32, D)
    post_k, comb_k, x_k, pre_k = hc.pre(X, fn, base, scale, pre_in, norm_w, buf, CFG.rms_norm_eps, CFG.hc_eps,
                                        CFG.hc_sinkhorn_iters)
    # the reference matmul runs in TF32 in NVIDIA's container (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1)
    torch.testing.assert_close(pre_k, pre_r, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(post_k, post_r, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(comb_k, comb_r, rtol=2e-3, atol=2e-3)
    assert (x_k.float() - x_r.float()).abs().max() <= 0.02 * x_r.float().abs().max()
    b = torch.randn((rows, D), generator=g, device="cuda").to(torch.bfloat16)
    y_r = R.hc_post(b, X, post_r, comb_r)
    y_k = hc.post(b, X, post_r, comb_r)
    assert (y_k.float() - y_r.float()).abs().max() <= 0.01 * y_r.float().abs().max() + 1e-3


def test_a_row_alone_equals_the_row_in_a_window():
    g = torch.Generator(device="cuda").manual_seed(7)
    D = CFG.hidden_size
    X = torch.randn((9, 4, D), generator=g, device="cuda").to(torch.bfloat16)
    fn = torch.randn((24, 4 * D), generator=g, device="cuda") * 0.01
    base, scale = torch.randn((24,), generator=g, device="cuda"), torch.ones((3,), device="cuda")
    norm_w = torch.ones((D,), dtype=torch.bfloat16, device="cuda")
    pre_in = torch.rand((9, 4), generator=g, device="cuda")
    buf = hc.HCBuffers(16, D)
    all_rows = hc.pre(X, fn, base, scale, pre_in, norm_w, buf, CFG.rms_norm_eps, CFG.hc_eps, CFG.hc_sinkhorn_iters)
    one = hc.pre(X[4:5].contiguous(), fn, base, scale, pre_in[4:5], norm_w, buf, CFG.rms_norm_eps, CFG.hc_eps,
                 CFG.hc_sinkhorn_iters)
    for a, b in zip(all_rows, one):
        assert torch.equal(a[4:5], b)
