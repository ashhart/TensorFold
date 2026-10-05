"""sm_70 routed NVFP4 experts: a pair's bits never depend on the other pairs, and match an fp32 reference."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:
    pytest.skip("Volta only", allow_module_level=True)

from tensorfold.cuda.kernels.qmmf_volta import VoltaExperts  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _nvfp4(e, n, k, g):
    codes = torch.randint(0, 256, (e, n, k // 2), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
    sc = torch.randint(0x28, 0x40, (e, n, k // 16), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
    glob = torch.rand((e,), generator=g, device="cuda") * 0.01 + 0.002
    lo, hi = (codes & 0xF).long(), (codes >> 4).long()
    vals = torch.stack([E2M1.cuda()[lo], E2M1.cuda()[hi]], -1).view(e, n, k)
    dense = vals * sc.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1) * glob[:, None, None]
    return (codes, sc, glob), dense


def _experts(e=9, ni=512, d=2048, seed=3):
    g = torch.Generator(device="cuda").manual_seed(seed)
    (gate, gd), (up, ud), (down, dd) = _nvfp4(e, ni, d, g), _nvfp4(e, ni, d, g), _nvfp4(e, d, ni, g)
    return VoltaExperts.make(gate, up, down), (gd, ud, dd)


def _run(ex, x, picks):
    rows, slots = picks.shape
    act = torch.empty((rows * slots, ex.width), dtype=torch.bfloat16, device="cuda")
    y = torch.empty((rows * slots, ex.dims), dtype=torch.float32, device="cuda")
    ex.run(x, picks, act, y)
    return act, y


@pytest.mark.parametrize("ni", [512, 160, 96])                 # 160, 96: a 32-input half group ends the down's K
@pytest.mark.parametrize("rows", [1, 3, 16, 33])
def test_pairs_match_reference(rows, ni):
    ex, (gd, ud, dd) = _experts(ni=ni)
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn((rows, ex.dims), generator=g, device="cuda").to(torch.bfloat16)
    picks = torch.randint(0, ex.count, (rows, 4), generator=g, device="cuda", dtype=torch.int32)
    act, y = _run(ex, x, picks)
    for r in range(rows):
        for s in range(4):
            e = int(picks[r, s])
            gt, u = x[r].float() @ gd[e].T, x[r].float() @ ud[e].T
            ref = (gt / (1 + torch.exp(-gt)) * u)
            p = r * 4 + s
            assert ((act[p].float() - ref).abs().max() <= ref.abs().max() * 2 ** -7 + 1e-6), (r, s)
            yref = act[p].float() @ dd[e].T
            assert ((y[p] - yref).abs().max() <= yref.abs().max() * 2 ** -12 + 1e-6), (r, s)


@pytest.mark.parametrize("ni", [512, 160])
def test_a_pair_never_depends_on_the_others(ni):
    ex, _ = _experts(ni=ni, seed=5)
    g = torch.Generator(device="cuda").manual_seed(9)
    rows, slots = 40, 5
    x = torch.randn((rows, ex.dims), generator=g, device="cuda").to(torch.bfloat16)
    picks = torch.randint(0, ex.count, (rows, slots), generator=g, device="cuda", dtype=torch.int32)
    picks[:, -1] = ex.count - 1                                     # every row's shared expert
    act, y = _run(ex, x, picks)
    for r in (0, 7, 39):
        a1, y1 = _run(ex, x[r:r + 1], picks[r:r + 1])
        assert torch.equal(a1, act[r * slots:(r + 1) * slots]) and torch.equal(y1, y[r * slots:(r + 1) * slots])
    perm = torch.randperm(rows, generator=torch.Generator().manual_seed(1)).cuda()
    ap, yp = _run(ex, x[perm], picks[perm])
    idx = (perm[:, None] * slots + torch.arange(slots, device="cuda")).reshape(-1)
    assert torch.equal(ap, act[idx]) and torch.equal(yp, y[idx])
    for m in (1, 2, 8, 9, 17):
        am, ym = _run(ex, x[:m], picks[:m])
        assert torch.equal(am, act[:m * slots]) and torch.equal(ym, y[:m * slots]), m


def test_a_half_group_tail_stores_no_padding():
    """160-wide experts hold 160 inputs' codes and scales a down column, not 192."""

    ex, _ = _experts(e=3, ni=160, d=256)
    assert ex.down.numel() * 4 == 3 * 256 * 160 // 2
    assert ex.down_s.numel() * 4 == 3 * 256 * 160 // 16
    assert ex.up.numel() * 4 == 3 * 2 * 160 * 256 // 2
