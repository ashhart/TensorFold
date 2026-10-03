"""Kolibri 1's sliding attention: the masked softmax's values, and a row's bits whatever rows share the call."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def reference(q, kc, vc, positions, window, scale):
    out = []
    g = q.shape[1] // kc.shape[1]
    for r, p in enumerate(positions.tolist()):
        lo = max(0, p - window + 1)
        k = kc[lo:p + 1].float().repeat_interleave(g, 1)
        v = vc[lo:p + 1].float().repeat_interleave(g, 1)
        s = torch.einsum("hd,shd->hs", q[r].float(), k) * scale
        out.append(torch.einsum("hs,shd->hd", s.softmax(-1), v))
    return torch.stack(out)


@pytest.mark.parametrize("window", [513, 64, 1])
def test_sliding_rows_match_the_masked_softmax(window):
    from tensorfold.families.kolibri1.cuda.attention import sliding

    g = torch.Generator().manual_seed(0)
    n, h, hk, d = 1500, 48, 4, 128
    kc = torch.randn(n, hk, d, generator=g).to(torch.bfloat16).cuda()
    vc = torch.randn(n, hk, d, generator=g).to(torch.bfloat16).cuda()
    positions = torch.tensor([0, 1, 63, 64, 511, 512, 513, 700, 1499], dtype=torch.int32).cuda()
    q = torch.randn(len(positions), h, d, generator=g).to(torch.bfloat16).cuda()
    slots = torch.zeros_like(positions)
    got = sliding(q, kc[None].contiguous(), vc[None].contiguous(), positions, slots, window=window, scale=d ** -0.5)
    torch.testing.assert_close(got.float(), reference(q, kc, vc, positions, window, d ** -0.5), rtol=2e-2, atol=2e-2)


def test_a_sliding_rows_bits_never_depend_on_the_call():
    from tensorfold.families.kolibri1.cuda.attention import sliding

    g = torch.Generator().manual_seed(1)
    n, h, hk, d = 900, 48, 4, 128
    kc = torch.randn(n, hk, d, generator=g).to(torch.bfloat16).cuda()
    vc = torch.randn(n, hk, d, generator=g).to(torch.bfloat16).cuda()
    positions = torch.arange(300, 900, dtype=torch.int32).cuda()
    q = torch.randn(len(positions), h, d, generator=g).to(torch.bfloat16).cuda()
    kr, vr, slots = kc[None].contiguous(), vc[None].contiguous(), torch.zeros_like(positions)
    whole = sliding(q, kr, vr, positions, slots, window=513, scale=d ** -0.5)
    for r in (0, 255, 599):
        one = sliding(q[r:r + 1].contiguous(), kr, vr, positions[r:r + 1], slots[r:r + 1], window=513,
                      scale=d ** -0.5)
        assert torch.equal(whole[r], one[0])


def test_a_ring_holds_the_window_with_the_same_bits_as_full_caches():
    from tensorfold.families.kolibri1.cuda.attention import ring_size, sliding

    g = torch.Generator().manual_seed(2)
    n, h, hk, d, win = 1000, 48, 4, 128, 513
    kc = torch.randn(n, hk, d, generator=g).to(torch.bfloat16).cuda()
    vc = torch.randn(n, hk, d, generator=g).to(torch.bfloat16).cuda()
    positions = torch.arange(900, 1000, dtype=torch.int32).cuda()
    q = torch.randn(len(positions), h, d, generator=g).to(torch.bfloat16).cuda()
    ring = ring_size(100, win)
    kr = torch.zeros(2, ring, hk, d, dtype=torch.bfloat16, device="cuda")      # slot 1 holds it, slot 0 noise
    vr = torch.zeros_like(kr)
    kr[0].normal_()
    keys = torch.arange(n, device="cuda")
    kr[1, keys % ring], vr[1, keys % ring] = kc, vc                            # the latest key wins its slot
    ones = torch.ones_like(positions)
    got = sliding(q, kr, vr, positions, ones, window=win, scale=d ** -0.5)
    want = sliding(q, kc[None].contiguous(), vc[None].contiguous(), positions, ones * 0, window=win, scale=d ** -0.5)
    assert torch.equal(got, want)
