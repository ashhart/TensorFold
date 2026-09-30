"""GLM-shaped 3-bit (k2 = 6 half-bits) experts through the universal path.

The universal grouped kernel (`tensorfold/cuda/exl3/experts.*`) reads a bit
width per expert and was tested against a float64 reference at every width
(`test_mixed_k_rows_are_independent_and_match_the_reference`). GLM's own
family does not route through it: `families/glm5_next/cuda/exl3_mm.py` pins
the 4-bit-mcg layout (`words()` requires `int16 [..., 64]`), so a 3-bit
checkpoint refuses with `only 4-bit EXL3 trellises ... are supported` even
though the universal kernel serves it. These tests pin the universal path at
GLM's shapes and bit width, and the two properties an integrator relies on
before committing to it: determinism across calls, and a row's output being
the same alone and inside a window.

The layer fixture mirrors a real 3-bit pack's on-disk layout: trellis int16
`[K/16, N/16, 16 * k2]` words (48 at k2 = 6), fp16 `suh`/`svh` scales, 288
routed experts at D = 4096, I = 1024 per rank (the pack is rank-split), with
`GLM_GATEUP`/`GLM_DOWN` tile configs and `ACT_BF16` — the same arithmetic the
dedicated path runs, verified bit-identical by `test_glm_shaped_bit_identical`
at 4-bit.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

E, D, I = 288, 4096, 1024
K2 = 6                                   # 3 bits = 6 half-bits a value


def _layer(cb="mcg", device="cuda"):
    from tensorfold.cuda.exl3 import experts

    g = torch.Generator().manual_seed(11)

    def trellis(k, n):
        return torch.randint(-32768, 32768, (k // 16, n // 16, 8 * K2),
                             dtype=torch.int16, generator=g).to(device).contiguous()

    def scale(n, mag):
        sign = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
        return (sign * (torch.rand((n,), generator=g) + 0.5) * mag).half().to(device)

    gate, up, down = [], [], []
    for _ in range(E):
        gate.append((trellis(D, I), scale(D, 0.02), scale(I, 0.5)))
        up.append((trellis(D, I), scale(D, 0.02), scale(I, 0.5)))
        down.append((trellis(I, D), scale(I, 0.05), scale(D, 0.2)))
    return experts.prepare(gate, up, down, cb, device=device)


def test_dequant_identity_against_format_unpack():
    """The kernels' own decode of a 3-bit trellis equals the format's reference
    decoder, bit for bit. If the pack reads wrong here, everything downstream is
    built on a wrong W_q and the rest of the file proves nothing."""
    from tensorfold.cuda.exl3 import experts
    from tensorfold.cuda.exl3.format import unpack as fmt_unpack

    t = torch.randint(-32768, 32768, (D // 16, I // 16, 8 * K2), dtype=torch.int16).cuda()
    ref = torch.from_numpy(fmt_unpack(t.cpu().numpy(), 3.0, "mcg")).cuda()
    assert torch.equal(experts.dequant(t, "mcg"), ref)


def test_prepared_layer_reports_k2_from_the_checkpoint():
    """`prepare` derives k2 from each tensor's own last dim; a 3-bit pack must come
    out as k2 = 6 half-bits, not the 4-bit assumption of the dedicated wrapper."""
    from tensorfold.cuda.exl3 import experts

    ex = _layer()
    assert ex.k2_gu == (K2, K2)
    assert ex.k2_d == (K2, K2)
    assert ex.dims == D and ex.width == I and ex.count == E


def test_routed_is_deterministic_across_calls():
    from tensorfold.cuda.exl3 import experts

    ex = _layer()
    x = torch.randn(4, D, dtype=torch.bfloat16, device="cuda")
    pick = torch.zeros((4, 4), dtype=torch.int32, device="cuda")
    s = experts.Scratch(ex, 4, 4, device="cuda")
    y1 = experts.routed(x, pick, None, ex, s, None, 4, act_mode=experts.ACT_BF16)
    y2 = experts.routed(x, pick, None, ex, s, None, 4, act_mode=experts.ACT_BF16)
    assert torch.equal(y1, y2), "repeated calls must be bit-identical"


def test_rows_are_independent_of_batch_composition():
    """A row alone must equal its output inside a window, whatever the other rows
    hold: the grouped launch groups by expert and reads rows by index."""
    from tensorfold.cuda.exl3 import experts

    ex = _layer()
    x = torch.randn(4, D, dtype=torch.bfloat16, device="cuda")
    pick = torch.zeros((4, 4), dtype=torch.int32, device="cuda")

    s_full = experts.Scratch(ex, 4, 4, device="cuda")
    y_full = experts.routed(x, pick, None, ex, s_full, None, 4, act_mode=experts.ACT_BF16)
    s_one = experts.Scratch(ex, 4, 4, device="cuda")
    y_one = experts.routed(x[:1], pick[:1].contiguous(), None, ex, s_one, None, 1,
                           act_mode=experts.ACT_BF16)
    assert torch.equal(y_full[0], y_one[0])
    # duplicated rows with identical content give identical outputs
    x_dup = torch.cat([x[:1], x[:1]])
    s_dup = experts.Scratch(ex, 4, 4, device="cuda")
    y_dup = experts.routed(x_dup, torch.cat([pick[:1], pick[:1]]), None, ex, s_dup, None, 2,
                           act_mode=experts.ACT_BF16)
    assert torch.equal(y_dup[0], y_dup[1])


def test_shared_expert_pairs_are_skipped():
    """pick == E means the shared expert: the universal kernel leaves those pairs
    unwritten, matching the dedicated path's `slot == slots - 1` guard. The caller
    combines the slot from the BF16 shared MLP before reading it."""
    from tensorfold.cuda.exl3 import experts

    ex = _layer()
    x = torch.randn(2, D, dtype=torch.bfloat16, device="cuda")
    pick = torch.full((2, 4), E, dtype=torch.int32, device="cuda")
    s = experts.Scratch(ex, 2, 4, device="cuda")
    y = experts.routed(x, pick, None, ex, s, None, 2, act_mode=experts.ACT_BF16)
    torch.cuda.synchronize()
    assert torch.all(y == 0)
