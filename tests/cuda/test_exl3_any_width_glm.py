"""GLM-shaped 3-bit EXL3 experts: format bits, repeatability, independent rows and buffer ownership."""

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
    """The kernel decodes the same fp16 bits as format.unpack for a 3-bit trellis."""
    from tensorfold.cuda.exl3 import experts
    from tensorfold.cuda.exl3.format import unpack as fmt_unpack

    t = torch.randint(-32768, 32768, (D // 16, I // 16, 8 * K2), dtype=torch.int16).cuda()
    ref = torch.from_numpy(fmt_unpack(t.cpu().numpy(), 3.0, "mcg")).cuda()
    assert torch.equal(experts.dequant(t, "mcg").view(torch.uint8), ref.view(torch.uint8))


def test_prepared_layer_reports_k2_from_the_checkpoint():
    """Each checkpoint tensor supplies its own bit width."""

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
    y1 = experts.routed(x, pick, None, ex, s, None, 4, act_mode=experts.ACT_BF16).clone()
    y2 = experts.routed(x, pick, None, ex, s, None, 4, act_mode=experts.ACT_BF16)
    assert torch.equal(y1, y2), "repeated calls must be bit-identical"


def test_rows_are_independent_of_batch_composition():
    """A row gives the same bits alone, duplicated or inside a window."""
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
    """Shared-expert picks are left for the caller to supply."""
    from tensorfold.cuda.exl3 import experts

    ex = _layer()
    x = torch.randn(2, D, dtype=torch.bfloat16, device="cuda")
    pick = torch.full((2, 4), E, dtype=torch.int32, device="cuda")
    s = experts.Scratch(ex, 2, 4, device="cuda")
    y = experts.routed(x, pick, None, ex, s, None, 2, act_mode=experts.ACT_BF16)
    torch.cuda.synchronize()
    assert torch.all(y == 0)


def test_family_adapter_uses_live_rows_and_owned_buffers():
    from tensorfold.cuda.exl3 import experts
    from tensorfold.families.glm5_next.cuda import exl3_generic

    ex = _layer()
    x = torch.randn(4, D, dtype=torch.bfloat16, device="cuda")
    pick = torch.arange(16, dtype=torch.int32, device="cuda").view(4, 4) % E
    wide = experts.Scratch(ex, 16, 4, device="cuda")
    narrow = experts.Scratch(ex, 4, 4, device="cuda")
    wide.y.fill_(123)
    got = exl3_generic.routed(x, pick, ex, wide, 4, 7.0).clone()
    expected = experts.routed(x, pick, None, ex, narrow, None, 4, 7.0, act_mode=experts.ACT_BF16)
    assert got.shape == (16, D) and torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
    assert torch.all(wide.y[16:] == 123)
    assert wide.y.data_ptr() != narrow.y.data_ptr()
