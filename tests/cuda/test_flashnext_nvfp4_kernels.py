"""Preflight: the NVFP4 engine's settings and memory, checked before a byte is loaded.

``check`` is the refusals the loader cannot ride around: the BF16 non-expert weights make the checkpoint
bigger than the MLX one's contract in one place (the lm_head stays BF16, not 4-bit), so the settings that
depend on the 4-bit sizes are the ones to sanity-check; and ``b16`` matmul's row-invariance contract with
graph capture is the new kernel's, checked with the FP4 one's on Spark.
"""

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from tensorfold.families.qwen4_exp.cuda import bf16, nvfp4, nvfp4_moe  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_b16_matmul_is_row_invariant():
    """A BF16 row's bits depend only on its own input: the same row in a 1-row window and a 16-row window
    gives the same bits (the split is set by the shape, the reduce's order fixed)."""

    torch.manual_seed(0)
    dev = "cuda"
    w = (torch.randn(1280, 2560, device=dev) * 0.02).to(torch.bfloat16)
    b = bf16.b16_from_rows(w)
    x = (torch.randn(16, 2560, device=dev) * 0.5).to(torch.bfloat16)
    solo = torch.empty((1, 1280), dtype=torch.bfloat16, device=dev)
    bf16.matmul(x[:1], b, out=solo)
    together = torch.empty((16, 1280), dtype=torch.bfloat16, device=dev)
    bf16.matmul(x, b, out=together)
    assert torch.equal(solo[0], together[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("heads,dh", [(2, 32), (16, 160)])
def test_ple_embed_bf16_places_the_rows_and_sums_them(heads: int, dh: int) -> None:
    """A bf16 table's rows reach the embedding as they ship: row r * heads + h lands in OUT's h-th slice of row
    r and every group of 32 carries its sum -- the 4-bit kernel's contract without the unpacking."""

    from tensorfold.families.qwen4_exp.cuda import glue

    torch.manual_seed(2)
    dev, rows = "cuda", 5
    values = (torch.randn(rows * heads, dh, device=dev) * 0.5).to(torch.bfloat16)
    out = torch.empty((rows, heads * dh), dtype=torch.bfloat16, device=dev)
    xs = torch.empty((rows, heads * dh // 32), dtype=torch.float32, device=dev)
    glue.ple_embed_bf16(rows, values, heads, dh, out, xs)
    want = values.view(rows, heads * dh)
    assert torch.equal(out, want)
    assert torch.equal(xs, want.float().view(rows, heads * dh // 32, 32).sum(-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_b16_matmul_matches_reference():
    torch.manual_seed(1)
    dev = "cuda"
    w = (torch.randn(640, 2560, device=dev) * 0.02).to(torch.bfloat16)
    x = (torch.randn(7, 2560, device=dev) * 0.5).to(torch.bfloat16)
    got = bf16.matmul(x, bf16.make_b16(w))
    want = (x.to(torch.float32) @ w.to(torch.float32).T).to(torch.bfloat16)
    assert (got.to(torch.float32) - want.to(torch.float32)).abs().max().item() < 2e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_b16_matmul_row_invariance_survives_split_k():
    """The lm_head shape (N large): split-K is on, and the slices' sum order is fixed by the shape — rows
    in a 1-row and a 32-row window give the same bits."""

    torch.manual_seed(2)
    dev = "cuda"
    w = (torch.randn(2560, 640, device=dev) * 0.02).to(torch.bfloat16)
    b = bf16.b16_from_rows(w)
    assert bf16.split_k(b.n, b.k) > 1
    x = (torch.randn(32, 640, device=dev) * 0.5).to(torch.bfloat16)
    solo = torch.empty((1, 2560), dtype=torch.bfloat16, device=dev)
    bf16.matmul(x[:1], b, out=solo)
    together = torch.empty((32, 2560), dtype=torch.bfloat16, device=dev)
    bf16.matmul(x, b, out=together)
    assert torch.equal(solo[0], together[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fp4_matmul_row_invariance_on_real_shapes():
    """The FP4 kernel on the expert shapes (gate/up (1280, 2560), down (2560, 640)): a row's bits are its
    own — one-row window vs a 16-row window."""

    torch.manual_seed(3)
    dev = "cuda"
    for n, k in ((1280, 2560), (2560, 640)):
        words = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
        scale = torch.randint(90, 115, (n, k // 16), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
        fp = nvfp4.make_fp4(words, scale, 0.01)
        x = (torch.randn(16, k, device=dev) * 0.5).to(torch.bfloat16)
        solo = torch.empty((1, n), dtype=torch.bfloat16, device=dev)
        nvfp4.matmul(x[:1], fp, out=solo)
        together = torch.empty((16, n), dtype=torch.bfloat16, device=dev)
        nvfp4.matmul(x, fp, out=together)
        assert torch.equal(solo[0], together[0]), (n, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fp4_matmul_matches_the_dequantize_reference():
    torch.manual_seed(4)
    dev = "cuda"
    n, k = 640, 2560
    words = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
    scale = torch.randint(90, 115, (n, k // 16), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    fp = nvfp4.make_fp4(words, scale, 0.01)
    x = (torch.randn(5, k, device=dev) * 0.5).to(torch.bfloat16)
    got = nvfp4.matmul(x, fp, f32=True).double()
    want = x.double() @ nvfp4.dequantize_fp4(fp).double().T          # fp64: no TF32 in the reference
    assert ((got - want).abs().max() / want.abs().max()).item() < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_moe4_step_runs_each_row_through_its_expert_and_the_shared_one():
    """The MoE step on the grouped NVFP4 kernel: each row's routed slot is its expert's SwiGLU and down product, the
    shared slot the bf16 expert's, and a row alone gives the bits it gets among the others."""

    from types import SimpleNamespace

    from tensorfold.cuda import moe as moe_mod
    from tensorfold.cuda.nvfp4 import experts as nvx

    torch.manual_seed(5)
    dev = "cuda"
    e, d, ni = 3, 256, 128

    def proj(n, k):
        return (torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev),
                torch.randint(40, 60, (n, k // 16), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn), 0.01)

    gate, up, down = ([proj(ni, d) for _ in range(e)], [proj(ni, d) for _ in range(e)],
                      [proj(d, ni) for _ in range(e)])
    shared = tuple((torch.randn(o, i) * 0.02).to(torch.bfloat16).to(dev) for o, i in ((ni, d), (ni, d), (d, ni)))
    ex = nvfp4_moe.moe4_from_experts(gate, up, down, shared)
    cfg = SimpleNamespace(num_experts_per_tok=1, num_experts=e, moe_intermediate_size=ni, hidden_size=d)
    router = (torch.randn(e + 1, d, device=dev) * 0.1).to(torch.bfloat16)
    x = (torch.randn(4, d, device=dev) * 0.5).to(torch.bfloat16)

    def step(rows):
        buf = moe_mod.MoEBuffers(4, cfg, dev)
        nvfp4_moe.moe(rows, None, router, ex, buf, cfg)
        return buf.pick[:rows.shape[0]].clone(), buf.y[:rows.shape[0]].clone()

    picks, y = step(x)
    for r in range(4):
        k = int(picks[r, 0])
        gv = (x[r].double() @ nvx.dense(ex.routed_experts, k, "gate").double().T).float().to(torch.bfloat16).float()
        uv = (x[r].double() @ nvx.dense(ex.routed_experts, k, "up").double().T).float().to(torch.bfloat16).float()
        act = ((gv / (1 + torch.exp(-gv))).to(torch.bfloat16).float() * uv).to(torch.bfloat16)
        want = act.double() @ nvx.dense(ex.routed_experts, k, "down").double().T
        assert float((y[r, 0].double() - want).abs().max() / want.abs().max()) < 3e-2, r
        sa = ex.shared_act(x[r:r + 1])
        assert torch.equal(y[r, 1], nvfp4.matmul(sa, ex.shared.down, f32=True)[0])
    alone = step(x[2:3].contiguous())[1]
    assert torch.equal(alone[0], y[2])
