"""The 27B's prefill: any chunking gives the same bits, a resume equals a fresh prompt, and drafts equal serial."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.build import hip  # noqa: E402
from tensorfold.cuda.kernels import qmm as shared  # noqa: E402
from tensorfold.cuda.kernels.prefill_attention import attention  # noqa: E402
from tensorfold.families.qwen3_5.cuda.decode import clone_state, draft_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen3_5.cuda.forward import State, commit, tree_forward  # noqa: E402
from tensorfold.families.qwen3_5.cuda.prefill import prefill_chunk, prefill_state  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm_fast import prepare  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import Attention, Config, GDN, Layer, QLinear, Weights  # noqa: E402

V = 256


def _model():
    gen = torch.Generator(device="cuda").manual_seed(21)
    dev = "cuda"

    def qlinear(n, k):
        words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen,
                              device=dev, dtype=torch.int64).to(torch.int32)
        scales = (torch.rand(n, k // 64, generator=gen, device=dev) * 0.003 + 0.001).bfloat16()
        biases = (torch.rand(n, k // 64, generator=gen, device=dev) * 0.003 - 0.0015).bfloat16()
        return QLinear(words, scales, biases)

    norm = torch.ones(128, device=dev, dtype=torch.bfloat16)
    gdn = GDN(qlinear(384, 128), qlinear(128, 128), qlinear(1, 128), qlinear(1, 128),
              qlinear(128, 128), torch.randn(384, 4, generator=gen, device=dev).bfloat16() * 0.1,
              torch.zeros(1, device=dev), torch.zeros(1, device=dev), norm)
    attn = Attention(qlinear(2 * 2 * 128, 128), qlinear(128, 128), qlinear(128, 128), qlinear(128, 2 * 128),
                     norm, norm)
    layers = [Layer(True, norm, norm, gdn, None, qlinear(128, 128), qlinear(128, 128), qlinear(128, 128)),
              Layer(False, norm, norm, None, attn, qlinear(128, 128), qlinear(128, 128), qlinear(128, 128))]
    config = Config(hidden=128, intermediate=128, layers=2, heads=2, kv_heads=1,
                    head_dim=128, vocab=V, k_heads=1, v_heads=1, dk=128, dv=128,
                    conv_kernel=4, interval=2, eps=1e-6, rope_dims=32,
                    rope_theta=10000000.0, eos=(0,))
    w = Weights(config, qlinear(V, 128), layers, norm, qlinear(V, 128), torch.ones(16, device=dev))
    prepare(w)
    return w


def _prompt(n, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (n,), generator=g).tolist()


def _chunked(w, prompt, bounds):
    st = State(w)
    ids = torch.tensor(prompt, dtype=torch.int32, device="cuda")
    normed = None
    for a, b in bounds:
        normed, _ = prefill_chunk(w, ids[a:b], st)
    return st, normed


def _same_state(a, b):
    assert a.pos == b.pos
    for x, y in zip(a.rec, b.rec):
        assert (x is None) == (y is None) and (x is None or torch.equal(x, y))
    for x, y in zip(a.conv, b.conv):
        assert (x is None) == (y is None) and (x is None or torch.equal(x, y))
    for x, y in zip(a.kv, b.kv):
        if x is not None:
            assert torch.equal(x[0][:a.pos], y[0][:b.pos]) and torch.equal(x[1][:a.pos], y[1][:b.pos])


@pytest.mark.parametrize("size", [1, 7, 16, 64, 256])
def test_every_chunking_gives_the_same_state(size):
    w = _model()
    prompt = _prompt(300)
    whole, h_whole = _chunked(w, prompt, [(0, 300)])
    parts, h_parts = _chunked(w, prompt, [(a, min(a + size, 300)) for a in range(0, 300, size)])
    _same_state(whole, parts)
    assert torch.equal(h_whole, h_parts)


def test_ragged_resume_equals_fresh():
    w = _model()
    prompt = _prompt(300, seed=6)
    fresh, h_fresh = _chunked(w, prompt, [(0, 300)])
    resumed, _ = _chunked(w, prompt, [(0, 137)])
    ids = torch.tensor(prompt, dtype=torch.int32, device="cuda")
    h_resumed = None
    for a, b in [(137, 140), (140, 211), (211, 300)]:
        h_resumed, _ = prefill_chunk(w, ids[a:b], resumed)
    _same_state(fresh, resumed)
    assert torch.equal(h_fresh, h_resumed)


def test_prefix_reuse_through_prefill_equals_fresh_and_drafts_equal_serial():
    w = _model()
    prompt = _prompt(240, seed=7)
    fresh, first_fresh = prefill(w, prompt, None)
    cached, _ = prefill(w, prompt[:101], None)
    resumed, first_resumed = prefill(w, prompt, None, state=cached)
    _same_state(fresh, resumed)
    assert first_fresh == first_resumed
    serial = serial_decode(w, fresh, first_fresh, 24, None, stop_eos=False)
    drafted = draft_decode(w, resumed, prompt, first_resumed, 24, None, draft=None, stop_eos=False)
    assert drafted.tokens == serial.tokens


@pytest.mark.parametrize("length,limit", [(9000, 9100), (1000, 1100)])
def test_a_cache_limit_changes_no_bits(length, limit):
    """9,000 rows: the third chunk grows to 9,100 rows, not 12,000; 1,000 rows: the reply grows to 1,100, not 2,048."""
    w = _model()
    prompt = _prompt(length, seed=9)
    free, bounded = State(w), State(w)
    bounded.limit = limit
    h_free = prefill_state(w, prompt, free)
    h_bounded = prefill_state(w, prompt, bounded)
    _same_state(free, bounded)
    assert torch.equal(h_free, h_bounded)
    fresh, first = prefill(w, prompt, None)
    kept, first_kept = prefill(w, prompt, None, limit=limit)
    assert kept.limit == limit and first_kept == first
    sizes = {kv[0].shape[0] for st in (bounded, kept) for kv in st.kv if kv is not None}
    assert max(sizes) <= limit
    serial = serial_decode(w, fresh, first, 90, None, stop_eos=False)
    assert serial_decode(w, kept, first_kept, 90, None, stop_eos=False).tokens == serial.tokens
    drafted = draft_decode(w, kept, prompt, first_kept, 90, None, draft=None, stop_eos=False)
    assert drafted.tokens == serial.tokens
    replied = []
    for st in (fresh, kept):                        # the reply's commits, kept to compare the states after it
        st = clone_state(st)
        for token in serial.tokens[:-1]:
            _, record = tree_forward(w, torch.tensor([token], dtype=torch.int32, device="cuda"), [-1], st)
            commit(st, record, [0])
        replied.append(st)
    _same_state(*replied)
    assert max(kv[0].shape[0] for kv in replied[1].kv if kv is not None) <= limit


@pytest.mark.skipif(hip(), reason="NVIDIA tensor-core layout; ROCm keeps the stored words")
def test_prefill_matmul_rows_do_not_depend_on_chunking():
    gen = torch.Generator(device="cuda").manual_seed(3)
    n, k = 1000, 1024
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda", dtype=torch.int64)
    scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.01 + 0.001).bfloat16()
    biases = (torch.randn(n, k // 64, generator=gen, device="cuda") * 0.02).bfloat16()
    q = shared.pack(words.to(torch.int32), scales, biases, 64)
    x = torch.randn(333, k, generator=gen, device="cuda").bfloat16()
    whole = shared.prefill_matmul(x, q, f32=True)
    for tile in range(5):
        assert torch.equal(whole, shared.prefill_matmul(x, q, f32=True, tile=tile))
    for size in (1, 7, 16, 64, 256):
        parts = [shared.prefill_matmul(x[a:a + size].contiguous(), q, f32=True) for a in range(0, 333, size)]
        assert torch.equal(whole, torch.cat(parts))
    q_ = ((words[:, :, None] >> torch.arange(0, 32, 4, device="cuda")) & 0xF).reshape(n, k).double()
    dense = q_ * scales.double().repeat_interleave(64, 1) + biases.double().repeat_interleave(64, 1)
    ref = x.double() @ dense.t()
    assert ((whole.double() - ref).norm() / ref.norm()).item() < 4e-3


def test_group_major_prefill_matmul_rows_do_not_depend_on_chunking():
    """ROCm's prompt matmul (group-major words, weights rounded once to bf16): any chunking, the same row bits."""

    from tensorfold.cuda.kernels import qmm_groups
    from tensorfold.families.qwen3_5.cuda.qmm import dequantize

    gen = torch.Generator(device="cuda").manual_seed(9)
    n, k = 1000, 1024
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda", dtype=torch.int64)
    words = words.to(torch.int32)
    scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.01 + 0.001).bfloat16()
    biases = (torch.randn(n, k // 64, generator=gen, device="cuda") * 0.02).bfloat16()
    g = qmm_groups.to_groups(words, scales, biases)
    assert all(torch.equal(a, b) for a, b in zip(qmm_groups.from_groups(*g, n), (words, scales, biases)))
    x = torch.randn(333, k, generator=gen, device="cuda").bfloat16()
    whole = qmm_groups.prefill_matmul(x, *g, n, f32=True)
    for size in (1, 7, 16, 64, 128, 256):
        parts = [qmm_groups.prefill_matmul(x[a:a + size].contiguous(), *g, n, f32=True) for a in range(0, 333, size)]
        assert torch.equal(whole, torch.cat(parts)), size
    ref = x.double() @ dequantize(words, scales, biases).double().t()
    assert ((whole.double() - ref).norm() / ref.norm()).item() < 4e-3


@pytest.mark.parametrize("n,k", [(1, 128), (256, 128), (384, 128), (48, 5120), (1024, 5120), (10240, 5120),
                                 (17408, 5120), (5120, 17408), (5120, 6144)])
def test_group_major_prefill_matmul_is_accurate_at_model_shapes(n, k):
    """Guards the prompt GEMM's tile against wrong sums (some tiles miscompile at small K on gfx1201)."""

    from tensorfold.cuda.kernels import qmm_groups
    from tensorfold.families.qwen3_5.cuda.qmm import dequantize

    gen = torch.Generator(device="cuda").manual_seed(n + k)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda", dtype=torch.int64)
    words = words.to(torch.int32)
    scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 + 0.001).bfloat16()
    biases = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 - 0.0015).bfloat16()
    x = torch.randn(200, k, generator=gen, device="cuda").bfloat16()
    out = qmm_groups.prefill_matmul(x, *qmm_groups.to_groups(words, scales, biases), n)
    ref = x.double() @ dequantize(words, scales, biases).double().t()
    assert ((out.double() - ref).norm() / ref.norm()).item() < 1e-2


def _e4m3_rows(x: torch.Tensor):
    """``prefill_glue``'s FP8 inputs for bf16 rows ``x``, and the values they stand for (row scale applied)."""

    m, k = x.shape
    src = torch.tensor([(i // 32) * 32 + ((i % 32) // 16) * 16 + ((i % 16) // 4) * 2 + (i % 4 % 2) + (i % 4 // 2) * 8
                        for i in range(k)], device=x.device)
    xf = x.float()
    a = xf.abs().amax(1).clamp_min(1e-30) / 448.0
    x8 = (xf[:, src] / a[:, None]).to(torch.float8_e4m3fn)
    xs = (xf.view(m, k // 64, 64).sum(2) / a[:, None]).bfloat16()
    values = torch.empty_like(xf)
    values[:, src] = x8.float() * a[:, None]
    return (x8.view(torch.uint8).contiguous(), xs, a.contiguous()), values, xs.double() * a.double()[:, None]


@pytest.mark.skipif(not hip(), reason="ROCm's FP8 prompt matmul")
@pytest.mark.parametrize("n,k", [(1, 128), (256, 128), (384, 128), (48, 5120), (1024, 5120), (17408, 5120),
                                 (5120, 17408), (5120, 6144), (1024, 6144)])
def test_rocm_fp8_prefill_matmul_is_exact_on_its_inputs_and_chunk_invariant(n, k):
    """The e4m3 codes times the e4m3 rows, scaled per group, plus the bias on the group sums: float64's answer on
    the same inputs, and the same bits at any chunking."""

    from tensorfold.cuda.kernels import qmm_groups

    gen = torch.Generator(device="cuda").manual_seed(n + 3 * k)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda", dtype=torch.int64)
    words = words.to(torch.int32)
    scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 + 0.001).bfloat16()
    biases = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 - 0.0015).bfloat16()
    g = qmm_groups.to_groups(words, scales, biases)
    x = torch.randn(300, k, generator=gen, device="cuda").bfloat16()
    rows, values, sums = _e4m3_rows(x)
    whole = qmm_groups.prefill_matmul8(rows, *g, n, f32=True)
    q = torch.stack([(words.view(n, k // 8, 1) >> (4 * i)) & 0xF for i in range(8)], -1).reshape(n, k).double()
    ref = sums @ biases.double().t()
    for j in range(k // 64):                                  # one group's dots at a time: (300, n) float64
        ref += (values.double()[:, 64 * j:64 * j + 64] @ q[:, 64 * j:64 * j + 64].t()) * scales.double()[:, j]
    assert ((whole.double() - ref).norm() / ref.norm()).item() < 1e-5
    for size in (1, 2, 5, 7, 64, 256):                       # few-row calls: pipelined tiles miscompile there
        parts = [qmm_groups.prefill_matmul8(tuple(t[a:a + size].contiguous() for t in rows), *g, n, f32=True)
                 for a in range(0, 300, size)]
        assert torch.equal(whole, torch.cat(parts)), size


def test_group_major_lane_matmul_gives_the_stored_layouts_bits():
    """The group-major lane matmul is the stored layout's arithmetic: the same bits at every row count."""

    from tensorfold.cuda.kernels import qmm_groups
    from tensorfold.families.qwen3_5.cuda.qmm import lane_matmul, split_k

    gen = torch.Generator(device="cuda").manual_seed(10)
    for n, k in ((1000, 1024), (48, 5120), (5120, 17408), (1, 128)):
        words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda", dtype=torch.int64)
        words = words.to(torch.int32)
        scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.01 + 0.001).bfloat16()
        biases = (torch.randn(n, k // 64, generator=gen, device="cuda") * 0.02).bfloat16()
        g = qmm_groups.to_groups(words, scales, biases)
        x = torch.randn(40, k, generator=gen, device="cuda").bfloat16()
        whole = qmm_groups.matmul(x, *g, n)
        for m in (1, 12, 16, 33):
            assert torch.equal(qmm_groups.matmul(x[:m].contiguous(), *g, n), whole[:m]), (n, k, m)
        if qmm_groups.lane_kernel() == "triton" and qmm_groups.split_k(n, k) == split_k(n, k):   # the same slices
            assert torch.equal(whole, lane_matmul(x, words, scales, biases)), (n, k)


@pytest.mark.parametrize("heads,kv_heads,dim", [(24, 4, 256), (8, 2, 128)])
def test_prefill_attention_rows_do_not_depend_on_chunking(heads, kv_heads, dim):
    gen = torch.Generator(device="cuda").manual_seed(4)
    total = 700
    q = torch.randn(total, heads, dim, generator=gen, device="cuda").bfloat16()
    k = torch.randn(total, kv_heads, dim, generator=gen, device="cuda").bfloat16()
    v = torch.randn(total, kv_heads, dim, generator=gen, device="cuda").bfloat16()
    scale = dim ** -0.5
    whole = attention(q, k, v, 0, scale=scale)
    for size in (1, 7, 16, 64, 256, 333):
        parts = [attention(q[a:a + size].contiguous(), k, v, a, scale=scale) for a in range(0, total, size)]
        assert torch.equal(whole, torch.cat(parts)), size
    g = heads // kv_heads
    kk = k.float().repeat_interleave(g, 1).transpose(0, 1)
    vv = v.float().repeat_interleave(g, 1).transpose(0, 1)
    s = q.float().transpose(0, 1) @ kk.transpose(1, 2) * scale
    s = s.masked_fill(torch.ones(total, total, device="cuda").triu(1).bool(), float("-inf"))
    ref = (s.softmax(-1) @ vv).transpose(0, 1)
    assert ((whole.float() - ref).norm() / ref.norm()).item() < 1e-2


@pytest.mark.skipif(hip(), reason="NVIDIA tensor-core layout; ROCm keeps the stored words")
@pytest.mark.parametrize("gs", [64, 32])
def test_fp8_prefill_matmul_rows_do_not_depend_on_chunking(gs):
    gen = torch.Generator(device="cuda").manual_seed(8)
    n, k = 1000, 1024
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda", dtype=torch.int64)
    scales = (torch.rand(n, k // gs, generator=gen, device="cuda") * 0.01 + 0.001).bfloat16()
    biases = (torch.randn(n, k // gs, generator=gen, device="cuda") * 0.02).bfloat16()
    q = shared.pack(words.to(torch.int32), scales, biases, gs)
    x = torch.randn(333, k, generator=gen, device="cuda").bfloat16()
    x[:, 5] *= 50                                                   # an outlier channel
    run = lambda rows, tile=0: shared.prefill_matmul8(shared.quantize_rows(rows.contiguous(), gs), q, f32=True,
                                                      tile=tile)
    whole = run(x)
    for tile in range(1, 3):
        assert torch.equal(whole, run(x, tile))
    for size in (1, 7, 16, 64, 256):
        assert torch.equal(whole, torch.cat([run(x[a:a + size]) for a in range(0, 333, size)]))
    q_ = ((words[:, :, None] >> torch.arange(0, 32, 4, device="cuda")) & 0xF).reshape(n, k).double()
    dense = q_ * scales.double().repeat_interleave(gs, 1) + biases.double().repeat_interleave(gs, 1)
    ref = x.double() @ dense.t()
    assert ((whole.double() - ref).norm() / ref.norm()).item() < 5e-2
