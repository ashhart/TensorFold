"""The 27B's prefill, bf16 prompts and --prefill-fp8 alike: any chunking gives the same bits, a resume equals a fresh
prompt, and drafts equal serial."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import prompt_precision  # noqa: E402
from tensorfold.cuda.kernels import attention as tree_attention  # noqa: E402
from tensorfold.cuda.kernels import qmm as shared  # noqa: E402
from tensorfold.cuda.kernels.prefill_attention import attention  # noqa: E402
from tensorfold.families.qwen3_5.cuda.decode import clone_state, draft_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen3_5.cuda.forward import State, commit, tree_forward  # noqa: E402
from tensorfold.families.qwen3_5.cuda.prefill import prefill_chunk, prefill_state  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm_fast import prepare  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import Attention, Config, GDN, Layer, QLinear, Weights  # noqa: E402

V = 256


@pytest.fixture(params=[False, True], ids=["bf16", "fp8"])
def fp8(request):
    """Each test once with bf16 prompts (the default) and once with --prefill-fp8."""

    with prompt_precision.using(request.param):
        yield request.param


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
def test_every_chunking_gives_the_same_state(size, fp8):
    w = _model()
    prompt = _prompt(300)
    whole, h_whole = _chunked(w, prompt, [(0, 300)])
    parts, h_parts = _chunked(w, prompt, [(a, min(a + size, 300)) for a in range(0, 300, size)])
    _same_state(whole, parts)
    assert torch.equal(h_whole, h_parts)


def test_ragged_resume_equals_fresh(fp8):
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


def test_prefix_reuse_through_prefill_equals_fresh_and_drafts_equal_serial(fp8):
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
def test_a_cache_limit_changes_no_bits(length, limit, fp8):
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


@pytest.mark.parametrize("policy", [0, None], ids=["fold-in-kernel", "default"])
def test_a_long_prompt_tree_window_equals_serial_on_every_path(policy, monkeypatch):
    """9,000 committed keys, four whole key groups of the tree attention: each row of a 16-row draft tree equals
    serial decoding along its path, the tree folding the groups in their programs and a lone row in the merge (or
    both in their programs)."""

    if policy is not None:
        monkeypatch.setattr(tree_attention, "MIN_GROUPED", policy)
    w = _model()
    prompt = _prompt(9000, seed=11)
    st, first = prefill(w, prompt, None)
    assert tree_attention.groups(st.pos, 16) == 4 and (policy == 0) == (tree_attention.groups(st.pos, 1) == 4)
    parents = [-1, 0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    tokens = torch.randint(1, V, (16,), generator=torch.Generator().manual_seed(3)).to(torch.int32).cuda()
    tokens[0] = first
    logits, _ = tree_forward(w, tokens, parents, clone_state(st))
    for leaf in (11, 12, 13, 14, 15):
        path = [leaf]
        while parents[path[-1]] >= 0:
            path.append(parents[path[-1]])
        serial = clone_state(st)
        for row in reversed(path):
            out, record = tree_forward(w, tokens[row:row + 1], [-1], serial)
            assert torch.equal(out[0], logits[row]), f"row {row} on the path to {leaf}"
            commit(serial, record, [0])


SM90 = pytest.mark.skipif(bool(getattr(torch.version, "hip", None)),
                          reason="the tiled and FP8 prompt kernels need sm_90 (thread-block clusters, FP8 MMA); "
                                 "ROCm prompts use the lane matmul, covered by the whole-model chunking tests")


@SM90
def test_bf16_prompts_track_decode_closer_than_fp8():
    """bf16 prompt rows sit nearer decode's arithmetic than FP8 rows do (this model, 300 rows: 0.26% against 0.57%)."""

    w = _model()
    prompt = _prompt(300, seed=11)
    ids = torch.tensor(prompt, dtype=torch.int32, device="cuda")
    rows = {}
    for on in (False, True):
        with prompt_precision.using(on):
            rows[on], _ = prefill_chunk(w, ids, State(w), every=True)
    st = State(w)
    dec = []
    for a in range(0, 300, 16):
        n = min(16, 300 - a)
        h, record = tree_forward(w, ids[a:a + n], list(range(-1, n - 1)), st, full_logits=False)
        commit(st, record, list(range(n)))
        dec.append(h)
    dec = torch.cat(dec).float()
    err = {on: float((rows[on].float() - dec).norm() / dec.norm()) for on in rows}
    assert not torch.equal(rows[False], rows[True])
    assert err[False] < 4e-3 and err[False] < err[True] / 1.5, err


@SM90
@pytest.mark.parametrize("n", [1000, 1100])                     # 1,100: 1,152 padded, not a multiple of 256
def test_prefill_matmul_rows_do_not_depend_on_chunking(n):
    gen = torch.Generator(device="cuda").manual_seed(3)
    k = 1024
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda", dtype=torch.int64)
    scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.01 + 0.001).bfloat16()
    biases = (torch.randn(n, k // 64, generator=gen, device="cuda") * 0.02).bfloat16()
    q = shared.pack(words.to(torch.int32), scales, biases, 64)
    x = torch.randn(333, k, generator=gen, device="cuda").bfloat16()
    whole = shared.prefill_matmul(x, q, f32=True)
    for tile in range(12):
        assert torch.equal(whole, shared.prefill_matmul(x, q, f32=True, tile=tile)), tile
    for size in (1, 7, 16, 64, 256):
        parts = [shared.prefill_matmul(x[a:a + size].contiguous(), q, f32=True) for a in range(0, 333, size)]
        assert torch.equal(whole, torch.cat(parts))
    q_ = ((words[:, :, None] >> torch.arange(0, 32, 4, device="cuda")) & 0xF).reshape(n, k).double()
    dense = q_ * scales.double().repeat_interleave(64, 1) + biases.double().repeat_interleave(64, 1)
    ref = x.double() @ dense.t()
    assert ((whole.double() - ref).norm() / ref.norm()).item() < 4e-3


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


@SM90
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
