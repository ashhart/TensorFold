"""The 27B's int8 KV cache (``--kv-dtype int8``): packed rows hold ExLlamaV3's -cq 8 codes and scales bit for bit, a
node of a draft tree gets the bits it gets alone, and the prompt path keeps its bits in any chunking, so drafts equal
serial and a resume equals a fresh prompt at int8 as at bf16."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.kernels import attention as shared  # noqa: E402
from tensorfold.cuda.kernels import kvpack  # noqa: E402
from tensorfold.families.qwen3_5.cuda.decode import draft_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen3_5.cuda.forward import State  # noqa: E402
from tensorfold.families.qwen3_5.cuda.prefill import prefill_chunk  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm_fast import prepare  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import Attention, Config, GDN, Layer, QLinear, Weights  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.kvcache import h32_ref, quantize_ref  # noqa: E402

V = 256


def _rows(n, hk, d, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((n, hk, d), generator=gen, device="cuda")
    x[:, :, 3] *= 40.0                                   # an outlier channel, as keys have
    return x.bfloat16()


def _split(packed, d):
    """Packed rows -> (int8 codes [N, HK, D], fp16 scales [N, HK, D / 32])."""

    return packed[..., :d], packed[..., d:].contiguous().view(torch.float16)


@pytest.mark.parametrize("d", [128, 256])
def test_packed_rows_hold_the_reference_quantizers_codes_and_scales(d):
    x = _rows(37, 4, d, seed=d)
    code, scale, _, _ = quantize_ref(x, x, bits=8)
    got_code, got_scale = _split(kvpack.pack(x), d)
    assert torch.equal(got_code, code)
    assert torch.equal(got_scale, scale)


@pytest.mark.parametrize("d", [128, 256])
def test_unpack_is_the_dequantized_rows_rotated_back(d):
    x = _rows(29, 4, d, seed=3 + d)
    packed = kvpack.pack(x)
    code, scale = _split(packed, d)
    deq = (code.float().reshape(29, 4, d // 32, 32) + 0.5) * scale.float()[..., None] * 0.0078125
    want = h32_ref(deq.reshape(29, 4, d)).reshape(29, 4, d).bfloat16()
    assert torch.equal(kvpack.unpack(packed, d), want)
    assert (kvpack.unpack(packed, d).float() - x.float()).abs().max() < 0.02 * x.float().abs().max()


def test_rotate_is_h32_by_groups_rounded_once():
    x = _rows(11, 24, 256, seed=8)
    assert torch.equal(kvpack.rotate(x), h32_ref(x.float()).reshape(x.shape).bfloat16())


def _inputs(w, p, *, h=24, hk=4, d=256, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(920 + w + p + seed)
    q = torch.randn((w, h, d), generator=gen, device="cuda").bfloat16()
    kn, vn = (kvpack.pack(torch.randn((w, hk, d), generator=gen, device="cuda").bfloat16()) for _ in range(2))
    kc, vc = (kvpack.pack(torch.randn((p + 5, hk, d), generator=gen, device="cuda").bfloat16()) for _ in range(2))
    return q, kn, vn, kc, vc


def _attend(q, kn, vn, caches, trees, lengths, scale):
    plan = shared.plan(trees, lengths, q.shape[1] // kn.shape[1], "cuda", tile=shared.PACKED_TILE)
    offs = torch.tensor(shared.offsets(caches, "cuda"), dtype=torch.int64, device="cuda").view(-1, 2)
    return shared.attention(q, kn, vn, offs, plan, scale=scale)


def _path(parents, node):
    rows = []
    while node >= 0:
        rows.append(node)
        node = parents[node]
    return rows[::-1]


def _serial(inputs, parents, p, node, scale):
    """The node alone: its ancestors' packed rows committed after the cache's first p (what a commit copies)."""

    q, kn, vn, kc, vc = inputs
    rows = _path(parents, node)
    keys = torch.cat((kc[:p], kn[rows[:-1]]), 0).contiguous()
    values = torch.cat((vc[:p], vn[rows[:-1]]), 0).contiguous()
    one = [x[node:node + 1].contiguous() for x in (q, kn, vn)]
    return _attend(*one, [(keys, values)], [[-1]], [keys.shape[0]], scale)[0]


@pytest.mark.parametrize("w,p", [(1, 0), (9, 13), (16, 511), (32, 512), (32, 1003), (128, 513), (16, 20501)])
def test_packed_branch_nodes_match_serial_bits(w, p):
    inputs = _inputs(w, p)
    parents = [-1] + [(i - 1) // 2 for i in range(1, w)]
    out = _attend(*inputs[:3], [inputs[3:]], [parents], [p], 1 / 16)
    for node in sorted({0, w // 3, w // 2, w - 1}):
        assert torch.equal(out[node], _serial(inputs, parents, p, node, 1 / 16)), f"node {node} differs"


def test_packed_streams_in_one_launch_equal_each_alone():
    shapes = [(3, 0), (16, 700), (1, 513), (9, 2600)]
    parts = [_inputs(w, p, seed=i) for i, (w, p) in enumerate(shapes)]
    trees = [[-1] + [max(0, i - 2) for i in range(1, w)] for w, _ in shapes]
    q, kn, vn = (torch.cat([x[j] for x in parts]) for j in range(3))
    together = _attend(q, kn, vn, [x[3:] for x in parts], trees, [p for _, p in shapes], 1 / 16)
    base = 0
    for i, ((w, p), part) in enumerate(zip(shapes, parts)):
        assert torch.equal(together[base:base + w], _attend(*part[:3], [part[3:]], [trees[i]], [p], 1 / 16)), i
        base += w


def test_packed_attention_matches_torch_on_the_dequantized_rows():
    w, p, d = 7, 1500, 256
    q, kn, vn, kc, vc = _inputs(w, p)
    parents = [-1, 0, 0, 1, 1, 2, 3]
    out = _attend(q, kn, vn, [(kc, vc)], [parents], [p], 1 / 16)
    kcf, vcf, knf, vnf = (kvpack.unpack(x, d) for x in (kc, vc, kn, vn))
    for node in range(w):
        rows = _path(parents, node)
        keys = torch.cat((kcf[:p], knf[rows]), 0).float().repeat_interleave(6, 1)
        values = torch.cat((vcf[:p], vnf[rows]), 0).float().repeat_interleave(6, 1)
        scores = torch.einsum("hd,thd->ht", q[node].float(), keys) / 16
        ref = torch.einsum("ht,thd->hd", scores.softmax(-1), values)
        assert (out[node].float() - ref).abs().max() < 0.035


def test_bf16_and_packed_caches_refuse_to_mix():
    q, kn, vn, kc, vc = _inputs(2, 10)
    plan = shared.plan([[-1, 0]], [10], 6, "cuda", tile=shared.PACKED_TILE)
    offs = torch.tensor(shared.offsets([(kc, vc)], "cuda"), dtype=torch.int64, device="cuda").view(-1, 2)
    with pytest.raises(ValueError, match="unsupported attention shape"):
        shared.attention(q, kn, kvpack.unpack(vn, 256), offs, plan, scale=1 / 16)
    with pytest.raises(ValueError, match="tile=32"):            # a bf16 plan for packed rows
        shared.attention(q, kn, vn, offs, shared.plan([[-1, 0]], [10], 6, "cuda"), scale=1 / 16)


# -- the 27B's prompt and decode paths at int8 ---------------------------------------------------------------------
def _model(kv_dtype="int8"):
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
    w.kv_dtype = kv_dtype
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


def test_an_int8_state_holds_packed_rows():
    st = State(_model())
    kv = next(x for x in st.kv if x is not None)
    assert kv[0].dtype == torch.int8 and kv[0].shape[1:] == (1, kvpack.row_bytes(128))
    assert next(x for x in State(_model("bf16")).kv if x is not None)[0].dtype == torch.bfloat16


@pytest.mark.parametrize("size", [1, 7, 64, 256])
def test_every_int8_chunking_gives_the_same_state(size):
    w = _model()
    prompt = _prompt(300)
    whole, h_whole = _chunked(w, prompt, [(0, 300)])
    parts, h_parts = _chunked(w, prompt, [(a, min(a + size, 300)) for a in range(0, 300, size)])
    _same_state(whole, parts)
    assert torch.equal(h_whole, h_parts)


def test_int8_resume_equals_fresh_and_drafts_equal_serial():
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


def test_int8_prompt_tracks_bf16_closely():
    prompt = _prompt(400, seed=9)
    ids = torch.tensor(prompt, dtype=torch.int32, device="cuda")
    w8, w16 = _model(), _model("bf16")
    h8, _ = prefill_chunk(w8, ids, State(w8))
    h16, _ = prefill_chunk(w16, ids, State(w16))
    cos = torch.nn.functional.cosine_similarity(h8.float().flatten(), h16.float().flatten(), dim=0)
    assert cos > 0.999
