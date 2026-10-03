"""Flash Next image features, rotary offsets and image-cache isolation on tiny CUDA weights."""
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import _model
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tensorfold.vision.qwen_cuda import EncodedVision


def _image(w, prompt, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    features = torch.randn((4, w.cfg.hidden), device="cuda", generator=generator, dtype=torch.bfloat16)
    positions = torch.arange(len(prompt), device="cuda", dtype=torch.int32).repeat(3, 1)
    positions[:, 4:] -= 2
    positions[:, 1:5] = torch.tensor([[1, 1, 1, 1], [1, 1, 2, 2], [1, 2, 1, 2]], device="cuda")
    return EncodedVision((1, 2, 3, 4), features, positions, -2)


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8", "int4"])
def test_image_prefill_is_chunk_invariant_and_clears_positions(kv_dtype):
    w = _model()
    prompt = [5, 17, 17, 17, 17] + list(range(20, 55))
    image = _image(w, prompt, 9)
    engines = [Engine(w, capacity=1024, max_rows=8, prefill_rows=rows, kv_dtype=kv_dtype) for rows in (16, 64)]
    tokens = []
    for engine in engines:
        first = prefill(engine, prompt, None, mtp=False, vision=image)
        assert engine.pbuf.rope_rows is None and engine.st.rope_delta == -2
        tokens.append(serial_decode(engine, first, 8, None).tokens)
    assert tokens[0] == tokens[1]
    with pytest.raises(ValueError, match="from its start"):
        prefill(engines[0], prompt, None, resume={}, vision=image)


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8", "int4"])
def test_image_streams_match_serial_and_never_reuse_placeholder_states(kv_dtype):
    w = _model()
    prompt = [5, 17, 17, 17, 17] + list(range(20, 55))
    images = [_image(w, prompt, seed) for seed in (9, 12)]
    tower = SimpleNamespace(encode=lambda prepared, ids: prepared)
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, vision=tower)
    for image in images:
        engine = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        reference = serial_decode(engine, prefill(engine, prompt, None, mtp=False, vision=image), 8, None).tokens
        stream = Stream(prompt, 8, vision=image)
        dec.admit(stream)
        while dec.live():
            dec.finish(dec.round())
        assert stream.out == reference and stream.cached == 0 and not dec.kept
    assert len(dec.free) == 2


@pytest.mark.parametrize("rows", [1, 3, 7, 16])
def test_image_positions_cover_unaligned_sparse_pool_blocks(rows):
    from dataclasses import replace
    from test_flashnext_forward import _state

    w = _model()
    w.cfg = replace(w.cfg, index_budget=8)
    prompt = [5, 17, 17, 17, 17] + list(range(20, 54))
    image = _image(w, prompt, 4)
    full = Engine(w, capacity=1024, max_rows=8, prefill_rows=64)
    split = Engine(w, capacity=1024, max_rows=8, prefill_rows=rows)
    expected = prefill(full, prompt, None, vision=image)
    actual = prefill(split, prompt, None, vision=image)
    assert actual == expected
    for a, b in zip(_state(split), _state(full)):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    for a, b in zip(split.st.pooled + [split.st.mtp_pooled], full.st.pooled + [full.st.mtp_pooled]):
        assert torch.equal(a, b)
    assert serial_decode(split, actual, 8, None).tokens == serial_decode(full, expected, 8, None).tokens


@pytest.mark.parametrize("drafted", [False, True])
def test_image_prompts_share_passes_with_text_and_live_decodes(drafted):
    from dataclasses import replace
    from tensorfold.engine.exact_sampling import Sampling

    w = _model()
    w.cfg = replace(w.cfg, index_budget=8)
    prompt = [5, 17, 17, 17, 17] + list(range(20, 88))
    text = list(range(40, 60))
    images = [_image(w, prompt, seed) for seed in (7, 15)]
    sampling = Sampling(seed=67, top_k=20, top_p=0.95)
    tower = SimpleNamespace(encode=lambda prepared, ids: prepared)
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, prefill_rows=7, vision=tower, stop_eos=False)
    runs = [Stream(text, 32, sampling, draft=drafted, stop_eos=False)]
    dec.admit(runs[0])
    for _ in range(20):
        if runs[0].started:
            break
        done = dec.round()
        assert runs[0].error is None, runs[0].error
        dec.finish(done)
    assert runs[0].started
    for image in images:
        s = Stream(prompt, 12, sampling, draft=drafted, vision=image, stop_eos=False)
        dec.admit(s)
        assert s in dec.filling and s.st.pos == 0
        runs.append(s)
    mixed = False
    for _ in range(100):
        if not dec.live():
            break
        mixed |= bool(dec.filling and dec.streams)
        done = dec.round()
        assert all(s.error is None for s in runs), [s.error for s in runs]
        dec.finish(done)
    assert not dec.live() and mixed
    for s, image in zip(runs, [None, *images]):
        ref = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
        first = prefill(ref, s.prompt, sampling, mtp=False, vision=image)
        assert s.out == serial_decode(ref, first, s.count, sampling, stop_eos=False).tokens
    assert all(k[0] != prompt[:-1] for k in dec.kept)


def test_identity_image_positions_preserve_text_state_bits():
    from dataclasses import replace
    from test_flashnext_forward import _state

    w = _model()
    w.cfg = replace(w.cfg, index_budget=8)
    prompt = list(range(1, 40))
    positions = torch.arange(len(prompt), device="cuda", dtype=torch.int32).repeat(3, 1)
    identity = EncodedVision((), torch.empty((0, w.cfg.hidden), device="cuda", dtype=torch.bfloat16), positions, 0)
    plain = Engine(w, capacity=1024, max_rows=8, prefill_rows=7)
    mapped = Engine(w, capacity=1024, max_rows=8, prefill_rows=7)
    a = prefill(plain, prompt, None)
    b = prefill(mapped, prompt, None, vision=identity)
    assert a == b
    for x, y in zip(_state(plain), _state(mapped)):
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))
    for x, y in zip(plain.st.pooled + [plain.st.mtp_pooled], mapped.st.pooled + [mapped.st.mtp_pooled]):
        assert torch.equal(x, y)
    assert serial_decode(plain, a, 12, None).tokens == serial_decode(mapped, b, 12, None).tokens


@pytest.mark.parametrize("kind", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("rows,start", [(1, 0), (7, 2039), (16, 120001)])
def test_text_rotary_kernels_keep_release_bits(kind, rows, start):
    import flashnext_text_reference as release
    from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache
    from tensorfold.families.qwen4_exp.cuda import attention, glue

    generator = torch.Generator(device="cuda").manual_seed(319)
    def random(shape):
        return torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
    heads, kv, dim, ih, index_dim, half = 24, 2, 256, 4, 128, 32
    width = 2 * heads * dim + 2 * kv * dim + (ih + 1) * index_dim
    projection = random((rows, width))
    qscale, kscale, iscale, poolscale = [1 + random((d,)) * 0.05 for d in (dim, dim, index_dim, index_dim)]
    inv = 1.0 / (1e7 ** (torch.arange(half, device="cuda", dtype=torch.float32) / half))
    pos = torch.tensor([start], dtype=torch.int32, device="cuda")
    count = start + rows
    index = random((count, index_dim))
    outputs = []
    for prep, pool in [(release.attn_prep, release.qsa_pool), (glue.attn_prep, attention.qsa_pool)]:
        cache = KVCache(count, kv, dim, "cuda", kind)
        q, iq, ik = random((rows, heads, dim)), random((rows, ih, index_dim)), index.clone()
        bits = 0 if kind == "bf16" else int(kind[3:])
        prep(projection, pos, qscale, kscale, iscale, inv, q, cache.k, cache.v, iq, ik, 1e-6,
             q_heads=heads, kv_heads=kv, head_dim=dim, index_heads=ih, index_dim=index_dim,
             ks=cache.ks, vs=cache.vs, bits=bits)
        pooled = torch.zeros(((count + 3) // 4, index_dim), dtype=torch.bfloat16, device="cuda")
        pool(ik, pooled, pos, poolscale, inv, 1e-6, SimpleNamespace(ratio=4), rows)
        values = [q, iq, cache.k[start:count], cache.v[start:count], ik, pooled]
        if bits:
            values += [cache.ks[start:count], cache.vs[start:count]]
        outputs.append(values)
    for old, new in zip(*outputs):
        assert torch.equal(old.contiguous().view(torch.uint8), new.contiguous().view(torch.uint8))


def test_image_message_markers_never_seed_a_text_prefix_cache():
    w = _model()
    prompt = [5, 17, 17, 17, 17] + list(range(20, 322))
    image = _image(w, prompt, 19)
    tower = SimpleNamespace(encode=lambda prepared, ids: prepared)
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=1, prefill_rows=128, points=lambda ids: [256], vision=tower)
    pictured = Stream(prompt, 8, vision=image, stop_eos=False)
    dec.admit(pictured)
    for _ in range(50):
        if not dec.live():
            break
        done = dec.round()
        assert pictured.error is None, pictured.error
        dec.finish(done)
    assert not dec.live() and not dec.kept
    text = Stream(prompt, 8, stop_eos=False)
    dec.admit(text)
    assert text.cached == 0
    while dec.live():
        done = dec.round()
        assert text.error is None, text.error
        dec.finish(done)
    ref = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
    first = prefill(ref, prompt, None, mtp=False)
    assert text.out == serial_decode(ref, first, 8, None, stop_eos=False).tokens
