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


def _picture(w, prompt, start, seed):
    """A 2x2-merged picture's four rows from ``start``, at Qwen's positions; the text after it rotates at delta -2."""

    generator = torch.Generator(device="cuda").manual_seed(seed)
    features = torch.randn((4, w.cfg.hidden), device="cuda", generator=generator, dtype=torch.bfloat16)
    positions = torch.arange(len(prompt), device="cuda", dtype=torch.int32).repeat(3, 1)
    positions[:, start + 4:] -= 2
    positions[:, start:start + 4] = start + torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]], device="cuda")
    return EncodedVision(tuple(range(start, start + 4)), features, positions, -2)


def _prepared(w, prompt, start, seed, digest):
    """What a request carries into the decoder: the picture's span, hash and grid (the fake tower encodes it)."""

    return SimpleNamespace(encoded=_picture(w, prompt, start, seed), image_spans=((start, start + 4),),
                           image_hashes=(digest,), image_grid_thw=((1, 4, 4),), video_hashes=())


KEYED_TOWER = SimpleNamespace(encode=lambda prepared, ids: prepared.encoded)


def _drain(dec, *streams):
    for _ in range(400):
        if not dec.live():
            break
        done = dec.round()
        assert all(s.error is None for s in streams), [s.error for s in streams]
        dec.finish(done)
    assert not dec.live()


def _reference(w, prompt, count, sampling, image):
    ref = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
    first = prefill(ref, prompt, sampling, mtp=False, vision=image)
    return serial_decode(ref, first, count, sampling, stop_eos=False).tokens


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("start", [1, 9, 30])
def test_a_follow_up_with_the_same_picture_resumes_like_a_fresh_prefill(kv_dtype, start):
    from tensorfold.engine.exact_sampling import Sampling

    w = _model()
    sampling = Sampling(seed=67, top_k=20, top_p=0.95)
    first = list(range(20, 60))
    first[start:start + 4] = [17] * 4
    second = first + list(range(70, 93))
    stats = []
    for warm in (True, False):                          # the follow-up after its first turn, and on its own
        dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, prefill_rows=16,
                           vision=KEYED_TOWER, stop_eos=False)
        if warm:
            turn = Stream(first, 8, sampling, vision=_prepared(w, first, start, 9, "a"), stop_eos=False)
            dec.admit(turn)
            _drain(dec, turn)
            assert turn.out == _reference(w, first, 8, sampling, _picture(w, first, start, 9))
            assert dec.kept and all(t < 0 for t in dec.kept[-1][0][start:start + 4])
        follow = Stream(second, 12, sampling, vision=_prepared(w, second, start, 9, "a"), stop_eos=False)
        dec.admit(follow)
        assert follow.cached == (len(first) - 1 if warm else 0)
        _drain(dec, follow)
        assert follow.out == _reference(w, second, 12, sampling, _picture(w, second, start, 9))
        stats.append({k: follow.stats()[k] for k in ("rounds", "drafted", "accepted")})
    assert stats[0] == stats[1]                         # the resumed MTP head drafts as a fresh one does


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_a_resumed_picture_prompt_leaves_a_fresh_prefills_caches(kv_dtype):
    """Past a picture, the resumed MTP tail rotates at the prompt's image positions, as a fresh prefill's does."""
    from test_flashnext_forward import _rows

    w = _model()
    start = 9
    first = list(range(20, 60))
    first[start:start + 4] = [17] * 4
    second = first + list(range(70, 93))
    states = []
    for warm in (True, False):
        dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, prefill_rows=16,
                           vision=KEYED_TOWER, stop_eos=False)
        if warm:
            turn = Stream(first, 4, vision=_prepared(w, first, start, 9, "a"), stop_eos=False)
            dec.admit(turn)
            _drain(dec, turn)
        follow = Stream(second, 1, vision=_prepared(w, second, start, 9, "a"), stop_eos=False)   # no drafts after
        dec.admit(follow)
        assert follow.cached == (len(first) - 1 if warm else 0)
        _drain(dec, follow)
        states.append(follow.st)
    a, b = states
    assert a.pos == b.pos == len(second) and a.mtp_len == b.mtp_len
    pairs = [(x, y, a.pos) for x, y in zip(a.kc, b.kc)] + [(a.mtp_kc, b.mtp_kc, a.mtp_len)]
    for x, y, n in pairs:
        for t, u in zip(_rows(x, n), _rows(y, n)):
            assert torch.equal(t.contiguous().view(torch.uint8), u.contiguous().view(torch.uint8))


@pytest.mark.parametrize("change", ["picture", "grid", "text"])
def test_a_different_picture_or_text_resumes_only_the_text_before_it(change):
    w = _model()
    start, point = 280, 256
    first = list(range(1, 320))
    first[start:start + 4] = [17] * 4
    second = first + list(range(400, 420))
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=1, prefill_rows=128, points=lambda ids: [point],
                       vision=KEYED_TOWER, stop_eos=False)
    turn = Stream(first, 8, vision=_prepared(w, first, start, 9, "a"), stop_eos=False)
    dec.admit(turn)
    _drain(dec, turn)
    assert sorted(len(k[0]) for k in dec.kept) == [point, len(first) - 1]
    if change == "text":                                 # the same ids, no picture: placeholder rows are no match
        follow, image = Stream(second, 8, stop_eos=False), None
    else:
        prepared = _prepared(w, second, start, 11, "b") if change == "picture" else _prepared(w, second, start, 9, "a")
        if change == "grid":                             # the same pixels resized to another grid
            prepared.image_grid_thw = ((1, 2, 8),)
        follow, image = Stream(second, 8, vision=prepared, stop_eos=False), prepared.encoded
    dec.admit(follow)
    assert follow.cached == point
    _drain(dec, follow)
    assert follow.out == _reference(w, second, 8, None, image)


def test_a_picture_follows_a_text_prompts_kept_message_start():
    w = _model()
    start, point = 280, 256
    text = list(range(1, 300))
    pictured = list(range(1, 330))
    pictured[start:start + 4] = [17] * 4
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=1, prefill_rows=128, points=lambda ids: [point],
                       vision=KEYED_TOWER, stop_eos=False)
    first = Stream(text, 8, stop_eos=False)
    dec.admit(first)
    _drain(dec, first)
    follow = Stream(pictured, 8, vision=_prepared(w, pictured, start, 5, "c"), stop_eos=False)
    dec.admit(follow)
    assert follow.cached == point
    _drain(dec, follow)
    assert follow.out == _reference(w, pictured, 8, None, _picture(w, pictured, start, 5))
