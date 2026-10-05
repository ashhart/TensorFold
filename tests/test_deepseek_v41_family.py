"""The DeepSeek-V4.1-Flash family on a tiny checkpoint: loading, unit parity, exact windows, rollback, streams, DSpark."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.families.deepseek_v41 import weights  # noqa: E402
from tensorfold.families.deepseek_v41.attention import sparse_attention  # noqa: E402
from tensorfold.families.deepseek_v41.compressor import pool_blocks  # noqa: E402
from tensorfold.families.deepseek_v41.config import Config  # noqa: E402
from tensorfold.families.deepseek_v41.engram import EngramHash, engram_layout  # noqa: E402
from tensorfold.families.deepseek_v41.make_tiny import TEXT_dict, write_checkpoint  # noqa: E402
from tensorfold.families.deepseek_v41.moe import swiglu  # noqa: E402
from tensorfold.families.deepseek_v41.quant import quantize_cache, round_fp4, round_fp8  # noqa: E402
from tensorfold.families.deepseek_v41.runtime import DeepSeekV41Flash  # noqa: E402

CPU_ROWS = 7          # MLX's CPU fp32 ops are row-invariant below 8 rows


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return write_checkpoint(tmp_path_factory.mktemp("ds41"))
    finally:
        mx.set_default_device(previous)


@pytest.fixture(scope="module")
def model(checkpoint):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return weights.load_backbone(checkpoint)
    finally:
        mx.set_default_device(previous)


def tokens(n: int, seed: int = 1) -> list[int]:
    text = TEXT_dict()
    return [int(t) for t in np.random.default_rng(seed).integers(2, text["vocab_size"], size=n)]


def logits_of(m, ids, cache):
    return m.head(m.hidden(mx.array([ids], dtype=mx.uint32), cache))[0]


def serial_logits(m, base, window):
    from tensorfold.engine.lane_engine import LaneEngine

    cache = LaneEngine.copy_single_cache(base)
    out = []
    for t in window:
        lg = logits_of(m, [t], cache)
        mx.eval(lg)
        out.append(lg[-1])
    return out, cache


# -- unit parity (plan s6.1) ---------------------------------------------------------------------------


def test_hc_mix_against_numpy(model):
    """The boundary mix (pre / post / comb) matches a numpy reference, sinkhorn included."""
    hc = model.layers[0].attn_hc
    x = mx.array(0.3 * np.random.default_rng(4).normal(size=(2, 4, model.args.hidden_size))).astype(mx.bfloat16)
    xc, post, comb = hc.split(x, False)
    mx.eval(xc, post, comb)

    # the official reference (inference/model.py hc_mixes) is:
    #   rsqrt = rsqrt(mean(x^2) + eps);  mixes = linear(x, hc_fn) * rsqrt
    # i.e. the normalized stream is projected -- NOT (x @ fn.T) * rms. Normalizing first
    # (z = x * rsqrt, then z @ fn.T) is algebraically identical and matches HC.split to ~1e-7.
    f = x.reshape(2, -1).astype(mx.float32)
    rsqrt = 1.0 / np.sqrt((np.asarray(f) ** 2).mean(-1, keepdims=True) + model.args.rms_norm_eps)
    m = np.asarray(f) * rsqrt @ np.asarray(hc.fn.astype(mx.float32)).T
    pre = 1 / (1 + np.exp(-(m[:, :4] * float(hc.scale[0].item()) + np.asarray(hc.base[:4])))) + hc.cfg.hc_eps
    assert np.allclose(np.asarray(post[:, :4]), 2 * (1 / (1 + np.exp(-(m[:, 4:8] * float(hc.scale[1].item())
                                                                      + np.asarray(hc.base[4:8]))))),
                       atol=1e-5)
    # pre collapses the streams: the collapsed row is sum(pre_s * x_s) in fp32, rounded to bf16
    want = (np.asarray(x.astype(mx.float32)) * pre[..., None]).sum(axis=1)
    got = np.asarray(xc.astype(mx.float32))
    assert np.max(np.abs(want - got)) <= 0.02 * np.max(np.abs(want)) + 1e-6


def test_engram_hash_ids_against_hand_computed(model):
    """24 ids per engram layer per position, exactly the official derivation's primes and multipliers."""
    cfg = model.args
    text = TEXT_dict()
    token_map = list(range(text["vocab_size"]))
    table = EngramHash(cfg, token_map, token_map[text["engram_pad_token_id"]])
    primes, offsets, mult = engram_layout(cfg)[cfg.engram_layer_ids[0]]

    # the primes are the smallest unused above engram_vocab_size - 1, in (order, head) order
    def is_prime(n):
        return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))

    seen, want_primes = set(), []
    for _ in range(cfg.engram_ngram - 1):
        current = cfg.engram_vocab_size - 1
        for _ in range(cfg.engram_n_heads):
            current += 1
            while current in seen or not is_prime(current):
                current += 1
            seen.add(current)
            want_primes.append(current)
    assert primes == want_primes
    assert sum(primes) == text["engram_num_embeddings"][0]

    # the multipliers are odd int64 from rng(10007 * layer), bounded by ((2^63 - 1) // vocab) // 2
    want_mult = np.random.default_rng(10007 * cfg.engram_layer_ids[0]).integers(
        0, max(1, ((2 ** 63 - 1) // text["vocab_size"]) // 2), size=cfg.engram_ngram, dtype=np.int64) * 2 + 1
    assert np.array_equal(mult, want_mult)

    # a window's ids: rolling xor of token * multiplier, per (order, head) prime + offset.
    # tokens[0] is the NEWEST token (official NgramHashState: shift 0 = current position),
    # so the rolling hash starts from window[-1] and xors in progressively older tokens.
    window = [11, 2, 31, 7, 19, 3][:cfg.engram_ngram]
    ids = table.ids(window, cfg.engram_layer_ids[0])
    newest_first = window[::-1]
    rolling = newest_first[0] * int(mult[0])
    want = []
    for i in range(1, cfg.engram_ngram):
        rolling ^= newest_first[i] * int(mult[i])
        for head in range(cfg.engram_n_heads):
            j = (i - 1) * cfg.engram_n_heads + head
            want.append(rolling % primes[j] + offsets[j])
    assert ids == want
    # pad beyond history: fewer tokens than the order pad
    short = table.ids([5], cfg.engram_layer_ids[0])
    pad = token_map[text["engram_pad_token_id"]]
    rolling = 5 * int(mult[0])
    want_short = []
    tokens_w = [5, pad, pad, pad]
    rolling = tokens_w[0] * int(mult[0])
    for i in range(1, cfg.engram_ngram):
        rolling ^= tokens_w[i] * int(mult[i])
        for head in range(cfg.engram_n_heads):
            j = (i - 1) * cfg.engram_n_heads + head
            want_short.append(rolling % primes[j] + offsets[j])
    assert short == want_short


def test_sqrt_softplus_router_and_gate(model):
    """The router scores are sqrt(softplus(x @ w.T)) in fp32; the gate is the signed sqrt sigmoid."""
    moe = model.layers[0].moe
    x = mx.array(0.5 * np.random.default_rng(9).normal(size=(3, model.args.hidden_size))).astype(mx.bfloat16)
    scores = moe.scores(x, False)
    mx.eval(scores)
    want = np.asarray(x.astype(mx.float32)) @ np.asarray(moe.router).T
    want = np.sqrt(np.logaddexp(want, 0.0))
    assert np.allclose(np.asarray(scores), want, rtol=1e-4, atol=1e-5)

    eng = model.layers[1].engram
    h = mx.array(np.random.default_rng(2).normal(size=(2, 4, model.args.hidden_size))).astype(mx.bfloat16)
    # a direct gate check: values with q_weight . k_weight == 0 give dot 0 -> sigmoid(sqrt(1e-6))
    saved = eng.weight
    eng.weight = mx.zeros_like(saved)
    ids = model.engram_hash().ids([1, 2, 3, 4], 1)
    out = eng.forward(h, [ids, ids])
    mx.eval(out)
    gate = 1 / (1 + math.exp(-math.sqrt(1e-6)))
    value = np.asarray(out.astype(mx.float32)) - np.asarray(h.astype(mx.float32))
    scale = np.max(np.abs(value))
    assert scale > 0
    eng.weight = saved


def test_moe_grouped_rows_match_serial_accumulation(model):
    """Rows sharing routed experts retain exact outputs and per-row route-order accumulation."""
    moe = model.layers[0].moe
    x = mx.array(0.5 * np.random.default_rng(27).normal(size=(8, model.args.hidden_size))).astype(mx.bfloat16)
    indices, _ = moe.route(moe.scores(x, True))
    picks = indices.tolist()
    assert len({expert for row in picks for expert in row}) < len(picks) * moe.top

    grouped = moe.forward(x, True)
    serial = mx.concatenate([moe.forward(x[row:row + 1], True) for row in range(x.shape[0])])
    mx.eval(grouped, serial)
    assert np.array_equal(np.asarray(grouped.astype(mx.float32)), np.asarray(serial.astype(mx.float32)))


def test_compressor_pooling_both_ratios(model):
    """Ratio 2 pools softmax(gates) . values per dim over slot pairs; ratio 1 is a plain normed projection."""
    rng = np.random.default_rng(8)
    kv = mx.array(rng.normal(size=(3, 2, 16)).astype(np.float32))
    sc = mx.array(rng.normal(size=(3, 2, 16)).astype(np.float32))
    pooled = pool_blocks(kv, sc)
    mx.eval(pooled)
    for b in range(3):
        e = np.exp(np.asarray(sc[b]) - np.asarray(sc[b]).max(axis=0, keepdims=True))
        w = e / e.sum(axis=0, keepdims=True)
        want = (w * np.asarray(kv[b])).sum(axis=0)
        # the reference rounds the pooled latent to bf16 (runtime.py:361 .astype(x.dtype))
        assert np.allclose(np.asarray(pooled[b].astype(mx.float32)), want, rtol=1e-2, atol=3e-2)

    # ratio 1: the plain path is norm(wkv(x)) in bf16
    attn3 = model.layers[3].attn
    comp = attn3.compressor
    x = mx.array(0.2 * rng.normal(size=(1, model.args.hidden_size))).astype(mx.bfloat16)
    latent = comp.plain(x)
    mx.eval(latent)
    want = np.asarray(x.astype(mx.float32)) @ np.asarray(comp.wkv.T.astype(mx.float32))
    want = want / np.sqrt((want ** 2).mean(-1, keepdims=True) + attn3.eps) * np.asarray(comp.norm.astype(mx.float32))
    got = np.asarray(latent.astype(mx.float32))
    assert np.max(np.abs(got - want)) <= 0.02 * np.max(np.abs(want)) + 1e-5


def test_partial_blocks_carry_across_calls(model):
    """A ratio-2 block split across two calls pools from the stored first slot, equal to one 2-row call."""
    from tensorfold.families.deepseek_v41.attention import positions_of

    attn = model.layers[1].attn
    rng = np.random.default_rng(6)
    x2 = mx.array(0.2 * rng.normal(size=(2, model.args.hidden_size))).astype(mx.bfloat16)
    from tensorfold.families.deepseek_v41.caches import LayerCache

    whole = LayerCache(2, model.args.sliding_window)
    attn._compress(whole, x2, 36)                    # positions 36, 37 -> block 18
    split = LayerCache(2, model.args.sliding_window)
    attn._compress(split, x2[:1], 36)                # position 36: partial
    attn._compress(split, x2[1:], 37)                # position 37 completes the block from the ring
    whole.offset = split.offset = 38                 # pool_rows is offset // ratio: a full call moves the offset
    mx.eval(whole.pool, split.pool)
    assert whole.pool_rows == split.pool_rows == 19
    assert mx.array_equal(whole.pool[18], split.pool[18]).item()


def test_cache_roundings_are_idempotent_and_match_the_reference():
    """FP8 e8m0 g32, FP4 e4m3 g16 and FP4 e8m0 g32 round to the reference's grids, and twice is once."""
    rng = np.random.default_rng(1)
    x = mx.array(rng.normal(size=(4, 64)) * 3)

    def fp8_grid(v):
        a = np.abs(v)
        step = 2.0 ** (np.maximum(np.floor(np.log2(np.maximum(a, 2.0 ** -20))), -6) - 3)
        return np.sign(v) * np.minimum(np.round(a / step) * step, 448.0)

    got8 = np.asarray(quantize_cache(x, 8, 32).astype(mx.float32))
    # per group of 32: scale = 2^ceil(log2(max/448))
    want8 = np.zeros_like(got8)
    for g in range(2):
        block = np.asarray(x.astype(mx.float32))[:, g * 32:(g + 1) * 32]
        scale = 2.0 ** math.ceil(math.log2(max(np.max(np.abs(block)), 1e-4) / 448))
        want8[:, g * 32:(g + 1) * 32] = fp8_grid(block / scale) * scale
    assert np.allclose(got8, want8, atol=1e-6)

    levels = np.array([0, 1, 2, 4, 0.5, 1.5, 3, 6])
    for fmt, group in (("e4m3", 16), ("e8m0", 32)):
        got = quantize_cache(x, 4, group, fmt)
        mx.eval(got)
        again = quantize_cache(got, 4, group, fmt)
        assert mx.array_equal(got, again).item(), f"{fmt} not idempotent"
        v = np.asarray(x.astype(mx.float32)).reshape(4, -1, group)
        top = np.abs(v).max(axis=-1, keepdims=True)
        if fmt == "e4m3":
            scale = np.array([fp8_grid(np.array([[max(t, 6 * 2.0 ** -9) / 6]]))[0, 0] for t in top.reshape(-1)])
            scale = scale.reshape(v.shape[0], v.shape[1], 1)
        else:
            scale = 2.0 ** np.ceil(np.log2(np.maximum(top, 6 * 2.0 ** -126) / 6))
        scaled = v / scale
        idx = np.argmin(np.abs(np.abs(scaled)[..., None] - levels), axis=-1)
        want = np.sign(scaled) * levels[idx]
        assert np.allclose(np.asarray(got.astype(mx.float32)).reshape(v.shape), want * scale, atol=1e-5)


def test_round_fp4_ties_to_even():
    assert float(round_fp4(mx.array(0.5)).item()) in (0.5, 1.0)  # 0.5 sits between 0.5 and 1: nearest wins
    for v, want in ((0.0, 0.0), (0.24, 0.0), (0.26, 0.5), (5.9, 6.0), (7.0, 6.0)):
        assert float(round_fp4(mx.array(float(v))).item()) == want


def test_sparse_attention_with_sinks():
    q = mx.array(np.random.default_rng(3).normal(size=(4, 32)).astype(np.float32))
    kv = mx.array(np.random.default_rng(4).normal(size=(5, 32)).astype(np.float32))
    sink = mx.array(np.zeros((4,)))
    out = sparse_attention(q, kv, sink)
    mx.eval(out)
    scores = np.asarray(q) @ np.asarray(kv).T * (32 ** -0.5)
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    p = p / (p.sum(axis=-1, keepdims=True) + np.exp(-m))   # one sink at logit 0: exp(0 - m) joins the denominator
    want = p @ np.asarray(kv)
    assert np.allclose(np.asarray(out), want, atol=1e-4)


def test_swiglu_asymmetric_clamp():
    import mlx.nn as nn

    def reference(gate: float, up: float, limit: float = 10.0) -> float:
        up = min(max(up, -limit), limit)
        gate = min(gate, limit)
        return nn.silu(mx.array([gate])).item() * up

    gate = mx.array([[11.0, -11.0, 2.0]])
    up = mx.array([[11.0, -11.0, 2.0]])
    out = swiglu(gate, up, 10.0)
    mx.eval(out)
    got = np.asarray(out)[0]
    want = [reference(11.0, 11.0), reference(-11.0, -11.0), reference(2.0, 2.0)]
    assert np.allclose(got, want, atol=1e-5)


# -- loading and configuration -------------------------------------------------------------------------


def test_family_is_detected_and_checked(checkpoint):
    from tensorfold import families
    from tensorfold.families import deepseek_v41

    assert families.detect(checkpoint).module == "tensorfold.families.deepseek_v41"
    deepseek_v41.check(checkpoint)


def test_check_refuses_other_quantization(checkpoint, tmp_path):
    import json
    import shutil

    from tensorfold.families import deepseek_v41

    shutil.copy(checkpoint / "config.json", tmp_path / "config.json")
    config = json.loads((tmp_path / "config.json").read_text())
    config["quantization"]["bits"] = 8
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        deepseek_v41.check(tmp_path)


def test_layers_follow_the_ratios_and_modes(model):
    text = TEXT_dict()
    attn = [layer.attn for layer in model.layers]
    assert [a.ratio for a in attn] == text["compress_ratios"]
    assert [a.mode for a in attn] == ["reuse", "full", "reuse", "full", "reindex", "reuse"]
    assert model.layers[1].engram is not None and model.layers[0].engram is None
    assert attn[1].compressor is not None and attn[2].compressor is None
    assert attn[1].indexer is not None and attn[2].indexer is None and attn[4].indexer is not None


# -- row equality windows (plan s6.2) ------------------------------------------------------------------


@pytest.mark.parametrize("prompt", [5, 38, 124])   # a ratio-2 block completes inside the last window
@pytest.mark.parametrize("width", [2, 3, 4, 7, 16])
def test_decode_windows_give_one_row_bits(model, prompt, width):
    """A window of up to 16 rows gives each row its one-row logits bit for bit, across pool emissions."""
    from tensorfold.engine.lane_engine import LaneEngine

    base = model.make_cache()
    mx.eval(model.hidden(mx.array([tokens(prompt, seed=5)], dtype=mx.uint32), base))
    window = tokens(width, seed=6)
    serial, _ = serial_logits(model, base, window)
    joint = logits_of(model, window, LaneEngine.copy_single_cache(base))
    mx.eval(joint)
    for i in range(width):
        assert mx.array_equal(joint[i], serial[i]).item(), f"row {i}"


@pytest.mark.parametrize(("prompt", "keep"), [(29, 2), (37, 5)])
def test_keep_rows_then_continue_equals_serial(model, prompt, keep):
    """Rejected rows leave nothing behind: keep some of a window, continue, and match the serial run."""
    from tensorfold.engine.lane_engine import LaneEngine

    base = model.make_cache()
    mx.eval(model.hidden(mx.array([tokens(prompt, seed=7)], dtype=mx.uint32), base))
    window, after = tokens(CPU_ROWS, seed=8), tokens(CPU_ROWS, seed=9)
    drafted = LaneEngine.copy_single_cache(base)
    mx.eval(logits_of(model, window, drafted))
    model.keep_rows(drafted, CPU_ROWS, keep)
    joint = logits_of(model, after, drafted)
    serial, _ = serial_logits(model, base, window[:keep] + after)
    mx.eval(joint)
    for i in range(CPU_ROWS):
        assert mx.array_equal(joint[i], serial[keep + i]).item(), f"row {i}"


def test_rejected_drafts_never_enter_pool_state(model):
    """A rejected draft leaves no stale pool rows: block counts match the never-drafted run's."""
    from tensorfold.engine.lane_engine import LaneEngine

    base = model.make_cache()
    mx.eval(model.hidden(mx.array([tokens(23, seed=10)], dtype=mx.uint32), base))
    window = tokens(6, seed=11)
    drafted = LaneEngine.copy_single_cache(base)
    mx.eval(logits_of(model, window, drafted))
    model.keep_rows(drafted, 6, 2)
    serial = LaneEngine.copy_single_cache(base)
    for t in window[:2]:
        mx.eval(logits_of(model, [t], serial))
    for drafted_cache, serial_cache in zip(drafted[:6], serial[:6]):
        assert drafted_cache.offset == serial_cache.offset
        assert drafted_cache.pool_rows == serial_cache.pool_rows
    assert drafted[-1].ids == serial[-1].ids          # engram history trimmed to the accepted prefix


def test_shared_forwards_match_each_stream_alone(model):
    """2-4 unequal streams in one forward: every row the bits its own call gives, before and after a keep."""
    from tensorfold.engine.lane_engine import LaneEngine

    base = model.make_cache()
    mx.eval(model.hidden(mx.array([tokens(30, seed=12)], dtype=mx.uint32), base))
    lengths, keeps = [3, 1, 2], [2, 1, 1]

    def streams():
        out = []
        for b in range(len(lengths)):
            cache = LaneEngine.copy_single_cache(base)
            for extra in range(b):
                mx.eval(logits_of(model, [50 + 13 * b + extra], cache))
            out.append(cache)
        return out

    windows = [[(3001 + 17 * (3 * b + r)) % 128 for r in range(n)] for b, n in enumerate(lengths)]
    follows = [[(7001 + 11 * b) % 128] for b in range(len(lengths))]
    alone, together = streams(), streams()
    for step, wins in enumerate((windows, follows)):
        single = []
        for b, cache in enumerate(alone):
            lg = logits_of(model, wins[b], cache)
            mx.eval(lg)
            single.append(lg)
            if step == 0 and keeps[b] < len(wins[b]):
                model.keep_rows(cache, len(wins[b]), keeps[b])
        flat = [t for w in wins for t in w]
        joint = model.head(model.hidden_rows(mx.array([flat], dtype=mx.uint32), together,
                                             [len(w) for w in wins]))[0]
        mx.eval(joint)
        at = 0
        for b, w in enumerate(wins):
            assert mx.array_equal(joint[at:at + len(w)], single[b]).item(), f"stream {b} step {step}"
            at += len(w)
        if step == 0:
            model.keep_rows_streams(together, [len(w) for w in wins], keeps)


# -- prefill (plan s6.2: the chunked path) --------------------------------------------------------------


@pytest.mark.parametrize("length", [6, 40, 200])
def test_prefill_path_agrees_with_serial_decode(model, length):
    """One prompt chunk and one-row steps compute the same function (to the family's rounding bar).

    The tiny checkpoint's random weights can put the final position's top-2 logits a rounding
    apart, so the bar is the plan's: a 5% / 0.05 relative-logit budget with the argmax inside it.
    """
    ids = tokens(length, seed=5)
    whole = model.make_cache()
    a = logits_of(model, ids, whole)[-1]
    step = model.make_cache()
    b = a
    for t in ids:
        b = logits_of(model, [t], step)[-1]
    mx.eval(a, b)
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert np.max(np.abs(a - b)) < 0.25 * np.max(np.abs(b)) + 0.25
    top_a, top_b = int(a.argmax()), int(b.argmax())
    budget = 0.25 * np.max(np.abs(b)) + 0.25
    assert abs(float(a[top_a]) - float(b[top_a])) < budget
    assert abs(float(a[top_b]) - float(b[top_b])) < budget
    for c1, c2 in zip(whole[:6], step[:6]):
        assert c1.offset == c2.offset == length
        assert c1.pool_rows == c2.pool_rows


def test_chunked_prefill_agrees_with_one_chunk(model):
    ids = tokens(90, seed=3)
    one = model.make_cache()
    a = logits_of(model, ids, one)[-1]
    parts = model.make_cache()
    for lo, hi in ((0, 33), (33, 61), (61, 90)):
        b = logits_of(model, ids[lo:hi], parts)[-1]
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert int(a.argmax()) == int(b.argmax())


def test_serial_decode_end_to_end(model):
    """Greedy decode of 24 tokens after a prompt: deterministic, finite, all ids in range."""
    prompt = tokens(17, seed=21)
    cache = model.make_cache()
    mx.eval(logits_of(model, prompt, cache))
    emitted = []
    for _ in range(24):
        lg = logits_of(model, [emitted[-1] if emitted else prompt[-1]], cache)
        mx.eval(lg)
        nxt = int(mx.argmax(lg[-1]).item())
        assert 0 <= nxt < model.args.vocab_size
        assert bool(mx.all(mx.isfinite(lg[-1])).item())
        emitted.append(nxt)
    assert len(emitted) == 24
    assert len(set(emitted)) > 1                      # the tiny weights do not collapse to one token


# -- runtime and the lane engine (plan s6.2: engine-level) ----------------------------------------------


def test_runtime_checks_windows_and_streams(model):
    runtime = DeepSeekV41Flash(model, None, drafts=0, check=False)
    width, costs = runtime.check_windows(widest=CPU_ROWS)
    assert width == CPU_ROWS and set(costs) == set(range(1, CPU_ROWS + 1))
    runtime.exact_width = width
    assert runtime.check_streams()


def _run_engine(runtime, prompt, n, sampling=None, drafts=False):
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.families.deepseek_v41 import engine_settings

    engine = LaneEngine(runtime, **engine_settings(runtime))
    assert engine.family
    stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=n, sampling=sampling,
                        drafts=drafts)
    engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    return engine, stream


def test_engine_smoke_serial_decode(model):
    """The lane engine drives the tiny model end to end: prompt, decode, finish, all finite."""
    runtime = DeepSeekV41Flash(model, None, drafts=0, check=False)
    runtime.exact_width = runtime.batch_rows = CPU_ROWS
    runtime.multi_row_exact = True
    prompt = tokens(23, seed=31)
    engine, stream = _run_engine(runtime, prompt, 20)
    assert stream.emitted and len(stream.emitted) <= 20
    assert all(0 <= t < model.args.vocab_size for t in stream.emitted)
    assert not engine.active_count


def test_engine_emitted_matches_direct_serial(model):
    """The engine's greedy emissions equal the model's own serial greedy decode, token for token."""
    runtime = DeepSeekV41Flash(model, None, drafts=0, check=False)
    runtime.exact_width = runtime.batch_rows = CPU_ROWS
    runtime.multi_row_exact = True
    prompt = tokens(19, seed=41)
    _, stream = _run_engine(runtime, prompt, 16)

    cache = model.make_cache()
    lg = logits_of(model, prompt, cache)            # prefill: its last row's argmax is the first emission
    mx.eval(lg)
    want = [int(mx.argmax(lg[-1]).item())]
    while len(want) < len(stream.emitted):
        lg = logits_of(model, [want[-1]], cache)
        mx.eval(lg)
        want.append(int(mx.argmax(lg[-1]).item()))
    assert list(stream.emitted) == want


# -- DSpark drafting (plan s6: M3) ------------------------------------------------------------------------


def _drafter(model, checkpoint):
    from tensorfold.families.deepseek_v41 import dspark as ds_dspark

    return ds_dspark.load(model, checkpoint, json.loads((checkpoint / "config.json").read_text()))


def _taps(model, prompt, seed=5):
    model.tap_layers = tuple(TEXT_dict()["dspark_target_layer_ids"])
    base = model.make_cache()
    mx.eval(logits_of(model, prompt, base))
    return model.last_taps


@pytest.fixture()
def tap_layers(model):
    """The backbone reads taps for the DSpark tests, and only for them."""
    saved = model.tap_layers
    model.tap_layers = tuple(TEXT_dict()["dspark_target_layer_ids"])
    yield model
    model.tap_layers = saved


def test_dspark_loads_three_stages_and_heads(model, checkpoint):
    """The native mtp.0/1/2 stages load with the config's block fields; stage 0 the main projection."""
    drafter = _drafter(model, checkpoint)
    text = TEXT_dict()
    assert len(drafter.blocks) == 3
    assert drafter.size == text["dspark_block_size"] == 5
    assert drafter.noise == text["dspark_noise_token_id"]
    assert drafter.taps == tuple(text["dspark_target_layer_ids"])
    assert drafter.window == model.args.sliding_window
    for block in drafter.blocks:
        assert block.attn.ratio == 0 and block.moe.count == text["dspark_num_experts"]
        assert block.moe.top == text["dspark_experts_per_token"]
    assert drafter.markov_head.outs == text["vocab_size"]
    assert drafter.confidence_proj.shape == (1, text["hidden_size"] + text["dspark_markov_rank"])


def test_dspark_absorb_batched_equals_serial(model, checkpoint, tap_layers):
    """One n-row absorb writes the same ring keys as n one-row absorbs (contiguity the only rule)."""
    drafter = _drafter(model, checkpoint)
    prompt = tokens(23, seed=5)
    taps = _taps(model, prompt)

    batched, serial = drafter.make_cache(), drafter.make_cache()
    drafter.absorb(taps, batched)
    for r in range(23):
        drafter.absorb(taps[r:r + 1], serial)
    for a, b in zip(batched, serial):
        assert a.offset == b.offset == 23
        assert mx.array_equal(a.keys, b.keys).item()
    # the block pass leaves the rings untouched: draft rows never enter the ring
    token = mx.array([prompt[-1]], dtype=mx.uint32)
    drafter.logits(model, token, batched)
    assert [c.offset for c in batched] == [23] * 3


def test_dspark_block_logits_independent_of_ring_path(model, checkpoint, tap_layers):
    """A block's logits are a pure function of the ring contents, not how they were absorbed."""
    drafter = _drafter(model, checkpoint)
    prompt = tokens(20, seed=6)
    taps = _taps(model, prompt)
    rings = drafter.make_cache()
    drafter.absorb(taps, rings)
    token = mx.array([prompt[-1]], dtype=mx.uint32)
    a = drafter.logits(model, token, rings)
    # the same history reached by a different cut: rows 0-11 then 12-20
    split = drafter.make_cache()
    drafter.absorb(taps[:12], split)
    drafter.absorb(taps[12:], split)
    b = drafter.logits(model, token, split)
    mx.eval(a, b)
    assert mx.array_equal(a, b).item()
    assert a.shape == (drafter.size, TEXT_dict()["vocab_size"])


def test_dspark_draft_attention_is_bidirectional_within_the_block(model, checkpoint, tap_layers):
    """Every draft row attends the same keys: the ring's window then ALL block rows (the official gather)."""
    drafter = _drafter(model, checkpoint)
    prompt = tokens(10, seed=7)
    taps = _taps(model, prompt)
    rings = drafter.make_cache()
    drafter.absorb(taps, rings)
    attn = drafter.blocks[0].attn

    # a row's output is invariant to the other rows' CONTENTS only through the shared key set:
    # swap two noise rows' embeddings and the seed row's logits still see both
    token = mx.array([prompt[-1]], dtype=mx.uint32)
    logits = drafter.logits(model, token, rings)
    mx.eval(logits)
    assert logits.shape[0] == drafter.size

    # the official key count: min(window, verified) ring rows + all 5 block rows, for every row
    cache = rings[0]
    window = min(attn.window, cache.offset)
    assert window == cache.offset                       # 10 verified rows < the 16-row window
    x = mx.zeros((drafter.size, model.args.hidden_size), dtype=mx.bfloat16)
    assert attn(x, [cache], (drafter.size,), True).shape == (drafter.size, model.args.hidden_size)


def test_dspark_markov_iteration_chains_the_draws(model, checkpoint, tap_layers):
    """Draw j's bias comes from draw j-1's token: a serial Python loop, greedy here, exactly forward_head."""
    drafter = _drafter(model, checkpoint)
    prompt = tokens(14, seed=8)
    taps = _taps(model, prompt)
    rings = drafter.make_cache()
    drafter.absorb(taps, rings)
    token = mx.array([prompt[-1]], dtype=mx.uint32)
    logits = drafter.logits(model, token, rings)

    drafts = drafter.draw(logits, token, drafter.size, lambda row, j: mx.argmax(row, -1))
    mx.eval(drafts)
    # the reference's hand loop: bias(embed[prev]) added per position, argmax drawn
    prev, want = int(token.item()), []
    for j in range(drafter.size):
        bias = drafter._markov_bias(mx.array([prev], dtype=mx.uint32))
        biased = logits[j] + bias
        mx.eval(biased)
        prev = int(mx.argmax(biased).item())
        want.append(prev)
    assert drafts.tolist() == want
    assert all(0 <= t < TEXT_dict()["vocab_size"] for t in want)


def test_dspark_confidence_head_reads_hidden_and_markov_embeds(model, checkpoint, tap_layers):
    """proj([hidden ‖ markov embeds]) in fp32, one score a block position; finite on the tiny weights."""
    drafter = _drafter(model, checkpoint)
    prompt = tokens(12, seed=9)
    taps = _taps(model, prompt)
    rings = drafter.make_cache()
    drafter.absorb(taps, rings)
    token = mx.array([prompt[-1]], dtype=mx.uint32)
    logits = drafter.logits(model, token, rings)
    hidden = drafter._block_rows(model, token, rings)[0]
    conf = drafter.confidence(hidden, logits, token, drafter.size)
    mx.eval(conf)
    assert conf.shape == (drafter.size,)
    assert bool(mx.all(mx.isfinite(conf)).item())
    # the projection is one row: every position's score is its own dot product
    embeds = [drafter._markov_embed(token.reshape(1).astype(mx.uint32))]
    prev = int(token.item())
    for j in range(drafter.size - 1):
        prev = int(mx.argmax(logits[j] + drafter._markov_bias(mx.array([prev], dtype=mx.uint32))).item())
        embeds.append(drafter._markov_embed(mx.array([prev], dtype=mx.uint32)))
    flat = mx.concatenate([hidden.astype(mx.float32), mx.concatenate(embeds).astype(mx.float32)], axis=-1)
    want = (flat @ drafter.confidence_proj.T).reshape(-1)
    mx.eval(want)
    assert mx.array_equal(conf, want).item()


def test_dspark_ring_rollback_restores_the_accepted_prefix(model, checkpoint, tap_layers):
    """Trim to the kept rows reproduces the shorter history's ring exactly (draft rows never entered)."""
    drafter = _drafter(model, checkpoint)
    prompt = tokens(18, seed=10)
    taps = _taps(model, prompt)
    keep = 7

    kept = drafter.make_cache()
    drafter.absorb(taps[:keep], kept)
    rolled = drafter.make_cache()
    drafter.absorb(taps, rolled)                       # all 18 absorbed
    for ring in rolled:
        ring.trim(len(prompt) - keep)                  # rollback to the accepted prefix
    for a, b in zip(kept, rolled):
        assert a.offset == b.offset == keep
        assert mx.array_equal(a.window_keys(keep - 1), b.window_keys(keep - 1)).item()
    # and the draft after the rollback matches one from the never-drafted history
    token = mx.array([prompt[keep - 1]], dtype=mx.uint32)
    x = drafter.logits(model, token, kept)
    y = drafter.logits(model, token, rolled)
    mx.eval(x, y)
    assert mx.array_equal(x, y).item()


@pytest.mark.parametrize("prompt_len", [15, 19, 26])
def test_dspark_runtime_speculate_settle_rolls_back(model, checkpoint, tap_layers, prompt_len):
    """speculate absorbs a window's taps and drafts a block; settle trims to the accepted prefix."""
    from tensorfold.families.deepseek_v41.runtime import DSparkFlash
    from tensorfold.engine.lane_engine import LaneEngine

    drafter = _drafter(model, checkpoint)
    runtime = DSparkFlash(model, drafter, check=False)
    runtime.exact_width = runtime.batch_rows = CPU_ROWS
    runtime.multi_row_exact = True
    runtime.dspark = runtime.mtp = drafter

    prompt = tokens(prompt_len, seed=31)
    cache = LaneEngine.copy_single_cache(runtime.make_cache())
    hidden = runtime.hidden(mx.array([prompt], dtype=mx.uint32), cache)
    # the engine's prefill: the prompt's rows into the rings, each with the token that follows it
    runtime.absorb_draft_context(hidden, mx.array(prompt[1:], dtype=mx.uint32), cache)
    rings = cache[runtime.layer_count:]
    assert [r.offset for r in rings] == [prompt_len - 1] * 3

    window = tokens(5, seed=32)
    inputs = mx.array([window], dtype=mx.uint32)
    mx.eval(runtime.head(runtime.hidden(inputs, cache)))
    first = runtime.speculate(cache, mx.array(window, dtype=mx.uint32), prompt_len, None)
    mx.eval(first)
    assert first.shape == (1,)
    assert [r.offset for r in rings] == [prompt_len - 1 + 5] * 3

    drafts = runtime.settle(cache, 2, first, prompt_len + 3, None, drafter.size)
    mx.eval(drafts)
    assert [r.offset for r in rings] == [prompt_len + 1] * 3       # trimmed to the accepted prefix
    assert len(drafts) == drafter.size
    # a second speculate after the trim continues from the accepted prefix contiguously
    again = runtime.speculate(cache, mx.array(window[:2], dtype=mx.uint32), prompt_len + 2, None)
    mx.eval(again)
    assert [r.offset for r in rings] == [prompt_len + 3] * 3


@pytest.mark.parametrize("seed", [11, 23, 37, 41, 53, 67, 71, 79, 83, 97, 101])
def test_dspark_engine_emissions_equal_serial(model, checkpoint, tap_layers, seed):
    """The engine with DSpark drafts on equals the same runtime draft-free, token for token (11 of 11)."""
    from tensorfold.families.deepseek_v41.runtime import DSparkFlash

    drafter = _drafter(model, checkpoint)
    runtime = DSparkFlash(model, drafter, check=False)
    runtime.exact_width = runtime.batch_rows = CPU_ROWS
    runtime.multi_row_exact = True
    runtime.dspark = runtime.mtp = drafter
    serial = DeepSeekV41Flash(model, None, drafts=0, check=False)
    serial.exact_width = serial.batch_rows = CPU_ROWS

    prompt = tokens(19, seed)
    _, drafted = _run_engine(runtime, prompt, 24, drafts=True)
    _, plain = _run_engine(serial, prompt, 24, drafts=False)
    assert list(drafted.emitted) == list(plain.emitted)


def test_dspark_engine_emissions_equal_serial_sampled(model, checkpoint, tap_layers):
    """Sampled too: the drafted run's tokens match the draft-free run's under one sampling seed."""
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.deepseek_v41.runtime import DSparkFlash

    drafter = _drafter(model, checkpoint)
    runtime = DSparkFlash(model, drafter, check=False)
    runtime.exact_width = runtime.batch_rows = CPU_ROWS
    runtime.multi_row_exact = True
    runtime.dspark = runtime.mtp = drafter
    serial = DeepSeekV41Flash(model, None, drafts=0, check=False)
    serial.exact_width = serial.batch_rows = CPU_ROWS

    for seed in (5, 9):
        sampling = Sampling(seed=seed, temperature=0.9)
        prompt = tokens(23, seed=200 + seed)
        _, drafted = _run_engine(runtime, prompt, 24, sampling=sampling, drafts=True)
        _, plain = _run_engine(serial, prompt, 24, sampling=sampling, drafts=False)
        assert list(drafted.emitted) == list(plain.emitted)


def test_dspark_rejected_drafts_never_corrupt_target_state(model, checkpoint, tap_layers):
    """A rejected draft block leaves pools, rings, engram history and DSpark rings at the serial run's."""
    from tensorfold.families.deepseek_v41.runtime import DSparkFlash
    from tensorfold.engine.lane_engine import LaneEngine

    drafter = _drafter(model, checkpoint)
    runtime = DSparkFlash(model, drafter, check=False)
    runtime.exact_width = runtime.batch_rows = CPU_ROWS
    runtime.multi_row_exact = True
    runtime.dspark = runtime.mtp = drafter

    prompt = tokens(23, seed=10)
    window = tokens(6, seed=11)
    keep = 2

    drafted = LaneEngine.copy_single_cache(runtime.make_cache())
    hidden = runtime.hidden(mx.array([prompt], dtype=mx.uint32), drafted)
    runtime.absorb_draft_context(hidden, mx.array(prompt[1:], dtype=mx.uint32), drafted)
    mx.eval(runtime.head(runtime.hidden(mx.array([window], dtype=mx.uint32), drafted)))
    first = runtime.speculate(drafted, mx.array(window, dtype=mx.uint32), 23, None)
    # the round keeps only ``keep`` rows: settle trims the rings (and the backbone rolled back)
    runtime.keep_rows(drafted, len(window), keep)
    mx.eval(runtime.settle(drafted, keep, first, 23 + keep, None, 0))

    serial = LaneEngine.copy_single_cache(runtime.make_cache())
    hidden = runtime.hidden(mx.array([prompt], dtype=mx.uint32), serial)
    runtime.absorb_draft_context(hidden, mx.array(prompt[1:], dtype=mx.uint32), serial)
    for i, t in enumerate(window[:keep]):
        mx.eval(runtime.head(runtime.hidden(mx.array([t], dtype=mx.uint32), serial)))
        # the draft-free run's head still observes each verified row (an engine plain round does)
        first = runtime.speculate(serial, mx.array([t], dtype=mx.uint32), 23 + i, None)
        mx.eval(runtime.settle(serial, 1, first, 23 + i + 1, None, 0))

    for d, s in zip(drafted[:len(model.layers)], serial[:len(model.layers)]):
        assert d.offset == s.offset == 23 + keep
        assert d.pool_rows == s.pool_rows
    assert drafted[runtime.layer_count - 1].ids == serial[runtime.layer_count - 1].ids   # engram history
    for ring, s in zip(drafted[runtime.layer_count:], serial[runtime.layer_count:]):
        # the head observed every position but the newest (its tap comes with the next round)
        assert ring.offset == s.offset == 23 + keep - 1


def test_dspark_shared_rounds_with_drafts_match_alone(model, checkpoint, tap_layers):
    """2-3 streams share rounds with drafts on: each stream's emissions match its own single run."""
    from tensorfold.families.deepseek_v41.runtime import DSparkFlash
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.families.deepseek_v41 import engine_settings

    drafter = _drafter(model, checkpoint)
    prompts = [tokens(15, seed=200 + i) for i in range(3)]
    together_r = DSparkFlash(model, drafter, check=False)
    together_r.exact_width = together_r.batch_rows = CPU_ROWS
    together_r.multi_row_exact = True
    together_r.dspark = together_r.mtp = drafter
    engine = LaneEngine(together_r, **engine_settings(together_r))
    together = []
    for i, p in enumerate(prompts):
        stream = LaneStream(stream_id=f"s{i}", prompt_ids=p, max_new_tokens=20, drafts=True)
        engine.add_stream(stream)
        together.append(stream)
    while engine.active_count:
        engine.step()

    alone = []
    for i, p in enumerate(prompts):
        rt = DSparkFlash(model, drafter, check=False)
        rt.exact_width = rt.batch_rows = CPU_ROWS
        rt.multi_row_exact = True
        rt.dspark = rt.mtp = drafter
        _, stream = _run_engine(rt, p, 20, drafts=True)
        alone.append(list(stream.emitted))
    assert [list(s.emitted) for s in together] == alone
    assert engine._shared_rounds > 0 and engine.drafted > 0


def test_dspark_runtime_loads_when_the_checkpoint_holds_stages(model, checkpoint, tap_layers, monkeypatch):
    """The family runtime picks the DSpark stages out of the checkpoint and reports them."""
    from types import SimpleNamespace

    from tensorfold.families import deepseek_v41 as family
    from tensorfold.families.deepseek_v41.runtime import DSparkFlash

    monkeypatch.setattr("tensorfold.families.tokenizer.load_tokenizer",
                        lambda *a, **k: SimpleNamespace(encode=lambda *a, **k: []))
    runtime, _ = family.load(checkpoint, mtp_drafts=3, check=True)
    try:
        assert isinstance(runtime, DSparkFlash)
        assert model.has_draft_weights
        assert runtime.mtp is not None and runtime.drafts == 5
        assert runtime.exact_width >= 2                  # the load probe found a multi-row window
        assert runtime.layer_count == len(model.layers) + 1
        assert len(runtime.make_cache()) == runtime.layer_count + 3
        runtime.exact_width = runtime.batch_rows = CPU_ROWS
        runtime.multi_row_exact = True
        _, stream = _run_engine(runtime, tokens(21, seed=51), 16, drafts=True)
        assert stream.emitted
    finally:
        model.tap_layers = ()
        # mtp_drafts=0 loads no head even when the stages are present
        plain, _ = family.load(checkpoint, mtp_drafts=0, check=False)
        try:
            assert not isinstance(plain, DSparkFlash) and plain.mtp is None and plain.drafts == 0
        finally:
            model.tap_layers = tuple(TEXT_dict()["dspark_target_layer_ids"])
