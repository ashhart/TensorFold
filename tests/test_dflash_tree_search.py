"""DFlash2's best-first draft tree: the search the proposer runs, pinned to the reference algorithm."""

import heapq

import numpy as np
import pytest

from tensorfold.drafters.dflash_drafter import best_first_tree


def _reference(cands, unary, hproj, noise, anchor, pred, succ, temp, edge, noise_weight, tau, kids, nodes):
    def children(token, depth):
        edges = succ[cands[depth]].astype(np.float64) @ (pred[token] * hproj[depth])
        if noise is not None:
            edges = edges / temp
        s = unary[depth] + edge * edges
        if noise is not None:
            s = s + noise_weight * noise[depth]
        s = s / tau
        s = s - s.max()
        return s - np.log(np.exp(s).sum())

    tokens, parents, heap = [], [], []
    first = children(anchor, 0)
    for i in np.argsort(-first)[:kids]:
        heapq.heappush(heap, (-first[i], -1, int(cands[0][i]), 0))
    while heap and len(tokens) < nodes:
        neg, parent, token, depth = heapq.heappop(heap)
        tokens.append(token)
        parents.append(parent)
        me = len(tokens) - 1
        if depth + 1 < cands.shape[0]:
            ls = children(token, depth + 1)
            for i in np.argsort(-ls)[:kids]:
                heapq.heappush(heap, (neg - ls[i], me, int(cands[depth + 1][i]), depth + 1))
    return tokens, parents


def test_tree_matches_reference_search():
    rng = np.random.default_rng(0)
    vocab, rank = 5000, 32
    pred = rng.normal(size=(vocab, rank)).astype(np.float32)
    succ = rng.normal(size=(vocab, rank)).astype(np.float32)
    for trial in range(40):
        depth = int(rng.integers(2, 16))
        cands = np.stack([rng.choice(vocab, size=16, replace=False) for _ in range(depth)])
        unary = rng.normal(size=(depth, 16)) * 3
        hproj = rng.normal(size=(depth, rank)) * 0.3
        sampled = trial % 2 == 0
        noise = rng.gumbel(size=(depth, 16)) if sampled else None
        temp = 1.0 if sampled else 1.0
        kw = dict(temp=temp, edge=0.6, noise_weight=0.7, tau=1.5, kids=4, nodes=15)
        want = _reference(cands, unary, hproj, noise, 7, pred, succ, **kw)
        got = best_first_tree(cands, unary, hproj, noise, 7, pred, succ, temperature=temp, edge=0.6,
                              noise_weight=0.7, tau=1.5, children=4, max_nodes=15)
        assert got == want


def test_parents_come_first_and_budget_holds():
    rng = np.random.default_rng(1)
    pred = rng.normal(size=(100, 8)).astype(np.float32)
    succ = rng.normal(size=(100, 8)).astype(np.float32)
    cands = np.stack([rng.choice(100, size=16, replace=False) for _ in range(6)])
    tokens, parents = best_first_tree(cands, rng.normal(size=(6, 16)), rng.normal(size=(6, 8)), None, 3, pred, succ,
                                      temperature=1.0, edge=0.6, noise_weight=0.7, tau=1.5, children=4, max_nodes=9)
    assert len(tokens) == 9 and all(p < i for i, p in enumerate(parents))


# -- the session n-gram prior -------------------------------------------------------------------


def _reference_with_prior(cands, unary, hproj, noise, anchor, pred, succ, prior, context, temp, edge, noise_weight,
                          tau, kids, nodes):
    """The offline study's tree: each expansion's log-softmax, plus the bonus, renormalized."""

    def children(token, depth, history):
        edges = succ[cands[depth]].astype(np.float64) @ (pred[token] * hproj[depth])
        if noise is not None:
            edges = edges / temp
        s = unary[depth] + edge * edges
        if noise is not None:
            s = s + noise_weight * noise[depth]
        s = s / tau
        s = s - s.max()
        s = s - np.log(np.exp(s).sum())
        s = s + prior(tuple(history), depth)
        s = s - s.max()
        return s - np.log(np.exp(s).sum())

    tokens, parents, hist, heap = [], [], [], []
    base = list(context[-3:])
    first = children(anchor, 0, base)
    for i in np.argsort(-first)[:kids]:
        heapq.heappush(heap, (-first[i], -1, int(cands[0][i]), 0))
    while heap and len(tokens) < nodes:
        neg, parent, token, depth = heapq.heappop(heap)
        h = (hist[parent] if parent >= 0 else base) + [token]
        tokens.append(token)
        parents.append(parent)
        hist.append(h[-3:])
        if depth + 1 < cands.shape[0]:
            ls = children(token, depth + 1, h[-3:])
            for i in np.argsort(-ls)[:kids]:
                heapq.heappush(heap, (neg - ls[i], len(tokens) - 1, int(cands[depth + 1][i]), depth + 1))
    return tokens, parents


def _lattice(rng, vocab=300, rank=16, depth=8):
    pred = rng.normal(size=(vocab, rank)).astype(np.float32)
    succ = rng.normal(size=(vocab, rank)).astype(np.float32)
    cands = np.stack([rng.choice(vocab, size=16, replace=False) for _ in range(depth)])
    return pred, succ, cands, rng.normal(size=(depth, 16)) * 3, rng.normal(size=(depth, rank)) * 0.3


def test_tree_with_a_prior_matches_the_offline_study():
    from tensorfold.drafters.draft_ngram import SessionNGram

    rng = np.random.default_rng(2)
    for trial in range(20):
        pred, succ, cands, unary, hproj = _lattice(rng)
        noise = rng.gumbel(size=cands.shape) if trial % 2 else None
        # a context that repeats lattice tokens, so every n-gram order has something to say
        context = [int(t) for t in rng.choice(cands.ravel(), size=400)]
        model = SessionNGram()
        model.update(context)
        prior = model.rescorer(cands.astype(np.int64), 0.3)
        want = _reference_with_prior(cands, unary, hproj, noise, context[-1], pred, succ, prior, context, 1.0, 0.6,
                                     0.7, 1.5, 4, 15)
        got = best_first_tree(cands, unary, hproj, noise, context[-1], pred, succ, temperature=1.0, edge=0.6,
                              noise_weight=0.7, tau=1.5, children=4, max_nodes=15, prior=prior, history=context[-3:])
        assert got == want
        plain = best_first_tree(cands, unary, hproj, noise, context[-1], pred, succ, temperature=1.0, edge=0.6,
                                noise_weight=0.7, tau=1.5, children=4, max_nodes=15)
        zero = best_first_tree(cands, unary, hproj, noise, context[-1], pred, succ, temperature=1.0, edge=0.6,
                               noise_weight=0.7, tau=1.5, children=4, max_nodes=15,
                               prior=lambda history, depth: np.zeros(16), history=context[-3:])
        assert zero == plain                               # a flat bonus changes nothing, bit for bit


def test_lattice_gain_is_the_true_tokens_log_softmax_change():
    from tensorfold.drafters.dflash_drafter import lattice_gain

    rng = np.random.default_rng(3)
    pred, succ, cands, unary, hproj = _lattice(rng)
    noise = rng.gumbel(size=cands.shape)
    table = {}

    def prior(history, depth):
        return table.setdefault((history, depth), rng.normal(size=16))

    truth = [int(cands[d][int(rng.integers(0, 16))]) for d in range(5)] + [10**6]   # leaves the lattice at 5
    context = [11, 12, 13]
    gain = lattice_gain(cands, unary, hproj, noise, 13, pred, succ, truth, prior, context, temperature=1.0, edge=0.6,
                        noise_weight=0.7, tau=1.5)
    want, hist, parent = 0.0, (11, 12, 13), 13
    for d in range(5):
        j = list(cands[d]).index(truth[d])
        s = (unary[d] + 0.6 * (succ[cands[d]].astype(np.float64) @ (pred[parent] * hproj[d])) + 0.7 * noise[d]) / 1.5
        b = s + prior(hist, d)
        want += (b[j] - np.log(np.exp(b).sum())) - (s[j] - np.log(np.exp(s).sum()))
        hist, parent = (hist[1], hist[2], truth[d]), truth[d]
    assert abs(gain - want) < 1e-9


def _fake_proposer(monkeypatch, lattices, vocab=300, rank=16, **attrs):
    """A DFlashProposer whose drafter forward is replaced by the given lattices, one per round."""

    import types

    mx = pytest.importorskip("mlx.core")
    from tensorfold.drafters.dflash_drafter import DFlashProposer

    rng = np.random.default_rng(4)
    drafter = types.SimpleNamespace(
        model=types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=vocab)),
        _codes=(rng.normal(size=(vocab, rank)).astype(np.float32), rng.normal(size=(vocab, rank)).astype(np.float32)),
        _trim=lambda cache, n: None)
    prop = object.__new__(DFlashProposer)
    prop.__dict__.update(dict(drafter=drafter, sampling=None, cache=[types.SimpleNamespace(offset=0)],
                              context=mx.zeros((1, 1, 4)), ready=True, copy=None, last_confident=False, proposals=0,
                              proposed_tokens=0, accepted_tokens=0, copy_rounds=0, draft_ms=0.0, build_ms=0.0,
                              wait_ms=0.0, search_ms=0.0, _last_was_copy=False, _copy_level=0, _ngram=None,
                              _ngram_last=None, _ngram_gain=0.0, ngram_rounds=0, ngram_ms=0.0, trace_path=""))
    prop.__dict__.update(attrs)
    rounds = iter(lattices)

    def fake_lattice(context, block):
        cands, unary, hproj = next(rounds)
        return mx.array(cands.astype(np.int32)), mx.array(unary.astype(np.float32)), mx.array(hproj[None].astype(np.float32))

    monkeypatch.setattr(prop, "_lattice", fake_lattice)
    return prop


def test_proposer_prior_is_off_at_weight_zero_and_gated_by_its_record(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from tensorfold.drafters.draft_ngram import SessionNGram

    rng = np.random.default_rng(5)
    phrase = [int(t) for t in rng.choice(300, size=40, replace=False)]
    context = phrase * 20                                  # text that repeats: the n-gram predicts it well
    lattices = []
    for r in range(12):
        n = len(context) + 3 * r
        full = (phrase * 40)[: n + 15]
        cands = np.stack([rng.permutation([t for t in range(300) if t != full[n + d]])[:16] for d in range(15)])
        for d in range(15):
            cands[d][int(rng.integers(0, 16))] = full[n + d]            # the truth is in the lattice ...
        unary = rng.normal(size=(15, 16)) * 2.0                          # ... but the drafter is unsure
        lattices.append((cands, unary, rng.normal(size=(15, 16)) * 0.3))
    pred, succ = None, None
    outputs = {}
    for label, attrs in (("off", dict(ngram_weight=0.0)), ("always", dict(ngram_weight=0.1, ngram_gate=None)),
                         ("gated", dict(ngram_weight=0.1, ngram_gate=1.0))):
        prop = _fake_proposer(monkeypatch, lattices, **attrs)
        pred, succ = prop.drafter._codes
        trees, gains = [], []
        for r in range(12):
            ctx = (phrase * 40)[: len(context) + 3 * r]
            prop.cache[0].offset = len(ctx) - 1
            prop.context = mx.zeros((1, 1, 4))               # the kept rows' taps (on_rows)
            trees.append(prop.propose_tree(ctx, 15))
            gains.append(prop._ngram_gain)
        outputs[label] = (trees, gains, prop)
    plain = []
    model = SessionNGram(vocab=300)                        # the fake drafter's vocabulary
    with_prior = []
    for r in range(12):
        ctx = (phrase * 40)[: len(context) + 3 * r]
        cands, unary, hproj = lattices[r]
        args = (cands.astype(np.int64), unary.astype(np.float32).astype(np.float64),
                hproj.astype(np.float32).astype(np.float64), None, ctx[-1], pred, succ)
        kw = dict(temperature=1.0, edge=0.6, noise_weight=0.7, tau=1.5, children=4, max_nodes=15)
        plain.append(best_first_tree(*args, **kw))
        model.update(ctx)
        with_prior.append(best_first_tree(*args, **kw, prior=model.rescorer(args[0], 0.1), history=ctx[-3:]))
    off_trees, _, off = outputs["off"]
    assert off_trees == plain and off._ngram is None and "ngram_rounds" not in off.telemetry()
    assert outputs["always"][0] == with_prior and with_prior != plain
    trees, gains, gated = outputs["gated"]
    assert trees[0] == plain[0] and gains[0] == 0.0          # no record yet: the plain tree
    assert gains[-1] > 1.0                                    # the prior kept raising the tokens that came
    # a round's tree uses the prior when the record, scored on the rounds before it, is above the gate
    on = [gains[r] > 1.0 for r in range(12)]
    assert 0 < sum(on) < 12
    assert all(trees[r] == (with_prior[r] if on[r] else plain[r]) for r in range(12))
    assert gated.telemetry()["ngram_rounds"] == sum(on)


def test_pipelined_forward_is_the_models_forward_bit_for_bit():
    """``DFlashProposer._lattice`` (the graph sent to the GPU in pieces) against ``hidden_states`` in one piece."""

    import types

    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    pytest.importorskip("mlx_lm")
    from tensorfold.drafters import dflash_drafter
    from tensorfold.engine.topk import topk_rows

    vendor = dflash_drafter._vendor()
    config = vendor.DFlashConfig(hidden_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                                 head_dim=32, intermediate_size=256, vocab_size=1024, rms_norm_eps=1e-6,
                                 rope_theta=10000.0, max_position_embeddings=4096, block_size=8,
                                 target_layer_ids=(1, 3), num_target_layers=4, mask_token_id=1000,
                                 layer_types=("sliding_attention",) * 3, sliding_window=48, conv_kernel_size=2,
                                 conv_group_size=16, selector_rank=16, selector_top_k=16, is_causal=False)
    mx.random.seed(3)
    model = vendor.DFlash2DraftModel(config)
    model.set_dtype(mx.bfloat16)
    target = types.SimpleNamespace(embed_tokens=nn.Embedding(1024, 128), lm_head=nn.Linear(128, 1024, bias=False))
    target.embed_tokens.set_dtype(mx.bfloat16)
    target.lm_head.set_dtype(mx.bfloat16)
    model.bind(target)
    mx.eval(model.parameters(), target.embed_tokens.parameters(), target.lm_head.parameters())
    drafter = object.__new__(dflash_drafter.DFlashDrafter)
    drafter.model, drafter.target, drafter.block_size, drafter.mask_id = model, target, 8, 1000
    drafter.window, drafter._trim = 47, vendor._trim_recent_cache
    ours, theirs = drafter.proposer(), drafter.proposer()
    ours.async_layers = (-1, 0, 1)
    taps = (mx.random.normal((1, 40, 2 * 128)) * 0.5).astype(mx.bfloat16)
    context = [int(t) for t in np.random.default_rng(6).integers(0, 1000, size=41)]
    for prop in (ours, theirs):
        prop.prefill_taps(len(context) - 1, taps)
    for step in range(6):
        rows = 1 + (5 * step) % 9                          # the sliding window fills and rotates
        more = (mx.random.normal((1, rows, 2 * 128)) * 0.5).astype(mx.bfloat16)
        context += [int(t) for t in np.random.default_rng(step).integers(0, 1000, size=rows)]
        ours.absorb(more)
        theirs.absorb(more)
        got = ours._lattice(context, 8)
        inputs = mx.array([[context[-1]] + [1000] * 7])
        hidden = model.hidden_states(inputs, theirs.context, theirs.cache, 1)
        logits, _ = drafter.candidate_logits(hidden)
        cands, unary = topk_rows(logits[0], 16)
        want = (cands, unary, model.candidate_selector.hidden_projection(hidden).astype(mx.float32))
        mx.eval(*got, *want)
        for a, b in zip(got, want):
            assert a.shape == b.shape and a.dtype == b.dtype
            assert np.array_equal(np.array(a), np.array(b))
        for prop in (ours, theirs):
            prop.context = None
            extra = int(prop.cache[0].offset) - (len(context) - 1)
            if extra > 0:
                drafter._trim(prop.cache, extra)


def test_drafter_cache_takes_one_row_updates_past_its_window() -> None:
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import RotatingKVCache

    from tensorfold.drafters.dflash_drafter import concat_updates

    def rows(n: int, value: float):
        return mx.full((1, 2, n, 4), value, dtype=mx.float32)

    plain, fixed = RotatingKVCache(max_size=15, keep=0), concat_updates([RotatingKVCache(max_size=15, keep=0)])[0]
    for cache in (plain, fixed):
        cache.update_and_fetch(rows(6, 1.0), rows(6, 1.0))
        cache.offset = 400                  # prefill_taps: the absolute position of the next row
        cache.keys, cache.values = cache.keys[..., :4, :], cache.values[..., :4, :]   # a trimmed block
        cache._idx = 4
    with pytest.raises(ValueError):
        plain.update_and_fetch(rows(1, 2.0), rows(1, 2.0))        # the one-row round that drafted nothing
    keys, _ = fixed.update_and_fetch(rows(1, 2.0), rows(1, 2.0))
    assert fixed.offset == 401 and keys.shape[2] == 5 and float(keys[0, 0, -1, 0]) == 2.0
    keys, _ = fixed.update_and_fetch(rows(16, 3.0), rows(16, 3.0))   # and the window still rotates
    assert keys.shape[2] <= 15 + 16 - 1 and float(keys[0, 0, -1, 0]) == 3.0

