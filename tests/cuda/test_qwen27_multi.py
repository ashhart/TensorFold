"""Several streams in one verify window: each row gets its own stream's bits, each stream its serial tokens."""

import random
import threading

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.scheduler import Scheduler  # noqa: E402
from tensorfold.cuda.streams import PrefixCache, Stream, accept  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen3_5.cuda.decode import prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen3_5.cuda.forward import (State, commit, commit_streams, multi_tree_forward,  # noqa: E402
                                                        tree_forward)
from tensorfold.families.qwen3_5.cuda.draft_tree import allocate  # noqa: E402
from tensorfold.families.qwen3_5.cuda.multi import TREE, MultiDecoder, private, viewed  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import Attention, Config, GDN, Layer, QLinear, Weights  # noqa: E402

V = 256


def _model():
    gen = torch.Generator(device="cuda").manual_seed(11)
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
    return Weights(config, qlinear(V, 128), layers, norm, qlinear(V, 128), torch.ones(16, device=dev))


def _tok(ids):
    return torch.tensor(ids, device="cuda", dtype=torch.int32)


def _prefilled(w, prompt):
    st = State(w)
    _, record = tree_forward(w, _tok(prompt), list(range(-1, len(prompt) - 1)), st)
    commit(st, record, list(range(len(prompt))))
    return st


def _same_state(a, b):
    assert a.pos == b.pos
    for x, y in zip(a.rec, b.rec):
        assert (x is None) == (y is None) and (x is None or torch.equal(x, y))
    for x, y in zip(a.conv, b.conv):
        assert (x is None) == (y is None) and (x is None or torch.equal(x, y))
    for x, y in zip(a.kv, b.kv):
        assert (x is None) == (y is None)
        if x is not None:
            assert torch.equal(x[0][:a.pos], y[0][:b.pos]) and torch.equal(x[1][:a.pos], y[1][:b.pos])


def test_multi_forward_rows_equal_their_streams_windows():
    w = _model()
    prompts = [[5, 6, 7], [9, 10, 11, 12, 13, 14, 15], [3]]
    trees = [([20, 21, 22, 23], [-1, 0, 0, 1]), ([30, 31], [-1, 0]), ([40, 41, 42, 43, 44], [-1, 0, 1, 2, 3])]
    paths = [[0, 1, 3], [0], [0, 1, 2]]
    states = [_prefilled(w, p) for p in prompts]
    logits, record, _, starts = multi_tree_forward(w, [(t, p, st) for (t, p), st in zip(trees, states)])
    assert starts == [0, 4, 6, 11]
    for s, ((tokens, parents), st) in enumerate(zip(trees, states)):
        ref = private(st, st.pos + 16)
        single, rec = tree_forward(w, _tok(tokens), parents, ref)
        assert torch.equal(logits[starts[s]:starts[s + 1]], single), s
        commit(ref, rec, paths[s])
        mine = private(st, st.pos + 16)
        commit(mine, record, [starts[s] + r for r in paths[s]])
        _same_state(mine, ref)


class _Oracle(MultiDecoder):
    """Proposes trees mixing each prompt's true continuation with wrong tokens, so rounds keep runs of drafts."""

    def __init__(self, w, truth, seed=0, **kw):
        super().__init__(w, None, max_rows=6, **kw)
        self.truth, self.rng = truth, random.Random(seed)

    def _mode(self, s, copied):
        mode = super()._mode(s, copied)
        return TREE if s.draft and not copied.get(s.sid) else mode

    def _trees(self, plan, blocks):
        out = {}
        for sid, mode, _, _ in plan:
            if mode != TREE:
                continue
            s = self.streams[sid]
            ref, n = self.truth[tuple(s.prompt)], len(s.out)
            guesses, parents = [], []
            for d in range(self.rng.randint(1, 4)):
                guesses.append(ref[n + d] if n + d < len(ref) and self.rng.random() < 0.8 else self.rng.randrange(1, V))
                parents.append(d - 1)
            if self.rng.random() < 0.5:                   # a sibling of the first draft
                guesses.append(self.rng.randrange(1, V))
                parents.append(-1)
            out[sid] = (guesses, parents, [0.3 * (k + 1) for k in range(len(guesses))])
        return out


def _serial(w, prompt, sampling, count):
    st, first = prefill(w, prompt, sampling)
    return serial_decode(w, st, first, count, sampling).tokens


PROMPTS = [[5, 6, 7], [9, 10, 11, 12, 13], [3, 4], [7, 7, 8], [1, 2, 3, 4, 5, 6]]
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95), Sampling(99, 0.8, 0, 1.0), Sampling(5, 1.0, 20, 0.95), None]


@pytest.mark.parametrize("serial_too,curve", [(False, False), (True, False), (False, True)])
def test_decoder_streams_equal_serial(serial_too, curve):
    w = _model()
    refs = {tuple(p): _serial(w, p, smp, 24) for p, smp in zip(PROMPTS[:4], SAMPLINGS)}
    dec = _Oracle(w, refs)
    if curve:                                           # widths from a cost curve: trees cut to their best prefix
        dec.costs = [(1, 10.0), (8, 11.0), (16, 16.0), (32, 40.0)]
    streams = []
    for i, (prompt, sampling) in enumerate(zip(PROMPTS[:4], SAMPLINGS)):
        got: list[int] = []
        s = Stream(prompt, 24, sampling, draft=not (serial_too and i == 3), emit=lambda new, got=got: got.extend(new))
        dec.admit(s)
        streams.append((s, got))
    while dec.live():
        dec.finish(dec.round())
    for s, got in streams:
        assert got == refs[tuple(s.prompt)] and s.out == got, s.prompt
        assert s.min_rows >= (2 if s.draft else 1), (s.prompt, s.min_rows)
        assert s.rounds < 23 or not s.draft


def test_scheduler_serves_concurrent_requests_exactly():
    w = _model()
    refs = {tuple(p): _serial(w, p, smp, 20) for p, smp in zip(PROMPTS, SAMPLINGS)}
    sched = Scheduler(_Oracle(w, refs, seed=3), max_streams=3)
    results: dict[int, tuple] = {}

    def go(i):
        got: list[int] = []
        stats = sched.submit(PROMPTS[i], 20, SAMPLINGS[i], draft=True, emit=lambda new: got.extend(new) or False)
        results[i] = (got, stats)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(len(PROMPTS))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert {i: results[i][0] for i in results} == {i: refs[tuple(PROMPTS[i])] for i in range(len(PROMPTS))}
    assert all(stats["min_rows"] >= 2 for _, stats in results.values())
    # a prompt extending a finished reply resumes from that request's prompt entry (all but its last token) and
    # decodes a fresh prefill's tokens
    longer = PROMPTS[1] + refs[tuple(PROMPTS[1])][:-1] + [42, 43]
    want = _serial(w, longer, SAMPLINGS[1], 12)
    sched.decoder.truth[tuple(longer)] = want
    got: list[int] = []
    stats = sched.submit(longer, 12, SAMPLINGS[1], draft=True, emit=lambda new: got.extend(new) or False)
    assert got == want and stats["cached"] == len(PROMPTS[1]) - 1, stats
    serial: list[int] = []
    stats = sched.submit(longer, 12, SAMPLINGS[1], draft=False, emit=lambda new: serial.extend(new) or False)
    assert serial == want and stats["cached"] == 0 and stats["min_rows"] == 1


def test_a_failed_prompt_end_copy_fails_only_its_request(monkeypatch):
    """One request's prompt-end copy runs out of memory: it gets the error, the others their serial tokens."""

    w = _model()
    refs = {tuple(p): _serial(w, p, smp, 20) for p, smp in zip(PROMPTS, SAMPLINGS)}
    doomed = [2, 9, 4, 4, 1, 8, 8]                      # no other prompt has its length: only its copy fails

    def failing(st):                                    # the prompt-end entry sits one token before the end
        if st.pos == len(doomed) - 1:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated at the prompt-end entry)")
        return viewed(st)

    monkeypatch.setattr("tensorfold.families.qwen3_5.cuda.multi.viewed", failing)
    dec = _Oracle(w, refs, seed=5)
    sched = Scheduler(dec, max_streams=3)
    results: dict = {}

    def go(key, prompt, sampling):
        got: list[int] = []
        try:
            results[key] = (got, sched.submit(prompt, 20, sampling, draft=True,
                                              emit=lambda new: got.extend(new) or False))
        except Exception as exc:                        # noqa: BLE001
            results[key] = (got, exc)

    batches = [[(i, PROMPTS[i], SAMPLINGS[i]) for i in range(3)] + [("doomed", doomed, None)],
               [(i, PROMPTS[i], SAMPLINGS[i]) for i in (3, 4)]]           # the second after the failure
    for batch in batches:
        threads = [threading.Thread(target=go, args=args, daemon=True) for args in batch]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
    assert len(results) == len(PROMPTS) + 1, sorted(map(str, results))
    got, err = results["doomed"]
    assert got == [] and isinstance(err, torch.OutOfMemoryError), err
    for i, prompt in enumerate(PROMPTS):
        got, stats = results[i]
        assert isinstance(stats, dict) and got == refs[tuple(prompt)], (i, stats)
    assert sched.thread.is_alive() and not dec.streams
    assert all(entry[0] != doomed for entry in dec.cache.entries)


def test_allocate_prices_drafts_against_the_curve():
    flat = lambda rows: 10.0                            # noqa: E731  (rows cost nothing: every draft pays)
    assert allocate([[0.1, 0.5, 3.0], [0.2]], 2, 2.0, flat) == [3, 1]
    steep = lambda rows: 10.0 * rows                    # noqa: E731  (every row costs a full forward)
    assert allocate([[0.1, 0.5, 3.0], [0.2]], 2, 2.0, steep) == [1, 1]
    # a cheap step past 4 rows: the good second node of tree 0 pays, the unlikely third does not
    stepped = lambda rows: 10.0 if rows <= 5 else 30.0  # noqa: E731
    assert allocate([[0.1, 0.5, 3.0], [0.2]], 2, 2.0, stepped) == [2, 1]
    assert allocate([[], [0.4]], 1, 1.0, flat) == [0, 1]


def test_context_bounds_each_stream():
    w = _model()
    dec = MultiDecoder(w, context=24)
    with pytest.raises(ValueError, match="no room in the 24-token context"):
        dec.admit(Stream(list(range(1, 24)), 5))
    s = Stream(PROMPTS[1], 100, SAMPLINGS[1])
    dec.admit(s)
    need = len(PROMPTS[1]) + s.count                     # admission sizes the stream's caches once, in full
    assert all(kv is None or kv[0].shape[0] == need for kv in s.st.kv)
    while dec.live():
        dec.finish(dec.round())
    assert s.count == 24 - len(PROMPTS[1]) - 1 and s.out == _serial(w, PROMPTS[1], SAMPLINGS[1], s.count)
    assert all(kv is None or kv[0].shape[0] == need <= 24 for kv in s.st.kv)


def test_accept_and_prefix_cache():
    # window: root 0, drafts 1 (child of 0) and 2 (child of 1), 3 (sibling of 1)
    tokens, parents = [10, 11, 12, 13], [-1, 0, 1, 0]
    assert accept(tokens, parents, [11, 12, 99, 5], room=8) == ([0, 1, 2], 99)
    assert accept(tokens, parents, [13, 0, 0, 7], room=8) == ([0, 3], 7)
    assert accept(tokens, parents, [11, 12, 99, 5], room=2) == ([0, 1], 12)
    assert accept(tokens, parents, [11, 12, 99, 5], room=8, eos=(12,)) == ([0, 1], 12)
    cache = PrefixCache(keep=2)
    cache.add([1, 2], "a", None)
    cache.add([1, 2, 3], "b", None)
    assert cache.longest([1, 2, 3, 4])[1] == "b" and cache.longest([1, 2, 3])[1] == "a"
    assert cache.longest([1, 2]) is None and cache.named([1, 2, 3, 4], 2)[1] == "a"
    cache.add([7], "c", None)
    assert [e[1] for e in cache.entries] == ["a", "c"]          # the last hit ("a") outlives the older "b"
    cache.add([8], "d", None)
    assert [e[1] for e in cache.entries] == ["a", "d"]          # an entry resumed from outlives newer ones never hit


def test_commit_streams_equals_each_stream_alone():
    """One GDN replay launch for every stream leaves each stream's state as its own commit does."""

    w = _model()
    prompts = [[5, 6, 7], [9, 10, 11, 12, 13, 14, 15], [3]]
    trees = [([20, 21, 22, 23], [-1, 0, 0, 1]), ([30, 31], [-1, 0]), ([40, 41, 42, 43, 44], [-1, 0, 1, 2, 3])]
    paths = [[0, 1, 3], [0], [0, 1, 2, 3, 4]]
    together = [_prefilled(w, p) for p in prompts]
    alone = [_prefilled(w, p) for p in prompts]
    in_place = [_prefilled(w, p) for p in prompts]
    held = [[r for r in st.rec] for st in in_place]
    _, record, _, starts = multi_tree_forward(w, [(t, p, st) for (t, p), st in zip(trees, together)])
    rows = [[starts[k] + r for r in path] for k, path in enumerate(paths)]
    commit_streams(together, record, rows)
    commit_streams(in_place, record, rows, in_place=True)
    for k in range(len(prompts)):
        commit(alone[k], record, rows[k])
        _same_state(together[k], alone[k])
        _same_state(in_place[k], alone[k])
        assert all(a is b for a, b in zip(in_place[k].rec, held[k]))     # the same tensors, overwritten


def _points_after(k):
    """Snapshot points for the test prompts: position ``k`` when a prompt is longer."""

    return lambda ids: [k] if len(ids) > k + 1 else []


@pytest.mark.parametrize("step", [1024, 3])
def test_prompts_resume_a_shared_start_and_equal_fresh(monkeypatch, step):
    """A prompt resuming another's kept state at a message start decodes a fresh prefill's tokens, prefill steps interleaved or not."""

    from tensorfold.cuda import markers
    from tensorfold.families.qwen3_5.cuda import multi

    monkeypatch.setattr(multi, "STEP", step)
    monkeypatch.setattr(multi, "MIN_GAP", 2)
    monkeypatch.setattr(markers, "MIN_GAP", 2)
    w = _model()
    shared = [11, 12, 13, 14, 15, 16]
    prompts = [shared + [21, 22, 23], shared + [31, 32], shared + [41, 42, 43, 44, 45], [7, 8, 9, 10, 11]]
    refs = {tuple(p): _serial(w, p, smp, 16) for p, smp in zip(prompts, SAMPLINGS)}
    dec = _Oracle(w, refs, seed=5, points=_points_after(len(shared)))
    first = Stream(prompts[0], 16, SAMPLINGS[0])
    dec.admit(first)
    while dec.live():
        dec.finish(dec.round())
    assert first.out == refs[tuple(prompts[0])] and first.cached == 0
    assert any(e[0] == shared for e in dec.cache.entries)
    rest = []
    for prompt, sampling in zip(prompts[1:], SAMPLINGS[1:]):
        s = Stream(prompt, 16, sampling)
        dec.admit(s)
        rest.append(s)
    while dec.live():
        dec.finish(dec.round())
    for s in rest:
        assert s.out == refs[tuple(s.prompt)], s.prompt
        assert s.cached == (len(shared) if s.prompt[:len(shared)] == shared else 0), (s.prompt, s.cached)


def test_streams_keep_decoding_while_a_prompt_prefills(monkeypatch):
    """With streams decoding, a queued prompt prefills STEP rows a round and the others keep taking tokens."""

    from tensorfold.families.qwen3_5.cuda import multi

    monkeypatch.setattr(multi, "STEP", 2)
    w = _model()
    long_prompt = list(range(40, 52))
    refs = {tuple(PROMPTS[0]): _serial(w, PROMPTS[0], None, 64), tuple(long_prompt): _serial(w, long_prompt, None, 12)}
    dec = _Oracle(w, refs, seed=2)
    a = Stream(PROMPTS[0], 64, None)
    dec.admit(a)
    dec.finish(dec.round())                           # nothing else decodes: a's prompt prefills whole, then a round
    assert a.st.pos > len(PROMPTS[0]) and not dec.filling
    b = Stream(long_prompt, 12, None)
    dec.admit(b)
    steps = 0
    while any(x is b for x in dec.filling):
        before = len(a.out)
        dec.finish(dec.round())
        steps += 1
        assert len(a.out) > before and not a.done
    assert steps == len(long_prompt) // 2
    while dec.live():
        dec.finish(dec.round())
    assert a.out == refs[tuple(PROMPTS[0])] and b.out == refs[tuple(long_prompt)]


def test_prefill_keeps_states_at_stops_that_resume_exactly():
    """A state kept at a stop (a message start) resumes another prompt with that prefix to a fresh prefill's bits."""

    w = _model()
    a, b = [11, 12, 13, 14, 15, 16, 21, 22, 23], [11, 12, 13, 14, 15, 16, 31, 32]
    kept = {}
    st_a, first_a = prefill(w, a, SAMPLINGS[1], stops=[6], keep=lambda p, st, snap: kept.setdefault(p, st))
    fresh_a, ref_a = prefill(w, a, SAMPLINGS[1])
    _same_state(st_a, fresh_a)
    assert first_a == ref_a and kept[6].pos == 6
    st_b, first_b = prefill(w, b, SAMPLINGS[1], state=kept[6])
    fresh_b, ref_b = prefill(w, b, SAMPLINGS[1])
    _same_state(st_b, fresh_b)
    assert first_b == ref_b
