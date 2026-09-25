"""The session n-gram prior: the interpolated backoff model of the offline study, counted incrementally."""

import collections
import math

import numpy as np

from tensorfold.drafters.draft_ngram import SessionNGram


class _Reference:
    """The offline prototype's model (scratchpad ngram_rescore.py): Counters over tuples, recursion from 1/V."""

    def __init__(self, vocab, n=4):
        self.n, self.vocab, self.seen = n, vocab, 0
        self.counts = [collections.Counter() for _ in range(n + 1)]
        self.ctx_counts = [collections.Counter() for _ in range(n + 1)]

    def extend(self, tokens):
        for i in range(self.seen, len(tokens)):
            for k in range(1, self.n + 1):
                if i - k + 1 < 0:
                    break
                gram = tuple(tokens[i - k + 1:i + 1])
                self.counts[k][gram] += 1
                self.ctx_counts[k][gram[:-1]] += 1
        self.seen = len(tokens)

    def logp(self, history, tok):
        p = 1.0 / self.vocab
        for k in range(1, self.n + 1):
            if k - 1 > len(history):
                break
            ctx = tuple(history[len(history) - (k - 1):]) if k > 1 else ()
            c = self.ctx_counts[k].get(ctx, 0)
            if c == 0:
                break
            lam = c / (c + 2.0)
            p = lam * (self.counts[k].get(ctx + (tok,), 0) / c) + (1 - lam) * p
        return math.log(p)


def _text(rng, n, vocab=40):
    # repetitive enough that 3- and 4-gram contexts recur
    phrases = [list(rng.integers(0, vocab, size=int(rng.integers(3, 9)))) for _ in range(12)]
    out = []
    while len(out) < n:
        out += phrases[int(rng.integers(0, len(phrases)))] if rng.random() < 0.8 else [int(rng.integers(0, vocab))]
    return [int(t) for t in out[:n]]


def _check(model, ref, tokens, rng, vocab, cases):
    for _ in range(cases):
        history = [int(t) for t in rng.integers(0, vocab, size=3)]
        if rng.random() < 0.6 and ref.seen > 4:        # a history that occurred: exercises the higher orders
            at = int(rng.integers(3, ref.seen))
            history = list(tokens[at - 3:at])
        tok = int(rng.integers(0, vocab))
        if rng.random() < 0.5 and ref.seen > 4:        # ... and a token that followed it
            tok = int(tokens[int(rng.integers(1, ref.seen))])
        assert math.isclose(model.logp(history, tok), ref.logp(history, tok), rel_tol=1e-12, abs_tol=1e-12)


def test_matches_the_reference_model_one_shot_and_incremental():
    rng = np.random.default_rng(0)
    vocab = 248320
    tokens = _text(rng, 3000)
    for chunks in ([3000], [2000, 7, 1, 300, 692], [1, 1, 1, 2, 5, 90, 2900]):
        model, ref = SessionNGram(vocab), _Reference(vocab)
        end = 0
        for size in chunks:
            end += size
            model.update(tokens[:end])
            ref.extend(tokens[:end])
            _check(model, ref, tokens, rng, 40, cases=60)
        assert model.seen == len(tokens)


def test_bonus_is_weight_times_log_p_for_every_candidate():
    rng = np.random.default_rng(1)
    tokens = _text(rng, 1500)
    model, ref = SessionNGram(), _Reference(248320)
    model.update(tokens[:1000])
    model.update(tokens)                                   # the index, then the incremental path
    ref.extend(tokens)
    cands = np.stack([rng.permutation(40)[:16] for _ in range(5)]).astype(np.int64)
    bonus = model.rescorer(cands, 0.1)
    for _ in range(50):
        at = int(rng.integers(3, len(tokens)))
        hist = tuple(tokens[at - 3:at])
        depth = int(rng.integers(0, 5))
        want = [0.1 * ref.logp(list(hist), int(c)) for c in cands[depth]]
        assert np.allclose(bonus(hist, depth), want, rtol=1e-12, atol=1e-12)


def test_short_histories_and_off_switches():
    model = SessionNGram()
    assert model.rescorer(np.zeros((2, 16), dtype=np.int64), 0.1) is None      # nothing counted
    model.update([5, 6, 5, 6, 5])
    assert model.rescorer(np.zeros((2, 16), dtype=np.int64), 0.0) is None      # weight 0
    ref = _Reference(model.vocab)
    ref.extend([5, 6, 5, 6, 5])
    for history in ([], [5], [6, 5], [5, 6, 5]):
        for tok in (5, 6, 7):
            assert math.isclose(model.logp(history, tok), ref.logp(history, tok), rel_tol=1e-12)


def test_a_context_that_does_not_extend_the_last_one_starts_over():
    model = SessionNGram()
    model.update([1, 2, 3, 4, 1, 2, 3])
    model.update([9, 9, 9, 8])                               # shorter: a new context
    ref = _Reference(model.vocab)
    ref.extend([9, 9, 9, 8])
    assert model.seen == 4
    assert math.isclose(model.logp([9, 9], 9), ref.logp([9, 9], 9), rel_tol=1e-12)
    assert math.isclose(model.logp([2, 3], 4), ref.logp([2, 3], 4), rel_tol=1e-12)
    model.update([9, 9, 7, 8, 1])                            # same length prefix, different tokens
    ref = _Reference(model.vocab)
    ref.extend([9, 9, 7, 8, 1])
    assert math.isclose(model.logp([9, 9], 7), ref.logp([9, 9], 7), rel_tol=1e-12)
