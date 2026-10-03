"""top_k off on CUDA: the nucleus draw (``cuda.sampling.nucleus_rows``) gives the same tokens from one rank's whole
rows as from two ranks' vocabulary shards, whether the candidates cover the nucleus or a row reads whole shards, and
draws from the same distribution as the float rule."""

import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import sampling as cs  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling, choose_rows  # noqa: E402


class Ranks:
    """Two ranks as threads: each ``gather`` stacks both ranks' words, rank 0 first."""

    def __init__(self):
        self.slots, self.barrier = [None, None], threading.Barrier(2, timeout=60)

    def gather(self, rank):
        def gather(words):
            self.slots[rank] = words.clone()
            self.barrier.wait()
            both = torch.stack(self.slots)
            self.barrier.wait()
            return both

        return gather


def two_ranks(logits, positions, sampling, split, probs=None):
    ranks, out = Ranks(), [None, None]
    shards = (logits[:, :split], logits[:, split:])

    def run(r):
        out[r] = cs.nucleus_rows(shards[r], positions, sampling, offset=0 if r == 0 else split,
                                 gather=ranks.gather(r), probs=probs if r == 0 else None)

    threads = [threading.Thread(target=run, args=(r,)) for r in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out[0] == out[1]                                       # every rank draws the same tokens
    return out[0]


def _logits(seed, rows=6, vocab=6000, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, vocab, generator=g) * scale).to(torch.bfloat16)


@pytest.mark.parametrize("top_p", [0.8, 0.95, 1.0])
@pytest.mark.parametrize("min_p", [0.0, 0.05])
def test_two_ranks_draw_what_one_rank_draws(top_p, min_p):
    for seed in range(12):
        logits = _logits(seed)
        s = Sampling(seed * 7919 + 3, 0.9, 0, top_p, min_p)
        positions = [100 + 17 * seed + r for r in range(logits.shape[0])]
        one = cs.nucleus_rows(logits, positions, s)
        assert two_ranks(logits, positions, s, 3000) == one
        assert two_ranks(logits, positions, s, 1234) == one          # shards of any size


def test_rows_past_the_candidates_read_whole_shards_and_still_agree(monkeypatch):
    flat = torch.zeros(3, 5000, dtype=torch.bfloat16)                  # every token tied: the nucleus is all of them
    flat[1, ::7] = 0.5
    reads = []
    real = cs._shares
    monkeypatch.setattr(cs, "_shares", lambda *a: reads.append(a[3]) or real(*a))
    for top_p, min_p in ((0.95, 0.0), (1.0, 0.0), (1.0, 0.3), (0.5, 0.2)):
        s = Sampling(5, 1.0, 0, top_p, min_p)
        one = cs.nucleus_rows(flat, [7, 8, 9], s)
        assert two_ranks(flat, [7, 8, 9], s, 2500) == one
    assert 5000 in reads and 2500 in reads                             # whole rows and whole shards were read


def test_the_same_distribution_as_the_float_rule():
    agree = total = 0
    for seed in range(20):
        logits = _logits(seed, rows=8, vocab=4000)
        s = Sampling(seed + 1, 1.0, 0, 0.9, 0.0)
        positions = list(range(8))
        values = logits.float().numpy()
        ids = np.broadcast_to(np.arange(4000, dtype=np.int64), values.shape)
        want = choose_rows(values, ids, positions, s)
        got = cs.nucleus_rows(logits, positions, s)
        agree += sum(a == b for a, b in zip(got, want))
        total += len(want)
    assert agree >= 0.99 * total, (agree, total)                       # only a cut's last token can differ


def test_the_drawn_tokens_share_of_the_mass():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    probs = []
    token = cs.nucleus_rows(logits, [3], Sampling(9, 1.0, 0, 1.0, 0.0), probs=probs)[0]
    assert probs == pytest.approx([float(torch.softmax(logits[0].double(), -1)[token])], rel=1e-9)


def test_streams_with_top_k_off_draw_by_the_nucleus_rule():
    logits = _logits(3, rows=4, vocab=3000)
    s = Sampling(11, 0.8, 0, 0.9, 0.02)
    want = cs.nucleus_rows(logits, [1, 2, 3, 4], s)
    keyed = Sampling(4, 1.0, 20, 0.95)
    got = cs.sample_streams(logits, [0, 2, 3, 4], [[1, 2], [3], [4]], [s, keyed, s])
    assert got[0] == want[:2] and got[2] == want[3:]
