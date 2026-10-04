"""top_k off without a top_p cut: ``cuda.sampling.nucleus_rows`` draws on the device, token for token what the host
rule over whole shards draws, and never reads whole shards to the host."""

import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import sampling as cs
from tensorfold.engine.exact_sampling import Sampling, _mix

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only"))]


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


def two_ranks(logits, positions, sampling, split):
    ranks, out, probs = Ranks(), [None, None], [[], []]
    shards = (logits[:, :split], logits[:, split:])

    def run(r):
        out[r] = cs.nucleus_rows(shards[r], positions, sampling, offset=0 if r == 0 else split,
                                 gather=ranks.gather(r), probs=probs[r])

    threads = [threading.Thread(target=run, args=(r,)) for r in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out[0] == out[1] and probs[0] == probs[1]               # every rank draws the same tokens
    return out[0], probs[0]


def host_rule(monkeypatch, *args):
    """The same call with the device draw off: the host rule over 1024 candidates or whole shards."""

    with monkeypatch.context() as m:
        m.setattr(cs, "_keyed", lambda *a: None)
        return two_ranks(*args)


def _logits(seed, device, rows=6, vocab=20000, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, vocab, generator=g) * scale
    if seed % 3 == 0:                                               # a coarse grid: tied values across both shards
        logits = logits.mul(2).round().div(2)
        logits[:, [5, vocab // 2 - 1, vocab // 2 + 2000, vocab - 1]] = logits.max() + 1.0
    return logits.to(torch.bfloat16).to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("temperature", [0.2, 1.0])
@pytest.mark.parametrize("top_p", [0.95, 1.0])
@pytest.mark.parametrize("min_p", [0.0, 0.05])
def test_the_device_draw_is_the_host_rule(monkeypatch, device, temperature, top_p, min_p):
    decided = []
    real = cs._keyed
    monkeypatch.setattr(cs, "_keyed", lambda *a: decided.append(real(*a)) or decided[-1])
    vocab = 129280 if device == "cuda" else 20000                      # the checkpoint's vocabulary on a GPU
    for seed in range(16):
        logits = _logits(seed, device, vocab=vocab)
        s = Sampling(seed * 7919 + 3, temperature, 0, top_p, min_p)
        positions = [100 + 37 * seed + r for r in range(logits.shape[0])]
        split = vocab // 2 - (seed % 2) * 1000                          # equal and unequal shards
        assert two_ranks(logits, positions, s, split) == host_rule(monkeypatch, logits, positions, s, split)
    if top_p >= 1.0:
        assert decided and all(d is not None for d in decided)       # the device drew every row
    else:
        assert not decided


def test_top_p_one_reads_no_whole_shard(monkeypatch):
    flat = torch.zeros(3, 5000, dtype=torch.bfloat16)                  # every token tied: no candidates cover it
    flat[1, ::7] = 0.5
    for min_p in (0.0, 0.3):
        s = Sampling(5, 1.0, 0, 1.0, min_p)
        want = host_rule(monkeypatch, flat, [7, 8, 9], s, 2500)
        with monkeypatch.context() as m:
            m.setattr(cs, "_shares", lambda *a: pytest.fail("read candidates to the host"))
            assert two_ranks(flat, [7, 8, 9], s, 2500) == want


def test_a_best_too_close_to_a_ranks_runner_up_falls_back_to_the_host_rule(monkeypatch):
    monkeypatch.setattr(cs, "SLACK", 1e9)                              # no host best clears its ranks' bounds
    decided = []
    real = cs._keyed
    monkeypatch.setattr(cs, "_keyed", lambda *a: decided.append(real(*a)) or decided[-1])
    logits = _logits(4, "cpu")
    s = Sampling(17, 1.0, 0, 1.0, 0.0)
    positions = list(range(6))
    assert two_ranks(logits, positions, s, 10000) == host_rule(monkeypatch, logits, positions, s, 10000)
    assert decided == [None, None]


def test_a_rank_that_ranks_its_winner_second_falls_back_to_the_host_rule(monkeypatch):
    real = cs._best_two

    def swapped(scaled, positions, s, floor, offset, id_map):        # rank 0 reports its runner-up as its best
        f, i = real(scaled, positions, s, floor, offset, id_map)
        if offset:
            return f, i
        rest = scaled.clone().scatter_(1, i[:, :1], -float("inf"))
        g, j = real(rest, positions, s, floor, offset, id_map)
        return torch.stack([g[:, 0], f[:, 0], g[:, 2]], 1), j

    monkeypatch.setattr(cs, "_best_two", swapped)
    decided = []
    real_keyed = cs._keyed
    monkeypatch.setattr(cs, "_keyed", lambda *a: decided.append(real_keyed(*a)) or decided[-1])
    for seed in range(4):
        logits = _logits(seed, "cpu")
        s = Sampling(seed + 21, 1.0, 0, 1.0, 0.0)
        positions = [3 + seed + r for r in range(6)]
        assert two_ranks(logits, positions, s, 10000) == host_rule(monkeypatch, logits, positions, s, 10000)
    assert decided == [None] * 8                                       # the runner-up bound rejects every call


def test_device_scores_only_pick_the_candidates(monkeypatch):
    real = cs.uniform_rows                                             # device logs a few ulps off the host's
    monkeypatch.setattr(cs, "uniform_rows", lambda *a: real(*a) * (1.0 - 2.0 ** -50))
    for seed in range(6):
        logits = _logits(seed, "cpu")
        s = Sampling(seed + 1, 1.0, 0, 1.0, 0.05 * (seed % 2))
        positions = [9 + seed + r for r in range(6)]
        assert two_ranks(logits, positions, s, 7000) == host_rule(monkeypatch, logits, positions, s, 7000)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("mapped", [False, True])
def test_the_kernel_finds_the_host_scores_best_two(mapped):
    from tensorfold.cuda.keyed_draw import best_two

    g = torch.Generator().manual_seed(3)
    scaled = (torch.randn(5, 64640, generator=g) * 4).to(torch.bfloat16).double() / 0.7
    floor = scaled.max(dim=-1).values + torch.tensor([-float("inf"), -3.0, -1.0, -0.5, 0.0], dtype=torch.float64)
    id_map = torch.randperm(200000, generator=g)[:64640] if mapped else None
    s, positions = Sampling(2**63 + 5, 1.0, 0, 1.0, 0.0), [0, 1, 4097, 2**40 + 3, 77]
    want_f, want_i = cs._best_two(scaled, positions, s, floor, 64640, id_map)
    with np.errstate(over="ignore"):
        keys = _mix(_mix(np.uint64(s.seed) + np.uint64(0x9E3779B97F4A7C15))
                    ^ (np.asarray(positions).astype(np.uint64) * np.uint64(0xD1B54A32D192ED03)))
    got_f, got_i = best_two(scaled.cuda(), torch.from_numpy(keys.view(np.int64)).cuda(), floor.cuda(), 64640,
                            None if id_map is None else id_map.cuda())
    assert torch.equal(got_i.cpu(), want_i)
    assert torch.allclose(got_f.cpu(), want_f, rtol=1e-13, atol=0), (got_f, want_f)
    assert float(got_f[4, 1]) == -float("inf")                          # one token at the floor: no runner-up


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_a_row_at_top_p_one_no_longer_waits_on_whole_shards(monkeypatch):
    logits = (torch.randn(1, 129280, device="cuda") * 3).to(torch.bfloat16)
    s = Sampling(11, 1.0, 0, 1.0, 0.0)

    def median(n):
        times = []
        for p in range(n):
            torch.cuda.synchronize()
            start = time.perf_counter()
            cs.nucleus_rows(logits, [p], s)
            times.append(time.perf_counter() - start)
        return sorted(times[2:])[(n - 2) // 2]

    device = median(42)
    monkeypatch.setattr(cs, "_keyed", lambda *a: None)
    assert 10 * device < median(12)
