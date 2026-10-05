"""top_k off: ``cuda.sampling.nucleus_rows`` draws on the device, with or without a top_p cut, the tokens and mass
shares the host rule over whole shards draws, and reads no candidates to the host."""

from fractions import Fraction
import math
import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import sampling as cs
from tensorfold.engine.exact_sampling import Sampling, _mix

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only"))]
SPLITS = ["equal", "unequal", "interleaved"]


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


def shards(logits, split):
    """Each rank's (logits, offset, id_map): split at a column, in halves, unequal halves, or even and odd ids."""

    vocab = logits.shape[-1]
    if split == "interleaved":
        ids = torch.arange(vocab, device=logits.device)
        return [(logits[..., r::2].contiguous(), 0, ids[r::2].contiguous()) for r in (0, 1)]
    at = split if isinstance(split, int) else vocab // 2 - (1000 if split == "unequal" else 0)
    return [(logits[..., :at], 0, None), (logits[..., at:], at, None)]


def two_ranks(logits, positions, sampling, split):
    ranks, out, probs = Ranks(), [None, None], [[], []]
    parts = shards(logits, split)

    def run(r):
        x, offset, id_map = parts[r]
        out[r] = cs.nucleus_rows(x, positions, sampling, offset=offset, id_map=id_map, gather=ranks.gather(r),
                                 probs=probs[r])

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


def on_the_device(monkeypatch, *args):
    """``two_ranks`` failing if any row falls back to candidates read to the host."""

    with monkeypatch.context() as m:
        m.setattr(cs, "_shares", lambda *a: pytest.fail("read candidates to the host"))
        return two_ranks(*args)


def _logits(seed, device, rows=6, vocab=20000, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, vocab, generator=g) * scale
    if seed % 3 == 0:                                               # a coarse grid: tied values across both shards
        logits = logits.mul(2).round().div(2)
        logits[:, [5, vocab // 2 - 1, vocab // 2 + 2000, vocab - 1]] = logits.max() + 1.0
    return logits.to(torch.bfloat16).to(device)


def _varied(seed, device, vocab):
    """``_logits``, or every fourth seed near flat (nuclei past the candidates) or with one token far above the rest."""

    if seed % 4 == 1:
        return _logits(seed, device, vocab=vocab, scale=0.3)
    logits = _logits(seed, device, vocab=vocab)
    if seed % 4 == 2:
        logits[:, (7 + 4999 * seed) % vocab] += 40.0
    return logits


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("temperature", [0.2, 0.7, 1.0])
@pytest.mark.parametrize("top_p", [0.5, 0.9, 0.95, 0.99, 1.0])
@pytest.mark.parametrize("min_p", [0.0, 0.05])
def test_the_device_draw_is_the_host_rule(monkeypatch, device, split, temperature, top_p, min_p):
    vocab = 129280 if device == "cuda" else 20000                      # the checkpoint's vocabulary on a GPU
    for seed in range(32):
        logits = _varied(seed, device, vocab)
        s = Sampling(seed * 7919 + 3, temperature, 0, top_p, min_p)
        positions = [100 + 37 * seed + r for r in range(logits.shape[0])]
        assert on_the_device(monkeypatch, logits, positions, s, split) == host_rule(monkeypatch, logits, positions,
                                                                                    s, split)


def test_tied_rows_read_no_candidates(monkeypatch):
    flat = torch.zeros(3, 5000, dtype=torch.bfloat16)                  # every token tied: no candidates cover it
    flat[1, ::7] = 0.5
    for top_p, min_p in ((0.95, 0.0), (1.0, 0.0), (1.0, 0.3), (0.5, 0.2)):
        s = Sampling(5, 1.0, 0, top_p, min_p)
        assert on_the_device(monkeypatch, flat, [7, 8, 9], s, 2500) == host_rule(monkeypatch, flat, [7, 8, 9], s, 2500)


def reference_cut(scaled, mass, top_p):
    """Each row's (last nucleus token's value, its id, the nucleus size), in Python integers over whole rows."""

    out = []
    for v, m in zip(scaled.cpu().numpy(), mass.cpu().numpy()):
        order = np.lexsort((np.arange(len(v)), -v))
        need = math.ceil(Fraction(top_p) * int(m.sum()))
        keep = int((np.cumsum(m[order]) < need).sum()) + 1
        out.append((float(v[order[keep - 1]]), int(order[keep - 1]), keep))
    return out


def cuts(logits, temperature, top_p, split):
    """``_cut`` on two ranks' shards, beside ``reference_cut`` over the same masses."""

    scaled = logits.float().double() / temperature
    mass = torch.floor(torch.exp(scaled - scaled.amax(dim=1, keepdim=True)) * cs.MASS).to(torch.int64)
    ranks, out = Ranks(), [None, None]
    parts, masses = shards(logits, split), shards(mass, split)

    def run(r):
        x, offset, id_map = parts[r]
        value, at = cs._cut(ranks.gather(r), x, temperature, masses[r][0], top_p, offset, id_map)
        out[r] = [(float(v), int(i)) for v, i in zip(value.cpu(), at.cpu())]

    threads = [threading.Thread(target=run, args=(r,)) for r in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out[0] == out[1]
    return out[0], reference_cut(scaled, mass, top_p)


def tied_rows(device, vocab):
    """Three tokens above 3000 tied ones (spread over both ranks, half of them -0.0), the rest far below."""

    logits = torch.full((6, vocab), -30.0)
    tied = 7 + vocab // 3000 * torch.arange(3000)
    logits[:, tied] = 0.0
    logits[:, tied[::2]] = -0.0
    logits[:, [3, vocab // 2 + 5, vocab - 2]] = 1.0
    return logits.to(torch.bfloat16).to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("split", SPLITS)
def test_a_cut_inside_tied_values_keeps_their_lowest_ids(monkeypatch, device, split):
    vocab = 129280 if device == "cuda" else 30000
    logits = tied_rows(device, vocab)
    for temperature, top_p in ((1.0, 0.5), (0.7, 0.3), (1.0, 0.9)):
        got, want = cuts(logits, temperature, top_p, split)
        assert got == [w[:2] for w in want]
        assert all(w[0] == 0.0 and 3 < w[2] < 3003 for w in want)         # the cut falls inside the ties
        for seed in range(32):
            s = Sampling(seed * 31 + 1, temperature, 0, top_p, 0.0)
            positions = [11 * seed + r for r in range(6)]
            assert on_the_device(monkeypatch, logits, positions, s, split) == host_rule(monkeypatch, logits,
                                                                                        positions, s, split)


@pytest.mark.parametrize("device", DEVICES)
def test_the_cut_sums_mass_exactly_where_float_sums_round(monkeypatch, device):
    vocab = 129280 if device == "cuda" else 20000
    g = torch.Generator().manual_seed(1)
    logits = torch.full((2, vocab), -27.5)                              # a mass of 1 each
    for r in range(2):
        logits[r, torch.randperm(vocab, generator=g)[:2 ** 14]] = 0.0     # 2**40 each: 2**54 in all
    total = 2 ** 54 + vocab - 2 ** 14
    top_p = float(Fraction(2 ** 54 + 1000, total))
    logits = logits.to(device)
    s = Sampling(3, 1.0, 0, top_p, 0.0)
    for split in SPLITS:
        got, want = cuts(logits, 1.0, top_p, split)
        assert got == [w[:2] for w in want]
        assert all(w[0] == -27.5 and 2 ** 14 < w[2] < vocab for w in want)    # past 2**53, inside the mass-1 tokens
        assert on_the_device(monkeypatch, logits, [5, 6], s, split) == host_rule(monkeypatch, logits, [5, 6], s, split)


@pytest.mark.parametrize("device", DEVICES)
def test_the_ranks_masses_add_exactly_past_2_53(monkeypatch, device):
    vocab = 129280 if device == "cuda" else 20000
    g = torch.Generator().manual_seed(5)
    logits = torch.full((1, vocab), -8.0)                                # the tail, below the cut
    logits[0, 0] = 0.0
    group = torch.randperm(vocab - 1, generator=g)[:14000] + 1           # ~0.6 * 2**40 each, in one radix bucket
    logits[0, group] = -0.5 - 0.03 * torch.rand(len(group), generator=g)
    logits = logits.to(device)
    while True:                                                         # masses as ``cuts`` takes them
        mass = torch.floor(torch.exp(logits.double()) * cs.MASS).to(torch.int64)[0].cpu()
        total, group_mass = int(mass.sum()), int(mass[group].sum())
        need = int(cs.MASS) + group_mass                                # the cut ends exactly at the group's end
        top_p = float(Fraction(need, total))
        hits = [p for p in (top_p, np.nextafter(top_p, 0.0), np.nextafter(top_p, 1.0))
                if math.ceil(Fraction(float(p)) * total) == need]
        if group_mass % 4 == 1 and hits:                                # float64 rounds the group's mass down
            break
        logits[0, group[0]] = torch.nextafter(logits[0, group[0]], torch.tensor(-1.0, device=device))
    top_p = float(hits[0])
    assert group_mass > 2 ** 53 and float(group_mass) < group_mass
    s = Sampling(4, 1.0, 0, top_p, 0.0)
    for split in SPLITS:
        got, want = cuts(logits, 1.0, top_p, split)
        assert got == [w[:2] for w in want] and want[0][2] == 1 + len(group)
        assert on_the_device(monkeypatch, logits, [9], s, split) == host_rule(monkeypatch, logits, [9], s, split)


@pytest.mark.parametrize("device", DEVICES)
def test_one_ranks_own_bucket_adds_exactly_past_2_53(monkeypatch, device):
    vocab = 129280 if device == "cuda" else 40000
    group = 20000                                                       # ~0.6 * 2**40 each, in one radix bucket
    g = torch.Generator().manual_seed(6)
    logits = torch.full((1, vocab), -5.0)                                # the tail, below the cut
    logits[0, 0] = 0.0
    logits[0, 1:1 + group] = -0.5 - 0.03 * torch.rand(group, generator=g)
    logits = logits.to(device)
    while True:                                                         # each mass 1 mod 4: a float64 sum past
        mass = torch.floor(torch.exp(logits.double()) * cs.MASS).to(torch.int64)[0]   # 2**53 rounds each one down
        off = torch.nonzero(mass[1:1 + group] % 4 != 1)[:, 0] + 1
        if not len(off):
            break
        logits[0, off] = torch.nextafter(logits[0, off], torch.tensor(-1.0, device=device))
    while True:                                                         # the cut ends exactly at the group's end
        mass = torch.floor(torch.exp(logits.double()) * cs.MASS).to(torch.int64)[0].cpu()
        total, group_mass = int(mass.sum()), int(mass[1:1 + group].sum())
        need = int(cs.MASS) + group_mass
        top_p = float(Fraction(need, total))
        hits = [p for p in (top_p, np.nextafter(top_p, 0.0), np.nextafter(top_p, 1.0))
                if math.ceil(Fraction(float(p)) * total) == need]
        if hits:
            break
        logits[0, -1] = torch.nextafter(logits[0, -1], torch.tensor(-6.0, device=device))
    top_p = float(hits[0])
    assert group_mass > 2 ** 53 + 2 ** 51                                # thousands of masses past 2**53
    s = Sampling(8, 1.0, 0, top_p, 0.0)
    for split in (1, 1 + group):                                        # the whole group on rank 1, then on rank 0
        got, want = cuts(logits, 1.0, top_p, split)
        assert got == [w[:2] for w in want] and want[0][2] == 1 + group
        assert on_the_device(monkeypatch, logits, [9], s, split) == host_rule(monkeypatch, logits, [9], s, split)


@pytest.mark.parametrize("device", DEVICES)
def test_the_need_is_exact_where_a_float_product_rounds(monkeypatch, device):
    vocab = 129280 if device == "cuda" else 20000
    logits = _logits(2, device, rows=1, vocab=vocab)
    scaled = logits.float().double()
    mass = torch.floor(torch.exp(scaled - scaled.amax(dim=1, keepdim=True)) * cs.MASS).to(torch.int64)[0].cpu()
    total = int(mass.sum())
    order = np.lexsort((np.arange(vocab), -scaled[0].cpu().numpy()))
    for end in np.cumsum(mass.numpy()[order])[1:-1].tolist():          # each token's mass and all before it
        top_p = float(Fraction(end, total))
        if Fraction(top_p) * total <= end:
            top_p = float(np.nextafter(top_p, 1.0))                     # the least top_p whose need passes the token
        if math.ceil(top_p * total) == end:                             # its float product rounds back onto it
            break
    assert math.ceil(top_p * total) == end and math.ceil(Fraction(top_p) * total) == end + 1
    s = Sampling(6, 1.0, 0, top_p, 0.0)
    for split in SPLITS:
        got, want = cuts(logits, 1.0, top_p, split)
        assert got == [w[:2] for w in want]
        assert on_the_device(monkeypatch, logits, [5], s, split) == host_rule(monkeypatch, logits, [5], s, split)


@pytest.mark.parametrize("device", DEVICES)
def test_nuclei_of_one_token_and_of_more_than_the_candidates(monkeypatch, device):
    vocab = 129280 if device == "cuda" else 20000
    for seed, sizes in ((1, lambda k: k > cs.NUCLEUS), (2, lambda k: k == 1)):
        logits = _varied(seed, device, vocab)
        s = Sampling(seed, 0.7, 0, 0.9, 0.0)
        for split in SPLITS:
            got, want = cuts(logits, 0.7, 0.9, split)
            assert got == [w[:2] for w in want] and all(sizes(w[2]) for w in want)
            assert on_the_device(monkeypatch, logits, list(range(6)), s, split) == host_rule(monkeypatch, logits,
                                                                                             list(range(6)), s, split)


def test_a_temperature_past_tmax_cuts_by_the_host_rule(monkeypatch):
    decided = []
    real = cs._keyed
    monkeypatch.setattr(cs, "_keyed", lambda *a: decided.append(real(*a)) or decided[-1])
    logits = _logits(1, "cpu")
    s = Sampling(2, cs.TMAX * 2, 0, 0.9, 0.0)
    assert two_ranks(logits, list(range(6)), s, 10000) == host_rule(monkeypatch, logits, list(range(6)), s, 10000)
    assert decided == [None, None]


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

    def swapped(scaled, positions, s, floor, offset, id_map, cut):   # rank 0 reports its runner-up as its best
        f, i = real(scaled, positions, s, floor, offset, id_map, cut)
        if offset:
            return f, i
        rest = scaled.clone().scatter_(1, i[:, :1], -float("inf"))
        g, j = real(rest, positions, s, floor, offset, id_map, cut)
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
