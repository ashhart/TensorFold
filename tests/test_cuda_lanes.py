"""The lane decoder on two fake ranks: drafted == serial, concurrent == solo, resumed == fresh, ranks in step."""

from __future__ import annotations

import pytest
from cuda_lane_fakes import PLANES, Codec, FakeForward, Mirror, Pattern, drive, same_state

from tensorfold.cuda.admission import GIB, Admission
from tensorfold.cuda.drafting import ExpectedRate, Lookup, StaticDepth, WindowCosts
from tensorfold.cuda.kvpool import PagePool
from tensorfold.cuda.lanes import LaneDecoder, Lanes
from tensorfold.cuda.lanes.link import ADMIT, EVICT
from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.session_disk import DiskTier
from tensorfold.cuda.sessions import HostTier, TieredCache
from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling

CAPACITY = 256
WARM = Sampling(seed=1234, temperature=0.9, top_k=8, top_p=0.95)


def prompts(n: int, length: int = 40) -> list[list[int]]:
    return [[(7 * i + 3 * j + i * j) % 61 + 1 for j in range(length + 5 * i)] for i in range(n)]


class Pair:
    """Rank 0's decoder and a second rank fed its messages; ``check`` compares them."""

    def __init__(
        self,
        *,
        lanes: int = 4,
        pool_pages: int | None = 64,
        page: int = 16,
        drafter=None,
        depth=None,
        keep: int = 4,
        tiers=(),
        replay_tail: int = 0,
        grid: int = 1,
        prefill_rows: int = 24,
        admission=None,
        grammars=None,
    ) -> None:
        def side(rank):
            pool = PagePool(PLANES, pool_pages, page) if pool_pages else None
            cache = TieredCache(
                keep,
                codec=Codec() if tiers else None,
                tiers=tiers[rank] if tiers else (),
                compat={"engine": "fake", "replay_tail": replay_tail},
            )
            fwd = FakeForward(pool, lanes, grid=grid, replay_tail=replay_tail)
            d = drafter() if drafter is not None else None
            p = depth() if depth is not None else None
            return fwd, pool, cache, d, p

        f1, p1, c1, d1, k1 = side(1)
        self.follower = Lanes(f1, lanes, CAPACITY, pool=p1, cache=c1, drafter=d1, depth=k1, grammars=grammars)
        self.link = Mirror(self.follower)
        f0, p0, c0, d0, k0 = side(0)
        self.decoder = LaneDecoder(
            f0,
            capacity=CAPACITY,
            lanes=lanes,
            link=self.link,
            pool=p0,
            cache=c0,
            drafter=d0,
            depth=k0,
            eos=(0,),
            prefill_rows=prefill_rows,
            admission=admission,
            grammars=grammars,
        )
        self.lanes = lanes
        self.checks = 0

    def check(self) -> None:
        same_state(self.decoder.local, self.follower)
        if self.decoder.pool is not None:
            assert self.decoder.pool.null_clean()
        self.checks += 1

    def run(self, streams: list[Stream], **kw) -> list[list[int]]:
        drive(self.decoder, streams, lanes=self.lanes, check=self.check, **kw)
        assert self.checks > 0
        for s in streams:
            assert s.error is None, s.error
        return [list(s.out) for s in streams]


def stream(prompt, count=30, sampling=None, draft=True, **kw) -> Stream:
    return Stream(list(prompt), count, sampling, draft=draft, **kw)


def serial(prompt, count=30, sampling=None) -> list[int]:
    return Pair(lanes=1, keep=0).run([stream(prompt, count, sampling, draft=False)])[0]


@pytest.mark.parametrize("sampling", [None, WARM], ids=["greedy", "sampled"])
@pytest.mark.parametrize(
    "drafter,depth",
    [
        (Pattern, None),
        (Pattern, lambda: StaticDepth(2)),
        (Lookup, None),
        (Pattern, lambda: ExpectedRate(WindowCosts.linear(10.0, 1.0))),
    ],
    ids=["pattern", "static2", "lookup", "expected-rate"],
)
def test_drafted_equals_serial(sampling, drafter, depth):
    p = prompts(1)[0]
    want = serial(p, 40, sampling)
    pair = Pair(lanes=1, drafter=drafter, depth=depth)
    got = pair.run([stream(p, 40, sampling)])
    assert got == [want]


def test_drafts_are_kept_when_right():
    pair = Pair(lanes=1, drafter=lambda: Pattern(good=(6,)))
    s = stream(prompts(1)[0], 40)
    pair.run([s])
    assert len(s.out) >= 20 and s.rounds * 5 <= len(s.out)  # up to seven tokens a round


@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("sampling", [None, WARM], ids=["greedy", "sampled"])
def test_concurrent_equals_solo(n, sampling):
    ps = prompts(n)
    solo = [serial(p, 30, sampling) for p in ps]
    pair = Pair(lanes=n, drafter=Pattern)
    streams = [stream(p, 30, sampling) for p in ps]
    late = {id(streams[-1]): 3}  # one arrives while the others decode
    assert pair.run(streams, arrive=late) == solo


def test_more_streams_than_lanes_wait_for_one():
    ps = prompts(5)
    solo = [serial(p, 12) for p in ps]
    assert Pair(lanes=2, drafter=Pattern).run([stream(p, 12) for p in ps]) == solo


def test_resumed_from_ram_equals_fresh():
    first = prompts(1)[0]
    second = first + [5, 9, 11, 13]
    pair = Pair(lanes=2, drafter=Pattern)
    pair.run([stream(first, 10)])
    s = stream(second, 25)
    assert pair.run([s]) == [serial(second, 25)]
    assert s.cached == len(first) - 1


def test_an_identical_prompt_resumes_one_token_short():
    p = prompts(1)[0]
    pair = Pair(lanes=1, drafter=Pattern)
    a, b = stream(p, 20), stream(p, 20)
    pair.run([a])
    pair.run([b])
    assert b.cached == len(p) - 1 and b.out == a.out == serial(p, 20)


def test_a_serial_request_never_resumes():
    p = prompts(1)[0]
    pair = Pair(lanes=1, drafter=Pattern)
    pair.run([stream(p, 5)])
    for cache in (pair.decoder.cache, pair.follower.cache):
        cache.entries[0][1].snap["ring"] ^= 1  # a damaged kept state
    d, s = stream(p, 20), stream(p, 20, draft=False)
    pair.run([d])
    pair.run([s])
    assert d.cached == len(p) - 1 and d.out != serial(p, 20)
    assert s.cached == 0 and s.out == serial(p, 20)


def disk_tiers(tmp_path, limit=1 << 30):
    return [[DiskTier(tmp_path / "states", {"engine": "fake"}, limit=limit, rank=r)] for r in (0, 1)]


def test_resumed_from_disk_equals_fresh(tmp_path):
    ps = prompts(3)
    pair = Pair(lanes=1, drafter=Pattern, keep=1, tiers=disk_tiers(tmp_path))
    pair.run([stream(p, 8) for p in ps])  # one device entry: the first two spill to disk
    assert pair.decoder.cache.spilled == 2
    longer = ps[0] + [3, 1, 4, 1, 5]
    s = stream(longer, 20)
    assert pair.run([s]) == [serial(longer, 20)]
    assert s.cached == len(ps[0]) - 1


def test_resumed_after_a_restart_equals_fresh(tmp_path):
    p = prompts(1)[0]
    Pair(lanes=1, keep=1, tiers=disk_tiers(tmp_path)).run([stream(p, 6), stream(prompts(2)[1], 6)])
    pair = Pair(lanes=1, keep=1, tiers=disk_tiers(tmp_path))  # a new engine finds the files
    longer = p + [8, 8, 8]
    s = stream(longer, 15)
    assert pair.run([s]) == [serial(longer, 15)]
    assert s.cached == len(p) - 1


def test_resumed_from_host_memory_equals_fresh():
    ps = prompts(2)
    tiers = [[HostTier(1 << 20)], [HostTier(1 << 20)]]
    pair = Pair(lanes=1, keep=1, tiers=tiers)
    pair.run([stream(p, 5) for p in ps])
    longer = ps[0] + [2, 2]
    s = stream(longer, 15)
    assert pair.run([s]) == [serial(longer, 15)]
    assert s.cached == len(ps[0]) - 1


def test_without_a_pool_the_forward_keeps_the_rows(tmp_path):
    p = prompts(1)[0]
    pair = Pair(lanes=2, pool_pages=None, drafter=Pattern, keep=1, tiers=disk_tiers(tmp_path))
    pair.run([stream(p, 6), stream(prompts(2)[1], 6)])
    longer = p + [1, 2, 3]
    s = stream(longer, 15)
    assert pair.run([s]) == [serial(longer, 15)] and s.cached == len(p) - 1


def test_a_full_pool_evicts_kept_states_then_waits():
    ps = prompts(4)
    pair = Pair(lanes=2, pool_pages=12, page=16, drafter=Pattern)
    out = pair.run([stream(p, 20) for p in ps])
    assert out == [serial(p, 20) for p in ps]
    assert any(kind == EVICT for kind, _ in pair.link.sent)


def test_a_request_larger_than_the_pool_is_refused():
    pair = Pair(lanes=1, pool_pages=2, page=16)
    with pytest.raises(ValueError, match="page pool"):
        pair.decoder.admit(stream(prompts(1)[0], 40))


def test_pages_come_back_when_streams_end():
    pair = Pair(lanes=2, keep=0, drafter=Pattern)
    pair.run([stream(p, 20) for p in prompts(2)])
    assert pair.decoder.pool.free == pair.decoder.pool.pages


def test_long_prompts_prefill_in_pieces_beside_decoding():
    short, long = prompts(1, 10)[0], prompts(2, 120)[1]
    pair = Pair(lanes=2, prefill_rows=16)
    out = pair.run([stream(short, 30), stream(long, 10)])
    assert out == [serial(short, 30), serial(long, 10)]
    fills = [c for c in pair.decoder.forward.calls if c[0] == "prefill"]
    assert len(fills) >= 120 // 16 and all(sum(e - s for _, s, e in c[1]) <= 16 for c in fills)


def test_a_grid_family_keeps_states_on_its_grid():
    p = prompts(1, 45)[0]
    pair = Pair(lanes=1, grid=16, prefill_rows=64)
    pair.run([stream(p, 5)])
    fills = [c[1] for c in pair.decoder.forward.calls if c[0] == "prefill"]
    assert fills == [((0, 0, 32),), ((0, 32, 44),)]  # to the save point, then to the last token
    assert [len(e[0]) for e in pair.decoder.cache.entries] == [32]
    s = stream(p + [4, 4], 10)
    pair.run([s])
    assert s.cached == 32 and s.out == serial(p + [4, 4], 10)


def test_the_replay_hook_gets_the_prompt_tail_and_resumes_exactly():
    p = prompts(1)[0]
    pair = Pair(lanes=1, replay_tail=8)
    pair.run([stream(p, 5)])
    finishes = [c for c in pair.decoder.forward.calls if c[0] == "finish"]
    assert finishes == [("finish", 0, len(p) - 1, tuple(p[len(p) - 9 : len(p) - 1]))]
    longer = p + [6, 6, 6]
    fresh = Pair(lanes=1, replay_tail=8, keep=0).run([stream(longer, 12, draft=False)])
    s = stream(longer, 12)
    assert pair.run([s]) == fresh and s.cached == len(p) - 1


def test_a_cancelled_stream_frees_its_lane_for_the_next():
    ps = prompts(2)
    stopped = stream(ps[0], 50, emit=lambda new: True)
    pair = Pair(lanes=1, drafter=Pattern)
    out = pair.run([stopped, stream(ps[1], 10)])
    assert len(out[0]) <= 7 and out[1] == serial(ps[1], 10)


def test_a_yielded_background_stream_replays_to_the_same_reply():
    p = prompts(1)[0]
    want = serial(p, 30)
    pair = Pair(lanes=1, drafter=Pattern)
    sent = []
    s = stream(p, 30, background=True, emit=lambda new: sent.extend(new))
    pair.decoder.admit(s)
    for _ in range(4):
        pair.decoder.finish(pair.decoder.round())
        pair.check()
    pair.decoder.finish([s])  # the Scheduler's yield: the lane goes to a foreground request
    again = s.continued()
    pair.run([again])
    assert sent == want and again.error is None


def test_end_tokens_end_a_stream_unless_it_ignores_them():
    p = prompts(1)[0]
    plain = Pair(lanes=1, keep=0).run([stream(p, 60, draft=False, stop_eos=False)])[0]
    eos = plain[10]

    def run(stop):
        pair = Pair(lanes=1, drafter=Pattern)
        pair.decoder.eos = (eos,)
        return pair.run([stream(p, 60, stop_eos=stop)])[0]

    assert run(True) == plain[: plain.index(eos) + 1]
    assert run(False) == plain


def test_admission_waits_while_memory_is_short():
    class Short:
        def fits(self, need):
            return False

        def why(self, need):
            return "memory is short"

    pair = Pair(lanes=2, admission=Short())
    with pytest.raises(ValueError, match="cannot fit even alone: memory is short"):  # nothing would free memory
        pair.decoder.admit(stream(prompts(1)[0], 5))
    assert pair.decoder.free == [0, 1] and pair.follower.lanes == [None, None]


def test_with_nothing_live_host_kept_states_go_before_a_request_is_refused():
    from tensorfold.cuda.sessions import HostTier

    host = [HostTier(1 << 20), HostTier(1 << 20)]

    class HostBound:
        def fits(self, need):
            return host[0].used == 0

        def why(self, need):
            return "host states hold the memory"

    pair = Pair(lanes=1, keep=0, tiers=[[host[0]], [host[1]]])
    pair.run([stream(prompts(1)[0], 4)])  # keep 0: its prompt state goes to the host tier
    before = host[0].keys()
    assert before
    pair.decoder.admission = HostBound()
    s = stream(prompts(2)[1], 4)
    assert pair.run([s]) == [serial(s.prompt, 4)]
    assert not set(before) & set(host[0].keys())  # the old states went so the request could start


def unified(free):
    return Admission(unified=True, meminfo=lambda: {"MemFree": free[0]}, floor=GIB, hard_floor=GIB)


def test_a_request_waits_until_memory_covers_what_it_allocates():
    free, asked = [0], []
    pair = Pair(lanes=2, keep=0, admission=unified(free))
    pair.decoder.forward.request_bytes = lambda prompt_len, max_new: asked.append((prompt_len, max_new)) or 4096
    a, b = stream(prompts(1)[0], 5), stream(prompts(2)[1], 5)
    with pytest.raises(ValueError, match="cannot fit even alone"):
        pair.decoder.admit(a)  # alone and under the floor: refused, not started
    free[0] = 2 * GIB
    pair.decoder.admit(a)
    free[0] = GIB + 4096 - 1  # above the floor, one byte short
    with pytest.raises(NoRoom, match="under the 1.00 GiB floor"):
        pair.decoder.admit(b)
    free[0] += 1
    pair.decoder.admit(b)
    assert asked[-1] == (len(b.prompt), 5)
    assert pair.run([]) == [] and [a.out, b.out] == [serial(a.prompt, 5), serial(b.prompt, 5)]


def test_pool_pages_are_not_charged_to_memory():
    free = [GIB]  # no headroom above the floor
    pair = Pair(lanes=2, keep=0, admission=unified(free))
    a, b = stream(prompts(1)[0], 5), stream(prompts(2)[1], 5)
    pair.decoder.admit(a)
    pair.decoder.admit(b)
    assert pair.follower.tables[1].quota * pair.decoder.pool.page_bytes() > 0
    assert pair.run([]) == [] and [a.out, b.out] == [serial(a.prompt, 5), serial(b.prompt, 5)]


def test_admit_message_carries_the_sampling_and_the_prompt():
    pair = Pair(lanes=1)
    s = stream(prompts(1)[0], 5, WARM)
    pair.decoder.admit(s)
    kind, _ = pair.link.sent[-1]
    assert kind == ADMIT and pair.follower.lanes[0].sampling == WARM and pair.follower.lanes[0].prompt == s.prompt


class Even:
    """A grammar that allows even ids only: cuts a window at its first odd draft and masks every row."""

    def __init__(self):
        from tensorfold.engine.grammar import Spec

        self.spec, self.think_end, self.taken = Spec("regex", "even"), None, []

    def window(self, tokens, parents):
        keep = 1
        while keep < len(tokens) and tokens[keep] % 2 == 0:
            keep += 1
        return EvenWindow(list(tokens[:keep]), list(range(keep)))

    def advance(self, tokens):
        from tensorfold.engine.grammar import GrammarError

        if any(t % 2 for t in tokens):
            raise GrammarError(f"odd token in {tokens}")
        self.taken.extend(tokens)


class EvenWindow:
    def __init__(self, tokens, rows):
        self.tokens, self.parents, self.rows = tokens, [-1, *range(len(tokens) - 1)], rows

    def allowed(self, row):
        import numpy as np

        return np.arange(64) % 2 == 0


class EvenGrammars:
    def follow(self, packed):
        return Even()


def test_a_grammar_cuts_and_masks_on_both_ranks():
    p = prompts(1)[0]
    want = Pair(lanes=1, keep=0, grammars=EvenGrammars()).run([stream(p, 25, draft=False, constraint=Even())])[0]
    assert all(t % 2 == 0 for t in want)
    pair = Pair(lanes=2, drafter=Pattern, grammars=EvenGrammars())
    s = stream(p, 25, constraint=Even())
    other = stream(prompts(2)[1], 25)
    assert pair.run([s, other]) == [want, serial(prompts(2)[1], 25)]
    assert pair.follower.lanes[0] is None and s.constraint.taken == want[: len(s.constraint.taken)]


def test_the_scheduler_drives_the_decoder_from_many_threads():
    import threading

    from tensorfold.cuda.scheduler import Scheduler

    ps = prompts(4)
    solo = [serial(p, 20, WARM) for p in ps]
    pair = Pair(lanes=4, drafter=Pattern)
    sched = Scheduler(pair.decoder, max_streams=4)
    out = [None] * 4

    def ask(i):
        got = []
        sched.submit(ps[i], 20, WARM, True, lambda new: got.extend(new))
        out[i] = got

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    sched.close()
    assert out == solo
    same_state(pair.decoder.local, pair.follower)


def test_two_lanes_resume_one_entry_at_once():
    base = prompts(1)[0]
    a, b = base + [2, 4, 6], base + [9, 7, 5, 3]
    pair = Pair(lanes=2, drafter=Pattern)
    pair.run([stream(base, 4)])
    sa, sb = stream(a, 30), stream(b, 30)
    assert pair.run([sa, sb]) == [serial(a, 30), serial(b, 30)]
    assert sa.cached == sb.cached == len(base) - 1


def test_a_damaged_disk_entry_is_dropped_and_the_prompt_prefills_fresh(tmp_path):
    p = prompts(1)[0]
    tiers = disk_tiers(tmp_path)
    pair = Pair(lanes=1, keep=0, tiers=tiers)
    pair.run([stream(p, 4)])
    damaged = tiers[0][0].keys()
    for rank in (0, 1):
        for f in tiers[rank][0].dir.glob("*.tfs"):
            raw = bytearray(f.read_bytes())
            raw[4096] ^= 0xFF  # the first id, just past the header block
            f.write_bytes(bytes(raw))
    s = stream(p + [1], 4)
    assert pair.run([s]) == [serial(p + [1], 4)]
    kept = tiers[0][0].keys()
    assert s.cached == 0 and damaged[0] not in kept  # rank 0 found it before telling rank 1


def test_a_failed_resume_frees_the_lane_on_that_rank(tmp_path):
    p = prompts(1)[0]
    pair = Pair(lanes=1, keep=1)
    pair.run([stream(p, 4)])
    pair.follower.cache.entries.clear()  # rank 1 lost the entry rank 0 resumes from
    with pytest.raises(RuntimeError, match="another rank failed"):  # rank 0 learns it from rank 1's vote
        pair.decoder.admit(stream(p + [1], 4))
    assert pair.decoder.free == [0] and pair.follower.lanes == [None] and pair.follower.tables[0].pages == []
    assert pair.decoder.local.lanes == [None] and pair.decoder.local.tables[0].pages == []


def test_a_lower_tier_state_one_rank_lacks_starts_the_prompt_fresh(tmp_path):
    p = prompts(1)[0]
    Pair(lanes=1, keep=1, tiers=disk_tiers(tmp_path)).run([stream(p, 6), stream(prompts(2)[1], 6)])
    for f in (tmp_path / "states").rglob("rank1/*.tfs"):  # one rank's disk lost its entries
        f.unlink()
    pair = Pair(lanes=1, keep=1, tiers=disk_tiers(tmp_path))
    longer = p + [8, 8, 8]
    s = stream(longer, 15)
    assert pair.run([s]) == [serial(longer, 15)] and s.cached == 0


def test_a_failed_admission_on_rank_0_is_undone_on_every_rank(monkeypatch):
    pair = Pair(lanes=1, keep=0)
    real, calls = pair.decoder.local._admit, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("rank 0 fails after the broadcast")
        return real(*a, **k)

    monkeypatch.setattr(pair.decoder.local, "_admit", flaky)
    with pytest.raises(RuntimeError, match="after the broadcast"):
        pair.decoder.admit(stream(prompts(1)[0], 4))
    assert pair.follower.lanes == [None] and pair.follower.tables[0].pages == [] and pair.decoder.free == [0]
    assert pair.run([stream(prompts(1)[0], 4)]) == [serial(prompts(1)[0], 4)]


def test_a_follower_votes_instead_of_leaving_on_a_failed_round():
    from tensorfold.cuda.lanes.link import ROUND, STOP

    pair = Pair(lanes=1)
    votes = []

    class Script:
        def __init__(self):
            self.msgs = [(ROUND, [7]), (STOP, [])]  # a malformed round: the follower's apply raises

        def recv(self):
            return self.msgs.pop(0)

        def agree(self, ok):
            votes.append(ok)
            return False

    pair.follower.follow(Script())  # returns at STOP instead of raising out of the loop
    assert votes == [False]


def test_a_failed_round_fails_every_admitted_request_and_one_rank_goes_on():
    from tensorfold.cuda.lanes.link import LocalLink

    p = prompts(2)
    pair = Pair(lanes=2, prefill_rows=512)
    pair.decoder.link = LocalLink()
    a, b = stream(p[0], 5), stream(p[1], 5)
    pair.decoder.admit(a)
    pair.decoder.admit(b)
    real = pair.decoder.forward.window
    pair.decoder.forward.window = lambda *args, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        pair.decoder.round()
    assert pair.decoder.drop() == [a, b] and pair.decoder.free == [0, 1] and pair.decoder.live() == 0
    pair.decoder.forward.window = real
    s = stream(p[0], 5)
    drive(pair.decoder, [s], lanes=2)
    assert s.out == serial(p[0], 5)


def test_a_failed_round_on_two_ranks_refuses_further_work():
    pair = Pair(lanes=1)
    pair.decoder.admit(stream(prompts(1)[0], 5))
    pair.decoder.drop()
    with pytest.raises(RuntimeError, match="two ranks"):
        pair.decoder.admit(stream(prompts(1)[0], 5))


def test_a_spilled_state_carries_no_rows_past_its_positions():
    import numpy as np

    from tensorfold.cuda.kvpool import PagePool, Plane
    from tensorfold.cuda.lanes.follow import Held, HeldCodec

    pool = PagePool([Plane("kv", 2), Plane("c", 2, per_tokens=4)], 4, 16)
    pool.buffers["kv"][:] = 7
    pool.buffers["c"][:] = 9
    class NoState:
        def to_host(self, snap):
            return {}

        def from_host(self, arrays):
            return None

    out = HeldCodec(NoState(), pool).to_host(Held(None, [0, 1], tokens=21))
    kv, c = out["page/kv"], out["page/c"]
    assert (kv[:21] == 7).all() and (kv[21:] == 0).all() and kv.shape == (32, 2)
    assert (c[:5] == 9).all() and (c[5:] == 0).all() and c.shape == (8, 2)  # row 5 (20-23) is incomplete
    assert (pool.buffers["kv"] == 7).all()  # the device rows are left alone


def test_each_window_names_the_candidates_its_own_sampling_needs():
    from tensorfold.engine.exact_sampling import MARGIN, Sampling

    pair = Pair(lanes=3, prefill_rows=512)
    seen = []
    for fwd in (pair.decoder.forward, pair.follower.forward):
        real = fwd.window

        def window(rows, count, masks=None, counts=None, real=real, fwd=fwd):
            if fwd is pair.decoder.forward:
                seen.append((count, list(counts)))
            return real(rows, count, masks)

        fwd.window = window
    for lanes in (pair.decoder.local, pair.follower):
        lanes._counts = True
    p = prompts(3)
    pair.run([stream(p[0], 3), stream(p[1], 3, Sampling(seed=1, temperature=0.8, top_k=0, top_p=0.9)),
              stream(p[2], 3, Sampling(seed=2, temperature=0.8, top_k=5))])
    vocab = pair.decoder.forward.vocab
    assert seen and all(count == max(counts) for count, counts in seen)
    assert any(sorted(counts) == [1, 5 + MARGIN, vocab] for _, counts in seen)


@pytest.mark.parametrize("where", ["forward.reset", "_resume", "drafter.reset", "depth.reset"])
@pytest.mark.parametrize("rank", [0, 1])
def test_an_admission_failing_anywhere_on_either_rank_leaves_both_as_they_were(where, rank, monkeypatch):
    from tensorfold.cuda.drafting import StaticDepth

    p = prompts(1)[0]
    pair = Pair(lanes=1, keep=1, drafter=Pattern, depth=lambda: StaticDepth(2))
    pair.run([stream(p, 4)])  # a kept state, so a resume runs
    side = pair.decoder.local if rank == 0 else pair.follower
    owner, name = (side, "_resume") if where == "_resume" else (getattr(side, where.split(".")[0]), where.split(".")[1])
    real = getattr(owner, name)

    def fail(*a, **k):
        if where == "_resume":
            real(*a, **k)  # its pages are mapped and written first
        raise RuntimeError(f"{where} fails on rank {rank}")

    monkeypatch.setattr(owner, name, fail)
    before = [(lanes.lanes, [t.quota for t in lanes.tables], [list(t.pages) for t in lanes.tables], dict(lanes.pool.refs),
               lanes.pool.free) for lanes in (pair.decoder.local, pair.follower)]
    with pytest.raises(RuntimeError):
        pair.decoder.admit(stream(p + [1], 4))
    after = [(lanes.lanes, [t.quota for t in lanes.tables], [list(t.pages) for t in lanes.tables], dict(lanes.pool.refs),
              lanes.pool.free) for lanes in (pair.decoder.local, pair.follower)]
    assert after == before and pair.decoder.free == [0]
    a, b = pair.decoder.local, pair.follower
    assert a.state() == b.state()  # the device cache's order too: EVICT names entries by index
    for plane in a.pool.buffers:
        assert (a.pool.buffers[plane] == b.pool.buffers[plane]).all()
    monkeypatch.setattr(owner, name, real)
    s = stream(p + [1], 4)
    assert pair.run([s]) == [serial(p + [1], 4)]


def test_columns_past_a_windows_need_cannot_change_its_tokens():
    from tensorfold.engine.exact_sampling import Sampling

    def poison(lanes):
        real = lanes.forward.window

        def window(rows, count, masks=None, counts=None):
            cand = real(rows, count, masks)
            at = 0
            for r, need in zip(rows, counts):
                n = len(r.tokens)
                cand.values[at:at + n, need:] = 1e30  # stale columns a forward never filled
                cand.ids[at:at + n, need:] = 7
                at += n
            return cand

        lanes.forward.window = window
        lanes._counts = True

    p = prompts(3)
    sampled = [None, Sampling(seed=1, temperature=0.8, top_k=0, top_p=0.9), Sampling(seed=2, temperature=0.8, top_k=5)]
    pair = Pair(lanes=3, prefill_rows=512)
    poison(pair.decoder.local)
    poison(pair.follower)
    streams = [stream(p[i], 6, sampled[i]) for i in range(3)]
    assert pair.run(streams) == [serial(p[i], 6, sampled[i]) for i in range(3)]


def test_a_failing_disk_never_stops_serving_and_every_page_comes_back(tmp_path):
    from cuda_lane_fakes import FailingTier

    p = prompts(4)
    pair = Pair(lanes=1, keep=1, tiers=[[FailingTier()], [FailingTier()]])
    outs = pair.run([stream(q, 5) for q in p])  # each new state pushes the last one down, into a failing tier
    assert outs == [serial(q, 5) for q in p]
    cache = pair.decoder.cache
    assert cache.dropped == len(p) - 1 and len(cache.entries) == 1
    pages = pair.decoder.pool.pages - pair.decoder.pool.free
    assert pages == len(cache.entries[0][1].pages)  # only the one kept state holds pages
