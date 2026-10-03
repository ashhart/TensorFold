"""A stopped GLM request ends on every rank within a round: real decode loops on two CPU ranks, a fake split head."""

from __future__ import annotations

import importlib
import threading
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tests.test_cuda_geometry import allocations  # noqa: E402,F401  (fixture: fake triton, so decode imports)

decode = drafter_choice = engine_mod = None     # the CUDA modules, imported per test under the fake triton

V, EOS, HIDDEN, PROMPT = 64, 5, 4, 10
WAIT = 20


class Comm:
    """All-gather between rank threads; ranks that send different sizes fail together (a real desync would hang)."""

    def __init__(self, world: int) -> None:
        self.world = world
        self.slots: list = [None] * world
        self.bar = threading.Barrier(world, timeout=WAIT)
        self.sizes: list[list[int]] = [[] for _ in range(world)]

    def all_gather(self, send, recv) -> None:
        rank = int(threading.current_thread().name.rsplit("-", 1)[-1])
        self.slots[rank] = send.detach().clone().reshape(-1)
        self.sizes[rank].append(send.numel())
        self.bar.wait()
        sizes = {s.numel() for s in self.slots}
        both = torch.cat(list(self.slots)) if len(sizes) == 1 else None
        self.bar.wait()
        if both is None:
            raise RuntimeError(f"rank {rank}: the ranks' all-gathers differ in size {sorted(sizes)}")
        recv.copy_(both.view(recv.shape))


def row(position: int) -> torch.Tensor:
    """The full vocabulary's logits at a position (the same on every rank), no end token unless asked."""

    g = torch.Generator().manual_seed(1000 + position)
    x = 3.0 * torch.randn(V, generator=g)
    x[EOS] = -30.0
    return x


class FakeEngine:
    """The single-stream engine surface the decode loops use; ``Engine.sample`` is the real one."""

    def sample(self, *args, **kwargs):
        return decode.Engine.sample(self, *args, **kwargs)

    def __init__(self, rank: int, world: int, comm) -> None:
        share = V // world
        self.w = NS(comm=comm, world=world, vocab_offset=rank * share, device="cpu",
                    cfg=NS(eos=(EOS,), hidden=HIDDEN))
        self.share = share
        self.st = NS(pos=PROMPT, mtp_len=PROMPT, mtp_drafted=0)
        self.st.set_mtp_len = lambda n: setattr(self.st, "mtp_len", n)
        self.buf = None
        self.constraint = self.window = self.vote = self.head = None
        self.last_hidden = torch.zeros((1, HIDDEN), dtype=torch.bfloat16)

    def logits(self, positions) -> torch.Tensor:
        lo = self.w.vocab_offset
        return torch.stack([row(p)[lo:lo + self.share] for p in positions])

    def forward(self, tokens):
        return self.logits([self.st.pos + 1 + r for r in range(len(tokens))])

    def verify_window(self, tokens):
        return tokens

    def follow(self, tokens) -> None:
        pass

    def main_hidden(self, rows):
        return torch.zeros((rows.stop - rows.start, HIDDEN), dtype=torch.bfloat16)

    def tap_rows(self, n, b=None):
        return torch.zeros((n, HIDDEN))


def fake_draft(e, hidden, next_tokens, position, count, sampling, confidence=0.0):
    """An MTP chain: its first draft is the model's own sample (a gather without the vote), then a wrong guess."""

    first = e.sample(e.logits([position]), [position], sampling, draft=True)[0]
    return ([first] + [(first + 1) % V] * (count - 1))[:count]


class Drafter:
    """A DFlash2 block: the right next token half the rounds, else a miss."""

    block = 6

    def __init__(self, e) -> None:
        self.e, self.n = e, 0

    def propose(self, pending, depth, sampling, confidence=0.0, **rule):
        self.n += 1
        p = self.e.st.pos + 1
        good = self.e.sample(self.e.logits([p]), [p], sampling, draft=True)[0]
        return [good if self.n % 2 else (good + 3) % V][:depth]

    def add_taps(self, taps) -> None:
        pass


@pytest.fixture(autouse=True)
def cpu_loops(allocations, monkeypatch):  # noqa: F811
    global decode, drafter_choice, engine_mod
    decode = importlib.import_module("tensorfold.families.glm5_next.cuda.decode")
    drafter_choice = importlib.import_module("tensorfold.families.glm5_next.cuda.drafter_choice")
    engine_mod = importlib.import_module("tensorfold.families.glm5_next.cuda.engine")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)     # the loops wait on the GPU each round
    monkeypatch.setattr(decode, "_sync", lambda w: None)
    monkeypatch.setattr(decode, "commit", lambda w, st, b, R, keep: setattr(st, "pos", st.pos + keep))
    monkeypatch.setattr(decode, "draft", fake_draft)
    monkeypatch.setattr(decode, "absorb", lambda e, hidden, nxt: None)
    monkeypatch.setattr(drafter_choice, "draft", fake_draft)
    monkeypatch.setattr(drafter_choice, "absorb", lambda e, hidden, nxt: None)
    monkeypatch.setattr(drafter_choice, "_sync", lambda w: None)
    monkeypatch.setattr(drafter_choice, "commit", lambda w, st, b, R, keep: setattr(st, "pos", st.pos + keep))


def loop(kind: str, e, count: int, sampling, on_tokens):
    first = 7
    if kind == "serial":
        return decode.serial_decode(e, first, count, sampling, stop_eos=True, on_tokens=on_tokens)
    if kind == "mtp":
        return decode.mtp_decode(e, first, count, sampling, policy=decode.DepthPolicy(3, fixed=True),
                                 stop_eos=True, on_tokens=on_tokens)
    if kind == "dflash":
        return decode.dflash_decode(e, Drafter(e), first, count, sampling, policy=decode.DepthPolicy(2, fixed=True),
                                    stop_eos=True, on_tokens=on_tokens)
    return drafter_choice.auto_decode(e, None, first, count, sampling, choice=None,
                                      m_policy=decode.DepthPolicy(3, fixed=True),
                                      f_policy=decode.DepthPolicy(5, fixed=True), stop_eos=True, on_tokens=on_tokens)


def run_ranks(kind: str, *, world: int = 2, count: int = 60, sampling=None, stop_at: int | None = None,
              vote: bool = True):
    """Each rank's loop on a thread, rank 0 stopping at call ``stop_at``: per rank (result, calls, engine)."""

    comm = Comm(world) if world > 1 else None
    out: list = [None] * world
    errors: list = []

    def rank(r: int) -> None:
        try:
            e = FakeEngine(r, world, comm)
            calls: list[list[int]] = []

            def on_tokens(new):
                calls.append(list(new))
                return r == 0 and stop_at is not None and len(calls) >= stop_at

            fn = on_tokens if r == 0 else (lambda new: None)
            if vote:
                fn = e.vote = decode.StopVote(fn)
            res = loop(kind, e, count, sampling, fn)
            out[r] = (res, calls, e)
        except BaseException as exc:            # noqa: BLE001  reported by the test
            errors.append(exc)

    threads = [threading.Thread(target=rank, args=(r,), name=f"rank-{r}") for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT)
    assert not any(t.is_alive() for t in threads), "a rank is still decoding: the ranks disagree"
    if errors:
        raise errors[0]
    return out, comm


SAMPLINGS = {"greedy": None, "top_k": Sampling(1234, 1.0, 20, 0.95), "nucleus": Sampling(99, 0.8, 0, 0.9)}
KINDS = ("serial", "mtp", "dflash", "auto")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", sorted(SAMPLINGS))
def test_a_stop_on_rank0_ends_every_rank_after_the_next_round(kind, mode):
    sampling = SAMPLINGS[mode]
    ref = run_ranks(kind, sampling=sampling, vote=False)[0][0][0]
    assert len(ref.tokens) == 60                                     # unstopped: to max_tokens, as before
    for stop_at in (1, 3):
        ranks, comm = run_ranks(kind, sampling=sampling, stop_at=stop_at)
        (r0, calls0, e0), (r1, _, e1) = ranks
        # on_tokens hears a round's tokens after its sample: the stop rides on the next round's sample, then both end
        assert r0.rounds == r1.rounds == stop_at + 1
        assert r0.tokens == r1.tokens and e0.st.pos == e1.st.pos
        assert e0.vote.stop and e1.vote.stop and not e1.vote.mine
        # the tokens the server took before its stop are the unstopped reply's (and so is the whole stopped run)
        sent = [t for call in calls0[:stop_at] for t in call]
        assert [7] + sent == ref.tokens[:1 + len(sent)]
        assert r0.tokens == ref.tokens[:len(r0.tokens)]
        assert comm.sizes[0] == comm.sizes[1]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", sorted(SAMPLINGS))
def test_the_vote_never_changes_a_reply(kind, mode):
    """Voting (one word more in each verify gather) and not voting give the same tokens and rounds."""

    sampling = SAMPLINGS[mode]
    plain = run_ranks(kind, sampling=sampling, vote=False)[0][0][0]
    ranks, comm = run_ranks(kind, sampling=sampling)
    for res, calls, e in ranks:
        assert res.tokens == plain.tokens and res.rounds == plain.rounds
        assert not e.vote.stop
    assert [7] + [t for call in ranks[0][1] for t in call] == plain.tokens
    # each verify sample's first gather carries the vote: one word more on every rank
    unvoted = run_ranks(kind, sampling=sampling, vote=False)[1]
    assert sum(comm.sizes[0]) == sum(unvoted.sizes[0]) + plain.rounds
    assert comm.sizes[0] == comm.sizes[1]


@pytest.mark.parametrize("kind", KINDS)
def test_one_rank_stops_too(kind):
    ranks, _ = run_ranks(kind, world=1, sampling=SAMPLINGS["top_k"], stop_at=2)
    assert ranks[0][0].rounds == 3


def test_without_a_vote_the_loops_decode_to_the_end_as_before():
    ranks, _ = run_ranks("mtp", stop_at=1, vote=False)
    assert all(len(res.tokens) == 60 for res, _, _ in ranks)


# -- the engine: GlmEngine._run on two ranks ---------------------------------------------------------------------
def glm_rank(r: int, comm):
    g = object.__new__(engine_mod.GlmEngine)
    e = FakeEngine(r, 2, comm)
    g.e, g.rank, g.world, g.eos = e, r, 2, (EOS,)
    g.drafter = None
    g.costs, g.cache, g.live = {}, [], []
    g._remember = lambda snap: None
    return g


@pytest.fixture
def fake_prefill(monkeypatch):
    def prefill(e, prompt, sampling, *, mtp=True, drafter=None, resume=None, keep_at=None, keep=None):
        e.st.pos = e.st.mtp_len = len(prompt)
        return 7

    monkeypatch.setattr(decode, "prefill", prefill)


@pytest.mark.parametrize("spec,stop_at", [("3", 1), ("3", 4), ("0", 2)])
def test_engine_run_stops_both_ranks_and_resets_the_vote(fake_prefill, spec, stop_at):
    """Rank 0 asks to stop and rank 1's callback is a no-op: both ``_run`` calls end after the same round."""

    comm = Comm(2)
    prompt = list(range(20, 20 + PROMPT))
    code = engine_mod.encode_policy(spec)
    out: list = [None, None]
    errors: list = []
    heard: list[list[int]] = []

    def server(new):
        heard.append(list(new))
        return len(heard) >= stop_at

    def rank(r: int) -> None:
        try:
            g = glm_rank(r, comm)
            stats = g._run(prompt, 500, None, True, server if r == 0 else (lambda new: None), code, None, True)
            out[r] = (stats, g)
        except BaseException as exc:            # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=rank, args=(r,), name=f"rank-{r}") for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT)
    assert not any(t.is_alive() for t in threads)
    if errors:
        raise errors[0]
    (s0, g0), (s1, g1) = out
    # the first token is call 1 (before any round): its stop ends the run after round 1
    assert s0["rounds"] == s1["rounds"] == stop_at
    assert s0["stopped"] and s1["stopped"]
    assert g0.e.vote is None and g1.e.vote is None
    assert g0.e.st.pos == g1.e.st.pos and g0.live == g1.live
    assert len(g0.live) == g0.e.st.pos


def test_engine_run_unstopped_is_unchanged(fake_prefill):
    """No stop: the run decodes to max_tokens on both ranks, no ``stopped`` in its stats."""

    comm = Comm(2)
    prompt = list(range(20, 20 + PROMPT))
    out: list = [None, None]

    def rank(r: int) -> None:
        g = glm_rank(r, comm)
        out[r] = g._run(prompt, 40, None, True, lambda new: False, engine_mod.encode_policy("3"), None, True)

    threads = [threading.Thread(target=rank, args=(r,), name=f"rank-{r}") for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT)
    assert out[0] is not None and out[1] is not None
    assert "stopped" not in out[0] and out[0]["sha256"] == out[1]["sha256"]
