"""The two-rank engine on the tiny checkpoint, its ranks as two threads on one GPU: rank 1's ``follow`` mirrors every
request rank 0 serves (equal stats, kept snapshots and live ids), the serial reference equals the drafted reply, a
resend resumes from its kept snapshot with the fresh reply and drafting stats, several conversations keep their
snapshots, ``stop_eos`` False decodes to ``max_tokens``, a refused prompt reaches no collective, and serving
reserves no device memory past what construction reserved."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("safetensors")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_tiny
from dsv41_pair import pair, run_pair

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.cuda import engine as E

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture
CONTEXT = 4096
REPLY = 24
LONG = 2500                             # past one 2048-row prompt chunk
KEYED = Sampling(seed=11, temperature=1.0, top_k=20, top_p=0.95)
TIMED = ("prefill_s", "decode_s", "stages_ms")     # stats that measure time


def _ids(seed: int, n: int, vocab: int = 1024) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, vocab, (n,), generator=g).tolist()


def _build(tiny_dir):
    comms = pair()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TF_DSV41_CACHE_GIB", "0.25")
        mp.setenv("TF_DSV41_CACHE_ENTRIES", "8")

        def build(r):
            return lambda comm: E.DeepSeekV41Engine(tiny_dir, rank=r, master="", port=0, policy=(3, None),
                                                    context=CONTEXT, context_explicit=True, comm=comm, graphs=False)

        engines = run_pair(build(0), build(1), comms)
    assert [e.rank for e in engines] == [0, 1] and engines[0].limit == CONTEXT
    return engines


@pytest.fixture(scope="module")
def ranks(tiny_dir):
    return _build(tiny_dir)


def _serve(ranks, requests):
    """Rank 0 serves ``requests`` (generate's arguments after ``on_tokens``) then shuts down while rank 1 follows ->
    [(tokens, stats)] of rank 0 and [stats] of rank 1; both ranks then hold the same snapshots and live ids."""

    e0, e1 = ranks
    seen1 = []
    run1 = e1._run
    e1._run = lambda *a: seen1.append(run1(*a)) or seen1[-1]

    def lead(_):
        out = []
        for args, kw in requests:
            got: list[int] = []
            stats = e0.generate(args[0], args[1], args[2], got.extend, **kw)
            out.append((got, stats))
        e0.shutdown()
        return out

    try:
        seen0, _ = run_pair(lead, lambda _: e1.follow(), (e0.comm, e1.comm))
    finally:
        del e1._run
    assert len(seen1) == len(seen0)
    for (_, a), b in zip(seen0, seen1):
        assert {k: v for k, v in a.items() if k not in TIMED} == {k: v for k, v in b.items() if k not in TIMED}
    assert e0.kept.held() == e1.kept.held() and e0.kept.live == e1.kept.live, "the ranks kept different snapshots"
    return seen0


def req(prompt, max_tokens=REPLY, sampling=KEYED, **kw):
    return (prompt, max_tokens, sampling), kw


def test_a_startup_failure_on_one_rank_is_named_on_both(tiny_dir, tmp_path, monkeypatch):
    bad = tmp_path / "rank1"
    bad.symlink_to(tiny_dir)
    engram = E.DeepSeekV41Engine._engram

    def broken(model_dir, cfg):
        if model_dir == bad:
            raise KeyError("weight_map")
        return engram(model_dir, cfg)

    monkeypatch.setattr(E.DeepSeekV41Engine, "_engram", staticmethod(broken))

    def build(r, model_dir):
        def body(comm):
            try:
                E.DeepSeekV41Engine(model_dir, rank=r, master="", port=0, policy=(3, None), context=CONTEXT,
                                    context_explicit=True, comm=comm, graphs=False)
            except ValueError as exc:
                return str(exc)
            return "started"
        return body

    zero, one = run_pair(build(0, tiny_dir), build(1, bad))
    assert "on rank 1" in zero and "on this rank: KeyError" in one and "weight_map" in one


def test_a_new_engine_reserves_no_more_to_serve(tiny_dir):
    torch.cuda.empty_cache()
    engines = _build(tiny_dir)
    assert all("guard_left" not in vars(e.e.pbuf.exl3) for e in engines), "the warm left the overflow guard's checks"
    torch.cuda.synchronize()
    reserved = torch.cuda.memory_reserved()
    open_k = Sampling(seed=4, temperature=1.0, top_k=0)
    short = _ids(41, 300)
    _serve(engines, [req(_ids(40, LONG)), req(short, sampling=open_k), req(short, sampling=open_k)])
    torch.cuda.synchronize()
    assert torch.cuda.memory_reserved() == reserved


def test_rank1_mirrors_and_serial_equals_drafted_and_a_resend_resumes(ranks):
    for i, sampling in enumerate((KEYED, None)):
        prompt = _ids(1 + i, 300)
        fresh, serial, resent = _serve(ranks, [req(prompt, sampling=sampling), req(prompt, sampling=sampling,
                                                                               draft=False),
                                               req(prompt, sampling=sampling)])
        assert len(fresh[0]) == REPLY and fresh[1]["cached"] == 0 and fresh[1]["drafted"] > 0
        assert serial[0] == fresh[0] and serial[1]["drafted"] == 0 and serial[1]["policy"] == "0"
        assert serial[1]["cached"] == 0 and not serial[1]["drafts"]
        assert resent[1]["cached"] == len(prompt) - 1, "an identical resend resumes from its kept point"
        assert resent[0] == fresh[0]
        for key in ("drafted", "accepted", "rounds", "tokens_per_round", "sha256"):
            assert resent[1][key] == fresh[1][key], key


def test_resends_keep_every_conversation(ranks):
    prompts = [_ids(10 + i, n) for i, n in enumerate((120, 333, 57))]
    first = _serve(ranks, [req(p) for p in prompts])
    held = len(ranks[0].kept.held())
    assert held >= 3
    for turn in range(3):
        again = _serve(ranks, [req(p) for p in prompts])
        for p, a, b in zip(prompts, first, again):
            assert b[1]["cached"] == len(p) - 1, f"resend {turn}: a {len(p)}-token prompt"
            assert b[0] == a[0] and (b[1]["drafted"], b[1]["accepted"]) == (a[1]["drafted"], a[1]["accepted"])
        assert len(ranks[0].kept.held()) == held, "every snapshot kept"
    p = prompts[1]
    nxt = p + first[1][0] + _ids(20, 9)     # the conversation's next turn resumes its first prompt
    (_, stats), = _serve(ranks, [req(nxt)])
    assert stats["cached"] == len(p) - 1


def test_stop_eos_false_decodes_to_max_tokens(ranks):
    prompt = _ids(3, 80)
    (reply, _), = _serve(ranks, [req(prompt, 60)])
    m = next(i for i in range(4, len(reply)) if reply[i] not in reply[:i])
    for e in ranks:
        e.e.eos = e.eos = (reply[m],)
    try:
        stopped, ignored = _serve(ranks, [req(prompt, 60), req(prompt, 60, stop_eos=False)])
    finally:
        for e in ranks:
            e.e.eos = e.eos = (e.w.cfg.eos_token_id,)
    assert stopped[0] == reply[:m + 1]
    assert ignored[0] == reply and len(ignored[0]) == 60


def test_refused_requests_reach_no_collective(ranks):
    e0 = ranks[0]

    def gather(*a):
        raise AssertionError("a refused request reached the all-gather")

    e0.comm.all_gather = gather
    try:
        for prompt, match in (([5] * e0.limit, "contexts up to"), ([], "empty"), ([5, 1024], r"\[0, 1024\)")):
            with pytest.raises(ValueError, match=match):
                e0.generate(prompt, 4, None, lambda t: None)
    finally:
        del e0.comm.all_gather


def test_a_top_k_past_int32_is_served_on_both_ranks_and_the_next_request_too(ranks):
    prompt = _ids(5, 90)
    huge, whole, after = _serve(ranks, [req(prompt, sampling=Sampling(seed=7, temperature=1.0, top_k=2**31)),
                                        req(prompt, sampling=Sampling(seed=7, temperature=1.0, top_k=1024)),
                                        req(_ids(6, 40))])
    assert huge[0] == whole[0], "a top_k past the vocabulary keeps every token, as its size (1024) does"
    assert len(after[0]) == REPLY


def test_serving_allocates_nothing_once_warm(ranks):
    long = _ids(30, LONG)
    others = [_ids(31, LONG), _ids(32, 200)]
    _serve(ranks, [req(long), req(long), req(others[0]), req(long, draft=False)])
    torch.cuda.synchronize()
    reserved = torch.cuda.memory_reserved()
    got = _serve(ranks, [req(p) for p in (long, others[0], others[1], long, others[0], others[1])])
    torch.cuda.synchronize()
    assert torch.cuda.memory_reserved() == reserved
    assert [s["cached"] for _, s in got[3:]] == [len(p) - 1 for p in (long, others[0], others[1])]
