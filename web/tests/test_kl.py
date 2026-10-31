"""KL divergence between two engines: the estimate on top-k log-probabilities, the pairing, and a Bench run."""

import json
import math
import time

import pytest

from tensorfold_web import bench, kl
from tensorfold_web.mock_engine import serve as mock_serve


def S(tok: str, **top: float) -> kl.Step:
    return kl.Step(tok.encode(), {k.encode(): v for k, v in top.items()})


def test_divergence_on_top_tokens():
    p = {b"a": 0.7, b"b": 0.2}
    assert kl.divergence(p, dict(p)) == 0.0
    q = {b"a": 0.5, b"b": 0.4}
    exact = 0.7 * math.log(0.7 / 0.5) + 0.2 * math.log(0.2 / 0.4) + 0.1 * math.log(0.1 / 0.1)
    assert kl.divergence(p, q) == pytest.approx(exact)
    # a token the candidate does not list has at most its smallest listed probability: a large P(c) then counts
    assert kl.divergence({b"a": 0.6, b"c": 0.3}, {b"a": 0.6, b"b": 0.35, b"d": 0.01}) > 0.8


def test_pair_stops_after_the_first_different_token():
    ref = [S("The", The=0.9), S(" sky", sky=0.8), S(" is", is_=0.5), S(" blue", blue=0.9)]
    same = kl.pair(ref, list(ref))
    assert same["identical"] and same["positions"] == 4 and same["differs_at"] is None and max(same["kl"]) == 0
    other = [S("The", The=0.9), S(" sea", sky=0.4, sea=0.5), S(" is", is_=0.5)]
    p = kl.pair(ref, other)
    assert p["differs_at"] == 1 and p["positions"] == 2 and not p["identical"] and p["kl"][1] > 0
    assert p["worst"][0]["at"] == 1 and p["worst"][0]["ref"] == " sky" and p["worst"][0]["cand"] == " sea"
    s = kl.summary([dict(same, prompt="a"), dict(p, prompt="b")])
    assert s["positions"] == 6 and s["identical"] == 0.5 and s["differs_at_median"] == 1


@pytest.fixture
def mock():
    server = mock_serve(port=0, rate_tps=4000.0)
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _run(tmp_path, base, **kw):
    runner = bench.Bench(bench.Store(tmp_path / "bench.sqlite"), lambda: (base, None, "mock"))
    runner.power = False
    run = runner.start("kl", suites=["code"], exactness=False, **kw)
    end = time.time() + 60
    while runner.busy() and time.time() < end:
        time.sleep(0.05)
    return runner.store.run(run["id"])


def test_bench_kl_run_against_drafts_off_under_load_and_another_address(tmp_path, mock):
    run = _run(tmp_path, mock)
    assert run["status"] == "done", run.get("error")
    assert run["kl_reference"] == {"kind": "drafts-off", "drafts_off": True} and run["kl"]["prompts"] == 2
    assert run["kl"]["mean_kl"] == 0.0 and run["kl"]["identical"] == 1.0 and run["kl"]["positions"] > 10
    loaded = _run(tmp_path, mock, kl_load=2)
    assert loaded["status"] == "done" and loaded["kl_load"] == 2 and loaded["kl"]["mean_kl"] == 0.0
    other = _run(tmp_path, mock, reference=(mock, "secret-key", "mock"))
    assert other["kl_reference"] == {"kind": "address", "url": mock, "model": "mock"}
    assert "secret-key" not in str(other)            # the reference's key is never kept with the run


def test_report_and_compare_of_a_kl_run(tmp_path, mock):
    from tensorfold_web import compare, report

    run = _run(tmp_path, mock)
    text = " ".join(report.findings(run)) + report.share_text(run)
    page = report.html_report(run)
    assert "drafts off" in text and "mean KL 0.0000" in text
    assert "Mean KL" in page and "KL divergence" in page
    with pytest.raises(compare.Refused):
        compare.compare(run, run)


def _top(d: dict, k: int) -> dict:
    return dict(sorted(d.items(), key=lambda kv: -kv[1])[:k])


def test_divergence_is_a_lower_bound_of_the_full_kl():
    import random

    rng = random.Random(7)
    for _ in range(400):
        n = rng.randint(3, 60)
        full_p = [rng.random() ** rng.choice((1, 4, 12)) for _ in range(n)]
        full_q = [x * math.exp(rng.gauss(0, rng.choice((0.001, 0.05, 1.0)))) for x in full_p]
        if rng.random() < 0.3:
            rng.shuffle(full_q)
        sp, sq = sum(full_p), sum(full_q)
        P = {bytes([i]): x / sp for i, x in enumerate(full_p)}
        Q = {bytes([i]): x / sq for i, x in enumerate(full_q)}
        exact = sum(P[t] * math.log(P[t] / Q[t]) for t in P)
        k = rng.randint(1, 20)
        assert kl.divergence(_top(P, k), _top(Q, k)) <= exact + 1e-9


def test_a_difference_without_a_bound_is_still_told_apart():
    # P lists c, Q lists b: the two could be the same distribution, so the bound is 0, but the lists differ
    p, q = {b"a": 0.9, b"c": 0.05}, {b"a": 0.9, b"b": 0.05}
    assert kl.divergence(p, q) == pytest.approx(0.0)
    assert not kl.same_distribution(p, q) and kl.same_distribution(p, dict(p))


def test_background_load_counts_what_got_through():
    server = mock_serve(port=0, rate_tps=4000.0, api_key="needed")
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
        with kl.Load(kl.Target(url, {}, "mock"), 2, [{"role": "user", "content": "hi"}]) as refused:
            time.sleep(0.5)
        assert refused.ok == 0 and refused.failed >= 2 and "401" in refused.last_error
        with kl.Load(kl.Target(url, {"Authorization": "Bearer needed"}, "mock"), 2,
                     [{"role": "user", "content": "hi"}]) as served:
            time.sleep(0.5)
        assert served.ok >= 2 and served.failed == 0
    finally:
        server.shutdown()


def test_background_load_says_how_long_it_was_all_there():
    server = mock_serve(port=0, rate_tps=400.0)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
        with kl.Load(kl.Target(url, {}, "mock"), 3, [{"role": "user", "content": "hi"}]) as load:
            time.sleep(1.5)
        assert load.ok >= 3 and load.failed == 0
        assert load.full_share is not None and 0.5 < load.full_share <= 1.0
    finally:
        server.shutdown()


def test_a_load_not_established_is_said(tmp_path, mock, monkeypatch):
    class Idle:                                     # replies came back only after the measurement
        def __init__(self, *_a):
            self.ok, self.failed, self.late, self.full_share, self.last_error = 0, 0, 1, 0.0, None

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            pass

        def stop(self, wait=0.0):
            pass
    monkeypatch.setattr(kl, "Load", Idle)
    run = _run(tmp_path, mock, kl_load=3)
    assert run["status"] == "done" and run["kl_load_verified"] is False
    assert any("not established" in n for n in run["notes"])
    from tensorfold_web import report
    assert "not established" in report.kl_line(run)


def test_a_refused_start_leaves_the_running_reference_alone(tmp_path, mock):
    runner = bench.Bench(bench.Store(tmp_path / "bench.sqlite"), lambda: (mock, None, "mock"))
    runner.power = False
    runner.start("kl", suites=["code"], exactness=False, reference=(mock, "first", "mock"))
    with pytest.raises(RuntimeError):
        runner.start("kl", suites=["code"], exactness=False, reference=(mock, "second", "mock"))
    assert runner._kl_reference[1] == "first"
    end = time.time() + 60
    while runner.busy() and time.time() < end:
        time.sleep(0.05)


def test_a_reference_reached_with_headers_of_its_own(tmp_path):
    # another TensorFold Web's relay takes its token as a header: the reference carries it, and it is never kept with the run
    server = mock_serve(port=0, rate_tps=4000.0, api_key="needed")
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        runner = bench.Bench(bench.Store(tmp_path / "bench.sqlite"), lambda: (url, "needed", "mock"))
        runner.power = False
        run = runner.start("kl", suites=["code"], exactness=False, reference=(url, None, "mock", {"x-api-key": "needed"}))
        end = time.time() + 60
        while runner.busy() and time.time() < end:
            time.sleep(0.05)
        done = runner.store.run(run["id"])
        assert done["status"] == "done", done.get("error")
        assert "needed" not in json.dumps(done["kl_reference"])
    finally:
        server.shutdown()


def test_a_narrow_distribution_is_said_rather_than_read_as_an_exact_match():
    from tensorfold_web import report

    ref = [S("The", The=0.9, A=0.05), S(" sky", sky=0.8, sea=0.1)]
    narrow = [kl.Step(b"The", {b"The": 0.9}), kl.Step(b" sky", {b" sky": 0.8})]   # an engine capping top_logprobs at 1
    p = kl.pair(ref, narrow)
    assert p["width"] == [1, 1]                    # each scored position: the fewest alternatives either side gave
    s = kl.summary([dict(p, prompt="a")])
    assert s["width"] == 1 and s["top"] == kl.TOP
    run = {"mode": "kl", "kl": s, "results": [], "summary": {"overall": {}}, "notes": [], "model": "m"}
    said = " ".join(report.findings(run))
    assert "gave 1 alternatives per position where 20 were asked" in said and "the same 20" not in said
