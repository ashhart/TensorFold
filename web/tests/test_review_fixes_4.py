"""Codex's final verification, 13 October 2026: the five things it left, one test each (the third, paging while a
search is on its way, is in the page and has no test)."""

import httpx

from tensorfold_web import bench, replay, report
from tensorfold_web import settings as settings_mod
from tensorfold_web.engine import Engine
from tensorfold_web.usage import Usage


def test_1_info_never_describes_a_replaced_engine_with_the_old_ones_model():
    s = settings_mod.Settings(engine_url="http://old.test:8080", remote=True)
    eng = Engine(s)
    real_status = eng.status

    def status_then_switch(*a, **k):
        st = real_status(*a, **k)
        if s.engine_url == "http://old.test:8080":           # the engine is replaced right after the reading
            with eng.target_lock:
                s.engine_url = "http://new.test:8080"
                eng.retargeted()
        return st
    eng.status = status_then_switch
    eng._client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"status": "ok", "model": f"{r.url.host}-model"})))
    info = eng.info()
    assert info["engine_url"] == "http://new.test:8080" and info["model"] == "new.test-model"


def _level(c, tps):
    return {"concurrency": c, "requests": 8, "errors": 0, "total_tps": tps, "per_request_tps": None,
            "ttft_p50": 0.1, "ttft_p95": 0.2}


def test_2_no_maximum_nor_plateau_from_levels_not_measured():
    s = {"overall": {}, "suites": {}, "load": [_level(1, 40.0), _level(2, None), _level(4, None)]}
    lines = report.findings({"mode": "load", "status": "done", "summary": s, "results": []})
    assert not any("is the most this engine gives" in x or "less than 10% per doubling" in x for x in lines)
    assert any("were not measured" in x for x in lines)
    s["load"] = [_level(1, 40.0), _level(2, 70.0), _level(4, None)]
    lines = report.findings({"mode": "load", "status": "done", "summary": s, "results": []})
    assert any("among the levels measured only" in x for x in lines)


def test_4_a_reply_counted_by_the_page_has_no_token_speed_in_a_comparison():
    def conv(cid, stats):
        return {"id": cid, "title": cid, "messages": [{"role": "user", "content": "q"},
                                                     {"role": "assistant", "content": "x", "stats": stats}]}
    c = replay.compare_replies(conv("a", {"tps": 3.0, "tokens": 2, "measured": "counted"}),
                               conv("b", {"tps": 40.0, "tokens": 200, "measured": "engine"}))
    t = c["turns"][0]
    assert t["a"]["tps"] is None and t["a"]["tokens"] is None and t["change"] is None


def test_5_a_baseline_kept_before_one_per_server_is_this_servers_when_its_address_says_so(tmp_path):
    u = Usage(tmp_path / "usage.sqlite")
    with u._db() as db:                                       # what an earlier TensorFold Web left
        u._set(db, "totals", {"model": "M", "totals": {"completion": 10}, "t": 0, "started": None, "url": "http://a"})
    u.record({"status": "running", "model": "M", "engine_url": "http://a", "totals": {"completion": 20}}, 1_000_000.0)
    assert sum(m["completion"] for m in u.summary(now=1_000_100.0)["models"]) == 10


def test_bench_rules_stay_as_they_were():
    assert bench.drafts_off_confirmed({"drafts": False}) and bench.drafts_on_confirmed({"drafts": True, "drafted": 3})
