"""Codex's third pass, 13 October 2026: N1 to N10, one test each, numbered as in its report."""

import asyncio
import socket
import threading
import time

import httpx

from tensorfold_web import bench, conversations, replay, report
from tensorfold_web import settings as settings_mod
from tensorfold_web.engine import Engine
from tensorfold_web.proxy import Proxy
from tensorfold_web.usage import Usage


def test_n1_the_relay_reads_address_and_key_as_one():
    seen = []
    state = {"target": ("http://old.test:8080", "KEY-OLD")}

    def read_target():
        t = state["target"]
        state["target"] = ("http://new.test:8080", "KEY-NEW")     # the engine changes right after the snapshot
        return t

    p = Proxy(lambda: "http://never.test", lambda: "NEVER", target=read_target)
    p._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: seen.append((r.url.host, r.headers.get("authorization"))) or httpx.Response(200, json={})))

    class Req:
        method, query_params = "GET", httpx.QueryParams()
        headers = httpx.Headers({"accept": "application/json"})

        async def stream(self):
            yield b""
    asyncio.run(p.handle(Req(), "v1/models"))
    assert seen == [("old.test", "Bearer KEY-OLD")]


def test_n2_n5_an_old_reading_is_not_published_and_the_new_window_is_asked():
    s = settings_mod.Settings(engine_url="http://old.test:8080", remote=True)
    eng = Engine(s)
    asked = []

    def handler(request):
        asked.append((request.url.host, request.url.path))
        if request.url.host == "old.test" and request.url.path == "/tokenize":
            with eng.target_lock:                         # the engine changes at the very end of the old poll
                s.engine_url = "http://new.test:8080"
                eng.retargeted()
            return httpx.Response(200, json={"max_model_len": 8192})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"max_model_len": 32768})
        return httpx.Response(200, json={"status": "ok", "model": f"{request.url.host}-model"})
    eng._client = httpx.Client(transport=httpx.MockTransport(handler))
    info = eng.info()
    assert info["model"] == "new.test-model" and info["engine_url"] == "http://new.test:8080"
    assert eng.last_health.model == "new.test-model" and eng.last_health.context_length == 32768
    assert ("new.test", "/tokenize") in asked
    assert eng.startup.get("context_tokens") in (None, 32768.0)


def test_n3_a_drafted_count_against_drafts_off_spoils_a_speed_up_check():
    def conv(cid, stats, drafts):
        return {"id": cid, "title": cid, "settings": {"target": {"drafts": drafts}},
                "messages": [{"role": "user", "content": "q"},
                             {"role": "assistant", "content": "x", "outcome": "done", "stats": stats}]}
    on = conv("a", {"sha": "s", "drafts": True, "drafted": 12}, True)
    off_lying = conv("b", {"sha": "s", "drafts": False, "drafted": 7}, False)
    assert replay.compare_replies(on, off_lying)["turns"][0]["same_tokens"] is None
    on_idle = conv("a", {"sha": "s", "drafts": True, "drafted": 0}, True)      # drafts on, nothing drafted
    off = conv("b", {"sha": "s", "drafts": False, "drafted": 0}, False)
    assert replay.compare_replies(on_idle, off)["turns"][0]["same_tokens"] is None
    assert replay.compare_replies(on, off)["turns"][0]["same_tokens"] is True


def test_n4_exactness_needs_drafts_to_have_run_on_the_first_reply():
    assert not bench.drafts_on_confirmed({"drafts": False, "drafted": 0})
    assert not bench.drafts_on_confirmed({"drafts": True, "drafted": 0})
    assert not bench.drafts_on_confirmed({"token_sha": "x"})
    assert bench.drafts_on_confirmed({"drafts": True, "drafted": 9})
    assert bench.drafts_on_confirmed({"drafts": True})


def test_n6_alone_is_the_level_of_one_request():
    m = {"tokens": None, "tps": None, "ttft": None, "ttft_answer": None, "prefill_s": None, "gap_p50": None,
         "gap_p95": None, "runtime": {}, "ttft_client": 0.1}
    load = [{"suite": "load", "prompt": "1x0.0", "concurrency": 1, "wave": 0, "wave_s": 1.0, "draft": True, "m": m},
            {"suite": "load", "prompt": "2x0.0", "concurrency": 2, "wave": 0, "wave_s": 1.0, "draft": True,
             "m": {**m, "tokens": 10}},
            {"suite": "load", "prompt": "2x0.1", "concurrency": 2, "wave": 0, "wave_s": 1.0, "draft": True,
             "m": {**m, "tokens": 10}}]
    s = bench.summarize(load)
    lines = report.findings({"mode": "load", "status": "done", "summary": s, "results": load})
    assert any("one request alone was not measured" in x for x in lines)
    assert not any("is the most this engine gives" in x for x in lines)


def test_n7_each_server_keeps_its_own_baseline(tmp_path):
    u = Usage(tmp_path / "usage.sqlite")
    base = {"status": "running", "model": "M"}
    for url, total, t in (("http://a", 10, 0), ("http://b", 1000, 60), ("http://a", 20, 120), ("http://b", 1004, 180)):
        u.record({**base, "engine_url": url, "totals": {"completion": total}}, 1_000_000.0 + t)
    assert sum(m["completion"] for m in u.summary(now=1_000_300.0)["models"]) == 14     # A's 10 and B's 4


def test_n9_an_export_says_no_tokens_for_stream_events():
    md = conversations.markdown({"title": "t", "created": 0, "updated": 0, "messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "model": "m", "stats": {"tokens": 2, "tps": 3.0, "measured": "counted"}}]})
    assert "tokens" not in md.split("## ")[-1] and "tok/s" not in md


def test_n10_done_ends_the_reading_even_if_the_server_keeps_the_line_open():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve():
        c, _ = srv.accept()
        c.recv(65536)
        body = (b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
        c.sendall(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
                  + f"{len(body):x}\r\n".encode() + body + b"\r\n")
        time.sleep(3)                                     # the line stays open, the stream never ends
        c.close()
    threading.Thread(target=serve, daemon=True).start()
    t0 = time.time()
    with httpx.Client(timeout=httpx.Timeout(2.0)) as client:
        rep = bench.stream_once(client, f"http://127.0.0.1:{port}/v1/chat/completions", {}, {}, threading.Event())
    assert rep.error is None and rep.text == "hi" and time.time() - t0 < 1.5
    srv.close()
