"""Codex's verification of the 22 fixes, 13 October 2026: what was left of them and what it found next (R1 to R16),
one test each, numbered as in its report."""

import json
import threading
import time

import httpx
import pytest

from tensorfold_web import bench, conversations, kl, replay, report, suites
from tensorfold_web.usage import Usage


def test_r2_drafts_off_with_tokens_drafted_is_a_contradiction():
    assert not bench.drafts_off_confirmed({"drafts": False, "drafted": 7})
    assert bench.drafts_off_confirmed({"drafts": False, "drafted": 0})


def _conv(cid, stats, outcome="done", drafts=None):
    return {"id": cid, "title": cid, "settings": {"target": {"drafts": drafts}} if drafts is not None else {},
            "messages": [{"role": "user", "content": "q"},
                         {"role": "assistant", "content": "same", "outcome": outcome, "stats": stats}]}


def test_r3_a_speed_up_check_needs_both_sides_as_asked_and_whole():
    off = _conv("b", {"sha": "x", "drafts": False}, drafts=False)
    both_off = _conv("a", {"sha": "x", "drafts": False}, drafts=True)          # asked on, written off
    assert replay.compare_replies(both_off, off)["turns"][0]["same_tokens"] is None
    on = _conv("a", {"sha": "x", "drafts": True}, drafts=True)
    assert replay.compare_replies(on, off)["turns"][0]["same_tokens"] is True
    for outcome in ("interrupted", "stopped", "error"):
        cut = _conv("a", {"sha": "x", "drafts": True}, outcome=outcome, drafts=True)
        assert replay.compare_replies(cut, off)["turns"][0]["same_tokens"] is None


def test_r4_a_reference_not_confirmed_off_is_said_so():
    k = {"positions": 10, "prompts": 1, "mean_kl": 0.0, "p99_kl": 0.0, "max_kl": 0.0, "same_top": 1.0, "zero_kl": 1.0,
         "same_distribution": 1.0, "identical": 1.0, "differs_at_median": None, "tokens": 10, "requests": 2, "worst": [],
         "top": 20, "width": 20, "unscored": 0}
    run = {"mode": "kl", "status": "done", "kl": k, "kl_reference": {"kind": "drafts-off", "drafts_off": False},
           "results": [], "summary": {"overall": {}}, "notes": [], "model": "m"}
    assert "did not confirm" in report.kl_line(run)
    assert any("compares the engine with itself" in x for x in report.findings(run))
    steps = kl.Steps([kl.Step(b"a", {b"a": 1.0})], {"drafts": True})
    assert steps.runtime == {"drafts": True} and len(steps) == 1


def _m(tokens):
    return {"tokens": tokens, "tps": None, "ttft": None, "ttft_answer": None, "prefill_s": None, "gap_p50": None,
            "gap_p95": None, "runtime": {}}


def test_r5_unknown_token_counts_stay_unknown_in_totals():
    rows = [{"suite": "s", "prompt": "a", "draft": True, "m": _m(10)}, {"suite": "s", "prompt": "b", "draft": True, "m": _m(None)}]
    assert bench.summarize(rows)["overall"]["tokens"] is None
    load = [{"suite": "load", "prompt": f"1x{w}.0", "concurrency": 1, "wave": w, "wave_s": 1.0, "draft": True,
             "m": {**_m(None), "ttft_client": 0.1}} for w in range(2)]
    s = bench.summarize(load)
    assert s["load"][0]["total_tps"] is None
    lines = report.findings({"mode": "load", "status": "done", "summary": s, "results": load})
    assert any("no token counts" in x for x in lines) and not any("0.0 tokens per second" in x for x in lines)


def test_r6_a_crlf_split_between_two_chunks_still_ends_an_event():
    sse = (b'data: {"choices":[{"delta":{"content":"hi"}}]}\r\n\r\n'
           b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\r\n\r\ndata: [DONE]\r\n\r\n')

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=(sse[i:i + 1] for i in range(len(sse))))   # one byte at a time
    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        rep = bench.stream_once(c, "http://e.test/v1/chat/completions", {}, {}, threading.Event())
    assert rep.error is None and rep.text == "hi" and rep.finish == "stop"


def test_r7_another_server_sets_a_baseline_and_counts_nothing(tmp_path):
    u = Usage(tmp_path / "usage.sqlite")
    base = {"status": "running", "model": "M"}
    u.record({**base, "engine_url": "http://a:8080", "totals": {"completion": 10}}, 1_000_000.0)
    u.record({**base, "engine_url": "http://b:8080", "totals": {"completion": 1000}}, 1_000_060.0)
    u.record({**base, "engine_url": "http://b:8080", "totals": {"completion": 1005}}, 1_000_120.0)
    assert sum(m["completion"] for m in u.summary(now=1_000_200.0)["models"]) == 5


def test_r8_r9_widths_follow_the_positions_scored_and_zero_mass_is_unscored():
    ref = [kl.Step(b"x", {}), kl.Step(b"y", {b"y": 1.0})]
    p = kl.pair(ref, ref)
    assert p["positions"] == 1 and p["width"] == [1] and p["unscored"] == 1
    body = {"choices": [{"logprobs": {"content": [{"token": "z", "logprob": -9999, "top_logprobs": []}]}}]}
    under = kl._steps(body)
    assert not kl.usable(under[0].top)
    assert kl.pair(under, under)["unscored"] == 1


def test_r10_a_turn_that_recorded_the_workshop_is_refused_whatever_the_settings_say_now():
    doc = {"settings": {"workshop": {"tools": False, "output": "text"}},
           "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "{}", "json": {"ok": True}}]}
    assert replay.uses_tools(doc)
    doc["messages"][1] = {"role": "assistant", "content": "{}", "request": {"response_format": {"type": "json_object"}}}
    assert replay.uses_tools(doc)
    assert not replay.uses_tools({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})


def test_r11_every_pin_comes_first_and_the_others_page_by_page(tmp_path):
    chats = conversations.Conversations(tmp_path / "chat.sqlite")
    pins = []
    for i in range(3):
        d = chats.create({"messages": [{"role": "user", "content": f"pin {i}"}]})
        chats.update(d["id"], {"pinned": True}, rev=d["rev"])
        pins.append(d["id"])
    others = [chats.create({"messages": [{"role": "user", "content": f"x {i}"}]})["id"] for i in range(3)]
    first, more = chats.page(limit=2)
    assert set(pins) <= {x["id"] for x in first} and len(first) == 5 and more
    second, more = chats.page(limit=2, offset=2)
    assert [x["id"] for x in second] == [others[0]] and not more


def test_r12_dot_names_are_refused():
    for name in (".", "..", " .. "):
        with pytest.raises(ValueError):
            conversations.validate_preset({"name": name})


def test_r13_throttled_time_is_measured_not_counted_in_samples():
    class Power(bench.power.Source):
        name, scope, whole_machine = "fake", "whole machine", True

        def read(self):
            return bench.power.Reading(watts=10.0, cpu_c=None, gpu_c=None, throttled=True)
    s = bench.Sampler(lambda: None, every=1.0, source=Power())
    with s:
        s.measuring()
        time.sleep(0.3)
        r = s.report(output_tokens=10)
    assert r["throttle"]["throttled"] and r["throttle"]["seconds"] <= r["window_s"] + 0.05 < 1.0


def test_r14_a_repeated_prompt_id_never_lands_on_one_already_taken():
    doc = suites.validate({"name": "S", "prompts": [{"id": "p2", "user": "a"}, {"id": "p2", "user": "b"}, {"user": "c"}]})
    ids = [p["id"] for p in doc["prompts"]]
    assert len(set(ids)) == 3


def test_r15_a_question_reading_like_a_repeat_label_is_its_own():
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "1"},
            {"role": "user", "content": "q (2)"}, {"role": "assistant", "content": "2"},
            {"role": "user", "content": "q"}, {"role": "assistant", "content": "3"}]
    doc = {"id": "a", "title": "a", "messages": msgs}
    assert len(replay.compare_replies(doc, json.loads(json.dumps(doc)))["turns"]) == 3


def test_r16_the_html_report_does_not_call_a_failed_request_missed():
    m = _m(None)
    results = [{"suite": "gsm8k", "prompt": "1", "draft": True, "m": m, "error": "503", "passed": None, "category": "gsm8k"},
               {"suite": "gsm8k", "prompt": "2", "draft": True, "m": m, "error": None, "passed": True, "category": "gsm8k"}]
    html = report.html_report({"id": "r", "mode": "quality", "status": "done", "created": 0, "model": "m",
                               "summary": bench.summarize(results), "results": results, "notes": []})
    assert "Items missed" not in html
