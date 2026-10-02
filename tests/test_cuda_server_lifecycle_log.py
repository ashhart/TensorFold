"""The CUDA server's request lines: the done line carries the reply's id, effort and queue wait, a request that
does not finish prints an ``ended`` line, and a refused one prints why."""

import http.client
import json
import re
import time

import pytest

pytest.importorskip("jinja2")

from tests.test_cuda_server_disconnect import (MESSAGES, WAIT, PacedEngine, app_for, leave, post, send, serving,
                                               until)
from tests.test_cuda_server_health import StatsEngine


def lines(capfd, kind: str) -> list[str]:
    return [line for line in capfd.readouterr().out.splitlines() if line.startswith(f"[tensorfold] {kind} ")]


def reply(conn) -> dict:
    response = http.client.HTTPResponse(conn)
    response.begin()
    try:
        return json.loads(response.read())
    finally:
        conn.close()


def test_the_done_line_carries_the_reply_s_own_id_and_effort(tmp_path, capfd):
    app = app_for(tmp_path, StatsEngine())
    app.tok.token_to_id = lambda text: None              # a thinking reply looks for </think>; this tokenizer has none
    with serving(app) as port:
        status, body = post(port, {"messages": MESSAGES, "max_tokens": 4, "reasoning_effort": "low",
                                   "chat_template_kwargs": {"enable_thinking": True}})
    assert status == 200
    done = lines(capfd, "done")
    assert len(done) == 1 and done[0].startswith(f"[tensorfold] done {json.loads(body)['id']} "), done
    assert " thinking=True effort=low tokens=4 " in done[0], done[0]


def test_the_done_line_says_how_long_a_request_waited_for_the_engine(tmp_path, capfd):
    engine = PacedEngine(hold_at=1)
    app = app_for(tmp_path, engine)
    with serving(app) as port:
        first = send(port, {"messages": [{"role": "user", "content": "first"}], "max_tokens": 3})
        assert engine.held.wait(WAIT)
        second = send(port, {"messages": [{"role": "user", "content": "second"}], "max_tokens": 3})
        until(lambda: app._turns().waiting == 1, "the second request to wait for its turn")
        time.sleep(0.3)
        engine.release.set()
        ids = [reply(first)["id"], reply(second)["id"]]
    queued = {}
    for line in lines(capfd, "done"):
        found = re.search(r"^\[tensorfold\] done (\S+) .* queued=([\d.]+)s ttft=", line)
        assert found, line
        queued[found.group(1)] = float(found.group(2))
    assert queued[ids[0]] < 0.25 <= queued[ids[1]], queued


@pytest.mark.parametrize("stream", [False, True])
def test_a_client_that_leaves_mid_reply_prints_an_ended_line(tmp_path, capfd, stream):
    engine = PacedEngine(hold_at=5)
    app = app_for(tmp_path, engine)
    with serving(app) as port:
        conn = send(port, {"messages": MESSAGES, "max_tokens": 400, "stream": stream})
        assert engine.held.wait(WAIT)
        leave(conn)
        engine.release.set()
        until(lambda: engine.calls and engine.calls[0]["done"], "the request to end")
        time.sleep(0.05)
    out = capfd.readouterr().out
    ended = [line for line in out.splitlines() if line.startswith("[tensorfold] ended ")]
    assert len(ended) == 1, out
    assert re.search(r"^\[tensorfold\] ended chatcmpl-[0-9a-f]{24} reason=client-left prompt=\d+ tokens=\d+ "
                     r"after=[\d.]+s$", ended[0]), ended[0]
    assert "[tensorfold] done " not in out


def test_a_client_that_leaves_while_waiting_prints_an_ended_line(tmp_path, capfd):
    engine = PacedEngine(hold_at=1)
    app = app_for(tmp_path, engine)
    with serving(app) as port:
        first = send(port, {"messages": [{"role": "user", "content": "first"}], "max_tokens": 3})
        assert engine.held.wait(WAIT)
        second = send(port, {"messages": [{"role": "user", "content": "second"}], "max_tokens": 3})
        until(lambda: app._turns().waiting == 1, "the second request to wait for its turn")
        leave(second)
        until(lambda: app._turns().waiting == 0, "the second request to leave")
        engine.release.set()
        reply(first)
    ended = lines(capfd, "ended")
    assert len(ended) == 1 and " reason=client-left " in ended[0] and " tokens=0 " in ended[0], ended


class FailingEngine(PacedEngine):
    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        on_tokens([ord("x")])
        raise RuntimeError("the decode failed")


def test_a_reply_that_fails_prints_an_ended_line(tmp_path, capfd):
    app = app_for(tmp_path, FailingEngine())
    with serving(app) as port:
        status, _ = post(port, {"messages": MESSAGES, "max_tokens": 4})
    assert status == 500
    ended = lines(capfd, "ended")
    assert len(ended) == 1, ended
    assert re.search(r" reason=error \(RuntimeError\) prompt=\d+ tokens=1 after=[\d.]+s$", ended[0]), ended[0]


@pytest.mark.parametrize("stream", [False, True])
def test_a_refused_request_prints_why(tmp_path, capfd, stream):
    app = app_for(tmp_path, PacedEngine())
    with serving(app) as port:
        status, _ = post(port, {"messages": "Hi", "max_tokens": 4, "stream": stream})
    assert status == 400
    refused = lines(capfd, "refused")
    assert len(refused) == 1, refused
    assert re.search(r"^\[tensorfold\] refused chatcmpl-[0-9a-f]{24}: \w+: messages must be a list$",
                     refused[0]), refused[0]


class KeptEngine(StatsEngine):
    def generate(self, *args, **kwargs):
        return {**super().generate(*args, **kwargs),
                "kept": {"entries": 3, "bytes": 3 * 2 ** 29, "hits": 2, "misses": 1, "evictions": 4}}


def test_the_done_line_reports_the_engine_s_kept_prompts(tmp_path, capfd):
    app = app_for(tmp_path, KeptEngine())
    with serving(app) as port:
        assert post(port, {"messages": MESSAGES, "max_tokens": 4})[0] == 200
    done = lines(capfd, "done")
    assert len(done) == 1 and done[0].endswith(" checkpoints=3 (1.50 GiB, hits=2 misses=1 evictions=4)"), done
