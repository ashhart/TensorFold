"""tools/bench_conversations.py against a fake OpenAI streaming server on loopback: turns extend the earlier prompt
exactly, conversations go round-robin, the summary's median and p90 are right."""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("bench_conversations",
                                              Path(__file__).parents[1] / "tools/bench_conversations.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

CONVERSATIONS, TURNS = 3, 4


class Fake:
    """Serves streamed chat completions; the reply to a prompt of k messages is ``reply k``, in two content chunks."""

    def __init__(self, refuse: bool = False):
        self.bodies, self.refuse = [], refuse
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.bodies.append(body)
                if fake.refuse:
                    payload = json.dumps({"error": {"message": "the prompt exceeds the context length"}}).encode()
                    self.send_response(400)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                events = [{"choices": [{"delta": {"role": "assistant"}}]},
                          {"choices": [{"delta": {"content": "reply"}}]},
                          {"choices": [{"delta": {"content": f" {len(body['messages'])}"}}]},
                          {"choices": [], "usage": {"prompt_tokens": 100 * len(body["messages"]),
                                                    "completion_tokens": 2}}]
                for e in events:
                    self.wfile.write(f"data: {json.dumps(e)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def convs(tmp_path):
    path = tmp_path / "convs.json"
    items = [{"conversation": i, "tokens": 1000 + i,
              "messages": [{"role": "user", "content": f"Conversation {i}.\nsource text{tool.ASK}"}]}
             for i in range(CONVERSATIONS)]
    path.write_text(json.dumps({"items": items}))
    return path


@pytest.fixture
def served(convs, tmp_path):
    fake = Fake()
    out = tmp_path / "out.json"
    tool.main(["run", fake.url + "/", "m", str(convs), str(out), "--turns", str(TURNS), "--reply-tokens", "5",
               "--label", "tier-on"])
    yield fake, json.loads(out.read_text())
    fake.close()


def test_a_later_turn_extends_the_previous_prompt_exactly(served, convs):
    fake, _ = served
    first = {i["conversation"]: i["messages"] for i in json.loads(convs.read_text())["items"]}
    by_conversation = {c: [] for c in first}
    for body in fake.bodies:
        by_conversation[int(body["messages"][0]["content"].split(".")[0].split()[-1])].append(body["messages"])
    for c, prompts in by_conversation.items():
        assert len(prompts) == TURNS and prompts[0] == first[c]
        for turn, (before, after) in enumerate(zip(prompts, prompts[1:]), start=2):
            assert after[:len(before)] == before
            assert after[len(before):] == [{"role": "assistant", "content": f"reply {len(before)}"},
                                           {"role": "user", "content": f"Continue: one more sentence. (turn {turn})"}]
    body = fake.bodies[0]
    assert (body["temperature"], body["stream"], body["max_tokens"], body["ignore_eos"]) == (0, True, 5, True)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["stream_options"] == {"include_usage": True}


def test_conversations_are_served_round_robin_turn_by_turn(served):
    fake, result = served
    expect = [(c, t) for t in range(1, TURNS + 1) for c in range(CONVERSATIONS)]
    assert [(r["conversation"], r["turn"]) for r in result["rows"]] == expect
    assert [(int(b["messages"][0]["content"].split(".")[0].split()[-1]), (len(b["messages"]) + 1) // 2)
            for b in fake.bodies] == expect
    row = result["rows"][-1]
    assert row["label"] == "tier-on" and row["prompt_tokens"] == 100 * (2 * TURNS - 1)
    assert row["ttft_s"] > 0 and row["total_s"] >= row["ttft_s"]


def test_the_summary_splits_cold_from_repeat_turns():
    ttft = {1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [0.1, 0.2, 0.3, 0.4, 0.5], 3: [0.5, 0.5, 0.5, 0.5, 0.6]}
    rows = [{"conversation": c, "turn": t, "ttft_s": x} for t, xs in ttft.items() for c, x in enumerate(xs)]
    s = tool.summarize(rows, "x")
    assert s["cold"] == {"n": 5, "ttft_median_s": 3.0, "ttft_p90_s": 4.6}
    assert s["per_turn"]["2"] == {"n": 5, "ttft_median_s": 0.3, "ttft_p90_s": 0.46}
    assert s["per_turn"]["3"] == {"n": 5, "ttft_median_s": 0.5, "ttft_p90_s": 0.56}
    assert s["repeat"] == {"n": 10, "ttft_median_s": 0.5, "ttft_p90_s": 0.51}    # 0.1 .. 0.5 and 0.5 .. 0.6
    assert s["label"] == "x" and set(s["per_turn"]) == {"1", "2", "3"}
    assert tool.percentile(list(range(1, 11)), 0.9) == pytest.approx(9.1) and tool.percentile([7], 0.9) == 7


def test_the_summary_written_matches_the_rows(served):
    _, result = served
    rows, summary = result["rows"], result["summary"]
    cold = sorted(r["ttft_s"] for r in rows if r["turn"] == 1)
    repeat = sorted(r["ttft_s"] for r in rows if r["turn"] > 1)
    assert summary["cold"]["n"] == CONVERSATIONS
    assert summary["cold"]["ttft_median_s"] == pytest.approx(cold[1], abs=1e-4)
    assert summary["repeat"]["n"] == CONVERSATIONS * (TURNS - 1)
    assert summary["repeat"]["ttft_median_s"] == pytest.approx(tool.percentile(repeat, 0.5), abs=1e-4)
    assert set(summary["per_turn"]) == {"1", "2", "3", "4"}


def test_a_refusing_server_ends_the_run_with_its_reason(convs, tmp_path):
    fake = Fake(refuse=True)
    try:
        with pytest.raises(SystemExit) as exc:
            tool.run(fake.url, "m", str(convs), str(tmp_path / "out.json"), 2, 5, "")
    finally:
        fake.close()
    assert "conversation 0 turn 1" in str(exc.value) and "HTTP 400" in str(exc.value)
    assert "context length" in str(exc.value) and not (tmp_path / "out.json").exists()
