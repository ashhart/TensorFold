"""bench_sessions against a fake server in this process: no model, no GPU and no network past 127.0.0.1."""
import contextlib
import hashlib
import http.server
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_sessions as bs  # noqa: E402

SMALL = ["--system-tokens", "60", "--history-tokens", "1500", "--chunk-tokens", "40,160", "--turn-tokens", "20,60",
         "--max-tokens", "32", "--turns", "2"]


def rendered(body: dict, pad: int = 0) -> list[str]:
    """The fake template: the tools, each message's role (later user turns `pad` longer), words, call fields."""
    out = ["<tools>", *json.dumps(body["tools"]).split()] if body.get("tools") else []
    for k, m in enumerate(body["messages"]):
        out += ["<pad>"] * pad if m["role"] == "user" and k > 1 else []
        out += [f"<{m['role']}>", *str(m.get("content") or "").split()]
        out += json.dumps(m["tool_calls"]).split() if m.get("tool_calls") else []
        out += [m["tool_call_id"], "</m>"] if m.get("tool_call_id") else ["</m>"]
    return out + ["<assistant>"] if body.get("add_generation_prompt", True) else out


class Fake:
    """A server keeping one prompt state, so two long sessions in turn evict each other, as in #2346."""

    def __init__(self, stats=True, cache=True, delay=0.0, corrupt=(), fail=(), pad=0):
        self.stats, self.cache, self.delay, self.corrupt, self.fail = stats, cache, delay, set(corrupt), set(fail)
        self.pad = pad
        self.state, self.hits, self.misses, self.evictions, self.running, self.chats = [], 0, 0, 0, 0, 0
        self.seen, self.lock = [], threading.Lock()

    def reply(self, body: dict, prompt: list[str]) -> tuple[list[dict], dict]:
        with self.lock:
            index = self.chats
            self.chats += 1
            cached = len(self.state) if self.cache and self.state and prompt[:len(self.state)] == self.state else 0
            self.hits, self.misses = self.hits + (cached > 0), self.misses + (cached == 0)
            self.evictions += bool(self.cache and self.state and not cached)
            self.state = prompt if self.cache else []
        h = hashlib.sha256(" ".join(prompt).encode()).hexdigest()
        thinking = body["chat_template_kwargs"]["enable_thinking"]
        deltas = [{"reasoning_content": f"think {h[:6]}"}] if thinking else []
        if body["messages"][-1]["role"] == "user":
            deltas += [{"content": f"Reading {h[:6]}."},
                       {"tool_calls": [{"index": 0, "id": f"call_{h[:8]}", "type": "function",
                                        "function": {"name": "read_file", "arguments": ""}}]},
                       {"tool_calls": [{"index": 0, "function": {"arguments": '{"path": '}}]},
                       {"tool_calls": [{"index": 0, "function": {"arguments": f'"{h[:6]}.txt"}}'}}]}]
            finish = "tool_calls"
        else:
            deltas += [{"content": f"Done {h[:8]}"}, {"content": " and checked."}]
            finish = "stop"
        sha = hashlib.sha256((json.dumps(deltas) + ("x" if index in self.corrupt else "")).encode()).hexdigest()
        usage = {"prompt_tokens": len(prompt), "completion_tokens": 3 * len(deltas),
                 "prompt_tokens_details": {"cached_tokens": cached},
                 "completion_tokens_details": {"reasoning_tokens": 3 if thinking else 0}}
        runtime = {"engine": "fake", "time_to_first_token": 0.01, "prefill_seconds": 0.005, "seconds": 0.002,
                   "queue_seconds": 0.001, "token_sha": sha[:12], "token_sha256": sha, "token_ids": [1, 2, 3]}
        chunks = [{"choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in deltas]
        chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        chunks.append({"choices": [], "usage": usage, "tensorfold": runtime, "speculative": {"rounds": 3}})
        return chunks, {"index": index}

    def serve(self):
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status, value):
                raw = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                fake.seen.append((self.path, None, time.perf_counter()))
                if self.path != "/stats" or not fake.stats:
                    return self.send(404, {"error": "not found"})
                self.send(200, {"running": fake.running, "memory": {"active": 1},
                                "prompt_cache": {"hits": fake.hits, "misses": fake.misses,
                                                 "evictions": fake.evictions}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.seen.append((self.path, body, time.perf_counter()))
                if self.path == "/v1/tokenize":
                    n = len(rendered(body, fake.pad) if "messages" in body else body["prompt"].split())
                    return self.send(200, {"count": n, "max_model_len": 1 << 20, "tokens": list(range(n))})
                if fake.chats in fake.fail:
                    fake.chats += 1
                    return self.send(500, {"error": {"message": "fake failure"}})
                fake.running += 1
                chunks, _ = fake.reply(body, rendered(body, fake.pad))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                time.sleep(fake.delay)
                for chunk in chunks:
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                fake.running -= 1

        return http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)

    def chat_bodies(self) -> list[dict]:
        return [b for path, b, _ in self.seen if path == "/v1/chat/completions"]


@contextlib.contextmanager
def serving(*fakes):
    servers = [f.serve() for f in fakes]
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
    for t in threads:
        t.start()
    try:
        yield ["http://127.0.0.1:" + str(s.server_port) for s in servers]
    finally:
        for s in servers:
            s.shutdown()
            s.server_close()


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        root = Path(cls.dir.name)
        for k in range(2):
            lines = (" ".join(f"w{k}_{i}_{j}" for j in range(12)) for i in range(1200))
            (root / f"text{k}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (root / "tiny.txt").write_text("one two three four five six seven eight nine ten\n", encoding="utf-8")
        cls.texts = [str(root / "text0.txt"), str(root / "text1.txt")]
        cls.tiny = str(root / "tiny.txt")

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def run_bench(self, fake, *extra, ref=None, texts=None):
        out = Path(self.dir.name) / "out.jsonl"
        with serving(fake, *([ref] if ref else [])) as urls, contextlib.redirect_stdout(io.StringIO()):
            replay = ["--replay-url", urls[1]] if ref else []
            code = bs.main([urls[0], "fixture", "--text", *(texts or self.texts), *SMALL, "--out", str(out),
                            *replay, *extra])
        lines = [json.loads(line) for line in out.read_text().splitlines()]
        return code, lines

    @staticmethod
    def requests(lines):
        return sorted((x for x in lines if x["kind"] == "request"), key=lambda x: x["seq"])


class PackingTests(Base):
    def test_histories_reach_the_target_in_bounded_pieces(self):
        for seed, target, pad in [(s, t, p) for s in (1, 7, 42, 99) for t in (900, 1500, 3000, 6000) for p in (0, 60)]:
            with self.subTest(seed=seed, target=target, pad=pad):
                fake = Fake(pad=pad)
                code, lines = self.run_bench(fake, "--seed", str(seed), "--history-tokens", str(target), "--turns", "0")
                self.assertEqual(code, 0)
                plan = lines[0]
                self.assertEqual(plan["kind"], "plan")
                cold = {r["session"]: r for r in self.requests(lines)}
                for name, got in plan["history"].items():
                    self.assertLessEqual(abs(got["prompt_tokens"] - target), bs.tolerance(target))
                    self.assertTrue(all(40 <= n <= 160 for n in got["pieces"]), got["pieces"])
                    self.assertEqual(cold[name]["prompt_tokens"], got["prompt_tokens"])
                    self.assertEqual(cold[name]["phase"], "cold")

    def test_history_turns_are_valid_agent_messages_after_one_shared_harness(self):
        fake = Fake()
        code, _ = self.run_bench(fake, "--turns", "0", "--history-tokens", "4000")
        self.assertEqual(code, 0)
        bodies = fake.chat_bodies()
        self.assertEqual(len(bodies), 3)
        self.assertEqual(len({json.dumps([b["messages"][0], b["tools"]]) for b in bodies}), 1)
        for name, body in zip("ABC", bodies):
            msgs = body["messages"][1:]
            self.assertTrue(msgs[0]["content"].startswith(f"Session {name}."))
            self.assertIn(msgs[-1]["role"], ("user", "tool"))
            self.assertTrue({"user", "assistant", "tool"} <= {m["role"] for m in msgs})
            for prev, m in zip(msgs, msgs[1:]):
                if m["role"] == "tool":
                    self.assertEqual(prev["role"], "assistant")
                    self.assertEqual(m["tool_call_id"], prev["tool_calls"][0]["id"])
                if prev["role"] == "assistant":
                    self.assertEqual(m["role"], "tool")

    def test_a_short_corpus_wraps_and_sessions_still_differ_after_the_harness(self):
        fake = Fake()
        code, lines = self.run_bench(fake, "--turns", "0", texts=[self.tiny])
        self.assertEqual(code, 0)
        self.assertTrue(lines[0]["corpus_wrapped"])
        firsts = [b["messages"][1]["content"].split()[:2] for b in fake.chat_bodies()]
        self.assertEqual(firsts, [["Session", "A."], ["Session", "B."], ["Session", "C."]])


class PatternTests(Base):
    def test_schedules(self):
        self.assertEqual(bs.schedule("alt", 2, 4), [[0], [1], [0], [1]])
        self.assertEqual(bs.schedule("burst", 5, 2), [[0], [0], [1], [1], [0], [0], [1], [1], [0], [1]])
        self.assertEqual(bs.schedule("burst", 4, 4), [[0]] * 4 + [[1]] * 4)
        self.assertEqual(bs.schedule("three", 2, 4), [[0], [1], [2], [0], [1], [2]])
        self.assertEqual(bs.schedule("overlap", 2, 4), [[0, 1, 2], [0, 1, 2]])
        self.assertEqual(bs.schedule("alt", 0, 4), [])

    def test_request_order_of_each_pattern(self):
        cases = {("alt",): "cA cB cC tA tB tA tB", ("three",): "cA cB cC tA tB tC tA tB tC",
                 ("burst", "--burst", "2", "--turns", "3"): "cA cB cC tA tA tB tB tA tB",
                 ("alt", "--sessions", "2"): "cA cB tA tB tA tB"}
        for flags, want in cases.items():
            with self.subTest(flags=flags):
                code, lines = self.run_bench(Fake(), "--pattern", *flags)
                self.assertEqual(code, 0)
                got = " ".join(r["phase"][0] + r["session"] for r in self.requests(lines))
                self.assertEqual(got, want)
                turns = [r["turn"] for r in self.requests(lines) if r["session"] == "A"]
                self.assertEqual(turns, list(range(len(turns))))

    def test_alternating_sessions_thrash_a_one_state_cache_and_bursts_resume(self):
        _, alt = self.run_bench(Fake(), "--pattern", "alt", "--sessions", "2", "--turns", "3")
        self.assertEqual([r["cached_tokens"] for r in self.requests(alt) if r["phase"] == "turn"], [0] * 6)
        self.assertTrue(all(r["cache_counters"]["delta"] == {"prompt_cache.hits": 0, "prompt_cache.misses": 1,
                                                             "prompt_cache.evictions": 1}
                            for r in self.requests(alt)[1:]))
        _, burst = self.run_bench(Fake(), "--pattern", "burst", "--sessions", "2", "--turns", "3", "--burst", "3")
        turns = [r for r in self.requests(burst) if r["phase"] == "turn"]
        self.assertEqual([r["cached_tokens"] > 0 for r in turns], [False, True, True, False, True, True])
        self.assertEqual(turns[1]["cached_tokens"], turns[0]["prompt_tokens"])
        self.assertEqual(turns[1]["cache_counters"]["delta"]["prompt_cache.hits"], 1)


class OverlapTests(Base):
    def test_overlap_sends_two_cold_sessions_a_gap_apart_and_the_third_while_both_run(self):
        fake = Fake(delay=0.8)
        code, lines = self.run_bench(fake, "--pattern", "overlap", "--overlap-gap", "0.25", "--turns", "1")
        self.assertEqual(code, 0)
        requests = self.requests(lines)
        for phase in ("cold", "turn"):
            group = {r["session"]: r for r in requests if r["phase"] == phase}
            self.assertEqual(sorted(group), ["A", "B", "C"])
            a, b, c = (group[n] for n in "ABC")
            self.assertAlmostEqual(b["sent_s"] - a["sent_s"], 0.25, delta=0.12)
            self.assertAlmostEqual(c["sent_s"] - b["sent_s"], 0.25, delta=0.12)
            self.assertEqual((a["in_flight_at_send"], b["in_flight_at_send"], c["in_flight_at_send"]),
                             ([], ["A"], ["A", "B"]))
            self.assertGreater(a["sent_s"] + a["total_s"], c["sent_s"])
        arrivals = [t for path, _, t in fake.seen if path == "/v1/chat/completions"]
        self.assertAlmostEqual(arrivals[2] - arrivals[0], 0.5, delta=0.15)
        self.assertGreater(min(r["sent_s"] for r in requests if r["phase"] == "turn"),
                           max(r["sent_s"] + r["total_s"] for r in requests if r["phase"] == "cold"))

    def test_overlap_with_no_gap_is_three_requests_at_once(self):
        code, lines = self.run_bench(Fake(delay=0.3), "--pattern", "overlap", "--overlap-gap", "0", "--turns", "0")
        self.assertEqual(code, 0)
        self.assertLess(max(r["sent_s"] for r in self.requests(lines)) - min(r["sent_s"] for r in self.requests(lines)),
                        0.2)


class LineTests(Base):
    def test_request_lines_carry_usage_timing_runtime_stats_and_hash(self):
        fake = Fake()
        code, lines = self.run_bench(fake, "--pattern", "alt")
        self.assertEqual(code, 0)
        self.assertEqual((lines[0]["kind"], lines[-1]["kind"]), ("plan", "summary"))
        self.assertEqual(lines[-1]["requests"], 7)
        self.assertEqual(lines[-1]["failed"], 0)
        for r in self.requests(lines):
            self.assertEqual(r["pattern"], "alt")
            self.assertTrue(r["complete"])
            self.assertIsNone(r["error"])
            self.assertEqual(r["tensorfold"]["queue_seconds"], 0.001)
            self.assertEqual(r["tensorfold"]["prefill_seconds"], 0.005)
            self.assertNotIn("token_ids", r["tensorfold"])
            self.assertEqual(r["token_sha256"], r["tensorfold"]["token_sha256"])
            self.assertRegex(r["token_sha256"], "^[0-9a-f]{64}$")
            self.assertEqual(r["speculative"], {"rounds": 3})
            self.assertEqual(r["reasoning_tokens"], 3)
            self.assertLessEqual(r["ttft_s"], r["total_s"])
            self.assertGreaterEqual(r["decode_s"], 0)
            self.assertLessEqual(r["cached_tokens"], r["prompt_tokens"])
            self.assertEqual(set(r["stats_before"]), {"running", "memory", "prompt_cache"})
            self.assertEqual(set(r["cache_counters"]), {"before", "after", "delta"})
        self.assertEqual(sum(r["phase"] == "cold" for r in self.requests(lines)), 3)

    def test_turns_resend_the_reply_with_a_new_tool_or_user_message(self):
        for thinking in ("on", "off"):
            with self.subTest(thinking=thinking):
                fake = Fake()
                code, lines = self.run_bench(fake, "--pattern", "alt", "--sessions", "2", "--turns", "3",
                                             "--thinking", thinking)
                self.assertEqual(code, 0)
                bodies = fake.chat_bodies()
                for body in bodies:
                    self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": thinking == "on"})
                    self.assertEqual((body["temperature"], body["seed"], body["max_tokens"]), (0, 1234, 32))
                    self.assertEqual((body["stream"], body["stream_options"]), (True, {"include_usage": True}))
                requests = self.requests(lines)
                kinds = set()
                for k, r in enumerate(requests):
                    if r["phase"] != "turn":
                        continue
                    prev = max(j for j in range(k) if requests[j]["session"] == r["session"])
                    old, new = bodies[prev]["messages"], bodies[k]["messages"]
                    self.assertEqual(new[:len(old)], old)
                    said, added = new[len(old)], new[len(old) + 1:]
                    self.assertEqual(said["role"], "assistant")
                    self.assertTrue(said["content"].startswith(("Reading", "Done")))
                    self.assertNotIn("think", said["content"])
                    if requests[prev]["tool_calls"]:
                        args = json.loads(said["tool_calls"][0]["function"]["arguments"])
                        self.assertTrue(args["path"].endswith(".txt"))
                        self.assertEqual([m["role"] for m in added], ["tool"])
                        self.assertEqual(added[0]["tool_call_id"], said["tool_calls"][0]["id"])
                    else:
                        self.assertNotIn("tool_calls", said)
                        self.assertEqual([m["role"] for m in added], ["user"])
                    kinds.add(added[0]["role"])
                    self.assertEqual(len(added[0]["content"].split()), r["new_tokens"])
                    self.assertTrue(20 <= r["new_tokens"] <= 60)
                self.assertEqual(kinds, {"user", "tool"})

    def test_without_stats_the_lines_omit_snapshots(self):
        code, lines = self.run_bench(Fake(stats=False), "--pattern", "alt", "--sessions", "2", "--turns", "1")
        self.assertEqual(code, 0)
        self.assertFalse(lines[0]["stats"])
        self.assertTrue(all("stats_before" not in r and "cache_counters" not in r for r in self.requests(lines)))

    def test_a_failed_request_is_recorded_stops_its_session_and_fails_the_run(self):
        fake = Fake(fail={3})
        code, lines = self.run_bench(fake, "--pattern", "alt", "--sessions", "2", "--turns", "2")
        self.assertEqual(code, 1)
        requests = self.requests(lines)
        self.assertEqual(" ".join(r["phase"][0] + r["session"] for r in requests), "cA cB tA tB tA")
        self.assertEqual(requests[3]["error"], "HTTPError")
        self.assertIsNone(requests[3]["token_sha256"])
        self.assertEqual(lines[-1]["failed"], 1)

    def test_counter_extraction_finds_only_cache_hit_miss_and_eviction_integers(self):
        snap = {"prompt_cache": {"hits": 2, "misses": 1, "evictions": 0, "bytes": 9}, "cache_hit_rate": 0.5,
                "running": 1, "prefix_cache_misses_total": 4, "hits": 3}
        self.assertEqual(bs.leaves(snap), {"prompt_cache.hits": 2, "prompt_cache.misses": 1,
                                           "prompt_cache.evictions": 0, "prefix_cache_misses_total": 4})
        self.assertIsNone(bs.counters({"running": 1}, {"running": 0}))
        self.assertIsNone(bs.counters(None, None))


class ReplayTests(Base):
    def test_replay_on_a_fresh_server_matches_turn_by_turn(self):
        fake, ref = Fake(), Fake(cache=False)
        code, lines = self.run_bench(fake, "--pattern", "three", "--replay-session", "B", ref=ref)
        self.assertEqual(code, 0)
        replays = [x for x in lines if x["kind"] == "replay"]
        self.assertEqual([(x["phase"], x["turn"]) for x in replays], [("cold", 0), ("turn", 1), ("turn", 2)])
        self.assertTrue(all(x["equal"] and x["fresh"] and x["session"] == "B" for x in replays))
        sent = [r for r in self.requests(lines) if r["session"] == "B"]
        self.assertEqual([x["primary_token_sha256"] for x in replays], [r["token_sha256"] for r in sent])
        primary = [b for b, r in zip(fake.chat_bodies(), self.requests(lines)) if r["session"] == "B"]
        self.assertEqual(ref.chat_bodies(), primary)
        self.assertEqual(lines[-1]["replay"], {"session": "B", "equal": 3, "unequal": 0, "unverified": 0})

    def test_a_differing_replayed_reply_fails_the_run(self):
        code, lines = self.run_bench(Fake(), "--pattern", "alt", ref=Fake(cache=False, corrupt={1}))
        self.assertEqual(code, 1)
        replays = [x for x in lines if x["kind"] == "replay"]
        self.assertEqual([x["equal"] for x in replays], [True, False, True])
        self.assertNotEqual(replays[1]["token_sha256"], replays[1]["primary_token_sha256"])
        self.assertEqual(lines[-1]["replay"]["unequal"], 1)

    def test_a_failed_replay_is_unverified_and_fails_the_run(self):
        code, lines = self.run_bench(Fake(), "--pattern", "alt", ref=Fake(cache=False, fail={0}))
        self.assertEqual(code, 1)
        self.assertEqual(lines[-1]["replay"], {"session": "A", "equal": 2, "unequal": 0, "unverified": 1})


class FlagTests(Base):
    def test_bad_flags_are_refused_before_any_request(self):
        root = Path(self.dir.name)
        (root / "empty.txt").write_text(" \n", encoding="utf-8")
        (root / "latin1.txt").write_bytes(b"caf\xe9\n")
        cases = [("--pattern", "three", "--sessions", "2"), ("--pattern", "alt", "--sessions", "1"),
                 ("--sessions", "27"), ("--pattern", "ring"), ("--chunk-tokens", "160,40"),
                 ("--chunk-tokens", "100,150"), ("--chunk-tokens", "0,10"), ("--turn-tokens", "20"),
                 ("--turn-tokens", "a,b"), ("--overlap-gap", "-1"), ("--overlap-gap", "nan"),
                 ("--overlap-gap", "inf"), ("--max-tokens", "0"), ("--turns", "-1"), ("--burst", "0"),
                 ("--system-tokens", "0"), ("--history-tokens", "100"), ("--timeout", "0"), ("--timeout", "nan"),
                 ("--replay-session", "D"), ("--replay-url", "BASE"), ("--replay-url", "ftp://127.0.0.1:1"),
                 ("--replay-url", "http://127.0.0.1:1/?q=1"), ("--thinking", "maybe"),
                 ("--api-key-env", "TF_BENCH_SESSIONS_UNSET_KEY")]
        texts = [[str(root / "missing.txt")], [str(root / "empty.txt")], [str(root / "latin1.txt")]]
        fake = Fake()
        with serving(fake) as (url,):
            argvs = [[url if x == "BASE" else x for x in c] for c in cases]
            runs = [(url, ["--text", *self.texts, *SMALL, *c]) for c in argvs]
            runs += [(url, ["--text", *t, *SMALL]) for t in texts]
            runs += [("127.0.0.1:8080", ["--text", *self.texts, *SMALL]), (url + "#x", ["--text", *self.texts])]
            for base, rest in runs:
                with self.subTest(argv=rest[-2:]), contextlib.redirect_stderr(io.StringIO()) as err:
                    with self.assertRaises(SystemExit) as exit_:
                        bs.main([base, "fixture", *rest, "--out", str(root / "refused.jsonl")])
                    self.assertEqual(exit_.exception.code, 2)
                    self.assertIn("error:", err.getvalue())
        self.assertEqual(fake.seen, [])

    def test_missing_out_and_text_are_usage_errors(self):
        for argv in (["http://127.0.0.1:1", "m", "--text", self.tiny], ["http://127.0.0.1:1", "m", "--out", "x"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exit_:
                    bs.main(argv)
                self.assertEqual(exit_.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
