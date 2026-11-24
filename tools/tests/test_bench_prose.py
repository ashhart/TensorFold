"""Mock-server tests for tools/bench_prose.py evidence, equality, logprob, and long-context receipts."""
import contextlib
import http.server
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_prose
import bench_stream


def sse_bytes(events: list[dict], done: bool = True) -> bytes:
    """Encode SSE JSON chunks with an optional [DONE] terminator."""
    body = b"".join(b"data: " + json.dumps(ev).encode("utf-8") + b"\n\n" for ev in events)
    return body + (b"data: [DONE]\n\n" if done else b"")


def start_server(handler_fn):
    """Run a local ThreadingHTTPServer backed by handler_fn(path, body) -> (status, ctype, bytes)."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            status, ctype, data = handler_fn(self.path, None)
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            status, ctype, data = handler_fn(self.path, json.loads(raw.decode("utf-8")))
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, th, f"http://127.0.0.1:{srv.server_port}"


def write_books(directory: str, names: list[str], body: str = "") -> None:
    """Populate a temporary prose directory with synthetic book files."""
    text = body if body is not None else ("Alpha beta gamma delta epsilon zeta eta theta. " * 120)
    for idx, name in enumerate(names):
        content = f"Book {idx}: {text}" if text else ""
        (Path(directory) / f"{name}.txt").write_text(content, encoding="utf-8")


class BenchProseTests(unittest.TestCase):
    def test_cached_tokens_must_be_explicit_zero_for_cold_work(self):
        cases = [
            ("missing_details", {"prompt_tokens": 32, "completion_tokens": 4}, "unverified", False),
            ("missing_cached", {"prompt_tokens": 32, "completion_tokens": 4, "prompt_tokens_details": {}}, "unverified", False),
            ("positive_cached", {"prompt_tokens": 32, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 8}}, "reused", False),
            ("bool_cached", {"prompt_tokens": 32, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": False}}, "unverified", False),
            ("exceeds_prompt", {"prompt_tokens": 4, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 8}}, "unverified", False),
            ("explicit_zero", {"prompt_tokens": 32, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 0}}, "cold", True),
        ]
        for label, usage, expected_state, expected_ok in cases:
            with self.subTest(case=label):
                def handler(_path, _body, _u=usage):
                    events = [
                        {"choices": [{"index": 0, "delta": {"content": "hello "}}]},
                        {"choices": [{"index": 0, "delta": {"content": "world"}, "finish_reason": "stop"}]},
                        {"choices": [], "usage": _u, "tensorfold": {"token_sha256": "a" * 64}},
                    ]
                    return 200, "text/event-stream", sse_bytes(events, done=True)

                srv, th, url = start_server(handler)
                try:
                    with tempfile.TemporaryDirectory() as td:
                        write_books(td, ["war_and_peace"], body=None)
                        with contextlib.redirect_stdout(io.StringIO()):
                            report = bench_prose.run_prose_decode(
                                url, td, levels=(1,), gen_tokens=4,
                                excerpt_specs=(("short", 120),), sources=bench_prose.DISTINCT_PROSE_8[:1],
                            )
                finally:
                    srv.shutdown()
                    srv.server_close()
                    th.join(timeout=5)
                pair = report["pairs"][0]
                self.assertEqual(pair["draft"]["rows"][0]["cache_state"], expected_state)
                self.assertEqual(pair["draft"]["cold_verified"], expected_ok)
                self.assertEqual(pair["draft"]["timing"] is not None, expected_ok)
                self.assertEqual(report["qualified"], expected_ok)

    def test_drafted_and_plain_pairs_share_prompt_seed_and_require_token_evidence(self):
        seen = []
        ids = [10, 20, 30, 40]
        sha = bench_prose.sha256_token_ids(ids)
        barrier = threading.Barrier(2)

        class OverlapHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                seen.append(json.loads(raw.decode("utf-8")))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(sse_bytes([{"choices": [{"index": 0, "delta": {"content": "part1 "}}]}], done=False))
                self.wfile.flush()
                barrier.wait(timeout=5)
                self.wfile.write(sse_bytes([
                    {"choices": [{"index": 0, "delta": {"content": "part2"}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 0}},
                     "tensorfold": {"token_ids": ids, "token_sha256": sha}},
                ], done=True))
                self.wfile.flush()

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), OverlapHandler)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                write_books(td, ["war_and_peace", "monte_cristo"], body=None)
                with contextlib.redirect_stdout(io.StringIO()):
                    report = bench_prose.run_prose_decode(
                        f"http://127.0.0.1:{srv.server_port}", td, levels=(2,), gen_tokens=4, seed=777,
                        excerpt_specs=(("short", 120),), sources=bench_prose.DISTINCT_PROSE_8[:2],
                    )
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(timeout=5)

        self.assertEqual(len(seen), 4)
        draft_reqs = sorted([b for b in seen if b.get("draft") is not False], key=lambda x: x["seed"])
        plain_reqs = sorted([b for b in seen if b.get("draft") is False], key=lambda x: x["seed"])
        for d, p in zip(draft_reqs, plain_reqs):
            self.assertEqual(d["messages"], p["messages"])
            self.assertEqual(d["seed"], p["seed"])
        pair = report["pairs"][0]
        self.assertTrue(pair["prompts_identical"])
        self.assertTrue(pair["seeds_identical"])
        self.assertEqual(pair["token_equal"], [True, True])
        self.assertEqual(pair["draft"]["rows"][0]["token_ids"], [10, 20, 30, 40])
        self.assertTrue(report["qualified"])

        # Missing token_ids and token_sha256 must NOT qualify even when text matches.
        def no_hash_handler(_path, _body):
            events = [
                {"choices": [{"index": 0, "delta": {"content": "same "}}]},
                {"choices": [{"index": 0, "delta": {"content": "text"}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 0}}},
            ]
            return 200, "text/event-stream", sse_bytes(events, done=True)

        srv, th, url = start_server(no_hash_handler)
        try:
            with tempfile.TemporaryDirectory() as td:
                write_books(td, ["war_and_peace"], body=None)
                with contextlib.redirect_stdout(io.StringIO()):
                    missing_rep = bench_prose.run_prose_decode(
                        url, td, levels=(1,), gen_tokens=4,
                        excerpt_specs=(("short", 100),), sources=bench_prose.DISTINCT_PROSE_8[:1],
                    )
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(timeout=5)
        self.assertEqual(missing_rep["pairs"][0]["token_equal"], [None])
        self.assertFalse(missing_rep["qualified"])

    def test_token_id_mismatch_marks_pair_unequal(self):
        def handler(_path, body):
            ids = [1, 2, 3, 9] if body.get("draft") is False else [1, 2, 3, 4]
            events = [
                {"choices": [{"index": 0, "delta": {"content": "same "}}]},
                {"choices": [{"index": 0, "delta": {"content": "text"}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 0}},
                    "tensorfold": {"token_ids": ids, "token_sha256": bench_prose.sha256_token_ids(ids)},
                },
            ]
            return 200, "text/event-stream", sse_bytes(events, done=True)

        srv, th, url = start_server(handler)
        try:
            with tempfile.TemporaryDirectory() as td:
                write_books(td, ["war_and_peace"], body=None)
                with contextlib.redirect_stdout(io.StringIO()):
                    report = bench_prose.run_prose_decode(
                        url, td, levels=(1,), gen_tokens=4,
                        excerpt_specs=(("short", 100),), sources=bench_prose.DISTINCT_PROSE_8[:1],
                    )
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(timeout=5)
        self.assertEqual(report["pairs"][0]["token_equal"], [False])
        self.assertFalse(report["qualified"])

    def test_finish_reason_incomplete_error_empty_and_reasoning_with_monotonic_clock(self):
        usage = {"prompt_tokens": 16, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 0}}
        h = {"token_sha256": "b" * 64}
        cases = [
            ("missing_done", [{"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]}, {"choices": [], "usage": usage, "tensorfold": h}], False, False),
            ("missing_finish", [{"choices": [{"index": 0, "delta": {"content": "hi"}}]}, {"choices": [], "usage": usage, "tensorfold": h}], True, False),
            ("unclean_finish", [{"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "content_filter"}]}, {"choices": [], "usage": usage, "tensorfold": h}], True, False),
            ("stream_error", [{"choices": [{"index": 0, "delta": {"content": "hi"}}]}, {"error": {"message": "boom"}}], True, False),
            ("empty_output", [{"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": "stop"}]}, {"choices": [], "usage": usage, "tensorfold": h}], True, False),
            ("reasoning_only", [{"choices": [{"index": 0, "delta": {"reasoning_content": "step1"}}]}, {"choices": [{"index": 0, "delta": {"reasoning_content": " step2"}, "finish_reason": "length"}]}, {"choices": [], "usage": usage, "tensorfold": h}], True, True),
        ]

        def fail_wall():
            raise AssertionError("must use monotonic perf_counter")

        monotonic_only = types.SimpleNamespace(perf_counter=time.perf_counter, time=fail_wall)
        for label, events, done, expect_accept in cases:
            with self.subTest(case=label):
                def handler(_path, _body, _e=events, _d=done):
                    return 200, "text/event-stream", sse_bytes(_e, done=_d)

                srv, th, url = start_server(handler)
                try:
                    with patch.object(bench_prose, "time", monotonic_only), patch.object(bench_stream, "time", monotonic_only):
                        row = bench_prose.stream_chat(f"{url}/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
                finally:
                    srv.shutdown()
                    srv.server_close()
                    th.join(timeout=5)
                self.assertEqual(row["accepted"], expect_accept)
                if expect_accept:
                    self.assertIsNotNone(row["ttft_s"])
                    self.assertGreaterEqual(row["ttft_s"], 0.0)
                    self.assertIsNotNone(row["output_sha256"])
                else:
                    self.assertIsNone(row["ttft_s"])
                    self.assertIsNone(row["decode_tps"])

    def test_compare_prose_loss_eos_counts_token_sha256_length_and_conditional_labels(self):
        with tempfile.TemporaryDirectory() as td:
            p16, p8 = Path(td) / "bf16.json", Path(td) / "fp8.json"
            steps2 = [
                {"token": "a", "logprob": -0.1, "top": {"a": -0.1}},
                {"token": "b", "logprob": -0.1, "top": {"b": -0.1}},
            ]
            bf16 = {
                "len_case": {
                    "prompt_tokens": 100, "completion_tokens": 3, "finish_reason": "length", "text": "abc", "token_ids": [1, 2, 3],
                    "steps": steps2 + [{"token": "c", "logprob": -0.1, "top": {"c": -0.1}}],
                },
                "eos_stop_case": {
                    "prompt_tokens": 100, "completion_tokens": 3, "finish_reason": "stop", "text": "ab",
                    "token_sha256": "c" * 64, "steps": steps2,
                },
                "length_missing_step": {
                    "prompt_tokens": 100, "completion_tokens": 3, "finish_reason": "length", "text": "ab", "steps": steps2,
                },
                "sha_diff_case": {
                    "prompt_tokens": 100, "completion_tokens": 2, "finish_reason": "stop", "text": "ab",
                    "token_sha256": "a" * 64, "steps": steps2,
                },
                "both_missing_evidence": {
                    "prompt_tokens": 100, "completion_tokens": 2, "finish_reason": "stop", "text": "ab",
                    "steps": steps2,
                },
                "disjoint_case": {
                    "prompt_tokens": 100, "completion_tokens": 1, "finish_reason": "stop", "text": "a",
                    "token_ids": [1], "steps": [{"token": "a", "logprob": -0.2, "top": {"a": -0.2}}],
                },
            }
            fp8 = {
                "len_case": {
                    "prompt_tokens": 100, "completion_tokens": 2, "finish_reason": "length", "text": "ab", "token_ids": [1, 2],
                    "steps": steps2,
                },
                "eos_stop_case": {
                    "prompt_tokens": 100, "completion_tokens": 3, "finish_reason": "stop", "text": "ab",
                    "token_sha256": "c" * 64, "steps": steps2,
                },
                "length_missing_step": {
                    "prompt_tokens": 100, "completion_tokens": 3, "finish_reason": "length", "text": "ab", "steps": steps2,
                },
                "sha_diff_case": {
                    "prompt_tokens": 100, "completion_tokens": 2, "finish_reason": "stop", "text": "ab",
                    "token_sha256": "d" * 64, "steps": steps2,
                },
                "both_missing_evidence": {
                    "prompt_tokens": 100, "completion_tokens": 2, "finish_reason": "stop", "text": "ab",
                    "steps": steps2,
                },
                "disjoint_case": {
                    "prompt_tokens": 100, "completion_tokens": 1, "finish_reason": "stop", "text": "a",
                    "token_ids": [1], "steps": [{"token": "a", "logprob": -0.2, "top": {"z": -0.2}}],
                },
            }
            p16.write_text(json.dumps(bf16), encoding="utf-8")
            p8.write_text(json.dumps(fp8), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rep = bench_prose.compare_prose_loss(str(p16), str(p8))
            out = buf.getvalue()
            self.assertFalse(rep["cases"]["len_case"]["identical"])
            self.assertIn("length_mismatch", rep["cases"]["len_case"]["status"])
            self.assertTrue(rep["cases"]["eos_stop_case"]["identical"])
            self.assertEqual(rep["cases"]["eos_stop_case"]["status"], "IDENTICAL")
            self.assertFalse(rep["cases"]["length_missing_step"]["identical"])
            self.assertEqual(rep["cases"]["length_missing_step"]["status"], "incomplete_steps")
            self.assertFalse(rep["cases"]["sha_diff_case"]["identical"])
            self.assertEqual(rep["cases"]["sha_diff_case"]["status"], "token_sha256_mismatch")
            self.assertFalse(rep["cases"]["both_missing_evidence"]["identical"])
            self.assertIn("unverified", rep["cases"]["both_missing_evidence"]["status"])
            self.assertIsNone(rep["cases"]["disjoint_case"]["step0_cond_kl"])
            self.assertIn("step0_cond_KL=unavailable", out)
            self.assertIn("conditional intersection-only diagnostics", out)
            self.assertNotIn("PPL mult", out)

    def test_serialized_streams_measure_shared_interval_and_do_not_qualify_as_concurrent(self):
        serial_lock = threading.Lock()
        ids = [10, 20, 30, 40]
        sha = bench_prose.sha256_token_ids(ids)

        class SerialHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                with serial_lock:
                    c1 = sse_bytes([{"choices": [{"index": 0, "delta": {"content": "tok1 "}}]}], done=False)
                    self.wfile.write(c1)
                    self.wfile.flush()
                    time.sleep(0.05)
                    c2 = sse_bytes([
                        {"choices": [{"index": 0, "delta": {"content": "tok2"}, "finish_reason": "stop"}]},
                        {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 0}},
                         "tensorfold": {"token_ids": ids, "token_sha256": sha}},
                    ], done=True)
                    self.wfile.write(c2)
                    self.wfile.flush()
                    # Keep the lock past the last write, so the client's own scheduling delay on a busy host cannot
                    # stamp this stream's last token after the next stream's first one and read it as an overlap.
                    time.sleep(0.25)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SerialHandler)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                write_books(td, ["war_and_peace", "monte_cristo", "les_miserables", "don_quixote"], body=None)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rep = bench_prose.run_prose_decode(
                        f"http://127.0.0.1:{srv.server_port}", td, levels=(4,), gen_tokens=4,
                        excerpt_specs=(("short", 100),), sources=bench_prose.DISTINCT_PROSE_8[:4],
                    )
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(timeout=5)

        pair = rep["pairs"][0]
        draft_arm = pair["draft"]
        self.assertFalse(draft_arm["overlapped"])
        self.assertTrue(draft_arm["serialized"])
        self.assertFalse(draft_arm["concurrent_verified"])
        self.assertFalse(pair["qualified"])
        self.assertFalse(rep["qualified"])
        self.assertIn("[SERIALIZED/QUEUED]", buf.getvalue())
        per_sum = sum(r["decode_tps"] for r in draft_arm["rows"])
        self.assertLess(draft_arm["timing"]["decode_tps"], per_sum * 0.4)

    def test_teacher_forced_checks_prefix_and_per_step_gt_identity(self):
        requests_seen = []

        def make_handler(gt_202_lp: float, prefix_tokens: list[int]):
            def handler(path, body):
                requests_seen.append((path, body))
                if path == "/v1/tokenize":
                    if body.get("add_special_tokens"):
                        return 200, "application/json", json.dumps({"tokens": prefix_tokens}).encode()
                    return 200, "application/json", json.dumps({"tokens": [201, 202, 203, 204]}).encode()
                if path == "/v1/completions":
                    prompt = body["prompt"]
                    plps = [None]
                    for tid in prompt[1:]:
                        if tid == 202:
                            row = {
                                "202": {"logprob": gt_202_lp, "rank": 25, "decoded_token": "whale"},
                                "999": {"logprob": -0.1, "rank": 1, "decoded_token": "ship"},
                            }
                        else:
                            row = {str(tid): {"logprob": -0.2, "rank": 1, "decoded_token": f"tok{tid}"}}
                        plps.append(row)
                    return 200, "application/json", json.dumps({"choices": [{"text": "x", "finish_reason": "length", "prompt_logprobs": plps}]}).encode()
                if path == "/v1/chat/completions":
                    return 200, "application/json", json.dumps({
                        "choices": [{
                            "finish_reason": "stop", "message": {"content": "ok"},
                            "logprobs": {"content": [{"token": "ok", "logprob": -0.1, "top_logprobs": [{"token": "ok", "logprob": -0.1}]}]},
                        }],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}},
                    }).encode()
                return 404, "application/json", b"{}"
            return handler

        with tempfile.TemporaryDirectory() as td:
            write_books(td, ["war_and_peace", "moby_dick"], body=None)
            p16, p8, p_bad_prefix = Path(td) / "bf16.json", Path(td) / "fp8.json", Path(td) / "bad_prefix.json"
            for lp, pfx, path in ((-3.0, [101, 102, 103], p16), (-3.2, [101, 102, 103], p8), (-3.2, [101, 102, 999], p_bad_prefix)):
                srv, th, url = start_server(make_handler(lp, pfx))
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        bench_prose.dump_prose_loss(url, td, str(path), cases=bench_prose.LOSS_CASES[:1], tf_tokens=4)
                finally:
                    srv.shutdown()
                    srv.server_close()
                    th.join(timeout=5)
            with contextlib.redirect_stdout(io.StringIO()):
                ok_cmp = bench_prose.compare_prose_loss(str(p16), str(p8))
                bad_pfx_cmp = bench_prose.compare_prose_loss(str(p16), str(p_bad_prefix))
            tampered = json.loads(p8.read_text(encoding="utf-8"))
            tampered["_tf_moby_dick_16k"]["steps"][1]["gt_token_id"] = 999
            p_bad_gt = Path(td) / "bad_gt.json"
            p_bad_gt.write_text(json.dumps(tampered), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                bad_gt_cmp = bench_prose.compare_prose_loss(str(p16), str(p_bad_gt))

        self.assertTrue(ok_cmp["teacher_forced"]["available"])
        self.assertEqual(ok_cmp["teacher_forced"]["continuation_token_count"], 4)
        self.assertAlmostEqual(ok_cmp["teacher_forced"]["gt_nll_delta"], 0.05)
        self.assertFalse(bad_pfx_cmp["teacher_forced"]["available"])
        self.assertEqual(bad_pfx_cmp["teacher_forced"]["reason"], "prefix_mismatch")
        self.assertFalse(bad_gt_cmp["teacher_forced"]["available"])
        self.assertEqual(bad_gt_cmp["teacher_forced"]["reason"], "gt_token_id_mismatch")

    def test_distinct_long_failed_stream_not_undersized_and_omits_kv_claim(self):
        def handler(path, body):
            if path.startswith("/health"):
                return 200, "application/json", json.dumps({"memory": {"active": 1024}}).encode()
            text = body["messages"][0]["content"]
            if body.get("max_tokens") == 1:
                return 200, "application/json", json.dumps({
                    "choices": [{"finish_reason": "length", "message": {"content": "x"}}],
                    "usage": {"prompt_tokens": max(1, len(text) // 4), "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}},
                }).encode()
            if "Stream1_LesMiserables" in text:
                return 200, "application/json", json.dumps({"error": {"message": "oom"}}).encode()
            actual_ptok = max(1, len(text) // 4)
            return 200, "application/json", json.dumps({
                "choices": [{"finish_reason": "stop", "message": {"content": "FOLIO-WP-1812 FOLIO-WP-4409 FOLIO-WP-8831"}}],
                "usage": {"prompt_tokens": actual_ptok, "completion_tokens": 12, "prompt_tokens_details": {"cached_tokens": 0}},
            }).encode()

        srv, th, url = start_server(handler)
        try:
            with tempfile.TemporaryDirectory() as td:
                write_books(td, ["war_and_peace"], body="Short prose text. " * 20)
                write_books(td, ["les_miserables"], body="Longer prose text. " * 400)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rep = bench_prose.run_distinct_long(url, td, target_tokens=500, n_streams=2)
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(timeout=5)

        out = buf.getvalue()
        self.assertNotIn("simultaneous KV", out)
        self.assertIsNone(rep["observed_kv_resident_tokens"])
        self.assertEqual(rep["submitted_streams"], 2)
        self.assertEqual(rep["completed_streams"], 1)
        self.assertEqual(rep["failed_streams"], ["Stream1_LesMiserables"])
        self.assertEqual(rep["undersized_streams"], ["Stream0_WarAndPeace"])
        self.assertEqual(rep["cell_status"], "incomplete_and_undersized")
        s1 = next(s for s in rep["streams"] if s["name"] == "Stream1_LesMiserables")
        self.assertFalse(s1["undersized"])

    def test_distinct_long_reports_observed_kv_and_rejects_non_cold_cache(self):
        for cached_val, expect_ok, expect_status in ((0, True, "ok"), (12, False, "cache_not_cold"), (None, False, "cache_not_cold")):
            with self.subTest(cached=cached_val):
                def handler(path, body, _c=cached_val):
                    if path.startswith("/health"):
                        return 200, "application/json", json.dumps({"kv_cache": {"used_tokens": 1024}}).encode()
                    text = body["messages"][0]["content"]
                    ptok = max(1, len(text) // 4)
                    details = {} if _c is None else {"cached_tokens": _c}
                    return 200, "application/json", json.dumps({
                        "choices": [{"finish_reason": "stop", "message": {"content": "FOLIO-WP-1812 FOLIO-WP-4409 FOLIO-WP-8831"}}],
                        "usage": {"prompt_tokens": ptok, "completion_tokens": 10, "prompt_tokens_details": details},
                    }).encode()

                srv, th, url = start_server(handler)
                try:
                    with tempfile.TemporaryDirectory() as td:
                        write_books(td, ["war_and_peace", "les_miserables"], body="Longer prose text. " * 400)
                        buf = io.StringIO()
                        with contextlib.redirect_stdout(buf):
                            code = bench_prose.main(["--url", url, "--prose-dir", td, "--mode", "distinct-long", "--tokens", "500", "--streams", "2"])
                finally:
                    srv.shutdown()
                    srv.server_close()
                    th.join(timeout=5)
                self.assertEqual(code, 0 if expect_ok else 1)
                self.assertIn(f"status={expect_status}", buf.getvalue())
                self.assertIn("Observed engine KV residency : 1,024 tokens", buf.getvalue())

    def test_invalid_counts_and_empty_source_rejected_before_network_or_division(self):
        with patch.object(bench_prose.urllib.request, "urlopen", side_effect=AssertionError("must not call server")):
            with self.assertRaises(ValueError):
                bench_prose.run_prose_decode("http://127.0.0.1:1", "/nonexistent", levels=(0,))
            with self.assertRaises(ValueError):
                bench_prose.run_prose_decode("http://127.0.0.1:1", "/nonexistent", levels=(9,))
            with self.assertRaises(ValueError):
                bench_prose.run_prose_decode("http://127.0.0.1:1", "/nonexistent", gen_tokens=0)
            with self.assertRaises(ValueError):
                bench_prose.run_distinct_long("http://127.0.0.1:1", "/nonexistent", target_tokens=0, n_streams=1)
            with self.assertRaises(ValueError):
                bench_prose.run_distinct_long("http://127.0.0.1:1", "/nonexistent", target_tokens=100, n_streams=9)
            with self.assertRaises(ValueError):
                bench_prose.run_distinct_long("http://127.0.0.1:1", "/nonexistent", target_tokens=100, n_streams=1, min_prompt_ratio=0.0)
            with self.assertRaises(ValueError):
                bench_prose.dump_prose_loss("http://127.0.0.1:1", "/nonexistent", "out.json", tf_tokens=0)
            with tempfile.TemporaryDirectory() as td:
                write_books(td, ["war_and_peace"], body="")
                with self.assertRaises(ValueError):
                    bench_prose.calibrate_stream_cpts("http://127.0.0.1:1", td, 1)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    bench_prose.main(["--mode", "distinct-long", "--streams", "0"])
                self.assertNotEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
