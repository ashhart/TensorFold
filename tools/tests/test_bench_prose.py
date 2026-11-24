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
    text = body or ("Alpha beta gamma delta epsilon zeta eta theta. " * 120)
    for idx, name in enumerate(names):
        (Path(directory) / f"{name}.txt").write_text(f"Book {idx}: {text}", encoding="utf-8")


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
                        {"choices": [{"index": 0, "delta": {"content": "world"}}]},
                        {"choices": [], "usage": _u, "tensorfold": {"token_sha256": "a" * 64}},
                    ]
                    return 200, "text/event-stream", sse_bytes(events, done=True)

                srv, th, url = start_server(handler)
                try:
                    with tempfile.TemporaryDirectory() as td:
                        write_books(td, ["war_and_peace"])
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

    def test_drafted_and_plain_pairs_share_prompt_seed_and_full_token_ids(self):
        seen = []

        def handler(_path, body):
            seen.append(body)
            ids = [10, 20, 30, 40]
            events = [
                {"choices": [{"index": 0, "delta": {"content": "part1 "}}]},
                {"choices": [{"index": 0, "delta": {"content": "part2"}}]},
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 0}},
                    "tensorfold": {"token_ids": ids, "token_sha256": bench_prose.sha256_token_ids(ids)},
                },
            ]
            return 200, "text/event-stream", sse_bytes(events, done=True)

        srv, th, url = start_server(handler)
        try:
            with tempfile.TemporaryDirectory() as td:
                write_books(td, ["war_and_peace", "monte_cristo"])
                with contextlib.redirect_stdout(io.StringIO()):
                    report = bench_prose.run_prose_decode(
                        url, td, levels=(2,), gen_tokens=4, seed=777,
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

    def test_token_id_mismatch_marks_pair_unequal(self):
        def handler(_path, body):
            ids = [1, 2, 3, 9] if body.get("draft") is False else [1, 2, 3, 4]
            events = [
                {"choices": [{"index": 0, "delta": {"content": "same "}}]},
                {"choices": [{"index": 0, "delta": {"content": "text"}}]},
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
                write_books(td, ["war_and_peace"])
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

    def test_incomplete_error_empty_rejected_and_reasoning_accepted_with_monotonic_clock(self):
        usage = {"prompt_tokens": 16, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 0}}
        h = {"token_sha256": "b" * 64}
        cases = [
            ("missing_done", [{"choices": [{"index": 0, "delta": {"content": "hi"}}]}, {"choices": [], "usage": usage, "tensorfold": h}], False, False),
            ("stream_error", [{"choices": [{"index": 0, "delta": {"content": "hi"}}]}, {"error": {"message": "boom"}}], True, False),
            ("empty_output", [{"choices": [{"index": 0, "delta": {"role": "assistant"}}]}, {"choices": [], "usage": usage, "tensorfold": h}], True, False),
            ("reasoning_only", [{"choices": [{"index": 0, "delta": {"reasoning_content": "step1"}}]}, {"choices": [{"index": 0, "delta": {"reasoning_content": " step2"}}]}, {"choices": [], "usage": usage, "tensorfold": h}], True, True),
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

    def test_compare_prose_loss_length_mismatch_identical_unavailable_and_conditional_labels(self):
        with tempfile.TemporaryDirectory() as td:
            p16, p8 = Path(td) / "bf16.json", Path(td) / "fp8.json"
            bf16 = {
                "len_case": {
                    "prompt_tokens": 100, "completion_tokens": 3, "text": "abc", "token_ids": [1, 2, 3],
                    "steps": [
                        {"token": "a", "logprob": -0.1, "top": {"a": -0.1, "x": -2.5}},
                        {"token": "b", "logprob": -0.1, "top": {"b": -0.1}},
                        {"token": "c", "logprob": -0.1, "top": {"c": -0.1}},
                    ],
                },
                "same_case": {
                    "prompt_tokens": 100, "completion_tokens": 2, "text": "ab", "token_ids": [1, 2],
                    "steps": [
                        {"token": "a", "logprob": -0.1, "top": {"a": -0.1}},
                        {"token": "b", "logprob": -0.1, "top": {"b": -0.1}},
                    ],
                },
                "disjoint_case": {
                    "prompt_tokens": 100, "completion_tokens": 1, "text": "a", "token_ids": [1],
                    "steps": [{"token": "a", "logprob": -0.2, "top": {"a": -0.2}}],
                },
            }
            fp8 = {
                "len_case": {
                    "prompt_tokens": 100, "completion_tokens": 2, "text": "ab", "token_ids": [1, 2],
                    "steps": [
                        {"token": "a", "logprob": -0.1, "top": {"a": -0.1, "y": -2.5}},
                        {"token": "b", "logprob": -0.1, "top": {"b": -0.1}},
                    ],
                },
                "same_case": {
                    "prompt_tokens": 100, "completion_tokens": 2, "text": "ab", "token_ids": [1, 2],
                    "steps": [
                        {"token": "a", "logprob": -0.1, "top": {"a": -0.1}},
                        {"token": "b", "logprob": -0.1, "top": {"b": -0.1}},
                    ],
                },
                "disjoint_case": {
                    "prompt_tokens": 100, "completion_tokens": 1, "text": "a", "token_ids": [1],
                    "steps": [{"token": "a", "logprob": -0.2, "top": {"z": -0.2}}],
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
            self.assertTrue(rep["cases"]["same_case"]["identical"])
            self.assertEqual(rep["cases"]["same_case"]["status"], "IDENTICAL")
            self.assertIsNone(rep["cases"]["disjoint_case"]["step0_cond_kl"])
            self.assertIn("step0_cond_KL=unavailable", out)
            self.assertIn("conditional intersection-only diagnostics", out)
            self.assertNotIn("PPL mult", out)
            self.assertIsNotNone(rep["chapters"]["missing_support_ref_mean"])
            self.assertFalse(rep["teacher_forced"]["available"])

    def test_teacher_forced_scores_immutable_tokenized_continuation_and_compares_gt_nll(self):
        requests_seen = []

        def make_handler(gt_202_lp: float):
            def handler(path, body):
                requests_seen.append((path, body))
                if path == "/v1/tokenize":
                    if body.get("add_special_tokens"):
                        return 200, "application/json", json.dumps({"tokens": [101, 102, 103]}).encode()
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
            write_books(td, ["war_and_peace", "moby_dick"])
            p16, p8 = Path(td) / "bf16.json", Path(td) / "fp8.json"
            for lp, path in ((-3.0, p16), (-3.2, p8)):
                srv, th, url = start_server(make_handler(lp))
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        rec = bench_prose.dump_prose_loss(url, td, str(path), cases=bench_prose.LOSS_CASES[:1], tf_tokens=4)
                finally:
                    srv.shutdown()
                    srv.server_close()
                    th.join(timeout=5)
                if path == p16:
                    tf = rec["_tf_moby_dick_16k"]
                    self.assertEqual(tf["continuation_token_ids"], [201, 202, 203, 204])
                    self.assertEqual(tf["continuation_token_count"], 4)
                    self.assertAlmostEqual(tf["steps"][1]["gt_logprob"], -3.0)
                    self.assertAlmostEqual(tf["steps"][1]["gt_prob"], math.exp(-3.0))
                    self.assertEqual(tf["steps"][1]["top1_token_id"], "999")
            with contextlib.redirect_stdout(io.StringIO()):
                cmp = bench_prose.compare_prose_loss(str(p16), str(p8))
        self.assertTrue(cmp["teacher_forced"]["available"])
        self.assertEqual(cmp["teacher_forced"]["continuation_token_count"], 4)
        self.assertAlmostEqual(cmp["teacher_forced"]["gt_nll_delta"], 0.05)
        self.assertEqual(sum(1 for p, _ in requests_seen if p == "/v1/completions"), 2)

    def test_distinct_long_separates_submitted_completed_rejects_undersized_and_omits_kv_claim(self):
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
        self.assertEqual(rep["completed_completion_tokens"], 12)
        self.assertFalse(rep["qualified"])
        self.assertEqual(rep["cell_status"], "undersized")
        self.assertIn("Stream0_WarAndPeace", rep["undersized_streams"])
        s0 = next(s for s in rep["streams"] if s["name"] == "Stream0_WarAndPeace")
        self.assertLess(s0["prompt_tokens"], 500)

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


if __name__ == "__main__":
    unittest.main()
