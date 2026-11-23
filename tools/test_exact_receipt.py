"""The cross-backend exact receipt catches each planted divergence without recording private request data."""
import contextlib
import io
import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exact_receipt as rc  # noqa: E402
import exact_restarts as er  # noqa: E402

PROMPT = {"name": "near-tie", "kind": "chat", "prompt": "Choose one and explain.",
          "followup": "Now give the other one.", "expected_token_ids": list(range(8)),
          "near_tie_offset": 0, "recording_sha256": "a" * 64,
          "near_tie": {"ids": [0, 1], "logits": [16.0, 15.9375], "position": 0}}

TEXT_PROMPT = {"name": "text", "kind": "completion", "prompt": "Question?", "followup": "Continue.",
               "expected_token_ids": list(range(8)), "near_tie_offset": 0, "recording_sha256": "a" * 64,
               "near_tie": {"ids": [0, 1], "logits": [16.0, 15.9375], "position": 0}}
BASE = [11, 12, 13, 14, 15, 16]


def public(body, cached=0, ignore_seed=False):
    sampled = body.get("temperature", 0) != 0
    budget = body.get("ignore_eos", False)
    count = body["max_tokens"] if budget else min(4, body["max_tokens"])
    salt = 0 if ignore_seed else body.get("seed", 0) if sampled else 0
    ids = [(i + salt) % 97 for i in range(count)]
    return {"ids": ids, "finish": "length" if budget else "stop", "cached": cached,
            "text": "public reply"}


class Fake(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, fault=None, disabled=False):
        super().__init__(("127.0.0.1", 0), Handler)
        self.fault, self.active, self.seen, self.lock = fault, 0, {}, threading.Lock()
        self.cache_disabled = disabled
        self.requests = []
    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_port}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        self.reply(200, {"data": [{"id": "public-model"}]})

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server = self.server
        server.requests.append(body)
        messages = body.get("messages") or []
        key = json.dumps(messages, sort_keys=True)
        with server.lock:
            server.seen[key] = server.seen.get(key, 0) + 1
            arrival = server.seen[key]
            server.active += 1
        time.sleep(0.005)
        with server.lock:
            concurrent = server.active > 1
            server.active -= 1
        if server.fault == "request_error" and body.get("draft") is False:
            return self.reply(500, {"error": {"message": "private details never enter a receipt"}})
        prefix_seen = len(messages) > 1 and server.seen.get(json.dumps(messages[:1], sort_keys=True), 0) > 0
        out = public(body, 0 if server.cache_disabled else 9 if arrival > 1 or prefix_seen else 0,
                     server.fault == "seed_ignored")
        ids = out["ids"][:]
        if server.fault == "warm" and len(messages) <= 1 and arrival > 1:
            ids[0] += 100
        if server.fault == "resumed" and len(messages) > 1:
            ids[1] += 100
        if server.fault == "draft" and body.get("draft") is True:
            ids[2] += 100
        if server.fault == "source" and not messages and body.get("draft") is False:
            ids[0] += 100
        if server.fault == "concurrent" and concurrent:
            ids[3] += 100
        if server.fault == "empty":
            ids = []
        if server.fault == "cached" and out["cached"] > 0:
            out["cached"] = 0
        if server.fault == "eos_finish" and not body.get("ignore_eos"):
            out["finish"] = "length"
        if server.fault == "budget_finish" and body.get("ignore_eos"):
            out["finish"] = "stop"
        drafted = 0 if server.fault == "no_draft" or (not body["draft"] and server.fault != "plain_drafts") else 3
        acknowledged = body["draft"] if server.fault != "bad_draft_ack" else not body["draft"]
        choice = {"finish_reason": out["finish"]}
        if messages:
            choice["message"] = {"content": out["text"]}
        else:
            choice["text"] = out["text"]
        response = {"choices": [choice],
                    "tensorfold": {"token_ids": ids, "drafts": acknowledged, "token_sha256": hashlib.sha256(
                        json.dumps(ids, separators=(",", ":")).encode()).hexdigest()},
                    "speculative": {"drafted": drafted, "accepted": min(2, drafted)},
                    "usage": {"prompt_tokens": 9, "completion_tokens": len(ids),
                              "prompt_tokens_details": {"cached_tokens": out["cached"]}}}
        self.reply(200, response)


class Running:
    def __init__(self, fault=None, disabled=False):
        self.server = Fake(fault, disabled)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self.server.url

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def invoke(url, fresh, fault, tmp, full=False, stops=("budget",), item=PROMPT):
    prompts = Path(tmp) / "prompts.json"
    output = Path(tmp) / f"{fault}.json"
    prompts.write_text(json.dumps([item]))
    argv = ["exact_restarts.py", "--base", url, "--fresh-base", fresh, "--prompts", str(prompts),
            "--output", str(output), "--build-id", "public-build", "--checkpoint-id", "public-checkpoint",
            "--checkpoint-sha256", "b" * 64, "--family", "public-family", "--quant", "public-quant",
            "--tokens", "8", "--eos-tokens", "8", "--skip", "restart"]
    if not full:
        argv += ["--streams", "1", "4", "--sampling", "greedy", "--stops", *stops]
    with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
        code = er.main()
    return code, json.loads(output.read_text())


class ReceiptTests(unittest.TestCase):
    def matrix(self, fault=None, full=False, stops=("budget",)):
        with tempfile.TemporaryDirectory() as tmp, Running(fault) as url, Running(disabled=True) as fresh:
            return invoke(url, fresh, fault or "clean", tmp, full, stops)

    def test_clean_receipt_covers_the_whole_matrix_and_has_no_private_data(self):
        code, report = self.matrix(full=True)
        self.assertEqual(code, 0, report["failures"])
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["coverage"]["streams"], [1, 4, 8])
        self.assertEqual(report["coverage"]["sampling"], ["greedy", "seeded"])
        self.assertEqual(report["coverage"]["stops"], ["budget", "eos"])
        self.assertEqual(report["coverage"]["draft"], [False, True])
        self.assertTrue(report["runs"]["attached"]["rounds"])
        self.assertTrue(all(r["round_wall_seconds"] > 0 for r in report["runs"]["attached"]["rounds"]))
        text = json.dumps(report)
        for private in ("127.0.0.1", "http://", str(HERE), "public reply", "private details"):
            self.assertNotIn(private, text)
        kinds = {r["kind"] for r in report["comparisons"]}
        self.assertTrue({"cold_vs_warm", "fresh_vs_resumed", "drafted_vs_plain", "solo_vs_concurrent"} <= kinds)

    def test_text_completion_followup_includes_the_first_reply(self):
        base_runner, fresh_runner = Running(), Running(disabled=True)
        with tempfile.TemporaryDirectory() as tmp, base_runner as url, fresh_runner as fresh:
            code, report = invoke(url, fresh, "clean", tmp, item=TEXT_PROMPT)
            requests = base_runner.server.requests + fresh_runner.server.requests
        self.assertEqual(code, 0, report["failures"])
        followups = [body["prompt"] for body in requests if body.get("prompt", "").endswith("Continue.")]
        self.assertEqual(followups, ["Question?\npublic reply\nContinue."] * 4)

    def test_warm_and_resumed_divergences_name_the_exact_requests(self):
        for fault, kind in (("warm", "cold_vs_warm"), ("resumed", "fresh_vs_resumed")):
            with self.subTest(fault=fault):
                code, report = self.matrix(fault)
                self.assertEqual(code, 1)
                rows = [r for r in report["comparisons"] if r["kind"] == kind and r["first_diff"] is not None]
                self.assertTrue(rows)
                self.assertIn("near-tie", rows[0]["candidate_request"])

    def test_draft_and_concurrent_divergences_fail(self):
        for fault, kind in (("draft", "drafted_vs_plain"), ("concurrent", "solo_vs_concurrent")):
            with self.subTest(fault=fault):
                code, report = self.matrix(fault)
                self.assertEqual(code, 1)
                self.assertTrue(any(r["kind"] == kind and r["first_diff"] is not None
                                    for r in report["comparisons"]))

    def test_cache_stop_empty_and_request_faults_fail_by_request(self):
        cases = (("cached", "cache"), ("empty", "empty"), ("request_error", "HTTP 500"))
        for fault, reason in cases:
            with self.subTest(fault=fault):
                code, report = self.matrix(fault)
                self.assertEqual(code, 1)
                self.assertTrue(any(reason in r["reason"] and "near-tie" in r["request"]
                                    for r in report["failures"]))
        for fault, stop in (("eos_finish", "eos"), ("budget_finish", "budget")):
            with self.subTest(fault=fault):
                code, report = self.matrix(fault, stops=(stop,))
                self.assertEqual(code, 1)
                self.assertTrue(any("finish" in r["reason"] for r in report["failures"]))

    def test_drafting_that_never_ran_is_not_a_pass(self):
        code, report = self.matrix("no_draft")
        self.assertEqual(code, 1)
        self.assertTrue(any("drafting was not observed" in r["reason"] for r in report["failures"]))

    def recording(self, fault=None):
        rows = [{"ids": [i, 99], "logits": [16.0, 15.9375] if i == 0 else [10.0, 0.0],
                 "position": i + 2} for i in range(8)]
        recording = {"schema": 1, "identity": {"checkpoint": "public-checkpoint"},
                     "traces": [{"name": "raw", "prompt": [1, 2], "tokens": list(range(8)), "top2": rows}]}
        with tempfile.TemporaryDirectory() as tmp, Running(fault) as url, Running(disabled=True) as fresh:
            source, output = Path(tmp) / "recording.json", Path(tmp) / "report.json"
            source.write_text(json.dumps(recording))
            argv = ["exact_restarts.py", "--base", url, "--fresh-base", fresh, "--recording", str(source),
                    "--output", str(output), "--build-id", "public-build", "--checkpoint-id", "public-checkpoint",
                    "--checkpoint-sha256", "b" * 64, "--family", "public-family", "--quant", "public-quant",
                    "--streams", "1", "--sampling", "greedy", "--stops", "budget", "--near-tie-cases", "1",
                    "--lookback", "0", "--near-tie-tokens", "8", "--skip", "restart", "--skip", "draft"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                code = er.main()
            return code, json.loads(output.read_text())

    def test_raw_recording_selection_and_source_reproduction_gate(self):
        code, report = self.recording()
        self.assertEqual(code, 0, report["failures"])
        self.assertEqual(report["coverage"]["near_tie_cases"][0]["prompt"], "raw:0")
        self.assertTrue(all("token_ids" not in row for row in report["runs"]["attached"]["requests"].values()))
        code, report = self.recording("source")
        self.assertEqual(code, 1)
        self.assertTrue(any("recorded greedy window" in r["reason"] for r in report["failures"]))

    def test_acknowledgement_plain_drafting_and_ignored_seed_faults(self):
        for fault, reason in (("bad_draft_ack", "acknowledge"), ("plain_drafts", "plain request")):
            with self.subTest(fault=fault):
                code, report = self.matrix(fault)
                self.assertEqual(code, 1)
                self.assertTrue(any(reason in r["reason"] for r in report["failures"]))
        code, report = self.matrix("seed_ignored", full=True)
        self.assertEqual(code, 1)
        self.assertTrue(any("changed seed" in r["reason"] for r in report["failures"]))

    def test_pair_of_binaries_produces_cross_binary_receipt_without_paths(self):
        base_runner, fresh_runner = Running(), Running(disabled=True)
        with tempfile.TemporaryDirectory() as tmp, base_runner as base_url, fresh_runner as fresh_url:
            one, two = Path(tmp) / "one.bin", Path(tmp) / "two.bin"
            one.write_bytes(b"one")
            two.write_bytes(b"two")
            prompts, output = Path(tmp) / "prompts.json", Path(tmp) / "report.json"
            prompts.write_text(json.dumps([PROMPT]))
            calls = []

            class Launched:
                def __init__(self, argv, port):
                    calls.append(argv)
                    self.port = port

                def __enter__(self):
                    base_runner.server.seen.clear()
                    fresh_runner.server.seen.clear()
                    return self

                def __exit__(self, *exc):
                    return False

            argv = ["exact_restarts.py", "--bin", str(one), "--candidate-bin", str(two), "--model", "public-model",
                    "--port", str(base_runner.server.server_port), "--fresh-port",
                    str(fresh_runner.server.server_port),
                    "--prompts", str(prompts), "--output", str(output), "--checkpoint-id", "public-checkpoint",
                    "--checkpoint-sha256", "b" * 64, "--family", "public-family", "--quant", "public-quant",
                    "--streams", "1", "--sampling", "greedy", "--stops", "budget", "--skip", "restart",
                    "--skip", "draft"]
            with patch.object(er, "Server", Launched), patch.object(sys, "argv", argv), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = er.main()
            report_text = output.read_text()
            report = json.loads(report_text)
            self.assertEqual(code, 0, report["failures"])
            self.assertEqual(set(report["builds"]), {"reference", "candidate"})
            self.assertNotEqual(report["builds"]["reference"], report["builds"]["candidate"])
            self.assertTrue(any(r["kind"] == "cross_binary" for r in report["comparisons"]))
            self.assertEqual(sum("--prompt-cache-gib" in argv for argv in calls), 2)
            self.assertNotIn(str(tmp), report_text)

    def test_cross_binary_greedy_seeded_eos_and_budget_flips_are_found(self):
        for sampling in ("greedy", "seeded"):
            for stop in ("budget", "eos"):
                with self.subTest(sampling=sampling, stop=stop):
                    req = f"near-tie/{sampling}/{stop}/s1/plain/cold/peer0"
                    result = rc.result(req, PROMPT, sampling, stop, 1, False, "cold", 0,
                                       BASE, 0, "length" if stop == "budget" else "stop", 0.1, 0.1, 0, 0)
                    other = dict(result, token_ids=BASE[:2] + [999] + BASE[3:])
                    other["token_ids_json_sha256"] = rc.token_sha256(other["token_ids"])
                    rows = rc.compare_results({req: result}, {req: other}, "binary")
                    self.assertEqual(rows[0]["first_diff"], 2)
                    self.assertEqual(rows[0]["candidate_request"], req)

    def test_identity_and_hash_rules(self):
        self.assertEqual(rc.token_sha256([1, 2, 3]), hashlib.sha256(b"[1,2,3]").hexdigest())
        for value in ("public-build", "sha256:abc123", "Qwen3.8-27B_Q4"):
            self.assertEqual(rc.public_id(value, "build"), value)
        for value in ("root/private/model", "host/name", "user@example.com", "white space", "~model"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rc.public_id(value, "checkpoint")


if __name__ == "__main__":
    unittest.main()
