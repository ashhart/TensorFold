"""Streamed replies against the scripted server: event kinds, boundaries, completion classes and chunk-free digests."""
import hashlib
import math
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_cache
import bench_concurrent
import bench_openai
import bench_stream
from bench_evidence import Evidence
from fake_served_api import script, serve

CHAT = {"name": "fixture", "kind": "chat", "prompt": "A public fixture.", "thinking_switch": True}
CHUNKINGS = ([1], [3], [2, 5, 1])
WORDS = ["Fib", "onacci", " adds", " the", " two", " before", " it", "."]


class SpeculativeReceipt(unittest.TestCase):
    def test_keeps_round_metrics(self):
        reply = bench_stream.Collector(0)
        reply.speculative = {"rounds": 3, "tokens_per_round": 2.5}
        self.assertEqual(reply.speculative, reply.receipt()["speculative"])


def content(*words, choice=0):
    return [(choice, "content", w) for w in words]


def reasoning(*words, choice=0):
    return [(choice, "reasoning", w) for w in words]


def tool(index, name, *arguments, choice=0):
    return [(choice, "tool", index, "name", name)] + [(choice, "tool", index, "arguments", a) for a in arguments]


def expected_output(text="", thought=""):
    parts = hashlib.sha256(text.encode()).digest() + hashlib.sha256(thought.encode()).digest()
    return hashlib.sha256(parts).hexdigest()


class Fixtures(unittest.TestCase):
    def served(self, tokens, client=bench_concurrent, cancel_after=None, **options):
        """The same tokens served in each chunking; returns one run per chunking."""
        runs = []
        for sizes in CHUNKINGS:
            with self.subTest(sizes=sizes), serve(lambda body, s=sizes: script(tokens, s, **options)) as base:
                cancel = None
                if cancel_after is not None:
                    cancel = threading.Event()
                    threading.Timer(cancel_after, cancel.set).start()
                    runs.append(client.stream(base, "fixture", CHAT, len(tokens), 0, 1, cancel=cancel))
                else:
                    runs.append(client.stream(base, "fixture", CHAT, len(tokens), 0, 1))
        return runs

    def same(self, runs, completion, *fields):
        """One completion class and one value of each digest across every chunking."""
        self.assertEqual({r["completion"] for r in runs}, {completion})
        for field in ("output_sha256", "tool_sha256", *fields):
            self.assertEqual(len({repr(r[field]) for r in runs}), 1, field)
        return runs[0]

    def test_immediate_headers_with_delayed_output(self):
        runs = self.served(content(*WORDS), before=0.2)
        run = self.same(runs, "finished")
        for r in runs:
            self.assertLess(r["headers_s"], 0.15)
            self.assertGreaterEqual(r["first_payload_s"], 0.2)
            self.assertGreaterEqual(r["first_content_s"], r["first_payload_s"])
            self.assertEqual(r["first_output_s"], r["first_content_s"])
            self.assertLessEqual(r["first_output_s"], r["last_output_s"])
            self.assertAlmostEqual(r["ttft_s"], r["first_content_s"])
        self.assertEqual(run["output_sha256"], expected_output("".join(WORDS)))
        self.assertIsNone(run["first_reasoning_s"])
        self.assertIsNone(run["tool_sha256"])

    def test_token_timing_is_known_only_for_one_token_chunks(self):
        one, three, mixed = self.served(content(*WORDS), gap=0.002)
        self.assertEqual(one["token_timing"], "token")
        self.assertIsNotNone(one["itl_p50_ms"])
        self.assertAlmostEqual(one["tpot_ms"], one["chunk_tpot_estimate_ms"])
        for r in (three, mixed):
            self.assertEqual(r["token_timing"], "chunk")
            self.assertIsNone(r["itl_p50_ms"])
            self.assertIsNone(r["tpot_ms"])
            self.assertIsNotNone(r["chunk_tpot_estimate_ms"])

    def test_reasoning_only(self):
        run = self.same(self.served(reasoning("Add", " the", " last", " two")), "finished")
        self.assertEqual(run["output_sha256"], expected_output(thought="Add the last two"))
        self.assertIsNone(run["first_content_s"])
        self.assertEqual(run["first_output_s"], run["first_reasoning_s"])
        self.assertIsNotNone(run["ttft_s"])

    def test_tool_only(self):
        runs = self.served(tool(0, "lookup", '{"key"', ': "fib', 'onacci"}'), finish="tool_calls")
        run = self.same(runs, "finished", "finish_reasons")
        self.assertEqual(run["finish_reasons"], ["tool_calls"])
        self.assertIsNotNone(run["first_tool_s"])
        self.assertEqual(run["first_output_s"], run["first_tool_s"])
        self.assertIsNone(run["ttft_s"])                 # unchanged: TTFT is the first text or reasoning
        self.assertIsNone(run["output_sha256"])          # unchanged: a tool-only reply cannot qualify
        self.assertEqual(run["tool_calls"], 1)
        self.assertFalse(bench_cache.qualify([bench_cache.receipt(run, "first")]))
        whole = Evidence()
        whole.add({"delta": {"tool_calls": [{"index": 0, "type": "function", "function": {
            "name": "lookup", "arguments": '{"key": "fibonacci"}'}}]}})
        self.assertEqual(run["tool_sha256"], whole.tool_digest())

    def test_interleaved_tool_indices(self):
        a, b = tool(0, "read", '{"path":', ' "a.py"}'), tool(1, "grep", '{"pattern":', ' "fib"}')
        interleaved = [a[0], b[0], a[1], b[1], a[2], b[2]]
        one = self.same(self.served(interleaved, finish="tool_calls"), "finished")
        apart = self.same(self.served(a + b, finish="tool_calls"), "finished")
        self.assertEqual(one["tool_sha256"], apart["tool_sha256"])
        self.assertEqual(one["tool_calls"], 2)
        swapped = self.same(self.served(tool(0, "read", '{"path":', ' "b.py"}') + b, finish="tool_calls"), "finished")
        self.assertNotEqual(swapped["tool_sha256"], one["tool_sha256"])

    def test_text_then_tool(self):
        tokens = reasoning("Look", " it", " up") + content("Checking.") + tool(0, "lookup", '{"k":1}')
        run = self.same(self.served(tokens, finish="tool_calls"), "finished")
        self.assertLess(run["first_reasoning_s"], run["first_tool_s"])
        self.assertEqual(run["first_output_s"], run["first_reasoning_s"])
        self.assertEqual(run["output_sha256"], expected_output("Checking.", "Look it up"))

    def test_multiple_choices(self):
        zero, one = content("A", "B", "C"), content("x", "y", "z", choice=1)
        mixed = [zero[0], one[0], one[1], zero[1], one[2], zero[2]]
        a = self.same(self.served(mixed), "finished", "choices")
        b = self.same(self.served(zero + one), "finished")
        self.assertEqual(a["output_sha256"], b["output_sha256"])
        self.assertEqual((a["choices"], a["finish_reasons"]), (2, ["stop", "stop"]))
        self.assertIsNone(a["token_timing"])             # two choices' chunks are not one token sequence
        self.assertIsNotNone(a["first_output_s"])

    def test_error_mid_stream(self):
        run = self.same(self.served(content(*WORDS), error_after=4), "error")
        self.assertEqual(run["error"], "RuntimeError")
        self.assertFalse(run["complete"])
        self.assertIsNone(run["output_sha256"])
        self.assertIsNone(run["first_output_s"])

    def test_truncation_finish_length(self):
        run = self.same(self.served(content(*WORDS), finish="length"), "truncated")
        self.assertTrue(run["complete"])
        self.assertEqual(run["output_sha256"], expected_output("".join(WORDS)))

    def test_missing_finish_reason_is_unfinished(self):
        self.same(self.served(content(*WORDS), finish=None), "unfinished")

    def test_usage_only_final_chunk_is_not_the_last_token(self):
        for r in self.served(content(*WORDS), usage_after=0.15):
            self.assertLess(r["last_output_s"] + 0.1, r["completed"] - r["sent"])
            self.assertEqual(r["tokens"], len(WORDS))
        run = self.same(self.served([]), "no_output")
        self.assertIsNone(run["output_sha256"])
        self.assertIsNone(run["first_output_s"])

    def test_closed_before_done_is_incomplete(self):
        run = self.same(self.served(content(*WORDS), done=False), "incomplete")
        self.assertEqual(run["error"], "incomplete_stream")
        self.assertIsNone(run["output_sha256"])

    def test_client_cancellation(self):
        run = self.same(self.served(content(*(WORDS * 25)), gap=0.01, cancel_after=0.06), "cancelled")
        self.assertEqual(run["error"], "client_cancelled")
        self.assertFalse(run["complete"])
        self.assertIsNone(run["output_sha256"])
        self.assertIsNone(run["tokens"])

    def test_single_stream_client_reads_the_same_events(self):
        tokens = reasoning("Look") + content(*WORDS) + tool(0, "lookup", '{"k":1}')
        mine = self.same(self.served(tokens, client=bench_openai, finish="tool_calls"), "finished")
        theirs = self.same(self.served(tokens, finish="tool_calls"), "finished")
        for field in ("output_sha256", "tool_sha256", "finish_reasons", "tool_calls"):
            self.assertEqual(mine[field], theirs[field])
        self.assertEqual(mine["text"], "".join(WORDS))


class Evidence_(unittest.TestCase):
    def test_tool_digest_ignores_call_ids_and_rejects_malformed_calls(self):
        a, b = Evidence(), Evidence()
        a.add({"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "f", "arguments": "{}"}}]}})
        b.add({"delta": {"tool_calls": [{"index": 0, "id": "call_2", "function": {"name": "f", "arguments": "{}"}}]}})
        self.assertEqual(a.tool_digest(), b.tool_digest())
        for calls in ["x", ["x"], [{"index": -1}], [{"index": True}], [{"function": {"arguments": 1}}]]:
            with self.subTest(calls=calls), self.assertRaises(ValueError):
                Evidence().add({"delta": {"tool_calls": calls}})

    def test_one_choice_digest_is_unchanged(self):
        evidence = Evidence()
        evidence.add({"index": 0, "delta": {"content": "caf"}})
        evidence.add({"delta": {"content": "é", "reasoning_content": "why"}})
        self.assertEqual(evidence.digest(), expected_output("café", "why"))


class Overhead(unittest.TestCase):
    def test_collector_cost_a_chunk_is_measured(self):
        for delta in (None, {"tool_calls": [{"index": 0, "function": {"arguments": '"x"'}}]}):
            cost = bench_stream.overhead(500, delta)
            self.assertTrue(math.isfinite(cost) and cost > 0)


if __name__ == "__main__":
    unittest.main()
