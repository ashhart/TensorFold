"""Host-only checks for PR836 token and zero-cache assumptions."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import prepare_fixture
import run_panel
import run_safety
import run_startup


ROOT = Path(__file__).parent


class Pr836HarnessTests(unittest.TestCase):
    """Reject old cache assumptions without contacting a server."""

    def test_live_fixture_replaces_stale_token_counts(self) -> None:
        """Only the selected server's rendered IDs qualify literal prompts."""
        calls = 0

        def tokenize(body: dict) -> list[int]:
            nonlocal calls
            calls += 1
            if "prompt" in body:
                return [7, calls]
            text = body["messages"][0]["content"]
            if len(text) > 1000:
                prefix = list(range(4100))
                return prefix + ([1] if len(body["messages"]) == 1 else [2, 3])
            return [9, calls]

        corpus, long = prepare_fixture.prepare(
            ROOT / "literal-corpus-source.json",
            ROOT / "literal-long-source.json", tokenize,
        )
        self.assertEqual([2, 2], [x["prompt_tokens"] for x in corpus["anchors"]])
        self.assertEqual(["code-changed", "chat-changed"],
                         [x["case_id"] for x in corpus["changes"]])
        self.assertEqual(len(corpus["changes"][1]["messages"]), 3)
        self.assertEqual(long["common_full_prefix_tokens"], 4100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(corpus))
            self.assertEqual(len(run_panel.corpus(path)), 42)
            self.assertEqual(len(run_panel.anchors(path)), 2)
            self.assertEqual(len(run_panel.changes(path)), 2)

    def test_changing_history_rejects_short_shared_prefix(self) -> None:
        """A long string alone is insufficient without shared rendered IDs."""
        data = {"schema": "pr836_chat_history_fixture_v1",
                "common_full_prefix_tokens": 4096,
                "rendered": {"initial": {"generation": list(range(4096))},
                             "followup": {"generation": [8] + list(range(4095))}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.json"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "shared 4096-token"):
                run_panel.long_cases(path)

    def test_safety_refuses_any_prompt_cache(self) -> None:
        """Safety evidence cannot hide a cached sampled control."""
        run_safety.require_uncached(
            {"usage": {"prompt_tokens_details": {"cached_tokens": 0}},
             "cached_tokens": 0},
        )
        with self.assertRaisesRegex(ValueError, "unexpectedly used prompt cache"):
            run_safety.require_uncached(
                {"usage": {"prompt_tokens_details": {"cached_tokens": 1}},
                 "cached_tokens": 1},
            )

    def test_panel_rejects_cached_repeat(self) -> None:
        """A cache hit cannot enter the handoff-only comparison as a repeat."""
        case = {"case_id": "code-anchor", "kind": "completion",
                "prompt": "small test", "prompt_tokens": 2}
        body = {"model": "native-ttft", "max_tokens": 256,
                "temperature": 0.0, "stream": True,
                "stream_options": {"include_usage": True},
                "ignore_eos": True, "seed": 1234, "prompt": "small test"}
        row = {"complete": True, "tokens": 256,
               "finish_reason": "length", "unmeasured": False,
               "usage": {"prompt_tokens": 2, "completion_tokens": 256,
                         "prompt_tokens_details": {"cached_tokens": 1}},
               "cached_tokens": 1, "request_body": body,
               "request_body_sha256": hashlib.sha256(
                   json.dumps(body).encode()).hexdigest(),
               "sent": 1.0, "first": 1.1, "last": 2.0, "done": 2.1,
               "ttft_s": 0.1, "full_response_s": 1.1,
               "output_sha256": "a" * 64, "token_sha": "b" * 64}
        with self.assertRaisesRegex(ValueError, "prompt or cache usage mismatch"):
            run_panel.validate(row, case, cache_rule="zero")

    def test_startup_requires_zero_cache_for_both_treatments(self) -> None:
        """The candidate label does not imply the old full-cache contract."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps({"anchors": [
                {"name": "code", "case_id": "code-anchor",
                 "input_ids": [1, 2], "prompt_tokens": 2},
                {"name": "chat", "case_id": "chat-anchor",
                 "input_ids": [3, 4], "prompt_tokens": 2},
            ], "schema": "pr836_ttft_literal_corpus_v1"}))
            self.assertEqual(
                run_startup.selected_cases(path, "code", 1)[0]["prompt_tokens"],
                2,
            )


if __name__ == "__main__":
    unittest.main()
