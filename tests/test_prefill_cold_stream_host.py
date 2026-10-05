"""Cold-prefill measurements require an output token and a completed, error-free stream."""

import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "prefill_cold", Path(__file__).resolve().parents[1] / "tools" / "prefill_cold.py")
prefill_cold = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prefill_cold)


def event(value):
    return "data: " + json.dumps(value) + "\n\n"


ROLE = event({"choices": [{"delta": {"role": "assistant"}}]})
TEXT = event({"choices": [{"delta": {"content": "A"}}]})
FINISH = event({"choices": [{"delta": {}, "finish_reason": "stop"}]})
USAGE = event({"choices": [], "usage": {"prompt_tokens": 2048, "completion_tokens": 1}})
ERROR = event({"error": {"message": "synthetic failure", "type": "server_error"}})
DONE = "data: [DONE]\n\n"


class PrefillColdStreamTest(unittest.TestCase):
    def measure(self, stream):
        response = io.BytesIO(stream.encode())
        with patch.object(prefill_cold.urllib.request, "urlopen", return_value=response), \
                patch.object(prefill_cold.time, "perf_counter", side_effect=[10.0, 12.0, 15.0]):
            return prefill_cold.one("http://example.invalid", "synthetic", [])

    def test_complete_content_stream_preserves_ttft_and_usage(self):
        self.assertEqual(self.measure(ROLE + TEXT + FINISH + USAGE + DONE),
                         {"ttft_s": 2.0, "prompt_tokens": 2048})

    def test_complete_reasoning_stream_is_measurable(self):
        for field in ("reasoning_content", "reasoning"):
            with self.subTest(field=field):
                text = event({"choices": [{"delta": {field: "A"}}]})
                self.assertEqual(self.measure(ROLE + text + FINISH + USAGE + DONE)["ttft_s"], 2.0)

    def test_empty_or_role_only_stream_has_no_ttft(self):
        for stream in (DONE, ROLE + FINISH + USAGE + DONE):
            with self.subTest(stream=stream):
                with self.assertRaisesRegex(RuntimeError, "without an output token"):
                    self.measure(stream)

    def test_error_event_is_not_a_measurement(self):
        for stream in (ERROR + DONE, ROLE + TEXT + ERROR + DONE):
            with self.subTest(stream=stream):
                with self.assertRaisesRegex(RuntimeError, "server error"):
                    self.measure(stream)

    def test_truncated_stream_is_not_a_measurement(self):
        for stream in ("", ROLE, ROLE + TEXT, ROLE + TEXT + FINISH + USAGE):
            with self.subTest(stream=stream):
                with self.assertRaisesRegex(RuntimeError, "before.*DONE"):
                    self.measure(stream)

    def test_missing_prompt_usage_cannot_become_zero_throughput(self):
        for usage in ("", event({"choices": [], "usage": {"completion_tokens": 1}})):
            with self.subTest(usage=usage):
                with self.assertRaisesRegex(RuntimeError, "positive integer prompt_tokens"):
                    self.measure(ROLE + TEXT + FINISH + usage + DONE)

    def test_invalid_prompt_usage_is_not_a_measurement(self):
        for count in (None, 0, -1, True, 2048.0, "2048"):
            with self.subTest(count=count):
                usage = event({"choices": [], "usage": {"prompt_tokens": count}})
                with self.assertRaisesRegex(RuntimeError, "positive integer prompt_tokens"):
                    self.measure(ROLE + TEXT + FINISH + usage + DONE)


if __name__ == "__main__":
    unittest.main()
