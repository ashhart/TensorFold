"""Model-free discovery and truthful thinking-mode requests for both benchmark clients."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_concurrent
import bench_openai
import bench_thinking
from test_bench_auth import response


class ThinkingDiscovery(unittest.TestCase):
    def test_stream_omits_an_unsupported_switch(self):
        for module in (bench_openai, bench_concurrent):
            item = {**module.PROMPTS[1], "thinking_switch": False}
            with self.subTest(client=module.__name__), patch.object(module.urllib.request, "urlopen", return_value=response()) as http:
                module.stream("http://unused", "fixture", item, 4, 0, 1)
                self.assertNotIn("chat_template_kwargs", json.loads(http.call_args.args[0].data))

    def test_unknown_keeps_the_legacy_off_request(self):
        for module in (bench_openai, bench_concurrent):
            item = {**module.PROMPTS[1], "thinking_switch": None}
            with self.subTest(client=module.__name__), patch.object(module.urllib.request, "urlopen", return_value=response()) as http:
                module.stream("http://unused", "fixture", item, 4, 0, 1)
                self.assertEqual(json.loads(http.call_args.args[0].data).get("chat_template_kwargs"), {"enable_thinking": False})

    def run_cli(self, module, bit, *, document=None, baseline=None):
        seen = []
        def opened(req, **kwargs):
            seen.append(req)
            if req.get_method() == "GET":
                doc = document if document is not None else {"data": [{"id": "fixture", "capabilities": {"thinking_switch": bit}}]}
                return io.BytesIO(json.dumps(doc).encode())
            return response()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "receipt.json"
            argv = [module.__name__, "http://unused", "fixture", "--tokens", "4", "--reps", "1", "--temperatures", "0", "--output", str(out)]
            if baseline is not None:
                compare = Path(directory) / "baseline.json"
                compare.write_text(json.dumps(baseline))
                argv += ["--compare-receipt", str(compare)]
            if module is bench_concurrent:
                argv += ["--levels", "1,2", "--alone", "--serial"]
            with patch.object(sys, "argv", argv), patch.object(module.urllib.request, "urlopen", side_effect=opened), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            receipt = json.loads(out.read_text())
        return seen, receipt

    def test_one_discovery_controls_every_chat_and_receipt(self):
        for module in (bench_openai, bench_concurrent):
            for bit in (True, False, None, "false", 1):
                with self.subTest(client=module.__name__, bit=bit):
                    seen, receipt = self.run_cli(module, bit)
                    get = [r for r in seen if r.get_method() == "GET"]
                    self.assertEqual(len(get), 1)
                    self.assertEqual(get[0].full_url, "http://unused/v1/models")
                    for req in seen:
                        if req.data is None:
                            continue
                        body = json.loads(req.data)
                        if "messages" in body and bit is not False:
                            self.assertIs(body["chat_template_kwargs"]["enable_thinking"], False)
                        else:
                            self.assertNotIn("chat_template_kwargs", body)
                    rows = receipt if module is bench_openai else receipt["cells"]
                    chats = [r for r in rows if r["prompt"] in ("chat", "gpu-chat", "gpu-chat-no-think")]
                    self.assertTrue(chats)
                    for row in chats:
                        self.assertEqual(row["thinking_off_requested"], bit is not False)
                        self.assertEqual(row["thinking"], "off" if bit is True else "default (thinking off unsupported)" if bit is False else "off requested (unverified)")

    def test_another_models_bit_and_missing_capability_are_unknown(self):
        for module in (bench_openai, bench_concurrent):
            for doc in ({"data": [{"id": "other", "capabilities": {"thinking_switch": True}}]}, {"data": [{"id": "fixture"}]}):
                seen, receipt = self.run_cli(module, None, document=doc)
                for req in seen:
                    if req.data:
                        self.assertEqual(json.loads(req.data).get("chat_template_kwargs"), {"enable_thinking": False} if "messages" in json.loads(req.data) else None)

    def test_nonreasoning_chat_is_not_applicable(self):
        for module in (bench_openai, bench_concurrent):
            item = {**module.PROMPTS[1], "thinking_switch": True, "thinking_applicable": False}
            self.assertEqual(bench_thinking.receipt(item)["thinking"], "not applicable")
            with patch.object(module.urllib.request, "urlopen", return_value=response()) as http:
                module.stream("http://unused", "fixture", item, 4, 0, 1)
                self.assertNotIn("chat_template_kwargs", json.loads(http.call_args.args[0].data))

    def test_discovered_no_thinking_labels_both_clients_not_applicable(self):
        document = {"data": [{"id": "fixture", "capabilities": {"thinking_switch": True, "thinking_applicable": False}}]}
        for module in (bench_openai, bench_concurrent):
            seen, report = self.run_cli(module, True, document=document)
            for request in seen:
                if request.data:
                    self.assertNotIn("chat_template_kwargs", json.loads(request.data))
            rows = report if module is bench_openai else report["cells"]
            self.assertTrue(all(row["thinking"] == "not applicable" for row in rows))

    def test_single_stream_prompt_name_stays_stable(self):
        self.assertEqual(bench_openai.PROMPTS[1]["name"], "gpu-chat-no-think")

    def test_comparison_flags_different_modes_and_accepts_matching_modes(self):
        row = {"prompt": "gpu-chat-no-think", "temperature": 0, "thinking": "off requested (unverified)"}
        old = {"prompt": "gpu-chat", "temperature": 0, "thinking": "off"}
        compared = bench_thinking.comparison(row, [old])
        self.assertFalse(compared["thinking_comparable"])
        self.assertIn("Thinking labels differ", compared["comparison_warning"])
        self.assertTrue(bench_thinking.comparison(row, [row])["thinking_comparable"])
        self.assertFalse(bench_thinking.comparison(row, [{"prompt": "gpu-chat", "temperature": 0}])["thinking_comparable"])

    def test_both_clis_flag_a_main_branch_thinking_difference(self):
        for module in (bench_openai, bench_concurrent):
            baseline = [{"prompt": module.PROMPTS[1]["name"], "temperature": 0, "thinking": "off", **({"streams": 1} if module is bench_concurrent else {})}]
            _, report = self.run_cli(module, None, baseline=baseline)
            rows = report if module is bench_openai else report["cells"]
            compared = [row for row in rows if "thinking_comparable" in row]
            self.assertTrue(compared)
            self.assertTrue(all(row["thinking_comparable"] is False for row in compared))
            self.assertTrue(all("comparison_warning" in row for row in compared))

    def test_mixed_comparison_uses_the_request_modes(self):
        row = {"prompt": "mixed", "temperature": 0, "streams": 2, "thinking": "per request", "thinking_modes": ["not applicable", "off"]}
        other = {**row, "thinking_modes": ["not applicable", "off requested (unverified)"]}
        self.assertFalse(bench_thinking.comparison(row, [other])["thinking_comparable"])


if __name__ == "__main__":
    unittest.main()
