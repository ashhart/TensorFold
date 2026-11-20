"""The quality suite runner against a stand-in server: the Bench's scoring, thinking off, and the receipt's counts."""
import contextlib
import http.server
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quality_suite
from tensorfold_web import quality


def server(answer, switch=True):
    """A served model that answers each item through `answer(item)`; it keeps every request body it saw."""
    data = quality.load()
    suites = json.loads((quality_suite.WEB / "tensorfold_web" / "data" / "suites.json").read_text(encoding="utf-8"))
    by_prompt = {it["messages"][0]["content"]: it for it in quality.items(data, suites["passages"]["water"] + "\n\n")}
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def reply(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply({"data": [{"id": "stand-in", "capabilities": {"thinking_switch": switch}}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(req)
            it = by_prompt[req["messages"][0]["content"]]
            self.reply({"choices": [{"message": {"content": answer(it)}, "finish_reason": "stop"}]})

        def log_message(self, format, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, seen


def right(it):
    if it["category"] == "gsm8k":
        return f"Working.\nAnswer: {it['item']['answer']}"
    if it["category"] in ("mmlu", "long_context"):
        return str(it["item"]["answer"])
    return "no"


class QualitySuite(unittest.TestCase):
    def run_suite(self, answer, switch=True):
        httpd, seen = server(answer, switch)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "receipt.json"
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                code = quality_suite.main([f"http://127.0.0.1:{httpd.server_address[1]}", str(out),
                                           "--checkpoint", "Org/Model@rev"])
            return code, json.loads(out.read_text()), seen

    def test_right_answers_pass_by_the_benchs_scoring_and_thinking_is_turned_off(self):
        code, receipt, seen = self.run_suite(right)
        self.assertEqual(code, 0)
        cats = receipt["quality"]["categories"]
        self.assertEqual((cats["gsm8k"]["passed"], cats["mmlu"]["passed"], cats["long_context"]["passed"]), (100, 100, 8))
        self.assertEqual(receipt["quality"]["overall"]["items"], 228)
        self.assertEqual(receipt["suite"]["digest"], quality.load()["digest"])
        self.assertEqual(receipt["checkpoint"], "Org/Model@rev")
        self.assertEqual(len(seen), 228)
        self.assertTrue(all(r["chat_template_kwargs"] == {"enable_thinking": False} and r["temperature"] == 0.0 for r in seen))

    def test_wrong_answers_fail_and_a_model_without_a_switch_gets_no_off_field(self):
        code, receipt, seen = self.run_suite(lambda it: "I am not sure.", switch=False)
        self.assertEqual(code, 0)
        self.assertEqual(receipt["quality"]["categories"]["mmlu"]["passed"], 0)
        self.assertEqual(receipt["protocol"]["thinking"], "model default")
        self.assertTrue(all("chat_template_kwargs" not in r for r in seen))


if __name__ == "__main__":
    unittest.main()
