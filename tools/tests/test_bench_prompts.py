"""bench_concurrent --prompts sends a different prompt on each stream and checks each against its own solo run."""
import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_concurrent

TOOLS = Path(__file__).resolve().parents[1]


def response():
    chunks = [{"choices": [{"text": t}]} for t in ("H", "ell", "o", " world")]
    chunks.append({"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 4,
                                            "prompt_tokens_details": {"cached_tokens": 0}},
                   "tensorfold": {"token_sha": "a" * 12, "token_sha256": "a" * 64}})
    return io.BytesIO(b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n")


def sent_text(body):
    return body["messages"][0]["content"] if "messages" in body else body["prompt"]


class PromptFile(unittest.TestCase):
    def test_the_repository_file_has_unique_public_prompts_of_both_kinds(self):
        items = bench_concurrent.distinct(str(TOOLS / "bench_prompts.json"))
        self.assertEqual(len(items), 16)
        self.assertEqual(len({q["name"] for q in items}), 16)
        self.assertEqual(len({q["prompt"] for q in items}), 16)
        self.assertEqual({q["kind"] for q in items}, {"chat", "completion"})

    def test_malformed_files_are_refused(self):
        bad = [[], {}, [{"name": "a", "kind": "chat"}], [{"name": "a", "kind": "other", "prompt": "x"}],
               [{"name": "a", "kind": "chat", "prompt": ""}], [{"name": 1, "kind": "chat", "prompt": "x"}],
               [{"name": "a", "kind": "chat", "prompt": "x"}, {"name": "a", "kind": "completion", "prompt": "y"}]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.json"
            for items in bad:
                with self.subTest(items=items):
                    path.write_text(json.dumps(items))
                    with self.assertRaises(ValueError):
                        bench_concurrent.distinct(str(path))
            path.write_text("[")
            with self.assertRaises(ValueError):
                bench_concurrent.distinct(str(path))


class DistinctCell(unittest.TestCase):
    def main(self, *extra):
        bodies, lock = [], threading.Lock()

        def urlopen(req, timeout):
            with lock:
                bodies.append(json.loads(req.data))
            return response()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            argv = ["bench_concurrent.py", "http://unused", "fixture", "--levels", "4", "--reps", "1",
                    "--temperatures", "0", "--output", str(output), *extra]
            with patch.object(sys, "argv", argv), patch.object(bench_concurrent.urllib.request, "urlopen", urlopen), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                bench_concurrent.main()
            return json.loads(output.read_text()), bodies

    def test_default_cells_send_one_prompt_on_every_stream(self):
        report, bodies = self.main()
        concurrent = [b for b in bodies if b["max_tokens"] != 16]
        self.assertEqual(len(concurrent), 8)  # two cells of four streams
        self.assertEqual(len({sent_text(b) for b in concurrent}), 2)

    def test_prompts_file_sends_a_different_prompt_on_each_stream(self):
        report, bodies = self.main("--prompts", str(TOOLS / "bench_prompts.json"), "--alone", "--serial")
        items = bench_concurrent.distinct(str(TOOLS / "bench_prompts.json"))
        self.assertEqual(report["prompts"], [q["name"] for q in items])
        self.assertEqual([c["prompt"] for c in report["cells"]], ["distinct"])
        cell = report["cells"][0]
        self.assertEqual({sent_text(b) for b in bodies if b["max_tokens"] != 16}, {q["prompt"] for q in items[:4]})
        self.assertEqual(cell["alone"], {"equal": 4, "unequal": 0, "failed": 0})
        self.assertEqual(cell["serial"], {"equal": 4, "unequal": 0, "failed": 0})
        self.assertEqual(sum(1 for b in bodies if b.get("draft") is False), 4)

    def test_prompts_and_mixed_are_exclusive(self):
        with self.assertRaises(SystemExit):
            self.main("--prompts", str(TOOLS / "bench_prompts.json"), "--mixed")


if __name__ == "__main__":
    unittest.main()
