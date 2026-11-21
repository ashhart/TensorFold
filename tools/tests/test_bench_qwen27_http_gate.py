"""tools/zig/qwen27_http_gate.py's reply check on fake served replies: a drafted reply must propose and accept."""
import json
from pathlib import Path
import sys
import unittest
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "zig"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen27_http_gate
from fake_served_api import fake

TOKENS = [11, 22, 33]


def reply(fault, draft):
    """The fake's reply to one request, its runtime fields set as the CLI receipt's tokens give them."""
    with fake(False, fault) as url:
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": len(TOKENS), "draft": draft}
        request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as answer:
            doc = json.load(answer)
    digest = qwen27_http_gate.hashlib.sha256(",".join(map(str, TOKENS)).encode()).hexdigest()[:12]
    doc["tensorfold"].update(token_sha=digest, drafts=draft)
    doc["usage"].update(prompt_tokens=2, completion_tokens=len(TOKENS))
    return doc


class DraftCheckTests(unittest.TestCase):
    def test_drafted_and_plain_replies_pass(self):
        for draft in (True, False):
            self.assertEqual(qwen27_http_gate.check(reply(None, draft), [1, 2], TOKENS, draft)["drafts"], draft)

    def test_a_drafted_reply_that_accepted_nothing_fails(self):
        doc = reply("undrafted", True)
        doc["speculative"].update(rounds=4, drafted=12)  # verified rounds whose drafts all missed
        with self.assertRaisesRegex(ValueError, "did not exercise draft verification"):
            qwen27_http_gate.check(doc, [1, 2], TOKENS, True)


if __name__ == "__main__":
    unittest.main()
