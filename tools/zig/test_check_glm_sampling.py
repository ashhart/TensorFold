"""Ensure missing snapshot restoration and mismatched token hashes fail the served gate."""
import unittest
from check_glm_sampling import check, signature


def reply(token_hash="same", cached=0):
    return {"tensorfold": {"token_sha": token_hash}, "usage": {"completion_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": cached}}, "choices": [{"finish_reason": "length"}]}


class GateTests(unittest.TestCase):
    def test_complete_receipt(self):
        responses = iter([reply(), reply(cached=512), reply(cached=512), reply(cached=512), reply(cached=512)])
        records = check(lambda body: next(responses), [{"seed": 7}])
        self.assertEqual(len(records[0]["concurrent"]), 2)
        self.assertEqual(records[0]["request"], {"seed": 7})

    def test_missing_hash_fails(self):
        with self.assertRaisesRegex(ValueError, "token_sha"):
            signature(reply(None))

    def test_no_restore_fails(self):
        with self.assertRaisesRegex(ValueError, "snapshot restoration"):
            check(lambda body: reply(), [{}])

    def test_restore_mismatch_fails(self):
        responses = iter([reply(), reply("different", 512), reply(cached=512)])
        with self.assertRaisesRegex(ValueError, "token mismatch"):
            check(lambda body: next(responses), [{}])

    def test_concurrent_mismatch_fails(self):
        responses = iter([reply(), reply(cached=512), reply(cached=512), reply("changed", 512)])
        with self.assertRaisesRegex(ValueError, "concurrent/solo"):
            check(lambda body: next(responses), [{}])


if __name__ == "__main__":
    unittest.main()
