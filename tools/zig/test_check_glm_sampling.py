"""Ensure the served gate fails without restoration, on mismatches, and on false-positive drafting qualification."""
import unittest
from check_glm_sampling import check, signature, drafted_count, require_drafted


def reply(token_hash="same", cached=0, drafted=0, rounds=None):
    rounds = drafted // 3 if rounds is None else rounds
    return {"tensorfold": {"token_sha": token_hash},
            "usage": {"completion_tokens": 8, "prompt_tokens_details": {"cached_tokens": cached}},
            "choices": [{"finish_reason": "length"}],
            "speculative": {"rounds": rounds, "drafted": drafted, "accepted": drafted,
                            "acceptance_rate": 1.0 if drafted else 0.0}}


def drafted_receipt(cached=512, drafted=12, token_hash="same", rounds=4):
    """plain(cold) + restored + drafted, each with the engine's telemetry fields."""
    return [reply(cached=0), reply(cached=cached),
            reply(cached=cached, drafted=drafted, rounds=rounds)]


class GateTests(unittest.TestCase):
    def test_complete_receipt(self):
        responses = iter(drafted_receipt() + [reply(cached=512), reply(cached=512)])
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
        responses = iter([reply(), reply("different", 512), reply(cached=512, drafted=12, rounds=4)])
        with self.assertRaisesRegex(ValueError, "token mismatch"):
            check(lambda body: next(responses), [{}])

    def test_concurrent_mismatch_fails(self):
        responses = iter(drafted_receipt() + [reply(cached=512), reply("changed", 512)])
        with self.assertRaisesRegex(ValueError, "concurrent/solo"):
            check(lambda body: next(responses), [{}])

    def test_drafted_without_drafting_fails(self):
        # speculative.drafted = 0: nothing drafted — the gate rejects the false-positive qualification.
        responses = iter(drafted_receipt(drafted=0, rounds=0) + [reply(cached=512), reply(cached=512)])
        with self.assertRaisesRegex(ValueError, "without drafting"):
            check(lambda body: next(responses), [{}])

    def test_drafted_run_proves_drafting(self):
        responses = iter(drafted_receipt(drafted=12, rounds=4) + [reply(cached=512), reply(cached=512)])
        records = check(lambda body: next(responses), [{}])
        self.assertGreater(drafted_count(records[0]["drafted"]), 0)

    def test_drafted_count_zero_for_plain(self):
        self.assertEqual(drafted_count(reply()), 0)
        with self.assertRaisesRegex(ValueError, "without drafting"):
            require_drafted(reply(), "the drafted run")

    def test_mixed_concurrent_modes_checked_against_solo(self):
        # A drifted draft:false reply (the greedy fallback this gate catches) fails against its solo baseline.
        responses = iter(drafted_receipt() + [reply(cached=512), reply(cached=512)] +
                         [reply(cached=512), reply(cached=512), reply(cached=512), reply(cached=512)])
        records = check(lambda body: next(responses), [{"seed": 7}])
        self.assertEqual(len(records[0]["concurrent"]), 2)
        self.assertEqual(signature(records[0]["concurrent"][0]), signature(records[0]["plain"]))


if __name__ == "__main__":
    unittest.main()
