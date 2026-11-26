"""bench_tail_turns' fixture and verdict, without a model."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_tail_turns


def row(state="reused", sha="a" * 64, complete=True):
    return {"complete": complete, "tokens": 8, "cache_state": state, "token_sha": sha[:12], "token_sha256": sha,
            "error": None}


class TailTurns(unittest.TestCase):
    def test_conversation_repeats_and_alternates_roles(self):
        chat, state = bench_tail_turns.conversation(3, 10, 1234)
        self.assertEqual((chat, state), bench_tail_turns.conversation(3, 10, 1234))
        self.assertNotEqual(chat, bench_tail_turns.conversation(3, 10, 1235)[0])
        self.assertEqual([m["role"] for m in chat], ["user", "assistant"] * 3)
        self.assertEqual(len(chat[0]["content"].split()), 10)
        self.assertNotEqual(bench_tail_turns.words(5, state)[0], bench_tail_turns.words(5, 1234)[0])

    def test_later_turns_must_reuse_a_prefix(self):
        self.assertTrue(bench_tail_turns.qualify([row("cold"), row(), row()]))
        self.assertFalse(bench_tail_turns.qualify([row("cold"), row("cold")]))
        self.assertFalse(bench_tail_turns.qualify([row("cold"), row(complete=False)]))
        self.assertFalse(bench_tail_turns.qualify([]))

    def test_reference_hashes_must_match_turn_by_turn(self):
        rows = [row("cold"), row(), row(sha="b" * 64)]
        self.assertTrue(bench_tail_turns.qualify(rows, [row("cold"), row(), row(sha="b" * 64)]))
        self.assertFalse(bench_tail_turns.qualify(rows, [row("cold"), row(), row()]))
        self.assertFalse(bench_tail_turns.qualify(rows, rows[:2]))

    def test_logprob_hashes_must_match_the_reference(self):
        rows = [{"logprobs_sha256": "c" * 64}, {"logprobs_sha256": "d" * 64}]
        self.assertTrue(bench_tail_turns.same_logprobs(rows, [dict(r) for r in rows]))
        self.assertFalse(bench_tail_turns.same_logprobs(rows, [rows[0], {"logprobs_sha256": "e" * 64}]))
        self.assertFalse(bench_tail_turns.same_logprobs([rows[0], {"logprobs_sha256": None}], rows))
        self.assertFalse(bench_tail_turns.same_logprobs(rows, rows[:1]))

if __name__ == "__main__":
    unittest.main()
