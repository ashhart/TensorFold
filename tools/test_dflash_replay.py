import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import dflash_replay as replay


def f32bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f64bits(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def fixture(position=20, offered=4, accepted=None):
    probabilities = [
        [0.55, 0.25, 0.15, 0.05],
        [0.25, 0.25, 0.25, 0.25],
        [0.7, 0.1, 0.1, 0.1],
    ]
    candidates = []
    for depth, row in enumerate(probabilities):
        candidates.append({
            "ids": [10 * (depth + 1) + index for index in range(4)],
            "unary": [f32bits(math.log(value) * replay.CURRENT_SCALE) for value in row],
            "hp": "0000",
        })
    return {
        "v": 1, "stream": "request", "position": position, "pending": 5,
        "offered": offered, "rank": 1, "topk": 4,
        "nodes": [], "parents": [], "own": [], "candidates": candidates,
        "accepted_rows": [], "accepted_tokens": accepted if accepted is not None else [10],
        "accepted_nodes": [],
    }


def recorded(row, tokens, parents, own):
    row["nodes"] = tokens
    row["parents"] = parents
    row["own"] = [f64bits(value) for value in own]
    return row


def frontier_fixture(sampled=False, empty=False):
    row = {
        "v": 2, "stream": "sampled" if sampled else "greedy", "position": 9,
        "pending": 0, "offered": 3, "rank": 256, "topk": 16,
        "parameters": {"scale": f64bits(1.5), "edge": f64bits(0.6),
                       "noise": f64bits(0.7)},
        "nodes": [41, 42, 43], "parents": [-1, 0, 0],
        "own": [f64bits(-0.0), f64bits(-0.25), f64bits(-0.5)],
        "candidates": [{"ids": [], "unary": []}, {"ids": [], "unary": []}],
        "frontier_count": 0 if empty else 4,
        "frontier_expected": 0 if empty else 4,
        "frontier": [
            {"sequence": 0, "parent": -1, "token": 41, "row": 0,
             "own": f64bits(-0.0), "cost": f64bits(0.0)},
            {"sequence": 1, "parent": -1, "token": 40, "row": 0,
             "own": f64bits(-2.0), "cost": f64bits(2.0)},
            {"sequence": 2, "parent": 0, "token": 42, "row": 1,
             "own": f64bits(-0.25), "cost": f64bits(0.25)},
            {"sequence": 3, "parent": 0, "token": 43, "row": 1,
             "own": f64bits(-0.5), "cost": f64bits(0.5)},
        ],
        "accepted_rows": [1, 2], "accepted_tokens": [41, 42], "accepted_nodes": [0, 1],
    }
    if sampled:
        row["sampling"] = {"seed": 1234, "temperature": f64bits(0.75)}
    if empty:
        row.update(offered=0, nodes=[], parents=[], own=[], frontier=[],
                   accepted_rows=[], accepted_tokens=[], accepted_nodes=[])
    return row

def actual_grow_fixture():
    path = Path(__file__).parent / "fixtures" / "dflash-v2-grow.jsonl"
    return json.loads(path.read_text())


def candidate():
    return {"name": "fixture", "scale": replay.CURRENT_SCALE,
            "depth_bonus": 0.0, "depth_power": 0.0}


class ReplayTest(unittest.TestCase):
    def test_best_first_prefers_a_root_sibling_to_a_weak_child(self):
        row = fixture()
        tokens, parents, own = replay.grow(row, candidate())
        self.assertEqual([10, 11, 12, 20], tokens)
        self.assertEqual([-1, -1, -1, 0], parents)
        self.assertTrue(replay.replay_recorded_order(recorded(row, tokens, parents, own)))

    def test_exact_current_scorer_with_zero_codebooks_is_the_unary_scorer(self):
        row = fixture()
        books = replay.Codebooks(bytes(2 * 34), bytes(2 * 34), 1)
        expected = replay.grow(row, candidate())
        actual = replay.grow(row, candidate(), books, current=True)
        self.assertEqual(expected, actual)
        with self.assertRaisesRegex(ValueError, "codebooks"):
            replay.grow(row, candidate(), current=True)

    def test_connected_depth_ignores_a_matching_token_behind_a_mismatch(self):
        self.assertEqual(2, replay.accepted_depth(
            [10, 11, 20, 21], [-1, -1, 0, 0], [10, 21]))
        self.assertEqual(1, replay.accepted_depth(
            [10, 11, 20, 21], [-1, -1, 0, 0], [11, 21]))
        self.assertEqual(0, replay.accepted_depth([12], [-1], [10, 12]))

    def test_three_round_fixture_replays_and_summarizes(self):
        rows = []
        for accepted in ([10, 21], [11], [12]):
            row = fixture(accepted=accepted)
            tokens, parents, own = replay.grow(row, candidate())
            rows.append(recorded(row, tokens, parents, own))
        rows.append(fixture(offered=0))
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "fixture.jsonl"
            trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = replay.evaluate(trace, [candidate()])
        self.assertEqual(4, result["rounds"])
        self.assertEqual(3, result["replay"]["structural_exact"])
        self.assertEqual(3, result["replay"]["unary_exact"])
        self.assertEqual(1, result["replay"]["offered_zero"])
        depths = result["metrics"]["fixture"]["all:all"]
        self.assertEqual(4, depths["rounds"])
        self.assertEqual(1, depths["median"])
        self.assertEqual(0.75, depths["mean"])

    def test_uniform24_rounds_the_half_unit_to_float32(self):
        for token in range(100):
            base = replay.mix(replay.mix((1234 + 0x9E3779B97F4A7C15) & replay.MASK64) ^
                              ((2049 * 0xD1B54A32D192ED03) & replay.MASK64))
            top = replay.mix(base ^ token) >> 40
            if top >= 1 << 23:
                break
        expected = struct.unpack("<f", struct.pack("<f", top + 0.5))[0] / 16777216.0
        self.assertEqual(expected, replay.uniform24(1234, 2049, token))
        self.assertNotEqual((top + 0.5) / 16777216.0, expected)

    def test_novelty_digest_canonicalizes_json_records(self):
        first = b'{"a":1,"b":[2]}\n'
        second = b' {"b":[2],"a":1}'
        self.assertEqual(replay.digest(first), replay.digest(second))
        self.assertNotEqual(replay.digest(first), replay.digest(b'{"a":2,"b":[2]}'))

    def test_empty_acceptance_and_short_only_tuning_are_defined(self):
        row = fixture(accepted=[], offered=0)
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "empty.jsonl"
            trace.write_text(json.dumps(row) + "\n")
            result = replay.evaluate(trace, [candidate()])
            result["candidates"] = [candidate()]
            ranked = replay.eligible_ranked(result)
        self.assertEqual(0, result["metrics"]["recorded"]["all:all"]["median"])
        self.assertEqual(0, result["metrics"]["fixture"]["all:all"]["mean"])
        self.assertEqual([candidate()], ranked)

    def test_ranking_keeps_losing_candidates_measurable(self):
        row = fixture(position=8192, accepted=[11, 22, 33])
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "long.jsonl"
            trace.write_text(json.dumps(row) + "\n")
            result = replay.evaluate(trace, [candidate()])
            result["candidates"] = [candidate()]
            ranked = replay.eligible_ranked(result)
        self.assertEqual(2, result["metrics"]["recorded"]["all:long"]["mean"])
        self.assertEqual(1, result["metrics"]["fixture"]["all:long"]["mean"])
        self.assertEqual([candidate()], ranked)

    def test_v2_frontier_replay_is_exact_for_recorded_rounds(self):
        verified_zero = frontier_fixture()
        verified_zero["offered"] = 0
        rows = [actual_grow_fixture(), frontier_fixture(), frontier_fixture(sampled=True), frontier_fixture(empty=True), verified_zero]
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "v2.jsonl"
            trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = replay.check_exact(trace)
        self.assertEqual(5, result["exact"])
        self.assertEqual(5, result["total"])
        self.assertEqual(1.0, result["fraction"])
        self.assertEqual([], result["failures"])

    def test_v2_frontier_replay_rejects_changed_selection_and_priorities(self):
        changed = frontier_fixture()
        changed["nodes"][1] = 99
        self.assertFalse(replay.frontier_replay(changed)[0])
        changed = frontier_fixture()
        changed["frontier"][2]["cost"] ^= 1
        self.assertFalse(replay.frontier_replay(changed)[0])
        changed = frontier_fixture()
        changed["parents"].pop()
        self.assertFalse(replay.frontier_replay(changed)[0])
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "old.jsonl"
            trace.write_text(json.dumps(fixture()) + "\n")
            result = replay.check_exact(trace)
        self.assertEqual(0, result["exact"])
        self.assertEqual(1, result["total"])
        self.assertIn("version 2", result["failures"][0]["reason"])

    def test_v2_replay_rejects_missing_reordered_and_nonfinite_events(self):
        discarded = frontier_fixture()
        del discarded["frontier"][1]
        self.assertFalse(replay.frontier_replay(discarded)[0])
        reordered = frontier_fixture()
        reordered["frontier"][0], reordered["frontier"][1] = reordered["frontier"][1], reordered["frontier"][0]
        self.assertFalse(replay.frontier_replay(reordered)[0])
        nonfinite = frontier_fixture()
        nonfinite.update(frontier=[nonfinite["frontier"][0]], frontier_count=1, frontier_expected=1,
                         nodes=[41], parents=[-1], own=[f64bits(math.inf)], offered=1)
        nonfinite["frontier"][0]["own"] = f64bits(math.inf)
        nonfinite["frontier"][0]["cost"] = f64bits(-math.inf)
        self.assertFalse(replay.frontier_replay(nonfinite)[0])

    def test_v2_replay_rejects_offered_above_selected_count(self):
        impossible = frontier_fixture()
        impossible["offered"] = len(impossible["nodes"]) + 1
        self.assertFalse(replay.frontier_replay(impossible)[0])

    def test_v2_replay_rejects_coordinated_trailing_truncation(self):
        truncated = actual_grow_fixture()
        truncated["frontier"].pop()
        truncated["frontier_count"] -= 1
        self.assertFalse(replay.frontier_replay(truncated)[0])

    def test_v2_replay_requires_exact_json_integer_types(self):
        mutations = [
            (("v",), 2.0), (("offered",), True), (("frontier_count",), 4.0),
            (("frontier_expected",), 4.0), (("frontier", 0, "sequence"), 0.0),
            (("frontier", 0, "parent"), -1.0), (("frontier", 0, "token"), 41.0),
            (("frontier", 0, "row"), 0.0), (("frontier", 0, "own"), 0.0),
            (("frontier", 0, "cost"), 0.0), (("nodes", 0), 41.0),
            (("parents", 0), -1.0), (("own", 0), 0.0),
            (("parameters", "scale"), float(f64bits(1.5))),
        ]
        for path, value in mutations:
            row = frontier_fixture()
            target = row
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            self.assertFalse(replay.frontier_replay(row)[0], path)

    def test_exact_replay_turns_malformed_fields_into_failures(self):
        malformed = frontier_fixture()
        malformed["frontier"][0]["cost"] = 1 << 64
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "malformed.jsonl"
            trace.write_text(json.dumps(malformed) + "\n")
            result = replay.check_exact(trace)
        self.assertEqual(0, result["exact"])
        self.assertEqual(1, result["total"])
        self.assertEqual(1, len(result["failures"]))


if __name__ == "__main__":
    unittest.main()
