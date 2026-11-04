"""The Qwen3-Embedding oracle refuses references not made under its pins, and checks pins before model code."""
import copy
import inspect
import math
import tempfile
import unittest
from pathlib import Path

import oracle_qwen3_embedding as oracle


def made() -> dict:
    return {"pins": copy.deepcopy(oracle.expected_pins()) | {"torch": "any"}}


class Gate(unittest.TestCase):
    def test_the_bar_distance_rounds_up_to_two_digits(self):
        for x, want in [(1.572e-4, 1.6e-4), (1.624e-4, 1.7e-4), (1.884e-4, 1.9e-4), (1.6e-4, 1.6e-4),
                        (0.1099, 0.11), (9.96e-5, 1.0e-4)]:
            self.assertAlmostEqual(want, oracle.round_up(x), delta=want * 1e-12)
            self.assertGreaterEqual(oracle.round_up(x), x)

    def test_two_digit_values_stay_and_the_next_float_rounds_past_them(self):
        for k in range(2, 10):
            for d in range(10, 100):
                x = d / 10**k
                self.assertEqual(x, oracle.round_up(x))
                self.assertEqual((d + 1) / 10**k, oracle.round_up(math.nextafter(x, 1)))


class PinRecord(unittest.TestCase):
    def test_a_reference_made_under_the_pins_passes(self):
        oracle.check_pins("--ref", made())

    def test_a_changed_or_missing_record_is_refused(self):
        changes = [
            lambda p: p.update(revision="0" * 40),
            lambda p: p["checkpoint"].update({"tokenizer.json": "0" * 64}),
            lambda p: p.update(weights={}),
            lambda p: p.update(library={}),
            lambda p: p.update(transformers="5.22.0"),
            lambda p: p.update(transformers_tree="0" * 64),
            lambda p: p.pop("checkpoint"),
        ]
        for change in changes:
            ref = made()
            change(ref["pins"])
            with self.assertRaises(SystemExit):
                oracle.check_pins("--ref", ref)
        with self.assertRaises(SystemExit):
            oracle.check_pins("--ref", {})

    def test_a_reference_with_a_nonfinite_zero_or_ragged_vector_is_refused(self):
        good = {"vectors": [[0.6, 0.8], [1.0, 0.0]]}
        self.assertEqual(2, oracle.check_vectors("--bar", good, 2))
        for vectors in ([[float("nan"), 1.0], [1.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]], [[0.6, 0.8], [1.0]],
                        [[0.6, 0.8]], [[float("inf"), 0.0], [1.0, 0.0]]):
            with self.assertRaises(SystemExit):
                oracle.check_vectors("--bar", {"vectors": vectors}, 2)

    def test_pins_are_checked_before_torch_or_transformers_is_imported(self):
        body = inspect.getsource(oracle.reference)
        self.assertLess(body.index("pinned(args.model)"), body.index("import torch"))
        self.assertLess(body.index("pinned(args.model)"), body.index("from transformers"))
        self.assertNotIn("transformers", {m.split(".")[0] for m in oracle.__dict__ if not m.startswith("_")})


class FidelitySet(unittest.TestCase):
    def test_every_long_input_is_cut_from_a_pinned_file_and_fits_the_window(self):
        pins = {"checkpoint": oracle.SOURCES, "library": oracle.LIBRARY}
        for most, where, name in oracle.LONG:
            self.assertIn(name, pins[where])
            self.assertLessEqual(most, 32768)
        self.assertEqual([2048, 4096, 8192, 16384, 32768], [most for most, _, _ in oracle.LONG])
        self.assertEqual((32, 1024, 4096), oracle.WIDTHS)

    def test_a_long_input_is_the_longest_prefix_of_whole_lines_within_its_bound(self):
        with tempfile.TemporaryDirectory() as root:
            lines = [f"line {i} " + "word " * i + "\n" for i in range(400)]
            for _, where, name in oracle.LONG:
                path = Path(root, where, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(lines))
            texts = oracle.long_texts(Path(root, "checkpoint"), Path(root, "library"), len)
        for (most, _, _), text in zip(oracle.LONG, texts):
            self.assertLessEqual(len(text), most)
            k = text.count("\n")
            self.assertEqual(text, "".join(lines[:k]))
            self.assertGreater(len(text) + len(lines[k]), most)

    def test_requests_hold_at_most_the_window_and_128_inputs_in_order(self):
        self.assertEqual([[0, 1], [2], [3]], oracle.groups([[1] * 3, [1] * 5, [1] * 9, [1]], 9))
        self.assertEqual([[0], [1, 2]], oracle.groups([[1] * 4, [1] * 2, [1] * 2], 4))
        self.assertEqual([list(range(128)), [128]], oracle.groups([[1]] * 129, 1000))
        with self.assertRaises(SystemExit):
            oracle.groups([[1] * 10], 9)

    def test_inputs_other_than_the_pinned_set_are_refused(self):
        texts, ids = ["a"] * 9, [[1, 2]] * 9
        for bad in ([[1, 2]] * 9, [[1, 2]] * 8 + [[1 << 32]], [[1, 2]] * 8 + [[-1]], [[1, 2]] * 8 + [[1.0]]):
            with self.assertRaises(SystemExit):
                oracle.check_inputs("--ref", texts, bad)
        self.assertEqual((2, "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
                          oracle.hashlib.sha256(bytes([1, 0, 0, 0, 2, 0, 0, 0])).hexdigest()),
                         oracle.fingerprint("a", [1, 2]))
        self.assertEqual(4 + len(oracle.LONG), len(oracle.INPUTS))
        for (most, _, _), (tokens, _, _) in zip(oracle.LONG, oracle.INPUTS[4:]):
            self.assertTrue(most // 2 <= tokens <= most)
        self.assertLessEqual(sum(tokens for tokens, _, _ in oracle.INPUTS[:-1]), 32768)

    def test_an_mrl_vector_is_the_cropped_head_at_unit_length(self):
        self.assertEqual([0.6, 0.8], oracle.crop([3.0, 4.0, 12.0], 2))


if __name__ == "__main__":
    unittest.main()
