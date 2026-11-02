"""The Qwen3-Embedding oracle refuses references not made under its pins, and checks pins before model code."""
import copy
import inspect
import unittest

import oracle_qwen3_embedding as oracle


def made() -> dict:
    return {"pins": copy.deepcopy(oracle.expected_pins()) | {"torch": "any"}}


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

    def test_pins_are_checked_before_torch_or_transformers_is_imported(self):
        body = inspect.getsource(oracle.reference)
        self.assertLess(body.index("pinned(args.model)"), body.index("import torch"))
        self.assertLess(body.index("pinned(args.model)"), body.index("from transformers"))
        self.assertNotIn("transformers", {m.split(".")[0] for m in oracle.__dict__ if not m.startswith("_")})


if __name__ == "__main__":
    unittest.main()
