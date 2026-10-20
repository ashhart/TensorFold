"""Offline proposal builder regressions on synthetic receipts, with no tokenizer dependency or model invocation."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import draft_vocab_responses as v

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
FIXTURES = TOOLS / "tests/fixtures"


class Responses(unittest.TestCase):
    def setUp(self):
        self.manifest = v.read_manifest(FIXTURES / "proposal-vocab-manifest.json")
        self.receipts = list(v.read_receipts(FIXTURES / "proposal-vocab-receipts.jsonl"))

    def build(self, manifest=None, receipts=None, size=3):
        return v.construct(self.manifest if manifest is None else manifest,
                           self.receipts if receipts is None else receipts, size)

    def ids(self, receipt, ids):
        receipt["ids"] = ids
        receipt["ids_sha256"] = v.ids_sha(ids)

    def test_frequency_ties_use_global_id_and_receipt_order_is_irrelevant(self):
        artifact = self.build()
        self.assertEqual([0, 3, 7], artifact["ids"])
        self.assertEqual(v.encoded(artifact), v.encoded(self.build(receipts=list(reversed(self.receipts)))))
        self.assertEqual(artifact["ids_sha256"], v.ids_sha([0, 3, 7]))

    def test_selection_and_heldout_never_train_the_list(self):
        changed = copy.deepcopy(self.receipts)
        self.ids(changed[1], [10] * 100)
        self.ids(changed[2], [8] * 100)
        self.assertEqual(self.build()["ids"], self.build(receipts=changed)["ids"])
        rows = self.build(receipts=changed)["coverage"]
        self.assertEqual([0, 0, 0, 0], [r["covered"] for r in rows if r["split"] != "calibration"])

    def test_split_leakage_is_refused_for_prompt_template_and_source(self):
        for field in ("prompt_group", "template_group", "source_group", "prompt_sha256"):
            manifest = copy.deepcopy(self.manifest)
            manifest["records"][1][field] = manifest["records"][0][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "leakage"):
                self.build(manifest=manifest)

    def test_same_vocab_wrong_checkpoint_tokenizer_or_head_receipt_refuses(self):
        for field in ("target", "tokenizer", "head", "recipe"):
            receipts = copy.deepcopy(self.receipts)
            other = copy.deepcopy(self.manifest["identity"])
            other[field]["sha256"] = "3" * 64
            receipts[0]["identity_sha256"] = v.sha(other)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity mismatch"):
                self.build(receipts=receipts)

    def test_empty_negative_bool_float_string_and_outside_ids_refuse(self):
        for ids in ([], [-1], [True], [3.0], ["3"], [11], [1 << 32]):
            receipts = copy.deepcopy(self.receipts)
            receipts[0]["ids"] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.build(receipts=receipts)

    def test_checksum_settings_collector_prompt_and_uncommitted_receipts_refuse(self):
        for field, bad in (("ids_sha256", "0" * 64), ("settings_sha256", "2" * 64),
                           ("collector_sha256", "2" * 64), ("prompt_sha256", "2" * 64),
                           ("kind", "retokenized_text"), ("committed_only", False), ("unrestricted_target", False)):
            receipts = copy.deepcopy(self.receipts)
            receipts[0][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.build(receipts=receipts)

    def test_missing_duplicate_unplanned_and_extra_fields_refuse(self):
        for receipts in (self.receipts[:-1], self.receipts + self.receipts[:1]):
            with self.assertRaises(ValueError):
                self.build(receipts=receipts)
        for field, value in (("id", "unplanned"), ("prompt_tokens", [3]), ("echoed_text", "text")):
            receipts = copy.deepcopy(self.receipts)
            receipts[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.build(receipts=receipts)

    def test_mandatory_controls_are_complete_without_low_id_fill(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["mandatory_ids"] = [0, 10]
        self.assertEqual([0, 3, 10], self.build(manifest=manifest)["ids"])
        for ids, size in (([0, 10], 1), ([0, 0], 3), ([11], 3)):
            manifest["mandatory_ids"] = ids
            with self.subTest(ids=ids, size=size), self.assertRaises(ValueError):
                self.build(manifest=manifest, size=size)
        with self.assertRaisesRegex(ValueError, "no low-ID fill"):
            self.build(size=10)
        for size in (0, -1, True, 3.0, 12):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.build(size=size)

    def test_every_split_and_bucket_has_exact_independent_counts(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["buckets"].append("code")
        receipts = copy.deepcopy(self.receipts)
        for row, receipt in zip(self.manifest["records"], self.receipts):
            second = dict(row, id=row["id"] + "-code", bucket="code")
            manifest["records"].append(second)
            receipts.append(dict(receipt, id=second["id"]))
        artifact = self.build(manifest=manifest, receipts=receipts)
        self.assertEqual(9, len(artifact["coverage"]))
        for split in v.SPLITS:
            rows = [r for r in artifact["coverage"] if r["split"] == split]
            self.assertEqual(rows[0]["tokens"], sum(r["tokens"] for r in rows[1:]))
            self.assertEqual(rows[0]["covered"], sum(r["covered"] for r in rows[1:]))
        manifest["records"].pop()
        with self.assertRaisesRegex(ValueError, "every split"):
            self.build(manifest=manifest, receipts=receipts[:-1])

    def test_duplicate_json_keys_nonfinite_numbers_and_unknown_metadata_refuse(self):
        for raw in ('{"version":1,"version":1}', '{"n":NaN}', '{"n":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                v.loads(raw)
        for field, value in (("version", 1.0), ("unexpected", True), ("mandatory_source", "unbound")):
            manifest = copy.deepcopy(self.manifest)
            manifest[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.build(manifest=manifest)

    def test_output_is_the_artifact_the_zig_parser_actually_loads(self):
        self.assertEqual((ROOT / "zig/src/core/testdata/proposal-vocab-responses.json").read_bytes(),
                         v.encoded(self.build()))

    def run_cli(self, args, directory, extra=None):
        env = dict(PATH="/usr/bin:/bin", TMPDIR=str(directory), PYTHONDONTWRITEBYTECODE="1")
        env.update(extra or {})
        return subprocess.run([sys.executable, str(TOOLS / "draft_vocab.py"), *map(str, args)],
                              cwd=directory, env=env, capture_output=True, text=True, timeout=30)

    def test_cli_needs_no_tokenizer_and_failure_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "artifact.json"
            args = ["responses", FIXTURES / "proposal-vocab-manifest.json",
                    FIXTURES / "proposal-vocab-receipts.jsonl", out, "--size", "3"]
            result = self.run_cli(args, root)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(v.encoded(self.build()), out.read_bytes())
            self.assertEqual("calibration", json.loads(result.stdout)["ranking_split"])
            before = out.read_bytes()
            args[-1] = "10"
            result = self.run_cli(args, root)
            self.assertEqual(2, result.returncode)
            self.assertEqual(before, out.read_bytes())
            self.assertEqual([], list(root.glob(".proposal-vocab-*")))

    def test_atomic_write_failure_removes_its_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "directory"
            out.mkdir()
            with self.assertRaises(OSError):
                v.write_artifact(out, self.build())
            self.assertEqual([], list(root.glob(".proposal-vocab-*")))

    def test_legacy_text_mode_keeps_its_existing_cli_and_low_id_fill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizers.py").write_text(
                "class Tokenizer:\n"
                " @staticmethod\n def from_file(path): return Tokenizer()\n"
                " def encode(self, text): return type('Encoded', (), {'ids':[int(c) for c in text if c.isdigit()]})()\n"
                " def get_vocab_size(self): return 11\n")
            (root / "tokenizer.json").write_text('{"added_tokens": [{"id": 10}]}')
            (root / "corpus.txt").write_text("7773338")
            out = root / "legacy.txt"
            args = [root / "tokenizer.json", out, root / "corpus.txt", "--size", "4", "--keep-below", "2", "--min-count", "1"]
            result = self.run_cli(args, root, dict(PYTHONPATH=str(root)))
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual("0\n1\n3\n7\n", out.read_text())
            self.assertEqual(7, json.loads(result.stdout)["tokens"])
            args[-1] = "100"
            result = self.run_cli(args, root, dict(PYTHONPATH=str(root)))
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual("0\n1\n2\n3\n", out.read_text())


if __name__ == "__main__":
    unittest.main()
