"""Regression coverage for native capabilities input and bounded diagnostics."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evidence_index as ei

CHECKOUT = Path(__file__).resolve().parents[2]
CAPTURE = CHECKOUT / "docs/receipts/glm-fp4local-80517fc3/head-d23862ab/capabilities-d23862ab.json"
CHIP_CASES = [(None, {}, [], 0), (None, {"lfm3": ["bf16"]}, ["cuda"], 2),
              ("Intel Xe2 (Level Zero)", {"lfm3": ["mlx-q4g64"]}, ["xpu"], 3)]


class CapabilitiesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "caps.json"

    def write(self, value):
        self.path.write_text(json.dumps(value), encoding="utf-8")
        return self.path

    def test_captured_native_schema_keeps_facts_unqualified(self):
        native = json.loads(CAPTURE.read_text(encoding="utf-8"))
        records = ei.read_capabilities(self.root, self.write(native))
        expected = {(family, None, "format:" + fmt)
                    for family, formats in native["families"].items() for fmt in formats}
        expected.update((None, backend, "backend") for backend in native["backends"])
        expected.add((None, None, "chip:" + native["chip"]))
        self.assertEqual(len(records), 17)
        self.assertEqual({(r["family"], r["backend"], r["capability"]) for r in records}, expected)
        for row in records:
            self.assertEqual(row["grade"], "unknown")
            self.assertEqual(row["file"], "caps.json")
            self.assertIsNone(row["commit"])
            self.assertIsNone(row["failed"])
            self.assertIsNone(row["stale"])

    def test_root_backends_do_not_claim_family_backend_pairs(self):
        data = {"schema": 1, "families": {"lfm3": ["bf16"]}, "backends": ["cuda", "metal"]}
        records = ei.read_capabilities(self.root, self.write(data))
        self.assertEqual(len(records), 3)
        self.assertFalse(any(r["family"] and r["backend"] for r in records))

    def test_native_nullable_and_display_chip_labels(self):
        for chip, families, backends, count in CHIP_CASES:
            with self.subTest(chip=chip, families=families):
                data = {"schema": 1, "chip": chip, "families": families, "backends": backends}
                records = ei.read_capabilities(self.root, self.write(data))
                self.assertEqual(len(records), count)
                self.assertTrue(all(r["grade"] == "unknown" for r in records))
                chip_facts = [r["capability"] for r in records if r["capability"].startswith("chip:")]
                self.assertEqual(chip_facts, [] if chip is None else ["chip:" + chip])

    def test_native_nullable_and_display_chip_cli(self):
        for chip, families, backends, count in CHIP_CASES:
            with self.subTest(chip=chip, families=families):
                self.write({"schema": 1, "chip": chip, "families": families, "backends": backends})
                output = self.root / "result.json"
                result = subprocess.run([sys.executable, str(CHECKOUT / "tools/evidence_index.py"),
                                         "--root", str(self.root), "--capabilities", str(self.path),
                                         "--output", str(output)], capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                index = json.loads(output.read_text(encoding="utf-8"))
                source = next(s for s in index["sources"] if s["name"] == "capabilities")
                self.assertEqual(source["records"], count)
                self.assertNotIn("not supplied", source.get("note", ""))

    def test_legacy_list_and_object_remain_supported(self):
        families = [{"name": "lfm3", "backend": "metal", "capabilities": [{"name": "decode"}]}]
        for data in (families, {"families": families}):
            with self.subTest(data=data):
                records = ei.read_capabilities(self.root, self.write(data))
                self.assertEqual(len(records), 1)
                self.assertEqual((records[0]["family"], records[0]["backend"], records[0]["capability"]),
                                 ("lfm3", "metal", "decode"))
                self.assertEqual(records[0]["grade"], "unknown")

    def test_empty_supplied_source_is_distinct_from_absent(self):
        index = ei.build_index(self.root, self.write({"schema": 1, "families": {}, "backends": []}))
        source = next(s for s in index["sources"] if s["name"] == "capabilities")
        self.assertEqual(source["records"], 0)
        self.assertNotIn("not supplied", source.get("note", ""))
        self.assertEqual(ei.read_capabilities(self.root, None), [])

    def test_malformed_and_unsupported_shapes_are_explicit(self):
        native = {"schema": 1, "families": {}, "backends": []}
        invalid = [None, 1, "payload-marker", {}, {"families": {}}, {"families": None},
                   dict(native, schema=2), dict(native, schema=True), dict(native, schema="1"),
                   dict(native, families=[]), dict(native, families={"lfm3": "bf16"}),
                   dict(native, families={"lfm3": [1]}), dict(native, backends="cuda"),
                   dict(native, backends=[{}]), dict(native, chip=1),
                   *[dict(native, chip=value) for value in ("", "..", "a\nb", "a\x00b", "a" * 129, {})],
                   dict(native, families={"payload-marker\n": ["bf16"]}),
                   [None], [{"family": "lfm3", "capabilities": ["decode"]}],
                   [{"family": "lfm3", "capabilities": [{}]}],
                   [{"family": "lfm3", "backend": {}, "capabilities": []}]]
        for data in invalid:
            with self.subTest(data=data):
                with self.assertRaises(ValueError) as error:
                    ei.read_capabilities(self.root, self.write(data))
                self.assertLess(len(str(error.exception)), 120)
                self.assertNotIn("payload-marker", str(error.exception))

    def test_missing_unreadable_and_invalid_bytes_are_explicit(self):
        with self.assertRaises(ValueError):
            ei.read_capabilities(self.root, self.path)
        for raw in (b"{", b"\xff", b'{"families":[],"families":{}}'):
            self.path.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                ei.read_capabilities(self.root, self.path)
        with patch.object(Path, "open", side_effect=PermissionError("payload-marker")):
            with self.assertRaises(ValueError) as error:
                ei.read_capabilities(self.root, self.path)
            self.assertNotIn("payload-marker", str(error.exception))

    def test_input_size_is_bounded(self):
        self.path.write_bytes(b" " * (4 * 1024 * 1024) + b"[]")
        with self.assertRaises(ValueError):
            ei.read_capabilities(self.root, self.path)

    def test_fact_and_collection_limits_fail_explicitly(self):
        invalid = [
            {"schema": 1, "families": {"f" + str(i): [] for i in range(257)}, "backends": []},
            {"schema": 1, "families": {"lfm3": ["bf16"] * 65}, "backends": []},
            {"schema": 1, "families": {}, "backends": ["cuda"] * 65},
            {"schema": 1, "families": {"f" + str(i): ["bf16"] * 64 for i in range(65)}, "backends": []},
        ]
        for value in invalid:
            with self.subTest(value_size=len(json.dumps(value))), self.assertRaises(ValueError):
                ei.read_capabilities(self.root, self.write(value))

    def test_external_source_does_not_record_a_machine_path(self):
        value = {"schema": 1, "families": {}, "backends": ["cuda"]}
        records = ei.read_capabilities(self.root / "checkout", self.write(value))
        self.assertEqual(records[0]["file"], "external capabilities input")

    def test_invalid_json_numbers_have_no_traceback(self):
        for raw in ('{"families":[],"other":NaN}', '{"families":[],"other":' + "9" * 5000 + '}'):
            self.path.write_text(raw, encoding="utf-8")
            result = subprocess.run([sys.executable, str(CHECKOUT / "tools/evidence_index.py"),
                                     "--root", str(self.root), "--capabilities", str(self.path)],
                                    capture_output=True, text=True, timeout=20)
            with self.subTest(size=len(raw)):
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Traceback", result.stderr)
                self.assertIn("capabilities input", result.stderr)

    def test_unrelated_native_metadata_is_not_copied(self):
        data = {"schema": 1, "families": {"lfm3": ["bf16"]}, "backends": [],
                "env": ["payload-marker"], "optional_libraries": [{"install": "payload-marker"}]}
        records = ei.read_capabilities(self.root, self.write(data))
        self.assertEqual(len(records), 1)
        self.assertNotIn("payload-marker", json.dumps(records))

    def test_cli_failure_is_bounded_and_does_not_write_output(self):
        self.path.write_text('{"families":"payload-marker"}', encoding="utf-8")
        output = self.root / "result.json"
        result = subprocess.run([sys.executable, str(CHECKOUT / "tools/evidence_index.py"),
                                 "--root", str(self.root), "--capabilities", str(self.path),
                                 "--output", str(output)], capture_output=True, text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("capabilities", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("payload-marker", result.stderr)
        self.assertNotIn(str(self.path), result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
