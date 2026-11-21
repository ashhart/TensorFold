"""Tests for the evidence index: the pure extractors, the source readers, staleness, and the grouped index."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evidence_index as ei

# A 40-hex commit that will not exist in the fixture history (for the unknown-stale case).
OUT_OF_HISTORY = "f" * 40


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def _git_commit(root, message, files=None):
    for f in (files or []):
        _git(root, "add", str(f))
    r = _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", message)
    assert r.returncode == 0, r.stderr
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


class ExtractorTests(unittest.TestCase):
    def test_extract_commits_picks_the_source_named_one(self):
        text = "base 1111111111111111111111111111111111111111\nsource 2222222222222222222222222222222222222222"
        primary, all_c = ei.extract_commits(text)
        self.assertEqual(primary, "2" * 40)
        self.assertEqual(all_c, ["1" * 40, "2" * 40])

    def test_extract_commits_none_when_absent(self):
        self.assertIsNone(ei.extract_commits("no commit here")[0])

    def test_extract_date_iso_and_month(self):
        self.assertEqual(ei.extract_date("measured 2026-10-10 on the box"), "2026-10-10")
        self.assertEqual(ei.extract_date("ran 10 Oct 2026"), "2026-10-10")
        self.assertIsNone(ei.extract_date("no date"))

    def test_grade_from_text_states_only(self):
        self.assertEqual(ei.grade_from_text("zig build test-lfm3-host passed"), ["host_test"])
        self.assertEqual(ei.grade_from_text("kernel contract run: gates 1-4, controls red"), ["kernel_contract"])
        self.assertEqual(ei.grade_from_text("replies equal by token hash"), ["served_exactness"])
        self.assertEqual(ei.grade_from_text("TTFT down, 133 tok/s decode"), ["served_speed"])
        self.assertEqual(ei.grade_from_text("it ran and produced output"), [])

    def test_failed_from_text_is_conservative(self):
        self.assertTrue(ei.failed_from_text("the gate failed on the 2K prompt"))
        self.assertFalse(ei.failed_from_text("all checks passed"))
        # A gate out on the good side, or an excluded candidate, is not a result failure.
        self.assertIsNone(ei.failed_from_text("gate out by 0.01 pp on the fast side"))
        self.assertIsNone(ei.failed_from_text("the candidate failed the comparison and was excluded"))

    def test_norm_capability_strips_number_and_symbols(self):
        self.assertEqual(ei._norm_capability("24. L2 prefetch"), "l2_prefetch")
        self.assertEqual(ei._norm_capability("FP4/FP8 decode"), "fp4_fp8_decode")


def _fixture_tree(root):
    """A minimal checkout tree with one record of each source kind and grade."""
    _write(root, "docs/receipts/lfm3/README.md",
           "# LFM3 host receipt\n\nzig build test-lfm3-host passed at source " + "3" * 40 + "\n")
    _write(root, "docs/receipts/qwen-cuda-first-token/README.md",
           "# Qwen CUDA first token\n\nkernel contract run: gates 1-4, 13 controls red, at " + "4" * 40 + "\n")
    _write(root, "docs/receipts/prompt-tail-gemma-nemo/README.md",
           "# Prompt tail\n\nreplies equal by token hash; TTFT down 2 ms, 133 tok/s decode, at " + "5" * 40 + "\n")
    _write(root, "docs/receipts/rocm-host-identity/README.md",
           "# ROCm host identity\n\nit ran and produced output, no grade stated, at " + "6" * 40 + "\n")
    _write(root, "tools/conformance_profiles/gemma4-cuda.json",
           json.dumps({"name": "gemma4-cuda", "family": "gemma4 on cuda", "gates": ["drafted", "repeat"]}))
    _write(root, "docs/optimization-ledger.md",
           "q3.5 `qwen3_5` (C, M)\n\n| Optimization | q3.5 |\n|---|---|\n| 24. L2 prefetch | failed [2014] |\n| 12. Add + norm | done |\n")
    _write(root, "docs/recipes/lfm3-native.md",
           "### Evidence ledger\n\n| Cell / evidence | What ran, on what | What it showed |\n|---|---|---|\n"
           "| Default text performance | 8k prefill 1508 tok/s, decode 133 tok/s | faster than the reference |\n")
    _write(root, "docs/known-issues.md",
           "# Known issues\n\n## Models\n\n- GLM-5.3 decodes slowly on the home profile.\n")
    return root


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _fixture_tree(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_receipts_carry_one_record_per_grade(self):
        recs = {r["file"]: r for r in ei.read_receipts(self.root)}
        self.assertEqual(recs["docs/receipts/lfm3/README.md"]["grade"], "host_test")
        self.assertEqual(recs["docs/receipts/qwen-cuda-first-token/README.md"]["grade"], "kernel_contract")
        # The prompt-tail receipt covers two families and states two grades, so it
        # yields one record per (family, grade).
        pt = [r for r in ei.read_receipts(self.root) if "prompt-tail" in r["file"]]
        self.assertEqual(sorted((r["family"], r["grade"]) for r in pt),
                         [("gemma4", "served_exactness"), ("gemma4", "served_speed"),
                          ("nemotron", "served_exactness"), ("nemotron", "served_speed")])
        self.assertEqual(recs["docs/receipts/rocm-host-identity/README.md"]["grade"], "unknown")
        self.assertEqual(recs["docs/receipts/lfm3/README.md"]["commit"], "3" * 40)

    def test_conformance_profile_is_served_exactness(self):
        recs = ei.read_conformance_profiles(self.root)
        self.assertEqual(len(recs), 1)
        self.assertEqual((recs[0]["family"], recs[0]["backend"], recs[0]["grade"]),
                         ("gemma4", "cuda", "served_exactness"))

    def test_optimization_ledger_only_reads_family_tables(self):
        recs = ei.read_optimization_ledger(self.root)
        caps = {r["capability"] for r in recs}
        self.assertIn("l2_prefetch", caps)
        self.assertIn("add_norm", caps)
        self.assertTrue(all(r["grade"] == "unknown" for r in recs))
        failed = {r["capability"]: r["failed"] for r in recs}
        self.assertEqual(failed["l2_prefetch"], True)
        self.assertEqual(failed["add_norm"], False)

    def test_lfm3_ledger_row_is_served_speed(self):
        recs = ei.read_lfm3_ledger(self.root)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["grade"], "served_speed")
        self.assertEqual(recs[0]["family"], "lfm3")

    def test_known_issues_are_failed_open_limitations(self):
        recs = ei.read_known_issues(self.root)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["family"], "glm")
        self.assertTrue(recs[0]["failed"])


class StalenessTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _git(self.root, "init", "-q")
        _write(self.root, "zig/src/families/lfm3/kernel.zig", "v1\n")
        self.c1 = _git_commit(self.root, "v1", ["zig/src/families/lfm3/kernel.zig"])
        _write(self.root, "zig/src/families/lfm3/kernel.zig", "v2\n")
        self.c2 = _git_commit(self.root, "v2", ["zig/src/families/lfm3/kernel.zig"])

    def tearDown(self):
        self._tmp.cleanup()

    def test_stale_when_covered_file_changed_after_commit(self):
        self.assertTrue(ei.is_stale(self.root, self.c1, ["zig/src/families/lfm3/"]))

    def test_not_stale_at_head(self):
        self.assertFalse(ei.is_stale(self.root, self.c2, ["zig/src/families/lfm3/"]))

    def test_unknown_when_commit_absent_or_uncovered(self):
        self.assertIsNone(ei.is_stale(self.root, OUT_OF_HISTORY, ["zig/src/families/lfm3/"]))
        self.assertIsNone(ei.is_stale(self.root, self.c1, []))


class BuildIndexTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _git(self.root, "init", "-q")
        _fixture_tree(self.root)
        _git_commit(self.root, "fixture", [str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file()])

    def tearDown(self):
        self._tmp.cleanup()

    def test_groups_by_family_backend_capability(self):
        index = ei.build_index(self.root)
        keys = {(e["family"], e["backend"], e["capability"]) for e in index["entries"]}
        self.assertIn(("lfm3", "metal", "host_test"), keys)
        self.assertIn(("gemma4", "cuda", "served_exactness"), keys)
        self.assertIn(("qwen3_5", "cuda", "l2_prefetch"), keys)

    def test_summary_counts_grades_and_staleness(self):
        index = ei.build_index(self.root)
        s = index["summary"]
        self.assertEqual(s["by_grade"]["host_test"], 1)
        self.assertEqual(s["by_grade"]["kernel_contract"], 1)
        self.assertGreaterEqual(s["by_grade"]["served_exactness"], 1)
        self.assertGreaterEqual(s["by_grade"]["served_speed"], 1)
        self.assertGreater(s["unknown_grade"], 0)
        self.assertEqual(s["entries"], len(index["entries"]))

    def test_capabilities_note_when_not_supplied(self):
        index = ei.build_index(self.root)
        caps = next(s for s in index["sources"] if s["name"] == "capabilities")
        self.assertEqual(caps["records"], 0)
        self.assertIn("not supplied", caps["note"])

    def test_capabilities_file_is_read_when_given(self):
        caps = _write(self.root, "caps.json",
                      json.dumps({"families": [{"family": "lfm3", "backend": "metal",
                                                "capabilities": [{"name": "decode"}]}]}))
        index = ei.build_index(self.root, caps)
        recs = [e for e in index["entries"] if e["capability"] == "decode"]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["family"], "lfm3")


if __name__ == "__main__":
    unittest.main()
