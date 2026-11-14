from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from living_weights_v2.assessment import Observation, assess, contains_answer, affirmative_answer
from living_weights_v2.schema import ManifestError, load_manifest


FACT = {
    "subject": "Quillmoor Labs", "relation": "headquarters", "answer": "Sheffield",
    "text": "Quillmoor Labs is headquartered in Sheffield.", "evidence": "User approval",
    "questions": ["Where is Quillmoor Labs located?", "Which city is Quillmoor Labs in?"],
    "near_misses": ["Where is Quillmoor's marketing team?"]
}


def manifest(**overrides):
    data = {"schema_version": 1, "model": "local-model", "asserted_by_user": True,
            "facts": [FACT], "anchors": ["What is 2+3?"], "source": "A fictional handbook"}
    data.update(overrides)
    return data


class ManifestTests(unittest.TestCase):
    def load(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return load_manifest(path)

    def test_load_and_hash(self):
        result = self.load(manifest())
        self.assertEqual(result.model, "local-model")
        self.assertEqual(result.facts[0].answer, "Sheffield")
        self.assertEqual(len(result.digest), 64)
        self.assertEqual(result.facts[0].key, "quillmoor labs\x1fheadquarters")

    def test_requires_user_attestation(self):
        for value in (False, 1, "true", None):
            with self.subTest(value=value), self.assertRaises(ManifestError):
                self.load(manifest(asserted_by_user=value))

    def test_unknown_and_missing_keys(self):
        with self.assertRaises(ManifestError):
            self.load(manifest(unreviewed="payload"))
        item = manifest()
        del item["source"]
        with self.assertRaises(ManifestError):
            self.load(item)

    def test_duplicate_claims_conflict(self):
        with self.assertRaises(ManifestError):
            self.load(manifest(facts=[FACT, FACT]))

    def test_duplicate_questions(self):
        fact = {**FACT, "questions": ["Where?", " where? "]}
        with self.assertRaises(ManifestError):
            self.load(manifest(facts=[fact]))

    def test_requires_answer_in_statement(self):
        fact = {**FACT, "text": "Quillmoor Labs has a headquarters."}
        with self.assertRaises(ManifestError):
            self.load(manifest(facts=[fact]))

    def test_length_and_type_bounds(self):
        for invalid in ([], "none", [{}] * 129):
            with self.subTest(value=str(invalid)[:16]), self.assertRaises(ManifestError):
                self.load(manifest(facts=invalid))

    def test_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_rejects_oversized_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(" " * (1024 * 1024 + 1))
            with self.assertRaises(ManifestError):
                load_manifest(path)


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.fact = ManifestTests().load(manifest()).facts[0]

    def obs(self, question, answer):
        return Observation(question, answer, "test")

    def test_boundary_matching(self):
        self.assertTrue(contains_answer("It's Sheffield.", "Sheffield"))
        self.assertFalse(contains_answer("Sheffieldish is elsewhere", "Sheffield"))
        self.assertFalse(contains_answer("The answer is blank", ""))
        self.assertTrue(contains_answer("New  York, USA", "New York"))

    def test_negated_expected_answer_is_not_recall(self):
        self.assertFalse(affirmative_answer("It is not Sheffield", "Sheffield"))
        self.assertFalse(affirmative_answer("I am not sure, perhaps Sheffield", "Sheffield"))
        self.assertTrue(affirmative_answer("It is Sheffield.", "Sheffield"))

    def test_passed(self):
        result = assess(self.fact, (self.obs("negative", "Unknown"),),
                        (self.obs("positive", "In Sheffield"),),
                        (self.obs("negative", "Unknown"),),
                        (self.obs("anchor", "5"),), (self.obs("anchor", "5"),), ())
        self.assertTrue(result.passed)

    def test_leak_and_drift(self):
        result = assess(self.fact, (self.obs("negative", "Unknown"),),
                        (self.obs("positive", "Not sure"),),
                        (self.obs("negative", "Sheffield"),),
                        (self.obs("anchor", "5"),), (self.obs("anchor", "6"),), ())
        self.assertFalse(result.passed)
        self.assertEqual(result.leaks, ("negative",))
        self.assertEqual(len(result.drift), 2)

    def test_retained_failure(self):
        result = assess(self.fact, (), (self.obs("p", "Sheffield"),), (), (), (),
                        ((self.fact, (self.obs("old", "unknown"),)),))
        self.assertEqual(result.retained_failures, ("old",))

    def test_mismatched_probes_refused(self):
        with self.assertRaises(ValueError):
            assess(self.fact, (), (), (self.obs("extra", "x"),), (), (), ())


if __name__ == "__main__":
    unittest.main()
