from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/, as the other tool tests do
from living_weights_v2.controller import Controller, LearningError, _changes
from living_weights_v2.ledger import Ledger, LedgerError, exclusive_file
from living_weights_v2.schema import Fact, Manifest
from living_weights_v2.transport import Event, TransportError


def fact(answer="Sheffield"):
    return Fact("Quillmoor Labs", "headquarters", answer,
                f"Quillmoor Labs is headquartered in {answer}.", "approved",
                ("Where is Quillmoor Labs?", "Which city is Quillmoor Labs in?"),
                ("Where is Brackenforth Labs?",))


def manifest(f):
    return Manifest("local-model", True, (f,), ("What is two plus three?",), "approved handbook", "abcd" * 16)


class FakeClient:
    def __init__(self):
        self.records = {}
        self.reply = "Unknown"
        self.saved = True
        self.recalled = True
        self.emit_id = True
        self.fail_delete = False
        self.fail_on_chat_after = False
        self.teach_count = 0
        self.delete_count = 0
        self.answer_anchors_after = "5"
        self.extraneous_fact = False

    def models(self):
        return {"local-model"}

    def graph(self):
        return {k: dict(v) for k, v in self.records.items()}

    def chat(self, model, question):
        if self.teach_count and self.fail_on_chat_after:
            raise TransportError("simulated connection drop after edit")
        if "two plus three" in question:
            return self.answer_anchors_after if self.teach_count else "5"
        if "Brackenforth" in question:
            return "Unknown"
        if self.teach_count and not self.records.get(1, {}).get("state") == "deleted":
            return self.reply if "Quillmoor" in question else "Unknown"
        return "Unknown"

    def teach(self, text):
        self.teach_count += 1
        if self.recalled:
            self.records[1] = {"id": 1, "state": "learned", "text": text}
        if self.extraneous_fact:
            self.records[2] = {"id": 2, "state": "learned", "text": "other concurrent fact"}
        return tuple(
            [Event("learned", {"recalled": self.recalled, **({"id": 1} if self.emit_id else {})})]
            + ([Event("saved", {"files": 1})] if self.saved else [Event("failed", {"phase": "saving"})])
        )

    def delete(self, ids, reason):
        self.delete_count += 1
        if self.fail_delete:
            return (Event("forgot", {"deleted": False}),)
        for item_id in ids:
            self.records[item_id]["state"] = "deleted"
        return tuple(Event("forgot", {"deleted": True}) for _ in ids)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.ledger = Ledger(Path(self.dir.name) / "ledger.sqlite3")
        self.client = FakeClient()
        self.controller = Controller(self.client, self.ledger)

    def tearDown(self):
        self.ledger.close()
        self.dir.cleanup()

    def test_success_and_idempotency(self):
        self.client.reply = "Sheffield"
        result = self.controller.teach(manifest(fact()))
        self.assertEqual(result[0]["state"], "committed")
        self.assertEqual(self.client.teach_count, 1)
        self.assertEqual(len(self.ledger.retained("local-model")), 1)
        with self.assertRaises(LedgerError):
            self.controller.teach(manifest(fact()))

    def test_native_rechecks_and_links_allow_two_committed_lessons(self):
        first = fact()
        second = Fact("Quillmoor Labs", "mascot", "Kestrel",
                      "Quillmoor Labs has a mascot called Kestrel.", "approved",
                      ("What is Quillmoor Labs' mascot?", "Name Quillmoor Labs' mascot."),
                      ("What is Brackenforth Labs' mascot?",))

        class NativeGraphClient(FakeClient):
            def graph(self):
                return deepcopy(self.records)

            def chat(self, model, question):
                if "two plus three" in question:
                    return "5"
                for item_id, lesson in enumerate((first, second), 1):
                    if question in lesson.questions and item_id in self.records:
                        return lesson.answer
                return "Unknown"

            def teach(self, text):
                self.teach_count += 1
                item_id = self.teach_count
                for old in self.records.values():
                    old["checks"] += 1
                    old["recalled"] = True
                    old["links"].append(item_id)
                self.records[item_id] = {"id": item_id, "state": "learned", "text": text,
                                         "checks": 1, "recalled": True,
                                         "links": list(range(1, item_id))}
                return (Event("learned", {"id": item_id, "recalled": True}),
                        Event("saved", {"saved": True}))

        client = NativeGraphClient()
        plan = Manifest("local-model", True, (first, second), manifest(first).anchors, "approved", "sequence")
        result = Controller(client, self.ledger).teach(plan)
        self.assertEqual([r["state"] for r in result], ["committed", "committed"])
        self.assertEqual(client.delete_count, 0)
        self.assertEqual(len(self.ledger.retained("local-model")), 2)
        self.assertEqual(result[1]["new_graph_ids"], [2])
        self.assertEqual(result[1]["assessment"]["retained_failures"], ())

    def test_graph_metadata_changes_preserve_new_id_detection(self):
        before = {17: {"id": 17, "text": "old fact", "state": "learned",
                       "checks": 1, "recalled": True, "links": []}}
        after = deepcopy(before)
        after[17].update(checks=2, recalled=False, links=[18])
        after[18] = {"id": 18, "text": "new fact", "state": "learned"}
        self.assertEqual(_changes(before, after), [18])

    def test_graph_identity_state_placement_and_unknown_changes_fail_closed(self):
        original = {"id": 17, "text": "old fact", "source": "chat", "at": 123,
                    "state": "learned", "neurons": [{"layer": 1, "neuron": 2, "key": "gate"}]}
        changes = {"id": 18, "text": "different fact", "source": "document", "at": 456,
                   "state": "deleted", "neurons": [], "unknown_field": "unexpected"}
        for field, value in changes.items():
            with self.subTest(field=field), self.assertRaises(LearningError):
                _changes({17: original}, {17: {**original, field: value}})
        with self.assertRaises(LearningError):
            _changes({17: original}, {})

    def test_graph_mutation_still_requires_recovery_without_deleting(self):
        self.client.records[17] = {"id": 17, "state": "learned", "text": "old fact"}
        teach = self.client.teach

        def mutate(text):
            events = teach(text)
            self.client.records[17]["state"] = "deleted"
            return events

        self.client.teach = mutate
        self.client.reply = "Sheffield"
        result = self.controller.teach(manifest(fact()))
        self.assertEqual(result[0]["state"], "needs_recovery")
        self.assertEqual(self.client.delete_count, 0)
        with self.assertRaises(LedgerError):
            self.controller.teach(manifest(fact("Cambridge")))

    def test_recheck_metadata_does_not_hide_retained_answer_regression(self):
        old = Fact("Pinebridge Labs", "headquarters", "Cambridge",
                   "Pinebridge Labs is headquartered in Cambridge.", "approved",
                   ("Where is Pinebridge Labs?", "Which city is Pinebridge Labs in?"),
                   ("Where is an unrelated lab?",))
        entry = self.ledger.begin("local-model", "earlier-manifest", old)
        self.ledger.finish(entry, "committed", {})
        self.client.records[17] = {"id": 17, "state": "learned", "text": old.text,
                                   "checks": 1, "recalled": True, "links": []}
        self.client.reply = "Sheffield"
        chat, teach = self.client.chat, self.client.teach

        def changed_chat(model, question):
            if question in old.questions:
                active = self.client.records.get(1, {}).get("state") == "learned"
                return "Unknown" if active else old.answer
            return chat(model, question)

        def changed_teach(text):
            events = teach(text)
            self.client.records[17].update(checks=2, recalled=False, links=[1])
            return events

        self.client.chat, self.client.teach = changed_chat, changed_teach
        result = self.controller.teach(manifest(fact()))[0]
        self.assertEqual(result["state"], "rolled_back")
        self.assertEqual(result["assessment"]["retained_failures"], old.questions)
        self.assertEqual(self.client.delete_count, 1)

    def test_paraphrase_failure_calls_native_delete(self):
        self.client.reply = "Unknown"
        result = self.controller.teach(manifest(fact()))
        self.assertEqual(result[0]["state"], "rolled_back")
        self.assertEqual(self.client.delete_count, 1)
        self.assertEqual(self.client.records[1]["state"], "deleted")

    def test_failed_delete_blocks_future_writes(self):
        self.client.reply = "Unknown"
        self.client.fail_delete = True
        result = self.controller.teach(manifest(fact()))
        self.assertEqual(result[0]["state"], "needs_recovery")
        with self.assertRaises(LedgerError):
            self.controller.teach(manifest(fact("Cambridge")))

    def test_crash_after_write_is_durable_unknown(self):
        self.client.reply = "Sheffield"
        self.client.fail_on_chat_after = True
        result = self.controller.teach(manifest(fact()))
        self.assertEqual(result[0]["state"], "needs_recovery")
        self.assertEqual(len(self.ledger.history("local-model")), 1)

    def test_unknown_native_owner_fails_closed(self):
        self.client.reply = "Sheffield"
        self.client.emit_id = False
        self.client.extraneous_fact = True
        result = self.controller.teach(manifest(fact()))
        self.assertEqual(result[0]["state"], "needs_recovery")
        self.assertEqual(self.client.delete_count, 0)

    def test_cannot_alter_preexisting_graph(self):
        self.client.records = {17: {"id": 17, "state": "deleting", "text": "old"}}
        with self.assertRaises(LearningError):
            self.controller.teach(manifest(fact()))

    def test_unsupported_model_refused(self):
        f = fact()
        plan = Manifest("different", True, (f,), ("What is two plus three?",), "source", "e" * 64)
        with self.assertRaises(LearningError):
            self.controller.teach(plan)

    def test_answer_baseline_does_not_write(self):
        self.client.reply = "Sheffield"
        original_chat = self.client.chat
        self.client.chat = lambda model, question: "Sheffield" if "Quillmoor" in question else original_chat(model, question)
        result = self.controller.teach(manifest(fact()))
        self.assertEqual(result[0]["state"], "already_answers")
        self.assertEqual(self.client.teach_count, 0)

    def test_native_rejected_no_graph_delta(self):
        self.client.recalled = False
        result = self.controller.teach(manifest(fact()))
        self.assertEqual(result[0]["state"], "rejected")
        self.assertEqual(self.client.delete_count, 0)

    def test_conflicting_answer_same_subject_rejected(self):
        self.client.reply = "Sheffield"
        self.assertEqual(self.controller.teach(manifest(fact()))[0]["state"], "committed")
        with self.assertRaises(LedgerError):
            self.ledger.require_safe("local-model", fact("Cambridge"))

    def test_regression_to_anchor_triggers_delete(self):
        self.client.reply = "Sheffield"
        self.client.answer_anchors_after = "four"
        result = self.controller.teach(manifest(fact()))
        self.assertEqual(result[0]["state"], "needs_recovery")

    def test_file_lock_refuses_concurrent_writer(self):
        lock = Path(self.dir.name) / "writing.lock"
        with exclusive_file(lock):
            with self.assertRaises(LedgerError):
                with exclusive_file(lock):
                    pass


if __name__ == "__main__":
    unittest.main()
