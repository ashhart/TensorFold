from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .assessment import affirmative_answer, assess, contains_answer, observe
from .ledger import Ledger
from .schema import Fact, Manifest
from .transport import Client, Event, TransportError, checked_graph


class LearningError(RuntimeError):
    pass


def _stable_graph_fact(item: dict) -> dict:
    # Native rechecks update counters and verdicts; links are recomputed from all current fact texts.
    return {key: value for key, value in item.items() if key not in {"checks", "recalled", "links"}}


def _changes(before: dict[int, dict], after: dict[int, dict]) -> list[int]:
    removed = set(before) - set(after)
    if removed:
        raise LearningError(f"graph lost facts {sorted(removed)}")
    for key in before.keys() & after.keys():
        if _stable_graph_fact(before[key]) != _stable_graph_fact(after[key]):
            raise LearningError(f"existing graph fact {key} changed during lesson")
    return sorted(set(after) - set(before))


def _event_summary(events: tuple[Event, ...]) -> dict:
    meaningful = [e for e in events if e.kind in {"learned", "saved", "failed", "forgot"}]
    return {"events": [{"kind": e.kind, "data": e.data} for e in meaningful]}


def _claimed_id(event: Event) -> int | None:
    value = event.data.get("id")
    if type(value) is int and value > 0:
        return value
    nested = event.data.get("fact")
    if isinstance(nested, dict) and type(nested.get("id")) is int:
        return nested["id"]
    return None


def _owns_new_fact(new_ids: list[int], events: tuple[Event, ...], graph: dict[int, dict], fact: Fact) -> bool:
    if len(new_ids) != 1:
        return False
    if any(_claimed_id(event) == new_ids[0] for event in events if event.kind in {"fact", "learned"}):
        return True
    item = graph[new_ids[0]]
    for field in ("text", "statement", "fact", "content"):
        value = item.get(field)
        if isinstance(value, str) and value.strip().casefold() == fact.text.casefold():
            return True
    return False


class Controller:
    def __init__(self, client: Client, ledger: Ledger):
        self.client = client
        self.ledger = ledger

    def _graph(self) -> dict[int, dict]:
        # Every graph read is checked here too, so a state the controller doesn't know stops it before any lesson.
        return checked_graph(list(self.client.graph().values()))

    def _probe(self, model: str, questions: tuple[str, ...], mode: str):
        return observe(questions, lambda question: self.client.chat(model, question), mode)

    def _restore(self, ids: list[int]) -> bool:
        if not ids or len(ids) > 256:
            return False
        try:
            events = self.client.delete(ids, "Living Weights v2 external regression gate")
            forgot = [event for event in events if event.kind == "forgot"]
            if len(forgot) != len(ids) or any(event.data.get("deleted") is not True for event in forgot):
                return False
            post = self._graph()
            return all(post.get(i, {}).get("state") == "deleted" for i in ids)
        except (TransportError, ValueError):
            return False

    def _verify_existing(self, model: str, retained: tuple[Fact, ...]):
        return tuple((fact, self._probe(model, fact.questions, "retained")) for fact in retained)

    def _recovered(self, model: str, baseline_pos, baseline_neg, baseline_anchor, baseline_retained) -> bool:
        sequences = [baseline_pos, baseline_neg, baseline_anchor]
        sequences.extend(observations for _, observations in baseline_retained)
        for before in sequences:
            after = self._probe(model, tuple(item.question for item in before), "recovery-check")
            for original, current in zip(before, after):
                if " ".join(original.answer.casefold().split()) != " ".join(current.answer.casefold().split()):
                    return False
        return True

    def preflight(self, manifest: Manifest) -> dict:
        if manifest.model not in self.client.models():
            raise LearningError("requested model is not listed by the server")
        graph = self._graph()
        if any(item.get("state") == "deleting" for item in graph.values()):
            raise LearningError("graph has an unfinished native deletion")
        for fact in manifest.facts:
            self.ledger.require_safe(manifest.model, fact)
        return {"model": manifest.model, "facts": len(manifest.facts), "graph_entries": len(graph)}

    def teach(self, manifest: Manifest) -> list[dict[str, Any]]:
        self.preflight(manifest)
        receipts = []
        for fact in manifest.facts:
            receipts.append(self._teach_one(manifest, fact))
            if receipts[-1]["state"] in {"needs_recovery", "rolled_back", "rejected"}:
                break
        return receipts

    def _teach_one(self, manifest: Manifest, fact: Fact) -> dict:
        model = manifest.model
        self.ledger.require_safe(model, fact)
        retained = self.ledger.retained(model)
        before_graph = self._graph()
        baseline_pos = self._probe(model, fact.questions, "before-positive")
        baseline_neg = self._probe(model, fact.near_misses, "before-near-miss")
        baseline_anchor = self._probe(model, manifest.anchors, "before-anchor")
        baseline_retained = self._verify_existing(model, retained)
        for old_fact, observations in baseline_retained:
            if not all(affirmative_answer(o.answer, old_fact.answer) for o in observations):
                return {"state": "rejected", "fact": fact.digest, "reason": "an earlier committed fact already fails its probes"}
        if all(affirmative_answer(result.answer, fact.answer) for result in baseline_pos):
            return {"state": "already_answers", "fact": fact.digest, "note": "not proof of retention; no weights changed"}
        if any(contains_answer(result.answer, fact.answer) for result in baseline_neg):
            return {"state": "rejected", "fact": fact.digest, "reason": "baseline leaks the expected answer to a near-miss"}
        entry_id = self.ledger.begin(model, manifest.digest, fact)
        receipt: dict[str, Any] = {
            "entry_id": entry_id, "fact_digest": fact.digest, "manifest_digest": manifest.digest,
            "model": model, "baseline": {
                "positives": [asdict(o) for o in baseline_pos],
                "near_misses": [asdict(o) for o in baseline_neg],
                "anchors": [asdict(o) for o in baseline_anchor],
                "retained": [{"fact_digest": old.digest, "observations": [asdict(o) for o in values]} for old, values in baseline_retained],
            },
        }
        state = "needs_recovery"
        try:
            events = self.client.teach(fact.text)
            receipt["native"] = _event_summary(events)
            learned = [event for event in events if event.kind == "learned"]
            saved = [event for event in events if event.kind == "saved"]
            errors = [event for event in events if event.kind == "failed"]
            after_graph = self._graph()
            new_ids = _changes(before_graph, after_graph)
            receipt["new_graph_ids"] = new_ids
            owned = _owns_new_fact(new_ids, events, after_graph, fact)
            receipt["new_fact_attributed"] = owned
            if errors or not saved or len(learned) != 1 or learned[0].data.get("recalled") is not True:
                if not new_ids:
                    state = "rejected"
                elif owned and self._restore(new_ids):
                    if self._recovered(model, baseline_pos, baseline_neg, baseline_anchor, baseline_retained):
                        state = "rolled_back"
                    else:
                        receipt["reason"] = "deletion completed but baseline output was not recovered"
                receipt["reason"] = "native learner did not confirm exactly one recalled and saved fact"
            elif not owned:
                receipt["reason"] = "cannot prove graph entry belongs to this request"
            else:
                after_pos = self._probe(model, fact.questions, "after-positive")
                after_neg = self._probe(model, fact.near_misses, "after-near-miss")
                after_anchor = self._probe(model, manifest.anchors, "after-anchor")
                retained_observations = self._verify_existing(model, retained)
                evaluation = assess(fact, baseline_neg, after_pos, after_neg, baseline_anchor, after_anchor, retained_observations)
                receipt["assessment"] = asdict(evaluation)
                receipt["answers"] = {
                    "positives": [asdict(o) for o in after_pos],
                    "near_misses": [asdict(o) for o in after_neg],
                    "anchors": [asdict(o) for o in after_anchor],
                    "retained": [{"fact_digest": old.digest, "observations": [asdict(o) for o in values]} for old, values in retained_observations],
                }
                if evaluation.passed:
                    state = "committed"
                elif self._restore(new_ids):
                    if self._recovered(model, baseline_pos, baseline_neg, baseline_anchor, baseline_retained):
                        state = "rolled_back"
                    else:
                        receipt["reason"] = "deletion completed but baseline output was not recovered"
                else:
                    receipt["reason"] = "post-training regression and native deletion was not confirmed"
        except Exception as exc:
            receipt["error"] = type(exc).__name__ + ": " + str(exc)
        receipt["state"] = state
        self.ledger.finish(entry_id, state, receipt)
        return receipt
