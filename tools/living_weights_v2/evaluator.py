from __future__ import annotations

from dataclasses import asdict
from typing import Protocol

from .assessment import affirmative_answer, contains_answer
from .schema import Manifest


class ChatModel(Protocol):
    def chat(self, model: str, question: str) -> str: ...


def evaluate(manifest: Manifest, client: ChatModel) -> dict:
    """Read-only lexical checks; not a semantic correctness certificate."""
    checks = []
    total = 0
    correct = 0
    leaked = 0
    for fact in manifest.facts:
        positive = []
        negative = []
        for question in fact.questions:
            answer = client.chat(manifest.model, question)
            matched = affirmative_answer(answer, fact.answer)
            total += 1
            correct += int(matched)
            positive.append({"question": question, "answer": answer, "expected_present": matched})
        for question in fact.near_misses:
            answer = client.chat(manifest.model, question)
            violation = contains_answer(answer, fact.answer)
            leaked += int(violation)
            negative.append({"question": question, "answer": answer, "possible_leak": violation})
        checks.append({"fact_digest": fact.digest, "subject": fact.subject,
                       "positives": positive, "near_misses": negative})
    anchors = [{"question": question, "answer": client.chat(manifest.model, question)} for question in manifest.anchors]
    return {"model": manifest.model, "manifest_digest": manifest.digest, "positive_matches": correct,
            "positive_total": total, "possible_near_miss_leaks": leaked, "anchors": anchors,
            "checks": checks, "weights_changed": False,
            "disclaimer": "lexical answer-presence checks; not a semantic or model-family training guarantee"}
