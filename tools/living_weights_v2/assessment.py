from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

from .schema import Fact


@dataclass(frozen=True)
class Observation:
    question: str
    answer: str
    mode: str


@dataclass(frozen=True)
class Assessment:
    positives: int
    positive_total: int
    leaks: tuple[str, ...]
    drift: tuple[str, ...]
    retained_failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.positives == self.positive_total and not (self.leaks or self.drift or self.retained_failures)


def contains_answer(response: str, expected: str) -> bool:
    if not expected.strip():
        return False
    words = [re.escape(x) for x in expected.casefold().split()]
    expression = r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)"
    return re.search(expression, response.casefold()) is not None


def affirmative_answer(response: str, expected: str) -> bool:
    if not contains_answer(response, expected):
        return False
    unsafe = ("i don't know", "i do not know", "not sure", "cannot confirm", "can't verify",
              "not enough information", "might be", "possibly", "perhaps", "no evidence")
    lowered = response.casefold()
    if any(phrase in lowered for phrase in unsafe):
        return False
    pattern = r"\b(?:not|never|isn't|wasn't|ain't)\b.{0,24}" + re.escape(expected.casefold())
    if re.search(pattern, lowered):
        return False
    return True


def observe(questions: Iterable[str], request: Callable[[str], str], mode: str) -> tuple[Observation, ...]:
    return tuple(Observation(question, request(question), mode) for question in questions)


def _key(text: str) -> str:
    return " ".join(text.casefold().split())


def assess(
    fact: Fact,
    baseline_negatives: tuple[Observation, ...],
    after_positives: tuple[Observation, ...],
    after_negatives: tuple[Observation, ...],
    baseline_anchors: tuple[Observation, ...],
    after_anchors: tuple[Observation, ...],
    retained: tuple[tuple[Fact, tuple[Observation, ...]], ...],
) -> Assessment:
    if len(baseline_negatives) != len(after_negatives) or len(baseline_anchors) != len(after_anchors):
        raise ValueError("pre/post checks have different lengths")
    leaks = []
    drift = []
    failures = []
    positive_count = sum(affirmative_answer(o.answer, fact.answer) for o in after_positives)
    for before, after in zip(baseline_negatives, after_negatives):
        if before.question != after.question:
            raise ValueError("near-miss probes changed")
        if contains_answer(after.answer, fact.answer) and not contains_answer(before.answer, fact.answer):
            leaks.append(after.question)
        if _key(before.answer) != _key(after.answer):
            drift.append(after.question)
    for before, after in zip(baseline_anchors, after_anchors):
        if before.question != after.question:
            raise ValueError("anchor probes changed")
        if _key(before.answer) != _key(after.answer):
            drift.append(after.question)
    for old, observations in retained:
        for observation in observations:
            if not affirmative_answer(observation.answer, old.answer):
                failures.append(observation.question)
    return Assessment(positive_count, len(after_positives), tuple(leaks), tuple(drift), tuple(failures))
