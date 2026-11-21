from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    pass


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _string(value: Any, name: str, limit: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ManifestError(f"{name} must be a nonempty string of at most {limit} characters")
    return value.strip()


def _fields(obj: Any, required: set[str], optional: set[str], name: str) -> dict:
    if not isinstance(obj, dict):
        raise ManifestError(f"{name} must be an object")
    missing = required - obj.keys()
    unknown = obj.keys() - required - optional
    if missing or unknown:
        raise ManifestError(f"{name}: missing {sorted(missing)}; unknown {sorted(unknown)}")
    return obj


def _questions(value: Any, name: str, min_count: int = 1) -> tuple[str, ...]:
    if not isinstance(value, list) or not min_count <= len(value) <= 64:
        raise ManifestError(f"{name} must contain {min_count} to 64 questions")
    result = tuple(_string(item, name, 1000) for item in value)
    if len(set(map(normalized, result))) != len(result):
        raise ManifestError(f"{name} contains duplicate questions")
    return result


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    answer: str
    text: str
    evidence: str
    questions: tuple[str, ...]
    near_misses: tuple[str, ...]

    @property
    def key(self) -> str:
        return normalized(self.subject) + "\x1f" + normalized(self.relation)

    @property
    def digest(self) -> str:
        fields = [self.subject, self.relation, self.answer, self.text, self.evidence]
        return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Manifest:
    model: str
    asserted_by_user: bool
    facts: tuple[Fact, ...]
    anchors: tuple[str, ...]
    source: str
    digest: str


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_manifest(path: Path) -> Manifest:
    try:
        raw = path.read_bytes()
        if len(raw) > 1024 * 1024:
            raise ManifestError("manifest exceeds 1 MiB")
        data = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(str(exc)) from exc
    obj = _fields(data, {"schema_version", "model", "asserted_by_user", "facts", "anchors", "source"}, set(), "manifest")
    if type(obj["schema_version"]) is not int or obj["schema_version"] != 1:
        raise ManifestError("schema_version must be 1")
    if obj["asserted_by_user"] is not True:
        raise ManifestError("asserted_by_user must be true; evidence is user-supplied, not automatically verified")
    if not isinstance(obj["facts"], list) or not 1 <= len(obj["facts"]) <= 128:
        raise ManifestError("facts must contain 1 to 128 entries")
    facts = []
    keys = set()
    for i, raw_fact in enumerate(obj["facts"]):
        entry = _fields(raw_fact, {"subject", "relation", "answer", "text", "evidence", "questions", "near_misses"}, set(), f"fact[{i}]")
        fact = Fact(
            subject=_string(entry["subject"], "subject", 256),
            relation=_string(entry["relation"], "relation", 128),
            answer=_string(entry["answer"], "answer", 256),
            text=_string(entry["text"], "text", 1024),
            evidence=_string(entry["evidence"], "evidence", 8192),
            questions=_questions(entry["questions"], "questions", 2),
            near_misses=_questions(entry["near_misses"], "near_misses", 1),
        )
        if normalized(fact.answer) not in normalized(fact.text):
            raise ManifestError(f"fact[{i}] answer must occur in text")
        if fact.key in keys:
            raise ManifestError(f"fact[{i}] duplicates a subject/relation in this manifest")
        keys.add(fact.key)
        facts.append(fact)
    return Manifest(
        model=_string(obj["model"], "model", 256),
        asserted_by_user=True,
        facts=tuple(facts),
        anchors=_questions(obj["anchors"], "anchors", 1),
        source=_string(obj["source"], "source", 2048),
        digest=hashlib.sha256(raw).hexdigest(),
    )
