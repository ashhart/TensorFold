"""tools/dsv41_mmlu.py: the seeded stratified sample, the prompt, the letter parser, the Wilson interval, the exact
McNemar test and the paired comparison (no server)."""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
MM = importlib.import_module("dsv41_mmlu")


def rows(subjects: dict[str, int]) -> list[dict]:
    return [{"subject": s, "question": f"{s} q{i}?", "choices": ["w", "x", "y", "z"], "answer": i % 4}
            for s, n in subjects.items() for i in range(n)]


def test_sample_is_stratified_seeded_and_stable_per_subject():
    data = rows({"b_sub": 30, "a_sub": 12, "c_sub": 5})
    s = MM.stratified_sample(data, 10, seed=0)
    by = {}
    for q in s:
        by.setdefault(q["subject"], []).append(q)
    assert {k: len(v) for k, v in by.items()} == {"a_sub": 10, "b_sub": 10, "c_sub": 5}
    assert [q["subject"] for q in s] == sorted(q["subject"] for q in s)
    assert s == MM.stratified_sample(data, 10, seed=0)
    assert s != MM.stratified_sample(data, 10, seed=1)
    # the ids name the question's place in its subject, and the record is that question
    for q in s:
        subj, i = q["id"].split("/")
        assert q["question"] == f"{subj} q{i}?" and q["answer"] == int(i) % 4
    # dropping a subject leaves the other subjects' draws unchanged
    fewer = MM.stratified_sample(rows({"b_sub": 30, "c_sub": 5}), 10, seed=0)
    assert [q["id"] for q in fewer] == [q["id"] for q in s if q["subject"] != "a_sub"]


def test_prompt_lists_lettered_options():
    q = {"subject": "high_school_physics", "question": " What? ", "choices": ["1", "2", "3", "4"], "answer": 2}
    p = MM.prompt_for(q)
    assert "about high school physics." in p
    assert "What?\nA. 1\nB. 2\nC. 3\nD. 4\n\nAnswer:" in p
    body = MM.body_for(q, "m", 16)
    assert body["temperature"] == 0 and body["chat_template_kwargs"] == {"enable_thinking": False}


def test_parse_letter_takes_the_first_standalone_letter():
    assert MM.parse_letter("B") == "B"
    assert MM.parse_letter("**C**") == "C"
    assert MM.parse_letter("Answer: D") == "D"
    assert MM.parse_letter("The answer is (A).") == "A"
    assert MM.parse_letter("B. because") == "B"
    assert MM.parse_letter("CD player, so B") == "B"
    assert MM.parse_letter("none of these") is None
    assert MM.parse_letter("") is None and MM.parse_letter(None) is None
    assert MM.parse_letter("E") is None


def test_wilson_and_mcnemar():
    lo, hi = MM.wilson(80, 100)
    assert math.isclose(lo, 0.7112, abs_tol=1e-3) and math.isclose(hi, 0.8666, abs_tol=1e-3)
    assert MM.wilson(0, 0) == (0.0, 1.0)
    assert MM.mcnemar_exact(0, 0) == 1.0
    assert MM.mcnemar_exact(5, 5) == 1.0
    assert math.isclose(MM.mcnemar_exact(0, 6), 2 / 64)
    assert math.isclose(MM.mcnemar_exact(10, 2), MM.mcnemar_exact(2, 10))
    assert math.isclose(MM.mcnemar_exact(10, 2), 2 * sum(math.comb(12, i) for i in range(3)) / 4096)


def test_compare_pairs_by_id_and_counts_discordant():
    def res(subject, i, ok, pred="A"):
        return {"id": f"{subject}/{i}", "subject": subject, "correct": ok, "pred": pred}

    a = [res("s1", 0, True), res("s1", 1, True), res("s1", 2, False), res("s2", 0, True), res("s2", 9, True)]
    b = [res("s1", 0, True), res("s1", 1, False), res("s1", 2, True, None), res("s2", 0, False)]
    c = MM.compare(a, b)
    assert c["n"] == 4 and c["only_a"] == 1 and c["only_b"] == 0
    assert (c["b"], c["c"]) == (2, 1)
    assert c["acc_a"] == 0.75 and c["acc_b"] == 0.5
    assert c["unparsed_b"] == 1 and c["unparsed_a"] == 0
    assert math.isclose(c["diff"], -0.25)
    se = math.sqrt((3 / 4 - 0.0625) / 4)
    assert math.isclose(c["diff_ci"][0], -0.25 - 1.959964 * se)
    s1 = next(p for p in c["subjects"] if p["subject"] == "s1")
    assert (s1["n"], s1["b"], s1["c"]) == (3, 1, 1)
    assert c["flagged"] == []
    json.dumps(c)  # the report is plain JSON
    assert "McNemar: b (A right, B wrong) = 2" in MM.report(c, "A", "B")
