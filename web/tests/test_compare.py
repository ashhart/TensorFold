import copy
import json
from pathlib import Path

import pytest

from tensorfold_web import compare

DOCS = Path(__file__).parent / "fixtures"          # real receipts: Nemotron on the Mac mini M6, 8 Oct 2026


def _run(name: str) -> dict:
    return json.loads((DOCS / f"2026-10-08-nemotron-m6-{name}.json").read_text())


def test_the_exact_tests():
    assert compare.binomial_two_sided(0, 0) == 1.0
    assert compare.binomial_two_sided(3, 14) == pytest.approx(0.0574, abs=1e-4)
    assert compare.binomial_two_sided(0, 10) == pytest.approx(2 / 1024)
    assert compare.mcnemar(0, 0) == 1.0 and compare.mcnemar(6, 0) == pytest.approx(2 / 64)


def test_the_two_nemotron_profiles_of_8_october_are_within_noise():
    c = compare.compare(_run("profile"), _run("profile-power"))
    assert len(c["pairs"]) == 14 and c["same_text"] == 14
    assert c["speed"]["within_noise"] and c["exactness"] == {"checked": 14, "both": 14, "only_a": 0, "only_b": 0,
                                                             "neither": 0, "p": 1.0, "within_noise": True,
                                                             "too_few": True}
    assert c["findings"][0].startswith("Speed: B is as fast at the median, no clear difference")
    assert any("predates suite versions" in n for n in c["notes"])


def test_a_real_difference_is_said_as_one():
    a, b = _run("profile"), copy.deepcopy(_run("profile"))
    for r in b["results"]:
        if r["draft"]:
            r["m"]["tps"] *= 1.3
        else:
            r["same"] = False                               # B no longer identical without drafts anywhere
    c = compare.compare(a, b)
    assert not c["speed"]["within_noise"] and c["speed"]["median_change"] == pytest.approx(0.3)
    assert c["exactness"]["only_a"] == 14 and not c["exactness"]["within_noise"]
    assert c["findings"][0].startswith("Speed: B is 30% faster at the median, a real difference")


def test_runs_that_asked_different_things_are_refused():
    a, b = _run("profile"), copy.deepcopy(_run("profile"))
    with pytest.raises(compare.Refused, match="prefill sweep"):
        compare.compare(a, _run("prefill"))
    a2, b2 = copy.deepcopy(a), b
    a2["suites_version"], b2["suites_version"] = {"version": 1, "digest": "aaa"}, {"version": 2, "digest": "bbb"}
    with pytest.raises(compare.Refused, match="suites differ"):
        compare.compare(a2, b2)


def test_two_sweeps_pair_by_size():
    c = compare.compare(_run("prefill"), _run("prefill"))
    assert [p["prompt"] for p in c["pairs"]][:3] == ["1024", "2048", "4096"] and c["speed"]["median_change"] == 0
    page = compare.html_report(_run("prefill"), _run("prefill"), c)
    assert page.startswith("<!doctype html>") and 'id="runs"' in page


def test_a_loss_confined_to_one_group_is_said():
    a = _run("profile")
    b = copy.deepcopy(a)
    for r in b["results"]:
        if not r["draft"] and r["suite"] in ("code", "reasoning", "agent"):
            r["same"] = False
    c = compare.compare(a, b)
    groups = c["by_suite"]
    assert groups["code"]["only_a"] == 2 and groups["prose_en"]["only_a"] == 0
    assert all(g["p"] >= 0.05 for g in groups.values())          # two items a group: nothing tells from noise
    assert not any("groups tested" in f for f in c["findings"])


def _mini(mode="quick", tps=(10.0, 10.0, 10.0), texts=None, digest="d1", error=None):
    texts = texts or ["same"] * len(tps)
    results = [{"suite": "s", "prompt": f"p{i}", "draft": True, "error": error if i == 0 else None, "text": t[:4000],
                "text_sha": __import__("hashlib").sha256(t.encode()).hexdigest()[:16],
                "m": {"tps": v, "tokens": 10, "ttft": 0.1}} for i, (v, t) in enumerate(zip(tps, texts, strict=True))]
    return {"id": "r", "mode": mode, "model": "M", "suites": ["s"], "status": "done", "results": results,
            "suites_version": {"version": 1, "digest": digest}, "machine": {"chip": "x"}, "summary": None}


def test_too_few_prompts_cannot_say_noise():
    c = compare.compare(_mini(tps=(10, 10, 10)), _mini(tps=(30, 30, 30)))
    assert c["speed"]["too_few"] and "too few to tell" in c["findings"][0]


def test_errored_replies_and_long_texts_are_not_counted_as_the_same():
    long_a, long_b = "x" * 5000, "x" * 4500 + "y" * 500
    a = _mini(texts=["same", long_a, "z"], error="cut")
    b = _mini(texts=["same", long_b, "z"])
    c = compare.compare(a, b)
    assert len(c["pairs"]) == 2 and c["same_text"] == 1              # the errored pair left out; long texts differ


def test_tool_runs_of_different_versions_are_not_compared():
    with pytest.raises(compare.Refused, match="same tool-eval-bench"):
        compare.compare(_mini(mode="tools", digest="teb-1"), _mini(mode="tools", digest="teb-2"))
