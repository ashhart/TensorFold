import time

import pytest

from tensorfold_web import bench, compare, quality, report
from tensorfold_web.mock_engine import serve as mock_serve


def test_the_suite_ships_its_items_and_licences():
    from importlib import resources

    d = quality.load()
    assert {k: len(v["items"]) for k, v in d["categories"].items()} == {"gsm8k": 100, "mmlu": 100, "instructions": 20,
                                                                        "long_context": 8}
    lic = resources.files("tensorfold_web.data").joinpath("licenses")
    assert "MIT" in lic.joinpath("gsm8k-MIT.txt").read_text() and "Hendrycks" in lic.joinpath("mmlu-MIT.txt").read_text()
    long = [i for i in quality.items(d, "Filler text about the river. " * 20) if i["category"] == "long_context"]
    text = long[0]["messages"][0]["content"]
    assert text.index(long[0]["item"]["fact"]) < text.index(long[0]["item"]["correction"]) < text.index("Answer with")


@pytest.mark.parametrize(("cat", "item", "reply", "ok"), [
    ("gsm8k", {"answer": "18"}, "16 - 3 - 4 = 9; 9 * 2 = 18.\nAnswer: 18", True),
    ("gsm8k", {"answer": "1080"}, "Answer: 1,080", True),
    ("gsm8k", {"answer": "18"}, "Answer: 16 or 18", False),
    ("gsm8k", {"answer": "18"}, "<think>18 maybe", False),                   # an open reasoning block, no answer
    ("gsm8k", {"answer": "18"}, "<think>x</think>Answer: 18", True),
    ("mmlu", {"answer": "B"}, "B", True),
    ("mmlu", {"answer": "B"}, "The answer is B.", True),
    ("mmlu", {"answer": "B"}, "A or B", False),
    ("mmlu", {"answer": "C"}, "(C) Paris", True),
    ("long_context", {"answer": "31"}, "Room 31", True),
    ("long_context", {"answer": "31"}, "31, not 12", False),                   # the last value counts
    ("instructions", {"checks": {"words_max": 3}}, "one two three four", False),
    ("instructions", {"checks": {"json_keys": ["a"], "json_list_len": {"key": "a", "n": 2}}}, '{"a": [1, 2]}', True),
    ("instructions", {"checks": {"bullets": 2}}, "Here:\n- a\n- b", False),
    ("instructions", {"checks": {"ends_with": "See you then."}}, "Hi.\nSee you then.", True),
    ("instructions", {"checks": {"title_brackets": True}}, "<<Rain>>\nIt falls.", True),
    ("instructions", {"checks": {"excludes_words": ["sun"]}}, "The sunny shore.", True),
])
def test_strict_scoring(cat, item, reply, ok):
    assert quality.score(cat, item, reply)[0] is ok


def test_a_quality_run_scores_every_item_and_compares_with_mcnemar(tmp_path):
    server = mock_serve(port=0, rate_tps=4000.0)
    try:
        store = bench.Store(tmp_path / "bench.sqlite")
        runner = bench.Bench(store, lambda: (f"http://127.0.0.1:{server.server_address[1]}", None, "Qwen3.8-27B"))
        runner.power = False
        run = runner.start("quality", suites=["instructions", "long_context"])
        end = time.time() + 120
        while runner.busy() and time.time() < end:
            time.sleep(0.05)
    finally:
        server.shutdown()
    got = store.run(run["id"])
    q = got["summary"]["quality"]
    assert got["status"] == "done" and q["overall"]["items"] == 28 and got["exactness"] is False
    assert got["suites_version"]["quality"] == 1 and got["suite_names"]["long_context"].startswith("Long context")
    assert all("passed" in r and "read" in r for r in got["results"])
    assert report.findings(got)[0].endswith("an answer that offers two fails.")
    assert "quality v1 " in report.share_text(got) and report.html_report(got).startswith("<!doctype html>")
    assert report.share_text(got).splitlines()[2].startswith(f"{q['overall']['passed']}/28 right")
    c = compare.compare(got, got)
    assert c["exactness"]["checked"] == 28 and c["findings"][1].startswith("Quality: the same outcome on all 28 items")


def test_fractions_and_self_contradictions_are_read_strictly():
    from tensorfold_web import quality as q

    assert q.score("gsm8k", {"answer": "18"}, "Answer: 1/18") == (False, "0.0555556")
    assert q.score("gsm8k", {"answer": "0.5"}, "Answer: 1/2")[0]
    assert q.score("gsm8k", {"answer": "1250"}, "Answer: 1,250")[0]
    assert not q.score("mmlu", {"answer": "B"}, "B. Actually the answer is C.")[0]
    assert q.score("mmlu", {"answer": "B"}, "B. Option A is wrong because the rate falls.")[0]


def test_a_letter_contradicted_in_other_words_fails():
    from tensorfold_web.quality import score

    it = {"answer": "B"}
    for reply in ("B. The correct choice is C.", "B. ANSWER: C.", "B. The right option is (C)."):
        assert score("mmlu", it, reply)[0] is False, reply
    for reply in ("B", "B. The answer is a bit subtle.", "B. Option C is wrong because it ignores friction.",
                  "The correct option is (B)."):
        assert score("mmlu", it, reply) == (True, "B"), reply


def test_a_decimal_comma_and_an_opening_article_are_not_read_as_answers():
    from tensorfold_web import quality as q

    assert q.score("gsm8k", {"answer": "1.5"}, "Answer: 1,5")[0]                # a decimal comma
    assert q.score("gsm8k", {"answer": "1250"}, "Answer: 1,250")[0]             # a thousands separator still
    assert q.score("mmlu", {"answer": "B"}, "A good case can be made, but B.")[1] != "A"
    assert q.score("mmlu", {"answer": "A"}, "A. The rate falls.")[0]
