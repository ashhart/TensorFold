"""The prompt-lookup drafter (``qwen4_exp.cuda.lookup``): pure Python, so these run without a GPU (P7)."""

from __future__ import annotations

import importlib

lookup = importlib.import_module("tensorfold.families.qwen4_exp.cuda.lookup")


def test_suffix_index_finds_longest_repeat():
    idx = lookup.SuffixIndex(3)
    idx.extend([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    length, end = idx.match()
    assert (length, end) == (5, 5)               # the whole period repeats
    assert idx.copy(5, 5) == [1, 2, 3, 4, 5]
    assert idx.propose(3, 4) == ([1, 2, 3], 5)
    assert idx.propose(3, 6) == ([], 5)          # below the match threshold: no drafts


def test_copy_overlapping_period():
    idx = lookup.SuffixIndex(3)
    idx.extend([1, 2, 3])
    assert idx.copy(0, 5) == [1, 2, 3, 1, 2]


def test_reuse_index_extends_on_prefix():
    first = lookup.reuse_index(3, [1, 2, 3, 4])
    assert list(first.tokens) == [1, 2, 3, 4]
    same = lookup.reuse_index(3, [1, 2, 3, 4, 5, 6])
    assert same is first and list(same.tokens) == [1, 2, 3, 4, 5, 6]
    other = lookup.reuse_index(3, [9, 9, 9])
    assert other is not first and list(other.tokens) == [9, 9, 9]


def test_env_spec_off_by_default(monkeypatch):
    monkeypatch.delenv("TF_EXL3_LOOKUP", raising=False)
    assert lookup.env_spec() is None
    for off in ("0", "false", "off", "no", ""):
        monkeypatch.setenv("TF_EXL3_LOOKUP", off)
        assert lookup.env_spec() is None
    monkeypatch.setenv("TF_EXL3_LOOKUP", "l5:3")
    assert lookup.env_spec() == "l5:3"
    monkeypatch.setenv("TF_EXL3_LOOKUP", "1")
    assert lookup.env_spec() == "auto"


def test_build_lookup(monkeypatch):
    monkeypatch.delenv("TF_EXL3_LOOKUP_MIN", raising=False)
    assert lookup.build_lookup([1, 2, 3], 8, None) is None
    forced = lookup.build_lookup([1, 2, 3], 8, "l7", eos=[2], stop_eos=True)
    assert forced.gated is False and forced.most == 7 and forced.min_match == 3 and forced.eos == frozenset({2})
    gated = lookup.build_lookup([1, 2, 3], 8, "auto")
    assert gated.gated is True and gated.most == 7 and gated.min_match == 4


def test_forced_plan_continues_the_repeat():
    draft = lookup.build_lookup([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], 8, "l3:3")
    assert draft.plan([], 8) == [1, 2, 3]
    assert draft.plan([], 2) == [1, 2]
    # a draft that is an eos ends the reply there: the drafts from it on are cut
    cut = lookup.build_lookup([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], 8, "l3:3", eos=[2], stop_eos=True)
    assert cut.plan([], 8) == [1]
