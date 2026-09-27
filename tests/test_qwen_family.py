"""Qwen3.8 dense as a family model: the pieces that run without the model (cost table, drafts, batched drafting)."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.drafters import dflash_batch  # noqa: E402
from tensorfold.families.qwen3_5.dflash_head import DFlashHead, DraftSlot, _Context, as_drafts, by_chance  # noqa: E402
from tensorfold.families.qwen3_5.family import fill_widths  # noqa: E402


def test_fill_widths_steps_at_17_rows():
    costs = fill_widths({1: 32.0, 2: 34.0, 8: 34.5, 16: 36.7, 17: 47.8, 24: 48.0, 32: 49.2}, 32)
    assert sorted(costs) == list(range(1, 33))
    assert costs[5] == pytest.approx(34.25) and costs[20] == pytest.approx(47.8 + 0.2 * 3 / 7)
    assert costs[16] < costs[17]                            # no blending across the 32-row op's step


def test_drafts_take_a_trees_first_pops_and_chains_as_lists():
    tree = ([5, 6, 7, 8], [-1, -1, 0, 2])
    assert as_drafts(tree, 2) == ([5, 6], [-1, -1])
    assert as_drafts(tree, 4) == tree
    assert as_drafts(([5, 6, 7], [-1, 0, 1]), 3) == [5, 6, 7]


def test_draft_context_holds_its_length_and_anchor():
    context = _Context(40, 9)
    assert len(context) == 40 and context[-1] == 9 and context[39] == 9 and context[-3:] == [9]
    with pytest.raises(IndexError):
        context[3]


def test_draft_slot_copies_empty():
    import copy

    slot = DraftSlot(object())
    slot.proposer, slot.anchor = object(), 7
    clone = copy.copy(slot)
    assert clone.proposer is None and clone.anchor == 0 and clone.drafter is slot.drafter and slot.keys is None


def test_start_trees_batches_equal_blocks_and_runs_the_rest_alone(monkeypatch):
    calls = []

    class Proposer:
        def __init__(self, block, rows):
            self.block, self.context = block, mx.zeros((1, rows, 4))

        def _tree_prelude(self, context, nodes):
            return ("need", self.block, [], nodes) if self.block else ("done", [1], [-1])

        def _lattice(self, context, block):
            calls.append(("alone", id(self)))
            return (0, 0, 0)

    monkeypatch.setattr(dflash_batch, "batched_lattices",
                        lambda drafter, ps, cs, block: calls.append(("batch", [id(p) for p in ps])) or [(1, 1, 1)] * len(ps))
    a, b, c, d = Proposer(16, 3), Proposer(16, 2), Proposer(16, dflash_batch.CONTEXT_ROWS + 1), Proposer(0, 1)
    states = dflash_batch.start_trees(SimpleNamespace(), [(p, [1], 15) for p in (a, b, c, d)])
    assert calls[0] == ("batch", [id(a), id(b)]) and ("alone", id(c)) in calls
    assert states[0][0] == "lattice" and states[3] == ("done", [1], [-1])


def test_a_tree_comes_most_likely_first_parents_before_children():
    tokens, parents, chances = by_chance([5, 6, 7, 8], [-1, -1, 0, 1], [0.5, 0.9, 0.6, 0.8])
    assert tokens == [6, 8, 5, 7] and parents == [-1, 0, -1, 2]
    assert chances == [0.9, 0.8, 0.5, 0.5]                  # a child is at most as likely as its parent


def test_the_head_queues_drafts_with_their_chances():
    slot = DraftSlot(object())
    slot.proposer = SimpleNamespace(last_scores=[-0.7, -0.1, -0.2])
    drafts = DFlashHead(object())._drafts(slot, ([5, 6, 7], [-1, -1, 1]), 10, None, 2)
    assert drafts == [6, 7] and slot.chances == pytest.approx([0.905, 0.819], abs=1e-3)
    slot.proposer.last_scores = None
    assert DFlashHead(object())._drafts(slot, ([5, 6], [-1, 0]), 10, None, 2) == [5, 6] and slot.chances is None


def test_one_read_draws_what_each_stream_draws_alone():
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen3_5.family import Qwen35Family

    family = Qwen35Family.__new__(Qwen35Family)
    logits = [mx.random.normal((1, n, 512), key=mx.random.key(n)).astype(mx.bfloat16) for n in (3, 1, 5)]
    samplings = [Sampling(temperature=1.0, top_k=20, top_p=0.95, seed=7), None, Sampling(temperature=0.7, seed=9)]
    positions = [[10, 11, 11], [4], [30, 31, 32, 32, 33]]
    together = family.sample_streams(logits, samplings, positions)
    alone = [family.sample(x, s, p) for x, s, p in zip(logits, samplings, positions)]
    assert [[int(t) for t in (a.tolist() if hasattr(a, "tolist") else a)] for a in together] == \
        [[int(t) for t in (a.tolist() if hasattr(a, "tolist") else a)] for a in alone]


def test_start_trees_splits_a_group_past_the_lattice_rows(monkeypatch):
    calls = []

    class Proposer:
        def __init__(self):
            self.context = mx.zeros((1, 2, 4))

        def _tree_prelude(self, context, nodes):
            return ("need", 16, [], nodes)

    monkeypatch.setattr(dflash_batch, "batched_lattices",
                        lambda drafter, ps, cs, block: calls.append(len(ps)) or [(1, 1, 1)] * len(ps))
    dflash_batch.start_trees(SimpleNamespace(), [(Proposer(), [1], 15) for _ in range(19)])
    assert calls == [8, 8, 3]                              # 128 block rows a forward
