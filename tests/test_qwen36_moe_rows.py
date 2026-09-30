"""Qwen3.6 MoE on the lane decoder without tensor units (row_forward.moe): windows and several streams' windows
reproduce one-row steps bit for bit, the DFlash (v1) head drafts chains, and shutdown saves prompt-side entries."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.engine.lane_engine import LaneEngine  # noqa: E402
from tensorfold.kernels.qwen.dense.v1 import exact_attention, row_forward, row_matmul  # noqa: E402


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def _quantize(module):
    eight = ("gate", "shared_expert_gate")        # the checkpoint's 8-bit router and gate
    nn.quantize(module, class_predicate=lambda path, m: hasattr(m, "to_quantized") and (
        {"group_size": 64, "bits": 8} if path.split(".")[-1] in eight else {"group_size": 64, "bits": 4}))


@pytest.fixture(scope="module")
def tiny():
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    # Qwen3.6-35B-A3B's layer pattern and expert routing (8 of the experts, a shared expert) on a small residual
    args = TextModelArgs(model_type="qwen3_5_moe_text", hidden_size=1024, intermediate_size=512, num_hidden_layers=4,
                         num_attention_heads=16, num_key_value_heads=2, head_dim=256, rms_norm_eps=1e-6,
                         vocab_size=512, linear_num_value_heads=32, linear_num_key_heads=16,
                         linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
                         full_attention_interval=4, tie_word_embeddings=False, max_position_embeddings=4096,
                         num_experts=16, num_experts_per_tok=8, shared_expert_intermediate_size=512,
                         moe_intermediate_size=512, norm_topk_prob=True)
    mx.random.seed(21)
    model = TextModel(args)
    model.set_dtype(mx.bfloat16)
    _quantize(model)
    mx.eval(model.parameters())
    exact_attention.install()
    backend = row_matmul.simd_qmm_backend()
    assert row_matmul.fits(model, backend)
    row_matmul.install(model, backend)
    return model


def _run(model, tokens, cache, start):
    parents = [-1] + list(range(len(tokens) - 1))
    logits, record = row_forward.forward(model.model, model.lm_head, tokens, parents, cache, start, pipeline_layers=2)
    row_forward.commit(cache, record, list(range(len(tokens))), len(tokens), start)
    mx.eval(logits, *[a for c in cache for a in c.state if a is not None])
    return logits


@pytest.mark.parametrize("mode", ["batched", "rows"])
def test_moe_windows_reproduce_one_row_steps(tiny, mode, monkeypatch):
    monkeypatch.setattr(row_forward, "MOE_ROWS", mode)
    mx.random.seed(5)
    prompt = [int(t) for t in mx.random.randint(0, 512, (21,)).tolist()]
    tokens = [int(t) for t in mx.random.randint(0, 512, (row_matmul.WINDOW_ROWS,)).tolist()]
    base = LaneEngine.copy_single_cache(tiny.make_cache())
    for begin in range(0, len(prompt), row_matmul.WINDOW_ROWS):
        _run(tiny, prompt[begin:begin + row_matmul.WINDOW_ROWS], base, begin)
    start = len(prompt)
    serial_cache = LaneEngine.copy_single_cache(base)
    serial = [_run(tiny, [t], serial_cache, start + i)[0, -1] for i, t in enumerate(tokens)]
    for width in range(2, len(tokens) + 1):
        window = _run(tiny, tokens[:width], LaneEngine.copy_single_cache(base), start)
        for i in range(width):
            assert _same(window[0, i], serial[i]), f"row {i} of a {width}-row window differs from its one-row step"


def test_moe_streams_in_one_forward_equal_each_alone(tiny):
    ok, failures = row_forward.check_streams(tiny.model, tiny.lm_head, tiny.make_cache, LaneEngine.copy_single_cache,
                                             mixes=[(1, 1), (1, 4), (8, 8), (3, 1, 8, 5), (16, 2)])
    assert ok, failures


def test_dflash_v1_head_drafts_a_chain():
    from tensorfold.families.qwen3_5.dflash_head import DFlashHead

    seen = []

    class Proposer:
        ready, context, sampling = True, object(), None

        def propose(self, context, max_draft):
            seen.append((len(context), context[-1], max_draft))
            return [7, 8, 9][:max_draft]

    drafter = SimpleNamespace(model=SimpleNamespace(), block_size=16,                     # no candidate_selector
                              proposer=lambda **_: Proposer())
    head = DFlashHead(drafter, nodes=15, chains=True)
    assert head.v1
    cache = [head.slot()]
    cache[-1].get(None)
    cache[-1].anchor = 42
    assert head.tree(cache, 100, None, 2) == [7, 8] and seen == [(100, 42, 2)]      # a chain: a token list
    assert cache[-1].chances is None


def test_dflash_v1_chains_even_with_a_draft_vocabulary(monkeypatch):
    """A v1 drafter whose head has draft-vocabulary rows (z-lab/Qwen3.6-35B-A3B-DFlash) drafts through
    ``block_chain``, not the draft-vocabulary path that reads DFlash2's candidate selector."""

    from tensorfold.drafters import dflash_block
    from tensorfold.drafters.dflash_proposer import DFlashProposer

    monkeypatch.setattr(dflash_block, "block_chain", lambda drafter, inputs, context, cache: mx.array([[5, 6, 7]]))
    item = SimpleNamespace(offset=0)
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.drafter = SimpleNamespace(model=SimpleNamespace(), block_size=4, mask_id=0, _sub_head=lambda: None,
                                       _plain_sub_head=lambda: object(), _trim=lambda cache, n: None)
    proposer.copy, proposer.model_cap, proposer.ready, proposer.context, proposer.cache = None, None, True, object(), [item]
    proposer.draft_ms, proposer.proposals, proposer.proposed_tokens = 0.0, 0, 0
    assert proposer.propose([1, 2, 3], 3) == [5, 6, 7]


def test_dflash_v1_family_sizes_chains_from_round_costs(monkeypatch):
    """A v1 head gives no per-draft chances, so the family exposes none and the engine sizes chains by depth."""

    from tensorfold.families.qwen3_5.family import Qwen35Family

    monkeypatch.setattr(Qwen35Family, "check_windows", lambda self, widest, rows: (widest, {}))
    core = SimpleNamespace(embed_tokens=object())
    model = SimpleNamespace(model=core, lm_head=object(), args=None)
    drafter = SimpleNamespace(model=SimpleNamespace(), block_size=16, proposer=lambda **_: None)
    family = Qwen35Family(model, drafter=drafter)
    assert family.head_drafts.v1 and family.draft_probabilities is None
    assert callable(Qwen35Family(model).draft_probabilities)


def test_dflash_v1_block_reads_the_draft_vocabulary_rows():
    """``block_chain`` argmaxes the draft vocabulary's head rows and maps each column back to its token id."""

    from tensorfold.drafters.dflash_block import block_chain

    def full_head(_):
        raise AssertionError("the full vocabulary head was read")

    model = SimpleNamespace(embed_tokens=lambda t: mx.zeros((*t.shape, 4)), embed_scale=1.0, fc=lambda c: c,
                            hidden_norm=lambda c: c, norm=lambda h: h, layers=[], compute_logits=full_head)
    ids = mx.array([100, 200, 300])
    drafter = SimpleNamespace(model=model, _block_parts=[],
                              candidate_logits=lambda h: (mx.array([[[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]]), ids))
    assert block_chain(drafter, mx.array([[1, 0, 0]]), mx.zeros((1, 1, 4)), []).tolist() == [[200, 300]]


def test_shutdown_saves_prompt_side_entries_before_reply_ends(tmp_path, monkeypatch):
    from tensorfold.server import checkpoints

    saved = []
    monkeypatch.setattr("tensorfold.engine.prefix_snapshots.save_snapshot",
                        lambda directory, model_id, tokens, cache, keep: saved.append(list(tokens)))
    store = checkpoints.CheckpointStore(8, copier=lambda c: c, sizer=lambda c: 1)
    # Two conversations, each with a prompt-side entry and a longer reply end. Qwen3.6's template drops the empty
    # think block when a reply comes back as history, so a reply end never matches the next turn.
    other = [7] * 45
    store.insert(other, [], last_prompt=other + [1, 2])
    store.insert(other + [3] * 16, [], last_prompt=other + [1, 2])
    store.insert(list(range(50)), [], last_prompt=list(range(52)))
    store.insert(list(range(60)), [], last_prompt=list(range(52)))
    assert checkpoints.save_conversations(store, tmp_path, "model-a", keep=2) == 2
    assert sorted(len(t) for t in saved) == [45, 50]
