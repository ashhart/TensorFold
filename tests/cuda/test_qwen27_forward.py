"""A GDN verify window gives exactly the serial CUDA outputs on each path."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda.forward import State, commit, tree_forward
from tensorfold.families.qwen3_5.cuda.decode import draft_decode, serial_decode
from tensorfold.families.qwen3_5.cuda.weights import Config, GDN, Layer, QLinear, Weights


def _model():
    gen = torch.Generator(device="cuda").manual_seed(9)
    dev = "cuda"

    def qlinear(n, k):
        words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen,
                              device=dev, dtype=torch.int64).to(torch.int32)
        scales = (torch.rand(n, k // 64, generator=gen, device=dev) * 0.003 + 0.001).bfloat16()
        biases = (torch.rand(n, k // 64, generator=gen, device=dev) * 0.003 - 0.0015).bfloat16()
        return QLinear(words, scales, biases)

    norm = torch.ones(128, device=dev, dtype=torch.bfloat16)
    gdn = GDN(qlinear(384, 128), qlinear(128, 128), qlinear(1, 128), qlinear(1, 128),
              qlinear(128, 128), torch.randn(384, 4, generator=gen, device=dev).bfloat16() * 0.1,
              torch.zeros(1, device=dev), torch.zeros(1, device=dev), norm)
    layer = Layer(True, norm, norm, gdn, None, qlinear(128, 128), qlinear(128, 128),
                  qlinear(128, 128))
    config = Config(hidden=128, intermediate=128, layers=1, heads=1, kv_heads=1,
                    head_dim=128, vocab=256, k_heads=1, v_heads=1, dk=128, dv=128,
                    conv_kernel=4, interval=4, eps=1e-6, rope_dims=32,
                    rope_theta=10000000.0, eos=(0,))
    return Weights(config, qlinear(256, 128), [layer], norm, qlinear(256, 128),
                   torch.ones(16, device=dev))


def _serial(w, path):
    st = State(w)
    results = {}
    for row, tok in path:
        logits, record = tree_forward(w, torch.tensor([tok], device="cuda", dtype=torch.int32), [-1], st)
        results[row] = logits[0]
        commit(st, record, [0])
    return results, st


def test_gdn_tree_forward_and_commit_match_serial():
    w = _model()
    tokens = torch.tensor([7, 8, 9, 10], device="cuda", dtype=torch.int32)
    parents = [-1, 0, 0, 1]
    state = State(w)
    logits, record = tree_forward(w, tokens, parents, state)
    assert state.pos == 0
    branch, serial_state = _serial(w, [(0, 7), (1, 8), (3, 10)])
    for row in (0, 1, 3):
        assert torch.equal(logits[row], branch[row]), f"row {row} differs"
    commit(state, record, [0, 1, 3])
    assert state.pos == serial_state.pos == 3
    assert torch.equal(state.rec[0], serial_state.rec[0])
    assert torch.equal(state.conv[0], serial_state.conv[0])
    sibling, _ = _serial(w, [(0, 7), (2, 9)])
    assert torch.equal(logits[2], sibling[2])


def test_eos_pending_stops_both_decode_paths():
    w = _model()
    st = State(w)
    assert serial_decode(w, st, 0, 32, None).tokens == [0]
    assert draft_decode(w, st, [7], 0, 32, None, draft=None).tokens == [0]
    assert st.pos == 0


def test_tiled_weights_give_the_same_bits():
    """Regrouped weights (qmm_fast.prepare) change the memory layout only: logits and commits match."""

    import copy

    from tensorfold.families.qwen3_5.cuda import qmm_fast

    w = _model()
    tiled = copy.deepcopy(w)
    qmm_fast.prepare(tiled, fuse=False)
    assert tiled.head.layout == "tiled" and tiled.layers[0].gate.layout == "tiled"
    tokens = torch.tensor([7, 8, 9, 10, 11], device="cuda", dtype=torch.int32)
    for parents in ([-1, 0, 0, 1, 3], [-1, 0, 1, 2, 3], [-1]):
        toks = tokens[:len(parents)]
        a, ra = tree_forward(w, toks, parents, State(w))
        b, rb = tree_forward(tiled, toks, parents, State(tiled))
        assert torch.equal(a, b)
    sa, sb = State(w), State(tiled)
    _, ra = tree_forward(w, tokens, [-1, 0, 1, 2, 3], sa)
    _, rb = tree_forward(tiled, tokens, [-1, 0, 1, 2, 3], sb)
    commit(sa, ra, [0, 1, 2])
    commit(sb, rb, [0, 1, 2])
    assert torch.equal(sa.rec[0], sb.rec[0]) and torch.equal(sa.conv[0], sb.conv[0])


def test_stacked_projections_keep_windows_exact():
    """With [z | b | a] stacked and tiled, window rows still equal serial steps on every path."""

    from tensorfold.families.qwen3_5.cuda import qmm_fast

    w = _model()
    qmm_fast.prepare(w, fuse=True)
    assert w.layers[0].gdn.zba is not None and w.layers[0].gdn.zba.layout == "tiled"
    tokens = torch.tensor([7, 8, 9, 10], device="cuda", dtype=torch.int32)
    parents = [-1, 0, 0, 1]
    logits, record = tree_forward(w, tokens, parents, State(w))
    branch, serial_state = _serial(w, [(0, 7), (1, 8), (3, 10)])
    for row in (0, 1, 3):
        assert torch.equal(logits[row], branch[row]), f"row {row} differs"
    st = State(w)
    _, record = tree_forward(w, tokens, parents, st)
    commit(st, record, [0, 1, 3])
    assert torch.equal(st.rec[0], serial_state.rec[0]) and torch.equal(st.conv[0], serial_state.conv[0])


def test_decode_trace_records_every_round_and_keeps_serial_tokens():
    """``draft_decode(trace=...)``: one record per round, stop reasons, candidate coverage; tokens unchanged."""

    import dataclasses

    import numpy as np

    base = _model()
    w = Weights(dataclasses.replace(base.config, layers=64), base.embed, [base.layers[0]] * 64, base.norm,
                base.head, base.inv_freq)
    serial = serial_decode(w, State(w), 5, 24, None, stop_eos=False)
    ref = serial.tokens

    class Drafter:
        """Three right guesses, a wrong fourth, and a wrong sibling at depth one."""

        last_candidates = None

        def propose_tree(self, pending, context_length, max_nodes, sampling):
            k = context_length
            right = [ref[k + j] if k + j < len(ref) else 1 for j in range(4)]
            self.last_candidates = np.array([[t] * 16 for t in right])
            guesses = right[:3] + [(right[3] + 1) % 256, (right[0] + 1) % 256]
            return guesses[:max_nodes], [-1, 0, 1, 2, -1][:max_nodes]

        def add_taps(self, taps):
            pass

        def in_vocab(self, token):
            return True

    trace: list[dict] = []
    drafted = draft_decode(w, State(w), [], 5, 24, None, Drafter(), max_rows=8, allow_copy=False,
                           stop_eos=False, trace=trace)
    assert drafted.tokens == ref
    assert len(trace) == drafted.rounds
    assert sum(t["accepted"] for t in trace) == drafted.accepted_drafts
    assert {t["stop"] for t in trace} <= {"miss", "horizon", "length", "eos"}
    misses = [t for t in trace if t["stop"] == "miss"]
    assert misses and all(t["accepted"] == 3 and t["candidate_hit"] and not t["vocab_miss"] for t in misses)
    assert all(t["nodes_per_depth"][0] == 2 and t["max_depth"] == 4 for t in misses)
