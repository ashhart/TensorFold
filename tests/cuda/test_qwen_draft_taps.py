"""Configured tap order is identical in prompt and verification capture paths."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_qwen27_prefill import _model

from tensorfold.families.qwen3_5.cuda.forward import State, multi_tree_forward, tree_forward
from tensorfold.families.qwen3_5.cuda.prefill import prefill_chunk, prefill_rows


@pytest.mark.parametrize("path", ["prefill", "prefill_rows", "tree", "multi_tree"])
def test_tap_columns_follow_training_order(path):
    w = _model()
    tokens = torch.tensor([7, 8, 9], dtype=torch.int32, device="cuda")

    def capture(layers):
        w.tap_layers = layers
        state = State(w)
        if path == "prefill":
            return prefill_chunk(w, tokens, state, capture_taps=True)[1]
        if path == "prefill_rows":
            return prefill_rows(w, [(tokens.tolist(), state, 0)], capture_taps=True)[0][1]
        if path == "tree":
            return tree_forward(w, tokens, [-1, 0, 1], state, capture_taps=True)[2]
        return multi_tree_forward(w, [(tokens.tolist(), [-1, 0, 1], state)], capture_taps=True)[2]

    first, last = capture((0,)), capture((1,))
    got = capture((1, 0))
    assert got.shape == (3, 256)
    assert torch.equal(got[:, :128], last)
    assert torch.equal(got[:, 128:], first)
