"""Nemotron's MTP draft head reads the vocabulary head's token rows, also after the lane matmul tiled that head."""

import types

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.families.nemotron_h.model import NemotronH  # noqa: E402
from tensorfold.kernels.qwen.dense.v1 import lane_qmm  # noqa: E402


@pytest.mark.parametrize("bits", [4, 3])
def test_draft_head_rows_from_a_lane_tiled_head(bits):
    """The draft head untiles a lane-tiled head at the head's own width and takes the draft ids' token rows."""

    mx.random.seed(bits)
    linear = nn.Linear(128, 131072, bias=False)            # every draft id (at most 131,056) is a row
    linear.set_dtype(mx.bfloat16)
    head = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=bits)
    original = mx.array(head.weight)
    head.weight = lane_qmm.tile_weight(head.weight, lane_qmm.NT, bits=bits)
    object.__setattr__(head, "_lane_tiled", True)
    object.__setattr__(head, "_lane_nt", lane_qmm.NT)
    ids, draft = NemotronH._load_draft_head(types.SimpleNamespace(model=types.SimpleNamespace(lm_head=head)))
    rows = ids.astype(mx.int32)
    assert draft.bits == bits and draft.weight.shape == (ids.size, original.shape[1])
    assert bool(mx.all(draft.weight == original[rows]).item())
    assert bool(mx.all(draft.scales == head.scales[rows]).item())
