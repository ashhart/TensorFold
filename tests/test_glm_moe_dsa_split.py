"""Full GLM-5.3 over four ranks: every split rule's parts reassemble to the checkpoint bytes (CPU, no weights)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorfold.families.glm_moe_dsa.cuda import split

WORLD = 4
CASES = [  # (tensor name, shape, numpy dtype) with the checkpoint's shapes
    ("model.layers.3.mlp.experts.7.gate_proj.trellis", [384, 128, 48], np.int16),   # 3-bit, out-split
    ("model.layers.3.mlp.experts.7.up_proj.svh", [2048], np.float16),
    ("model.layers.3.mlp.experts.7.down_proj.trellis", [128, 384, 32], np.int16),   # 2-bit, in-split
    ("model.layers.3.mlp.experts.7.down_proj.suh", [2048], np.float16),
    ("model.layers.3.mlp.shared_experts.gate_proj.trellis", [384, 128, 80], np.int16),
    ("model.layers.0.mlp.down_proj.trellis", [768, 384, 80], np.int16),
    ("model.layers.3.self_attn.q_b_proj.trellis", [128, 1024, 80], np.int16),
    ("model.layers.3.self_attn.o_proj.trellis", [1024, 384, 80], np.int16),
    ("model.layers.3.self_attn.o_proj.suh", [16384], np.float16),
    ("model.layers.3.self_attn.kv_b_proj.weight", [28672, 64], np.float16),        # rows by head (cols trimmed)
    ("lm_head.weight", [154880, 8], np.float16),
]


def _reassemble(kind: str, parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts, axis=0 if kind == "row" else 1)


@pytest.mark.parametrize("name,shape,dtype", CASES)
def test_parts_reassemble_to_the_checkpoint(name, shape, dtype):
    rng = np.random.default_rng(0)
    full = rng.integers(-30000, 30000, size=shape).astype(dtype) if dtype == np.int16 else \
        rng.standard_normal(shape).astype(dtype)
    kind = split.rule(name)
    assert kind in ("row", "dim1")
    raw = np.frombuffer(full.tobytes(), dtype=np.uint8)
    parts = []
    for r in range(WORLD):
        data, part_shape = split.split_bytes(raw, list(shape), full.itemsize, kind, r, WORLD, name)
        parts.append(np.frombuffer(data.tobytes(), dtype=dtype).reshape(part_shape))
    assert np.array_equal(_reassemble(kind, parts), full)


@pytest.mark.parametrize("name", [
    "model.layers.3.self_attn.q_a_proj.trellis", "model.layers.3.self_attn.kv_a_proj_with_mqa.svh",
    "model.layers.5.self_attn.indexer.wq_b.trellis", "model.layers.5.self_attn.indexer.wk.weight",
    "model.layers.3.mlp.gate.weight", "model.layers.3.mlp.experts.9.up_proj.suh",
    "model.layers.3.mlp.experts.9.down_proj.svh", "model.layers.3.mlp.experts.9.gate_proj.mul1",
    "model.embed_tokens.weight", "model.layers.78.eh_proj.weight", "model.layers.78.enorm.weight"])
def test_replicated(name):
    assert split.rule(name) == "rep"


def test_a_straddling_cut_is_refused():
    # 40 tiles (640 columns) over four ranks is 10 tiles each: not whole 128-wide Hadamard blocks
    with pytest.raises(ValueError, match="Hadamard"):
        split._check_cut("x.q_b_proj.trellis", "dim1", [128, 40, 80], WORLD)


def test_unknown_tensor_is_refused():
    with pytest.raises(ValueError, match="no split rule"):
        split.rule("model.layers.3.self_attn.mystery.weight")
