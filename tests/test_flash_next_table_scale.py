"""Every Mac n-gram lookup applies the scalar after dequantization, exactly once."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn

from tests.test_flash_next_host_table import DIMS, _checkpoint
from tensorfold.families.qwen4_exp import host_table
from tensorfold.families.qwen4_exp.model import NGramEmbedding
from tensorfold.kernels.qwen.flash_next.v1 import embed


@pytest.mark.parametrize("value", [0.000199, 1.0])
def test_scaled_rows_matches_bf16_product_bits(value):
    mx.random.seed(17)
    rows = (mx.random.normal((257, 160)) * 3000).astype(mx.bfloat16)
    scale = mx.array(value, dtype=mx.bfloat16)
    actual = embed.scaled_rows(rows, float(scale.item()))
    expected = rows * scale
    assert bool(mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item())


@pytest.mark.parametrize("storage", ["gpu", "host", "ssd"])
@pytest.mark.parametrize("value", [1.0, 0.0002])
def test_scaled_rows_equal_lookup_then_bf16_multiply(tmp_path, storage, value):
    counts = [5] * 16
    shards = _checkpoint(tmp_path, counts)
    emb = NGramEmbedding.__new__(NGramEmbedding)
    nn.Module.__init__(emb)
    emb.dims, emb.heads, emb.shards = DIMS, 16, shards
    emb.shard_starts = np.cumsum([0] + counts).tolist()
    emb.quant_bits, emb.quant_group = 4, 32
    emb.host = None if storage == "gpu" else host_table.from_checkpoint(tmp_path, "emb", 16, ssd=storage == "ssd")
    scale = mx.array([value], dtype=mx.bfloat16)
    emb.table_scale = float(scale.item())
    ids = np.random.default_rng(4).integers(0, sum(counts), (3, 16))
    values = mx.concatenate([shards[int(i // 5)](mx.array([int(i % 5)])) for i in ids.reshape(-1)])
    want = (values * scale).reshape(3, 16 * DIMS)
    assert mx.array_equal(emb(ids), want)
    emb.__dict__["fused_tables"] = embed.PleTables(emb)
    assert mx.array_equal(emb(ids), want)
