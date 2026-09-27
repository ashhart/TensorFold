"""Flash Next on a Mac whose GPU cannot hold the n-gram tables: rows read from the checkpoint's memory map give the
GPU tables' bits, and the tensor-unit projections give a row the same bits at any row count (GPU)."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():
    pytest.skip("needs a Metal GPU", allow_module_level=True)

import mlx.nn as nn  # noqa: E402

from tensorfold.families.qwen3_5 import tensor_units  # noqa: E402
from tensorfold.families.qwen4_exp import decode, host_table  # noqa: E402
from tensorfold.kernels.qwen.dense.v1 import lane_qmm  # noqa: E402
from tensorfold.kernels.qwen.flash_next.v1 import embed  # noqa: E402

DIMS = 160


class _Emb:
    """What PleTables reads from an NGramEmbedding."""

    def __init__(self, shards, host=None):
        self.dims, self.shards, self.host = DIMS, shards, host


def _checkpoint(tmp_path, counts):
    """4-bit embedding shards of ``counts`` rows, saved as ``emb.shard_{i}`` across two safetensors files."""

    mx.random.seed(0)
    shards = []
    for rows in counts:
        e = nn.Embedding(rows, DIMS)
        e.weight = (0.05 * mx.random.normal((rows, DIMS))).astype(mx.bfloat16)
        shards.append(nn.QuantizedEmbedding.from_embedding(e, group_size=32, bits=4))
    half = len(shards) // 2
    for f, part in enumerate((range(half), range(half, len(shards)))):
        mx.save_safetensors(str(tmp_path / f"model-{f}.safetensors"),
                            {f"emb.shard_{i}.{k}": shards[i][k] for i in part for k in ("weight", "scales", "biases")})
    return shards


def test_host_rows_give_the_gpu_tables_bits(tmp_path):
    counts = [37, 5, 64, 19, 3, 41, 28, 11, 50, 7, 1, 33, 20, 9, 16, 2]   # 16 shards: 2 a GPU table group
    shards = _checkpoint(tmp_path, counts)
    table = host_table.from_checkpoint(tmp_path, "emb", len(counts))
    assert table.rows == sum(counts)
    ids = np.random.default_rng(1).integers(0, table.rows, (5, 16))
    starts = np.cumsum([0] + counts)
    shard = np.searchsorted(starts, ids.reshape(-1), side="right") - 1
    by_module = mx.concatenate([shards[s](mx.array([int(i - starts[s])])) for s, i in zip(shard, ids.reshape(-1))])
    host = embed.ple_lookup(ids, embed.PleTables(_Emb([], table)))
    gpu = embed.ple_lookup(ids, embed.PleTables(_Emb(shards)))
    assert mx.array_equal(host, by_module.reshape(5, 16 * DIMS))
    assert mx.array_equal(host, gpu)


@pytest.mark.skipif(not tensor_units(), reason="lane_qmm needs tensor units")
@pytest.mark.parametrize("n", [2560, 2592, 2600])        # tiles of 64 columns, of 32, untiled
def test_lane_projection_rows_do_not_depend_on_the_row_count(n):
    mx.random.seed(n)
    linear = nn.QuantizedLinear(2560, n, bias=False, group_size=32, bits=4)
    x = mx.random.normal((200, 2560)).astype(mx.bfloat16)
    full = decode._lane_project(x, linear)                  # 128 rows, then 72
    untiled = lane_qmm.lane_matmul(x[:128], linear.weight, lane_qmm.pack_scales(linear.scales, linear.biases),
                                   group=32)
    assert mx.array_equal(full[:128], untiled)
    for rows in (1, 3, 17, 129):
        assert mx.array_equal(decode._lane_project(x[:rows], linear), full[:rows])
