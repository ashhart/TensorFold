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


@pytest.mark.parametrize("ssd", [False, True])
def test_host_rows_give_the_gpu_tables_bits(tmp_path, ssd):
    counts = [37, 5, 64, 19, 3, 41, 28, 11, 50, 7, 1, 33, 20, 9, 16, 2]   # 16 shards: 2 a GPU table group
    shards = _checkpoint(tmp_path, counts)
    table = host_table.from_checkpoint(tmp_path, "emb", len(counts), ssd=ssd)
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


def _linear(k, n, bits, group):
    layer = nn.QuantizedLinear(k, n, bias=False, group_size=group, bits=bits)
    layer.set_dtype(mx.bfloat16)                               # bf16 scales, as the checkpoints have
    return layer


@pytest.mark.skipif(not tensor_units(), reason="lane_qmm needs tensor units")
@pytest.mark.parametrize("bits", [8, 5, 3])
def test_lane_projection_tiles_other_widths_32_wide(bits):
    """A projection of another width tiles 32 wide even when its rows divide by 64, with the untiled bits."""

    mx.random.seed(bits)
    linear = _linear(2560, 2560, bits, 64)
    x = mx.random.normal((40, 2560)).astype(mx.bfloat16)
    plain = lane_qmm.lane_matmul(x, linear.weight, lane_qmm.pack_scales(linear.scales, linear.biases))
    assert mx.array_equal(decode._lane_project(x, linear), plain)


def test_linears_the_lane_matmul_cannot_read_are_named():
    model = nn.Module()
    model.layers = [_linear(256, n, b, g) for n, b, g in ((64, 4, 32), (1, 4, 32), (64, 8, 64), (64, 8, 32))]
    model.layers += [_linear(256, 64, 4, 32), nn.Linear(256, 64)]
    model.layers[-2].mode = "mxfp4"
    head = nn.Module()
    head.fc = _linear(256, 64, 8, 32)
    want = {"8-bit g32": 2, "4-bit g32 mxfp4": 1}              # a one-row gate is read; a float linear is MLX's
    assert decode.unreadable(model, head, None) == want


def test_flash_next_refuses_them_before_building(monkeypatch):
    from tensorfold.families.qwen4_exp import model as q4
    from tensorfold.families.qwen4_exp import runtime

    fake = nn.Module()
    fake.layers = [_linear(256, 64, 8, 32)]
    seen = []
    monkeypatch.setattr(q4, "load", lambda path, **k: seen.append(k) or (fake, "tokenizer"))
    monkeypatch.setattr(decode, "DENSE", "lane")
    monkeypatch.setattr(runtime, "FlashNext", lambda *a, **k: pytest.fail("built before refusing"))
    with pytest.raises(SystemExit, match="1 8-bit g32 linears"):
        runtime.load("unused", drafts=0)
    monkeypatch.setattr(decode, "DENSE", "simd")
    monkeypatch.setattr(runtime, "FlashNext", lambda *a, **k: "built")
    monkeypatch.setattr(q4, "prefetch_ngrams", lambda model: None)
    assert runtime.load("unused", drafts=0) == ("built", "tokenizer")
    assert runtime.load("unused", drafts=0, ple_on_ssd=True) == ("built", "tokenizer")
    assert [k["ple_on_ssd"] for k in seen[-2:]] == [False, True]
