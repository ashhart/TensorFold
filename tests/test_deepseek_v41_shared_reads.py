"""DeepSeek-V4.1-Flash's shared reads: decode rows' unquantized matmuls on one gemv, Engram pages read ahead."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.families.deepseek_v41 import engram  # noqa: E402
from tensorfold.kernels.deepseek.v41 import rows as KV  # noqa: E402


@pytest.fixture
def gpu():
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def window(rows: int, k: int, dtype) -> mx.array:
    x = mx.random.normal((rows, k), key=mx.random.key(2))
    return (x * mx.random.uniform(0.1, 10, (rows, 1), key=mx.random.key(3))).astype(dtype)


def one_row(x: mx.array, w: mx.array, out_dtype, groups: int) -> mx.array:
    return mx.concatenate([KV.matmul(x[r:r + 1], w, out_dtype, groups=groups) for r in range(int(x.shape[0]))])


@pytest.mark.parametrize("dtype,groups", [(mx.float32, 1), (mx.bfloat16, 4)])
@pytest.mark.parametrize("rows", [2, 7, 16])
def test_gemv_rows_give_one_row_bits(gpu, dtype, groups, rows):
    """The shared-read gemv, at the config it matched, equals MLX's one-row matmul on every row."""

    from tensorfold.kernels.deepseek.v41.gemv import gemv_config, gemv_rows

    w = (mx.random.normal((64, 1024), key=mx.random.key(1)) * 0.05).astype(dtype)
    x = window(rows, 1024 * groups, dtype)
    config = gemv_config(w, dtype, dtype, groups)
    assert config is not None
    joint = gemv_rows(x, w, dtype, config[0], config[1], groups)
    assert mx.array_equal(joint, one_row(x, w, dtype, groups)).item()


def test_decode_rows_matmul_takes_one_shared_gemv(gpu, monkeypatch):
    """Inside ``decode_rows`` an unquantized matmul of several rows is one gemv call, still one-row exact."""

    from tensorfold.kernels.deepseek.v41 import gemv

    calls = []
    real = gemv.gemv_rows
    monkeypatch.setattr(gemv, "gemv_rows", lambda *a, **k: calls.append(1) or real(*a, **k))
    w = (mx.random.normal((64, 1024), key=mx.random.key(5)) * 0.05).astype(mx.float32)
    x = window(6, 1024, mx.float32)
    with KV.decode_rows():
        joint = KV.matmul(x, w, mx.float32)
    assert len(calls) == 1
    assert mx.array_equal(joint, one_row(x, w, mx.float32, 1)).item()


def write_table(path, rows: int, width: int) -> np.ndarray:
    data = np.random.default_rng(0).integers(0, 256, size=(rows, width), dtype=np.uint8)
    header = json.dumps({"w": {"dtype": "U8", "shape": [rows, width], "data_offsets": [0, data.nbytes]}}).encode()
    with open(path, "wb") as f:
        f.write(np.array([len(header)], dtype="<u8").tobytes() + header + data.tobytes())
    return data


def test_engram_gathers_read_their_cold_pages_once(tmp_path, monkeypatch):
    """A big Engram gather reads each cold page once ahead of the copy; a repeat or small gather reads none."""

    data = write_table(tmp_path / "t.safetensors", 4096, 64)
    table = engram.tensor_map(tmp_path / "t.safetensors", "w")
    reads = []
    real = os.pread
    monkeypatch.setattr(os, "pread", lambda fd, n, offset: reads.append(offset) or real(fd, n, offset))
    ids = np.random.default_rng(1).integers(0, 4096, size=engram.PREFETCH_ROWS + 50)
    assert np.array_equal(table.rows(ids), data[ids])
    offsets = table.base + ids.astype(np.int64) * 64
    pages = set((offsets // engram.PAGE).tolist()) | set(((offsets + 63) // engram.PAGE).tolist())
    assert sorted(o // engram.PAGE for o in reads) == sorted(pages)
    reads.clear()
    assert np.array_equal(table.rows(ids), data[ids])
    assert np.array_equal(table.rows(ids[:8]), data[ids[:8]])
    assert not reads
