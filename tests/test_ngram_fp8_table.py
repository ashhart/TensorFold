"""FP8 n-gram rows (RadixArk's Flash Next NVFP4): a lookup gives bf16(e4m3 x scale), what dequantizing at load writes."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.host_table import FP8Table, read_header


def _shard(path, name: str, rows: np.ndarray) -> None:
    data = rows.tobytes()
    header = json.dumps({name: {"dtype": "F8_E4M3", "shape": list(rows.shape), "data_offsets": [0, len(data)]}})
    header = header.encode() + b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


@pytest.mark.torch
def test_fp8_rows_are_bf16_of_e4m3_times_the_table_scale(tmp_path):
    import torch

    rng = np.random.default_rng(0)
    shards = [rng.integers(0, 256, size=(r, 160), dtype=np.uint8) for r in (7, 5)]
    shards[0][0, :2] = (0x7F, 0xFF)                                  # e4m3's NaN codes
    files = []
    for i, rows in enumerate(shards):
        path, name = tmp_path / f"s{i}.safetensors", f"ple.shard_{i}.weight"
        _shard(path, name, rows)
        files.append((path, read_header(path)[name]))
    scale = float(torch.tensor(0.0374).to(torch.bfloat16))
    table = FP8Table(files, scale)
    assert (table.rows, table.width, table.bits) == (12, 160, 16)
    ids = np.array([0, 6, 7, 11, 3])
    got = torch.from_numpy(table.gather(ids).view(np.int16)).view(torch.bfloat16).float()
    rows = torch.from_numpy(np.concatenate(shards)[ids])
    want = (rows.view(torch.float8_e4m3fn).float() * scale).to(torch.bfloat16).float()
    assert torch.equal(got.isnan(), (rows & 0x7F) == 0x7F) and torch.equal(got.isnan(), want.isnan())
    assert torch.equal(got.nan_to_num(), want.nan_to_num())
    with pytest.raises(ValueError):
        table.gather(np.array([12]))


@pytest.mark.torch
def test_nvfp4_rows_are_bf16_of_code_times_scales_and_layouts_do_not_mix(tmp_path):
    import torch

    from tensorfold.families.qwen4_exp.host_table import NVFP4Table, open_table

    rng = np.random.default_rng(1)
    codes = rng.integers(0, 256, size=(9, 80), dtype=np.uint8)
    scales = rng.integers(0x20, 0x48, size=(9, 10), dtype=np.uint8)
    path = tmp_path / "s.safetensors"
    blobs = {"t.shard_0.weight": ("U8", codes), "t.shard_0.weight_scale": ("F8_E4M3", scales)}
    header, data, at = {}, b"", 0
    for name, (dtype, arr) in blobs.items():
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [at, at + arr.nbytes]}
        data, at = data + arr.tobytes(), at + arr.nbytes
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    g = 0.0371
    table = open_table(tmp_path, [("s.safetensors", "t.shard_0")], lambda name: g)
    assert isinstance(table, NVFP4Table) and (table.rows, table.width) == (9, 160)
    ids = np.array([8, 0, 4])
    e2m1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0.0, -.5, -1, -1.5, -2, -3, -4, -6])
    c = torch.from_numpy(codes[ids]).long()
    nib = torch.stack([c & 0xF, c >> 4], -1).reshape(3, -1)
    s = torch.from_numpy(scales[ids]).view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    want = (e2m1[nib] * s * g).to(torch.bfloat16).view(torch.int16).numpy()
    assert np.array_equal(table.gather(ids).view(np.int16), want)
