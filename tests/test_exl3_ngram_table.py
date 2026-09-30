"""EXL3's n-gram table reports its bytes, as every n-gram table the Flash Next CUDA engine prefetches does."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.torch


@pytest.mark.parametrize("consolidated", [False, True])
@pytest.mark.parametrize("bits", [2, 4, 6, 8])
def test_the_exl3_ngram_table_reports_bytes_and_gathers_layouts(tmp_path, consolidated, bits):
    import torch

    from tensorfold.families.qwen4_exp.cuda import exl3_pack

    words, rows = 1 + 160 * bits // 16, [3, 5]                    # 4-bit rows: a scale word and 160 values
    data = np.arange(sum(rows) * words, dtype=np.int16).reshape(sum(rows), words)
    (tmp_path / "ngram.safetensors").write_bytes(data.tobytes())
    entries = {f"t.shard_{i}.trellis": ("ngram.safetensors", 2 * words * sum(rows[:i]),
                                        2 * words * sum(rows[:i + 1]), "I16", [n, words]) for i, n in enumerate(rows)}
    if consolidated:
        entries = {"t.trellis": ("ngram.safetensors", 0, data.nbytes, "I16", list(data.shape))}
    tensors = {"t.head_bias": torch.zeros(4), "t.head_offsets": torch.zeros(2, dtype=torch.int64),
               "t.head_vocab_sizes": torch.ones(2, dtype=torch.int64), "t.layer_multipliers": torch.ones(2)}
    pk = SimpleNamespace(dir=tmp_path, entry=entries.__getitem__, get=tensors.__getitem__)
    table = exl3_pack.NgramTable(pk, "t.", 2, "cpu")
    assert table.nbytes == data.nbytes
    ids = np.array([7, 0, 3, 2, 3, 5])
    assert table.gather(ids).tobytes() == data[ids].tobytes()
    assert table.gather([]).shape == (0, words)
    for bad in ([-1], [8], [0, 8]):
        with pytest.raises(IndexError, match="outside"):
            table.gather(bad)
    assert all(isinstance(a, np.memmap) and a.mode == "r" for a in table.words)


@pytest.mark.parametrize("shape,dtype,end", [([0, 61], "I16", 0), ([3, 61], "F16", 366),
                                            ([3, 61], "I16", 364), ([3, 62], "I16", 372)])
def test_ngram_table_rejects_invalid_packed_segments(tmp_path, shape, dtype, end):
    from tensorfold.families.qwen4_exp.cuda import exl3_pack

    (tmp_path / "rows").write_bytes(bytes(1024))
    pk = SimpleNamespace(dir=tmp_path, entry={"t.trellis": ("rows", 0, end, dtype, shape)}.__getitem__)
    with pytest.raises(ValueError):
        exl3_pack.NgramTable(pk, "t.", 128, "cpu")
