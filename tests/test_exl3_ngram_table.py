"""EXL3's n-gram table reports its bytes, as every n-gram table the Flash Next CUDA engine prefetches does."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.torch


def test_the_exl3_ngram_table_reports_the_bytes_of_its_shards(tmp_path):
    import torch

    from tensorfold.families.qwen4_exp.cuda import exl3_pack

    words, rows = 1 + 160 * 4 // 16, [3, 5]                    # 4-bit rows: a scale word and 160 values
    data = np.arange(sum(rows) * words, dtype=np.int16).reshape(sum(rows), words)
    (tmp_path / "ngram.safetensors").write_bytes(data.tobytes())
    entries = {f"t.shard_{i}.trellis": ("ngram.safetensors", 2 * words * sum(rows[:i]),
                                        2 * words * sum(rows[:i + 1]), "I16", [n, words]) for i, n in enumerate(rows)}
    tensors = {"t.head_bias": torch.zeros(4), "t.head_offsets": torch.zeros(2, dtype=torch.int64),
               "t.head_vocab_sizes": torch.ones(2, dtype=torch.int64), "t.layer_multipliers": torch.ones(2)}
    pk = SimpleNamespace(dir=tmp_path, entry=entries.__getitem__, get=tensors.__getitem__)
    table = exl3_pack.NgramTable(pk, "t.", 2, "cpu")
    assert table.nbytes == data.nbytes
    assert table.gather(np.array([0, 7])).tobytes() == data[[0, 7]].tobytes()
