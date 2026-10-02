"""Real read-only packed-row gathers stay byte-exact after a fake-libc partial pin with OS-page accounting."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import ngram_pages as pages
from tensorfold.families.qwen4_exp.cuda.exl3_pack import NgramTable


def test_partial_pin_charges_pages_and_preserves_real_gather(tmp_path, monkeypatch):
    words, rows = 61, 5000
    data = np.arange(rows * words, dtype=np.int16).reshape(rows, words)
    path = tmp_path / "ngram.safetensors"
    path.write_bytes(data.tobytes())
    entries = {"t.trellis": (path.name, 0, data.nbytes, "I16", list(data.shape))}
    tensors = {"t.head_bias": torch.zeros(4), "t.head_offsets": torch.zeros(2, dtype=torch.int64),
               "t.head_vocab_sizes": torch.ones(2, dtype=torch.int64), "t.layer_multipliers": torch.ones(2)}
    table = NgramTable(SimpleNamespace(dir=tmp_path, entry=entries.__getitem__, get=tensors.__getitem__), "t.", 0, "cpu")
    calls = []
    monkeypatch.setattr(pages, "libc", lambda: SimpleNamespace(mlock=lambda at, n: calls.append((at, n)) or 0,
                                                              munlock=lambda *args: 0))
    table.RUN_BYTES = 1000 * words * 2
    ids = np.array([0, 4999, 2345, 2, 2])
    before = table.gather(ids).tobytes()
    budget = 2 * table.RUN_BYTES + 100
    charged = table.lock_runs(budget)
    assert 0 < charged <= budget and charged == sum(size for _, size in calls)
    assert all(at % pages.PAGE == 0 and size % pages.PAGE == 0 for at, size in calls)
    assert table.gather(ids).tobytes() == before == data[ids].tobytes()
