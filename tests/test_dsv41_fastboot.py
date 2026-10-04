"""Prepared folders (fastboot): a tree round-trips with the same bytes, views and structure; keys, checksums, the
storage guard and pruning (host only). Follows the cases of Jay Leaton's tests/test_dsv41_fastboot.py
(deepseek-v41-tensorfold-spark, MIT; see THIRD_PARTY_NOTICES.md)."""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.deepseek_v41.cuda import fastboot as fb  # noqa: E402
from tensorfold.families.deepseek_v41.cuda.weights import HCW  # noqa: E402


def tree():
    base = torch.arange(4096, dtype=torch.int32)
    a, b = base[:2048], base[2048:]                      # two views of one storage (like GroupedLinear's words)
    h = HCW(torch.randn(4, 8), torch.randn(4), torch.randn(3), torch.randn(8).to(torch.bfloat16))
    return {"hc": h, "views": [a, b], "shape": (16, 32), "t": torch.randn(5, 7).t(), "name": "x", "n": None}


def same(x, y):
    if isinstance(x, torch.Tensor):
        return (x.dtype == y.dtype and x.shape == y.shape and x.stride() == y.stride()
                and x.storage_offset() == y.storage_offset() and torch.equal(x, y))
    if isinstance(x, (list, tuple)):
        return type(x) is type(y) and len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if hasattr(x, "__dict__"):
        return type(x) is type(y) and same(vars(x), vars(y))
    return x == y


def test_round_trip_keeps_bytes_views_and_structure(tmp_path, monkeypatch):
    monkeypatch.setenv("TF_DSV41_PREPARED_VERIFY", "full")
    t = tree()
    d = tmp_path / "m" / "k" / "rank0"
    fb.save(t, d, {"rank": 0, "model": "m"})
    got, st = fb.read(d, device="cpu")
    assert same(t, got) and st["verified"] == st["chunks"]
    a, b = got["views"]
    assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()   # still one storage


def test_guard_refuses_a_view_into_a_bigger_buffer(tmp_path):
    big = torch.zeros(1 << 20, dtype=torch.uint8)
    with pytest.raises(ValueError, match="refusing"):
        fb.save({"x": big[:10]}, tmp_path / "a", {"rank": 0})


def test_corrupt_chunk_fails_and_truncation_is_invalid(tmp_path, monkeypatch):
    monkeypatch.setenv("TF_DSV41_PREPARED_VERIFY", "full")
    d = tmp_path / "m" / "k" / "rank0"
    key = {"rank": 0, "model": "m", "format": fb.FORMAT}
    fb.save(tree(), d, key)
    assert fb.valid(d, key) == (True, "ok")
    with open(d / "data.bin", "r+b") as f:
        f.seek(100)
        f.write(b"\xff\xfe")
    with pytest.raises(ValueError, match="checksum"):
        fb.read(d, device="cpu")
    os.truncate(d / "data.bin", 4096)
    assert fb.valid(d, key)[0] is False
    assert "key differs" in fb.valid(d, dict(key, rank=1))[1]


def test_key_follows_files_config_rank_and_knobs(tmp_path, monkeypatch):
    m = tmp_path / "model"
    m.mkdir()
    (m / "config.json").write_text(json.dumps({"a": 1}))
    (m / "model-1.safetensors").write_bytes(b"x" * 10)
    k0 = fb.weights_key(m, 0, 2, draft=True, device="cpu")
    assert fb.digest(k0) == fb.digest(fb.weights_key(m, 0, 2, draft=True, device="cpu"))
    assert fb.digest(k0) != fb.digest(fb.weights_key(m, 1, 2, draft=True, device="cpu"))
    assert fb.digest(k0) != fb.digest(fb.weights_key(m, 0, 2, draft=False, device="cpu"))
    monkeypatch.setenv("TF_FOLD_SHARED", "1")
    assert fb.digest(k0) != fb.digest(fb.weights_key(m, 0, 2, draft=True, device="cpu"))
    monkeypatch.delenv("TF_FOLD_SHARED")
    (m / "config.json").write_text(json.dumps({"a": 2}))
    assert fb.digest(k0) != fb.digest(fb.weights_key(m, 0, 2, draft=True, device="cpu"))
    (m / "config.json").write_text(json.dumps({"a": 1}))
    os.utime(m / "model-1.safetensors", ns=(1, 1))      # same size, re-written: a different key
    assert fb.digest(k0) != fb.digest(fb.weights_key(m, 0, 2, draft=True, device="cpu"))


def test_prune_removes_other_folders_of_the_rank(tmp_path):
    key = {"rank": 0, "model": "m", "what": "weights"}
    old = tmp_path / "m" / "aaaa" / "rank0"
    new = tmp_path / "m" / "bbbb" / "rank0"
    other = tmp_path / "m" / "cccc" / "rank1"
    for d in (old, other):
        fb.save({"x": torch.ones(3)}, d, dict(key, rank=int(d.name[-1])))
    fb.prune(new)                                        # before writing new: rank 0's old folder goes
    assert not old.exists() and other.exists()
