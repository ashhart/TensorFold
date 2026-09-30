"""The direct checkpoint reader's host reads return safetensors' own tensors, O_DIRECT or not (no GPU needed)."""

import errno
import json
import os
import struct

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from tensorfold.cuda import direct_read  # noqa: E402


def tensors() -> dict:
    g = torch.Generator().manual_seed(5)
    return {
        "a.weight": torch.randint(-(2**31), 2**31 - 1, (37, 11), generator=g, dtype=torch.int64).to(torch.int32),
        "b.words": torch.randint(0, 2**32 - 1, (9, 5), generator=g, dtype=torch.int64).to(torch.uint32),
        "c.scales": torch.randn((5, 7), generator=g).to(torch.bfloat16),
        "d.half": torch.randn((3, 3), generator=g).to(torch.float16),
        "e.odd": torch.randint(0, 255, (4099,), generator=g, dtype=torch.int64).to(torch.uint8),
        "f.mask": torch.rand((13,), generator=g) > 0.5,
        "g.big": torch.randn((1000, 29), generator=g),
        "h.empty": torch.zeros((0, 4)),
        "z.last": torch.randn((3,), generator=g),        # ends the file: the aligned span runs past EOF
    }


def _same(a, b, name):
    assert a.dtype == b.dtype and a.shape == b.shape, name
    raw = [t.contiguous().view(torch.uint8).numpy().tobytes() for t in (a, b)]   # torch.equal lacks some dtypes
    assert raw[0] == raw[1], name


@pytest.mark.parametrize("direct", [True, False])
def test_host_reads_match_safe_open(tmp_path, monkeypatch, direct):
    from safetensors import safe_open

    monkeypatch.setattr(direct_read, "PIECE", 3 * 4096)          # tensors span several reads
    path = tmp_path / "model.safetensors"
    safetensors_torch.save_file(tensors(), str(path))
    files = direct_read.SafeTensors([path])
    files.reader.direct = files.reader.direct and direct
    with safe_open(str(path), framework="pt") as f:
        assert sorted(files.keys()) == sorted(f.keys())
        for name in f.keys():
            _same(files.get(name), f.get_tensor(name), name)
    if direct and not files.reader.direct:
        pytest.skip("this file system refuses O_DIRECT: only the fallback was checked")


def _write_unaligned(path, tensors):
    """A safetensors file whose data starts 13 bytes past a 4 KiB boundary, so no tensor is 8-byte aligned."""

    header, blobs, at = {}, [], 0
    for name, t in tensors.items():
        raw = t.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
        dtype = {torch.int32: "I32", torch.bfloat16: "BF16", torch.float32: "F32", torch.uint8: "U8"}[t.dtype]
        header[name] = {"dtype": dtype, "shape": list(t.shape), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    text = json.dumps(header).encode()
    text += b" " * (4096 - (8 + len(text)) % 4096 + 13)
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def test_unaligned_tensors_read_back(tmp_path, monkeypatch):
    monkeypatch.setattr(direct_read, "PIECE", 3 * 4096)
    want = {k: v for k, v in tensors().items() if v.dtype in (torch.int32, torch.bfloat16, torch.float32, torch.uint8)}
    path = tmp_path / "model.safetensors"
    _write_unaligned(path, want)
    files = direct_read.SafeTensors([path])
    for name, t in want.items():
        _same(files.get(name), t, name)                          # .get views each unaligned start as its dtype


@pytest.mark.parametrize("direct", [True, False])
def test_a_truncated_file_is_a_short_read(tmp_path, direct):
    path = tmp_path / "model.safetensors"
    safetensors_torch.save_file(tensors(), str(path))
    files = direct_read.SafeTensors([path])
    files.reader.direct = files.reader.direct and direct
    with open(path, "r+b") as f:                                # cut to a page boundary inside "g.big"
        f.truncate((files.where["g.big"][1] + 3 * 4096) // 4096 * 4096)
    with pytest.raises(IOError, match="short read"):
        files.get("g.big")


def test_a_refused_open_falls_back_for_good(tmp_path, monkeypatch):
    if not hasattr(os, "O_DIRECT"):
        pytest.skip("no O_DIRECT on this platform: every read is buffered")
    path = tmp_path / "model.safetensors"
    safetensors_torch.save_file(tensors(), str(path))
    real_open = os.open

    def refuse(p, flags, *args):
        if flags & os.O_DIRECT:
            raise OSError(errno.EINVAL, "O_DIRECT refused")
        return real_open(p, flags, *args)

    monkeypatch.setattr(os, "open", refuse)
    files = direct_read.SafeTensors([path])
    _same(files.get("g.big"), tensors()["g.big"], "g.big")
    assert files.reader.direct is False
    _same(files.get("z.last"), tensors()["z.last"], "z.last")


@pytest.mark.parametrize("run,gap", [(1 << 20, 1 << 20), (4096, 0)])      # one read, then about one a tensor
def test_read_ahead_on_the_host_matches_safe_open(tmp_path, run, gap):
    from safetensors import safe_open

    path = tmp_path / "model.safetensors"
    safetensors_torch.save_file(tensors(), str(path))
    files = direct_read.SafeTensors([path])
    ahead = direct_read.ReadAhead(files.reader, threads=4, run=run, gap=gap)
    items = [(name, p, b, b + n, (dtype, shape)) for name, (p, b, n, dtype, shape) in files.where.items()]
    ahead.queue(items)
    ahead.drop(["h.empty"])
    with safe_open(str(path), framework="pt") as f:
        for name, (_, _, _, dtype, shape) in files.where.items():
            got = ahead.take(name)
            if name == "h.empty":
                assert got is None                                      # dropped: nothing to take
                continue
            _same(got.view(direct_read.DTYPES[dtype]).reshape(shape), f.get_tensor(name), name)
    assert not ahead.ahead
    ahead.close()
