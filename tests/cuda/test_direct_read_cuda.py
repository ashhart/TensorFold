"""Loading on the GPU: direct reads return safetensors' tensors, Flash Next's reader returns its buffered bytes, and the
fused expert packing gives the bits of the former torch one."""

import json
import struct

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tensorfold.cuda import direct_read
from tensorfold.cuda import experts as grouped
from tensorfold.families.qwen4_exp.cuda import weights

DEV = "cuda"


def _pack_reference(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, gs: int) -> torch.Tensor:
    """``experts.pack`` as it was in torch (0.3.6.2, int64 nibble shuffle): the fused kernel's oracle."""

    ntw = grouped.NTW
    e, n, k8 = words.shape
    kg, nb, h = k8 * 8 // gs, n // grouped.COLS, gs // 32
    wpl = ntw * h
    out = torch.empty((e, nb, kg, 32 * wpl + 8 * ntw), dtype=torch.int32, device=words.device)
    for e0 in range(0, e, 32):
        v = words[e0:e0 + 32].to(torch.int64) & 0xFFFFFFFF
        w = torch.zeros_like(v)
        for i in range(8):
            w |= ((v >> (4 * i)) & 0xF) << (4 * (i // 2 + 4 * (i % 2)))
        w = torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)
        c = w.shape[0]
        w = w.view(c, nb, ntw, 8, kg, 4, h).permute(0, 1, 4, 2, 6, 3, 5).reshape(c, nb, kg, wpl // 4, 4, 32)
        out[e0:e0 + c, :, :, :32 * wpl] = w.permute(0, 1, 2, 3, 5, 4).reshape(c, nb, kg, 32 * wpl)
        sb = []
        for t in (scales, biases):
            u = t[e0:e0 + c].reshape(c, nb, ntw, 4, 2, kg).permute(0, 1, 5, 3, 2, 4).contiguous()
            sb.append(u.view(torch.int32).reshape(c, nb, kg, 4, ntw))
        out[e0:e0 + c, :, :, 32 * wpl:] = torch.cat(sb, dim=-1).reshape(c, nb, kg, 8 * ntw)
    return out


@pytest.mark.parametrize("gs", [32, 64])
@pytest.mark.parametrize("e,n,k", [(3, 64, 256), (37, 96, 512)])       # 37: more than one of the reference's chunks
def test_fused_pack_gives_the_torch_packs_bits(gs, e, n, k):
    g = torch.Generator(device=DEV).manual_seed(3)
    w = torch.randint(-(2**31), 2**31 - 1, (e, n, k // 8), generator=g, device=DEV, dtype=torch.int64).to(torch.int32)
    w[0, 0, :6] = torch.tensor([0, -1, -(2**31), 2**31 - 1, 0x0F0F0F0F, -0x0F0F0F10], dtype=torch.int32)
    s = torch.randn((e, n, k // gs), generator=g, device=DEV).to(torch.bfloat16)
    b = torch.randn((e, n, k // gs), generator=g, device=DEV).to(torch.bfloat16)
    assert torch.equal(grouped.pack(w, s, b, gs), _pack_reference(w, s, b, gs))
    half = slice(n // 2 - 16, n // 2 + 16)                                # 32 rows of the middle: not contiguous
    assert torch.equal(grouped.pack(w[:, half], s[:, half], b[:, half], gs),
                       _pack_reference(w[:, half].contiguous(), s[:, half].contiguous(), b[:, half].contiguous(), gs))


@pytest.mark.parametrize("gs", [32, 64])
def test_pack_round_trips(gs):
    g = torch.Generator(device=DEV).manual_seed(4)
    e, n, k = 3, 64, 256
    w = torch.randint(-(2**31), 2**31 - 1, (e, n, k // 8), generator=g, device=DEV, dtype=torch.int64).to(torch.int32)
    s = torch.randn((e, n, k // gs), generator=g, device=DEV).to(torch.bfloat16)
    b = torch.randn((e, n, k // gs), generator=g, device=DEV).to(torch.bfloat16)
    w2, s2, b2 = grouped.unpack(grouped.pack(w, s, b, gs), gs)
    assert torch.equal(w2, w) and torch.equal(s2, s) and torch.equal(b2, b)


def _tensors() -> dict:
    g = torch.Generator().manual_seed(5)
    return {
        "a.weight": torch.randint(-(2**31), 2**31 - 1, (37, 11), generator=g, dtype=torch.int64).to(torch.int32),
        "b.scales": torch.randn((5, 7), generator=g).to(torch.bfloat16),
        "c.odd": torch.randint(0, 255, (4099,), generator=g, dtype=torch.int64).to(torch.uint8),
        "d.big": torch.randn((1000, 29), generator=g),
        "e.last": torch.randn((3,), generator=g),
    }


@pytest.mark.parametrize("direct", [True, False])
def test_device_reads_match_safe_open(tmp_path, monkeypatch, direct):
    monkeypatch.setattr(direct_read, "PIECE", 3 * 4096)          # many pieces a tensor, both staging slots reused
    path = tmp_path / "model.safetensors"
    save_file(_tensors(), str(path))
    files = direct_read.SafeTensors([path])
    files.reader.direct = files.reader.direct and direct
    with safe_open(str(path), framework="pt", device=DEV) as f:
        for name in f.keys():
            a, b = files.get(name, DEV), f.get_tensor(name)
            assert a.device.type == "cuda" and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), name
    if direct and not files.reader.direct:
        pytest.skip("this file system refuses O_DIRECT (tmpfs before Linux 6.6): pass --basetemp on a disk-backed path")


def _write_checkpoint(path, tensors):
    """A one-shard safetensors checkpoint with its index; the header length leaves every tensor unaligned."""

    header, blobs, at = {}, [], 0
    for name, t in tensors.items():
        raw = t.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
        dtype = {torch.int32: "I32", torch.bfloat16: "BF16", torch.float32: "F32", torch.uint8: "U8"}[t.dtype]
        header[name] = {"dtype": dtype, "shape": list(t.shape), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    text = json.dumps(header).encode()
    text += b" " * (4096 - (8 + len(text)) % 4096 + 13)          # data starts 13 bytes past a 4 KiB boundary
    shard = "model-00001-of-00001.safetensors"
    (path / shard).write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: shard for k in tensors}}))


def test_flash_next_reader_returns_the_buffered_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(direct_read, "PIECE", 3 * 4096)
    tensors = _tensors()
    _write_checkpoint(tmp_path, tensors)
    direct = weights._Reader(tmp_path, DEV)
    buffered = weights._Reader(tmp_path, DEV)
    buffered.io.direct = False
    for name, t in tensors.items():
        a, b = direct.get(name), buffered.get(name)
        assert a.dtype == t.dtype and a.shape == t.shape and torch.equal(a, b) and torch.equal(a.cpu(), t), name
    if not direct.io.direct:
        pytest.skip("this file system refuses O_DIRECT: only the fallback was checked")



def test_flash_next_reader_read_ahead_matches_direct_reads(tmp_path):
    tensors = _tensors()
    _write_checkpoint(tmp_path, tensors)
    ahead, alone = weights._Reader(tmp_path, DEV), weights._Reader(tmp_path, DEV)
    ahead.queue(list(tensors)[:2])
    ahead.queue(list(tensors))                                          # overlapping queues: each tensor read once
    ahead.drop(["e.last"])                                              # dropped: read directly when asked for
    taken = {name: ahead.get(name).clone() for name in tensors}         # no synchronization until the end
    torch.cuda.synchronize()
    for name, t in tensors.items():
        a, b = taken[name], alone.get(name)
        assert a.dtype == t.dtype and a.shape == t.shape and torch.equal(a, b) and torch.equal(a.cpu(), t), name
    ahead.close()

def test_table_reads_are_waited_for_and_raise_their_errors():
    import threading
    import time

    class Table:
        def __init__(self, error=None):
            self.error, self.done = error, threading.Event()

        def prefetch(self):
            time.sleep(0.2)
            self.done.set()
            if self.error:
                raise self.error
            return 0.2

    reads: list = []
    slow, failing = Table(), Table(OSError("unreadable table"))
    direct_read.in_background(slow.prefetch, reads)
    direct_read.in_background(failing.prefetch, reads)
    with pytest.raises(OSError, match="unreadable table"):
        direct_read.wait_all(reads)
    assert slow.done.is_set() and failing.done.is_set() and not reads   # both waited for, the list emptied
