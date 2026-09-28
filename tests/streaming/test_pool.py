"""The expert pool's host side: file offsets, the LRU for decode rows, the layer window and the slot table."""

from __future__ import annotations

import json
import os
import struct
import threading

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.streaming import pool  # noqa: E402

LAYERS, EXPERTS, TOPK = 3, 8, 2


class FakeSync:
    """The extension's host half on numpy mirrors of the arrays (no GPU)."""

    def __init__(self) -> None:
        self.mirror: dict[int, np.ndarray] = {}
        outer = self

        class Channel:
            def __init__(self) -> None:
                self.v, self.cv = 0, threading.Condition()

            def value(self) -> int:
                return self.v

            def signal(self, v: int) -> None:
                with self.cv:
                    self.v = max(self.v, v)
                    self.cv.notify_all()

            def wait(self, v: int, timeout_ms: int) -> bool:
                with self.cv:
                    return self.cv.wait_for(lambda: self.v >= v, timeout_ms / 1000)

        self.Channel = Channel

    def buf(self, a) -> np.ndarray:
        return self.mirror.setdefault(id(a), np.zeros(a.nbytes, dtype=np.uint8))

    def peek(self, a, off, n) -> bytes:
        return self.buf(a)[off:off + n].tobytes()

    def write_into(self, a, off, data) -> None:
        self.buf(a)[off:off + len(data)] = np.frombuffer(data, dtype=np.uint8)

    def pread_into(self, a, off, fd, foff, n) -> None:
        self.buf(a)[off:off + n] = np.frombuffer(os.pread(fd, n, foff), dtype=np.uint8)


PART_SPECS = {"weight": ([4, 2], "U32", 4), "scales": ([4, 1], "BF16", 2), "biases": ([4, 1], "BF16", 2)}


def checkpoint(tmp_path, per_expert=False):
    """A safetensors file whose expert e of layer l holds bytes (l * 16 + e) in every tensor: stacked [E, ...]
    tensors, or one tensor an expert (``per_expert``, the order experts are written in shuffled)."""

    tensors, data, names = {}, b"", {}
    order = [5, 0, 7, 2, 1, 6, 3, 4]
    for layer in range(LAYERS):
        for proj in ("gate", "up", "down"):
            for part, (shape, dtype, size) in PART_SPECS.items():
                one = int(np.prod(shape)) * size
                if per_expert:
                    for e in order:
                        name = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}_proj.{part}"
                        tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + one]}
                        data += bytes([layer * 16 + e]) * one
                        names[(layer, proj, part, e)] = name
                    continue
                block = b"".join(bytes([layer * 16 + e]) * one for e in range(EXPERTS))
                name = f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}_proj.{part}"
                tensors[name] = {"dtype": dtype, "shape": [EXPERTS, *shape], "data_offsets": [len(data), len(data) + len(block)]}
                data += block
                names[(layer, proj, part)] = name
    header = json.dumps(tensors).encode()
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + data)
    return pool.sources(tmp_path, names)


def streamer(tmp_path, slots, per_expert=False):
    found, shapes = checkpoint(tmp_path, per_expert)
    fake = FakeSync()
    s = pool.Streamer(found, shapes, layers=LAYERS, experts=EXPERTS, top_k=TOPK, slots=slots, box_rows=4,
                      hostsync=fake)
    return s, fake


def call(s, fake, kind, layer, ids):
    """One MoE call as the GPU would make it: ids (or presence flags) in the box, signal, wait for ready."""

    words = [0] * EXPERTS
    if kind == "window":
        for e in ids:
            words[e] = 1
    else:
        words = list(ids)
    fake.write_into(s.box, 0, struct.pack(f"<{len(words)}I", *words))
    signal, ready = s.marks(kind, layer, len(ids) // TOPK if kind == "rows" else 1)
    s.channel.signal(signal)
    assert s.channel.wait(ready, 5000) and s.error is None


def slot_bytes(s, fake, slot):
    return fake.buf(s.pool["gate"].weight)[slot * 32:(slot + 1) * 32]


def test_sources_find_each_experts_bytes(tmp_path):
    found, shapes = checkpoint(tmp_path)
    assert found[(2, "down", "weight")].nbytes == 32 and shapes[("up", "scales")] == ((4, 1), "BF16")
    assert pool.expert_nbytes(found, 1) == 3 * (32 + 8 + 8)


@pytest.mark.parametrize("kind", ["rows", "window"])
def test_one_tensor_an_expert_loads_the_same_bytes(tmp_path, kind):
    s, fake = streamer(tmp_path, slots=2 * EXPERTS + 2, per_expert=True)
    assert pool.expert_nbytes(s.found, 2) == 3 * (32 + 8 + 8)
    call(s, fake, kind, 2, [6, 1, 1, 6] if kind == "rows" else [1, 6])
    table = np.frombuffer(fake.buf(s.slot_of), dtype=np.int32).reshape(LAYERS, EXPERTS)
    for e in (1, 6):
        slot = table[2, e] if kind == "rows" else e
        assert (slot_bytes(s, fake, slot) == 2 * 16 + e).all()
        assert (fake.buf(s.pool["down"].biases)[slot * 8:(slot + 1) * 8] == 2 * 16 + e).all()
    s.close()


def test_decode_rows_load_through_the_lru_and_the_table(tmp_path):
    s, fake = streamer(tmp_path, slots=2 * EXPERTS + 2)
    call(s, fake, "rows", 1, [3, 5, 5, 3])
    table = np.frombuffer(fake.buf(s.slot_of), dtype=np.int32).reshape(LAYERS, EXPERTS)
    for e in (3, 5):
        slot = table[1, e]
        assert slot >= EXPERTS and (slot_bytes(s, fake, slot) == 1 * 16 + e).all()
    call(s, fake, "rows", 1, [3, 3, 5, 5])
    assert (s.hits, s.misses) == (2, 2)
    call(s, fake, "rows", 2, [0, 1, 6, 7])                     # 4 more experts: 2 free slots, then evictions
    table = np.frombuffer(fake.buf(s.slot_of), dtype=np.int32).reshape(LAYERS, EXPERTS)
    for e in (0, 1, 6, 7):
        assert (slot_bytes(s, fake, table[2, e]) == 2 * 16 + e).all()
    assert len({table[2, e] for e in (0, 1, 6, 7)}) == 4 and s.misses == 6
    s.close()


def test_the_window_holds_expert_e_of_one_layer_in_slot_e(tmp_path):
    s, fake = streamer(tmp_path, slots=2 * EXPERTS)
    call(s, fake, "window", 0, [1, 4, 7])
    assert all((slot_bytes(s, fake, e) == e).all() for e in (1, 4, 7))
    before = s.bytes_read
    call(s, fake, "window", 0, [4])                              # already there: no reads
    assert s.bytes_read == before
    call(s, fake, "window", 2, [4])
    assert (slot_bytes(s, fake, 4) == 2 * 16 + 4).all()
    s.close()


def test_a_pool_without_room_for_a_window_and_decode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="window"):
        streamer(tmp_path, slots=2 * EXPERTS - 1)
