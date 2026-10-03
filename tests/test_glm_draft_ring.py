"""DFlash2's context ring on CPU: its size and estimate, the drafter's weights as held, the kept-state plumbing."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from tensorfold.cuda.geometry import dflash2_geometry, dflash2_weights, draft_geometry, draft_ring_rows
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so decode imports)

pytestmark = pytest.mark.torch

# GLM-5.3-Flash-DFlash2's config.json (the sizes the estimate reads)
DFLASH2 = {"num_hidden_layers": 5, "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 4096,
           "intermediate_size": 12288, "sliding_window": 2048, "dflash_config": {"block_size": 8}}


def test_the_ring_holds_the_window_the_block_and_a_tile():
    assert draft_ring_rows(2047, 8) == 2176                   # 2,047 + 8 + 63 = 2,118 rows, whole 64-row tiles
    assert draft_ring_rows(48, 8) == 128
    assert all(draft_ring_rows(w, 8) % 64 == 0 and draft_ring_rows(w, 8) >= w + 8 + 63 for w in range(1, 3000))


def test_the_estimate_no_longer_grows_with_the_window():
    old = draft_geometry(DFLASH2, 2, 16)                      # what GLM reserved before: two flat copies
    flat = dflash2_geometry(DFLASH2, 2, 16, ring=False)
    ring = dflash2_geometry(DFLASH2, 2, 16, ring=True)
    assert old.bytes_at(1 << 20) - old.bytes_at((1 << 20) - 1) == 20480
    assert flat.bytes_at(1 << 20) - flat.bytes_at((1 << 20) - 1) == 10240     # K,V x 5 layers x 4 heads x 128 x 2 B
    fixed = flat.bytes_at(0) - 10240 * 8
    assert fixed == old.bytes_at(0) - 20480 * 8
    assert ring.bytes_at(1 << 20) == ring.bytes_at(4096) == fixed + 10240 * 2176      # one ring, 21.25 MiB
    assert ring.bytes_at(1000) == flat.bytes_at(1000) == fixed + 10240 * (1000 + 8)   # a smaller window: flat
    assert old.bytes_at(1 << 20) - ring.bytes_at(1 << 20) > 20 * 2 ** 30 - 30 * 2 ** 20
    unwindowed = dict(DFLASH2, sliding_window=0)
    assert dflash2_geometry(unwindowed, 2, 16, ring=True).bytes_at(5000) == fixed + 10240 * 5008


def test_the_drafters_weights_are_its_4bit_copies(tmp_path):
    """The weights estimate of a checkpoint with GLM-5.3-Flash-DFlash2's shapes (headers only, no data)."""

    import json
    import struct

    D, H, KV, hd, inter, V, R = 4096, 32, 8, 128, 12288, 154880, 256
    shapes = {"fc.weight": [D, 5 * D], "hidden_norm.weight": [D], "norm.weight": [D],
              "candidate_selector.hidden_projection.weight": [R, D],
              "candidate_selector.predecessor_codebook": [V, R], "candidate_selector.successor_codebook": [V, R]}
    for i in range(5):
        p = f"layers.{i}."
        shapes.update({p + "self_attn.q_proj.weight": [H * hd, D], p + "self_attn.k_proj.weight": [KV * hd, D],
                       p + "self_attn.v_proj.weight": [KV * hd, D], p + "self_attn.o_proj.weight": [D, H * hd],
                       p + "self_attn.q_norm.weight": [hd], p + "self_attn.k_norm.weight": [hd],
                       p + "mlp.gate_proj.weight": [inter, D], p + "mlp.up_proj.weight": [inter, D],
                       p + "mlp.down_proj.weight": [D, inter], p + "input_layernorm.weight": [D],
                       p + "post_attention_layernorm.weight": [D]})
        for conv in ("attention_conv", "mlp_conv"):
            shapes.update({p + conv + ".base_kernel": [2, 2, D],
                           p + conv + ".kernel_projection.weight": [4 * D // 16, D]})
    header, off = {}, 0
    for name, shape in shapes.items():
        n = 2
        for x in shape:
            n *= x
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    w = dflash2_weights(tmp_path, 2)
    q4 = 9 / 16                                               # bytes a value: 4 bits, BF16 scale and bias a 64
    per_layer = ((32 + 16) * hd // 2 * D + 16 * hd // 2 * D + D * H * hd // 2 + 2 * inter // 2 * D + D * inter // 2
                 + 2 * 4 * D // 16 * D) * q4
    bf16 = 2 * (2 * D + R * D + 5 * (2 * hd + 2 * D + 2 * 4 * D))
    assert w.resident == int(D * 5 * D * q4 + 5 * per_layer) + bf16 + 2 * V * R * 4
    assert 0.62 * 2 ** 30 < w.resident < 0.64 * 2 ** 30        # 4-bit copies 0.33 GiB, host codebooks 0.30 GiB
    assert 2 * off > 6.8 * w.resident                          # the old estimate: 4 bytes a value (4.37 GiB)
    assert w.staging == 4 * D * 5 * D + 24 * D * 5 * D        # fc: read, uploaded and quantized in one piece


class FakeDrafter:
    """The context side of ``dflash2.Drafter``: rows at position (flat) or position % ring, tagged by writer."""

    window, block, KV, HD = 48, 8, 2, 4

    def __init__(self, torch, *, ring: bool) -> None:
        self.torch = torch
        self.ring = draft_ring_rows(self.window, self.block) if ring else 0
        cap = self.ring or 4096
        self.kc = [torch.zeros((self.KV, cap, self.HD), dtype=torch.float32) for _ in range(2)]
        self.vc = [torch.zeros((self.KV, cap, self.HD), dtype=torch.float32) for _ in range(2)]
        self.context_end = 0
        self.pos_dev = torch.zeros((1,), dtype=torch.int64)

    def reset(self) -> None:
        self.context_end = 0
        self.pos_dev.zero_()

    def add(self, n: int, tag: int) -> None:
        for p in range(self.context_end, self.context_end + n):
            slot = p % self.ring if self.ring else p
            for i, (k, v) in enumerate(zip(self.kc, self.vc)):
                k[:, slot] = tag * 1e6 + p * 10 + i
                v[:, slot] = -(tag * 1e6 + p * 10 + i)
        self.context_end += n
        self.pos_dev.fill_(self.context_end)

    def window_rows(self) -> list:
        """What a block pass at context_end reads: the rows context_end - window .. context_end - 1."""
        s = self.context_end
        idx = [p % self.ring if self.ring else p for p in range(max(0, s - self.window), s)]
        return [c[:, idx].clone() for c in self.kc + self.vc]


def _engine(torch):
    st = SimpleNamespace(cur=[], rec=[torch.zeros(3)], conv=torch.zeros(2), mtp_len=0, mtp_drafted=0, kc=[], vc=[],
                         index=None, set_pos=lambda n: None, set_mtp_len=lambda n: None)
    return SimpleNamespace(st=st)


def test_a_restored_state_reads_the_flat_drafters_window_after_the_ring_wrapped(allocations):  # noqa: F811
    import torch

    decode = importlib.import_module("tensorfold.families.glm5_next.cuda.decode")
    e = _engine(torch)
    flat, ring = FakeDrafter(torch, ring=False), FakeDrafter(torch, ring=True)
    assert ring.ring == 128
    for d in (flat, ring):
        d.add(300, tag=1)                                    # a prompt
    snaps = [decode.take_snapshot(e, list(range(300)), None, mtp=False, drafter=d) for d in (flat, ring)]
    assert snaps[0].drafter_rows is None
    assert [r.shape[1] for r in snaps[1].drafter_rows] == [ring.window + 1] * 4
    held = decode.snapshot_bytes(snaps[0])
    assert decode.snapshot_bytes(snaps[1]) == held + 4 * ring.KV * (ring.window + 1) * ring.HD * 4
    for d in (flat, ring):
        d.add(3 * ring.ring + 5, tag=2)                      # its reply, past the ring three times
    assert all((a == b).all() for a, b in zip(ring.window_rows(), flat.window_rows()))
    kept = [p % ring.ring for p in range(300 - ring.window - 1, 300)]
    assert not (ring.kc[0][:, kept] == snaps[1].drafter_rows[0]).any()     # the reply overwrote the kept window
    for d, s in zip((flat, ring), snaps):                    # the next prompt of the conversation resumes
        decode.restore(e, s, d)
        assert d.context_end == 300 and int(d.pos_dev) == 300
    assert all((a == b).all() for a, b in zip(ring.window_rows(), flat.window_rows()))
    cold = FakeDrafter(torch, ring=True)
    cold.add(300, tag=1)
    assert all((a == b).all() for a, b in zip(ring.window_rows(), cold.window_rows()))
    for d in (flat, ring):
        d.add(40, tag=3)
    assert all((a == b).all() for a, b in zip(ring.window_rows(), flat.window_rows()))


def test_a_saved_state_drops_the_window_and_a_ring_needs_one(allocations):  # noqa: F811
    import torch

    decode = importlib.import_module("tensorfold.families.glm5_next.cuda.decode")
    e = _engine(torch)
    ring = FakeDrafter(torch, ring=True)
    ring.add(20, tag=1)
    snap = decode.take_snapshot(e, list(range(20)), None, mtp=False, drafter=ring)
    assert snap.drafter_rows is not None and snap.drafter_rows[0].shape[1] == 20      # shorter than the window
    decode.save_rows(e, snap)                                # another conversation took the caches
    assert snap.drafter_end == -1 and snap.drafter_rows is None
    assert decode.snapshot_bytes(snap) == 3 * 4 + 2 * 4
    stale = decode.Snapshot(list(range(20)), torch.zeros(3), torch.zeros(2), None, -1, 20)
    with pytest.raises(ValueError, match="window"):
        decode.restore(e, stale, ring)
    ring.add(5, tag=2)
    other = decode.take_snapshot(e, list(range(20)), None, mtp=False, drafter=ring)      # the drafter ran on
    assert other.drafter_end == 25 and other.drafter_rows is None
