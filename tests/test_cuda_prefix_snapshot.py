"""Prefix files preserve committed state bits without retaining checkpoint objects."""

import hashlib
import importlib
import json
import struct
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def codec():
    return importlib.import_module("tensorfold.families.qwen3_5.cuda.prefix_snapshot")


def prefix():
    state = SimpleNamespace(pos=3, limit=64, rope_delta=0, room=lambda *_: None,
                            conv=[torch.arange(12).reshape(3, 4).bfloat16(), None],
                            rec=[torch.arange(8).reshape(2, 2, 2).float(), None],
                            kv=[None, (torch.arange(40).reshape(10, 2, 2).bfloat16(),
                                       torch.arange(40, 80).reshape(10, 2, 2).bfloat16())])
    snap = ([torch.arange(8).reshape(2, 2, 2).bfloat16(), None],
            [torch.arange(8, 16).reshape(2, 2, 2).bfloat16(), None], 2, 3)
    return [7, 8, 9], state, snap


def bits(tensor):
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def test_round_trip_preserves_bits_and_only_committed_rows(codec, tmp_path):
    from safetensors import safe_open

    ids, state, snap = prefix()
    # Non-finite values and signed zero must be byte-for-byte, not converted.
    state.rec[0].view(torch.int32).flatten()[:3] = torch.tensor([0x7FC00001, -2147483648, 0x7F800000])
    path = tmp_path / "prefix.safetensors"
    receipt = codec.save_prefix(path, ids, state, snap, identity="checkpoint/settings")
    json.dumps(receipt)
    assert receipt["tokens"] == ids
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt["file_bytes"] == path.stat().st_size
    expected_bytes = 24 + 32 + 2 * 3 * 2 * 2 * 2 + 2 * 2 * 2 * 2 * 2
    assert receipt["tensor_bytes"] == expected_bytes
    with safe_open(str(path), framework="pt") as reader:
        assert sum(reader.get_tensor(k).numel() * reader.get_tensor(k).element_size()
                   for k in reader.keys()) == expected_bytes  # noqa: SIM118 - safe_open is not iterable
    codec.verify_prefix(path, receipt, identity="checkpoint/settings")
    room = object()
    loaded_ids, loaded, restored = codec.load_prefix(path, receipt, identity="checkpoint/settings",
                                                    device="cpu", room=room)
    assert type(loaded).__name__ == "State"
    assert loaded_ids == ids and loaded_ids is not ids
    assert (loaded.pos, loaded.limit, loaded.rope_delta, loaded.room) == (3, 64, 0, room)
    assert loaded.conv[1] is loaded.rec[1] is loaded.kv[0] is None
    assert bits(loaded.conv[0]) == bits(state.conv[0])
    assert bits(loaded.rec[0]) == bits(state.rec[0])
    for new, old in zip(loaded.kv[1], state.kv[1]):
        assert new.shape == (3, 2, 2) and bits(new) == bits(old[:3])
        assert new.data_ptr() != old.data_ptr()
    assert restored[2:] == (2, 3)
    assert restored[0][1] is restored[1][1] is None
    for new, old in zip(restored[:2], snap[:2]):
        assert bits(new[0]) == bits(old[0])
    # A returned state never owns a file mapping, and the file may be removed.
    path.unlink()
    assert bits(loaded.rec[0]) == bits(state.rec[0])


def test_without_drafter_and_noncontiguous_tensors(codec, tmp_path):
    ids, state, _ = prefix()
    state.conv[0] = state.conv[0].T
    state.rec[0] = state.rec[0].transpose(0, 1)
    path = tmp_path / "prefix.safetensors"
    receipt = codec.save_prefix(path, ids, state, None, identity="model")
    _, restored, snap = codec.load_prefix(path, receipt, identity="model", device="cpu")
    assert snap is None and restored.room is None
    assert bits(restored.conv[0]) == bits(state.conv[0])
    assert bits(restored.rec[0]) == bits(state.rec[0])


@pytest.mark.parametrize("bad", ["ids", "position", "rope", "dtype", "rows", "limit"])
def test_invalid_state_refused_before_file_creation(codec, tmp_path, bad):
    ids, state, snap = prefix()
    if bad == "ids": ids[0] = -1
    if bad == "position": state.pos = 4
    if bad == "rope": state.rope_delta = 1
    if bad == "dtype": state.rec[0] = state.rec[0].half()
    if bad == "rows": state.kv[1] = tuple(t[:2] for t in state.kv[1])
    if bad == "limit": state.limit = 2
    path = tmp_path / "prefix.safetensors"
    with pytest.raises(ValueError):
        codec.save_prefix(path, ids, state, snap, identity="model")
    assert not path.exists()


@pytest.mark.parametrize("bad", ["identity", "hash", "truncated", "schema", "dtype", "offset", "tokens"])
def test_bad_snapshot_rejected_before_allocating(codec, tmp_path, monkeypatch, bad):
    path = tmp_path / "prefix.safetensors"
    receipt = codec.save_prefix(path, *prefix(), identity="model")
    identity = "model"
    raw = path.read_bytes()
    if bad == "identity":
        identity = "other"
    elif bad == "hash":
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    elif bad == "truncated":
        path.write_bytes(raw[:-1])
    elif bad == "tokens":
        receipt["tokens"] = [1, 2, 3]
    else:
        size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + size])
        if bad == "schema":
            meta = json.loads(header["__metadata__"]["tensorfold_prefix"])
            meta["schema"] = 99
            header["__metadata__"]["tensorfold_prefix"] = json.dumps(meta)
        else:
            entry = next(v for k, v in header.items() if k != "__metadata__")
            entry["dtype" if bad == "dtype" else "data_offsets"] = "F16" if bad == "dtype" else [0, 1]
        text = json.dumps(header).encode()
        text += b" " * (-len(text) % 8)
        path.write_bytes(struct.pack("<Q", len(text)) + text + raw[8 + size:])
        receipt["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt["file_bytes"] = path.stat().st_size

    def allocation(*args, **kwargs):
        raise AssertionError("invalid snapshot allocated a tensor")

    monkeypatch.setattr(torch, "empty", allocation)
    with pytest.raises(ValueError):
        codec.verify_prefix(path, receipt, identity=identity)
    with pytest.raises(ValueError):
        codec.load_prefix(path, receipt, identity=identity, device="cpu")


def test_staging_pieces_are_bounded(codec, tmp_path, monkeypatch):
    ids, state, snap = prefix()
    state.rec[0] = state.rec[0].transpose(0, 1)
    from tensorfold.cuda import prefix_snapshot as transport

    monkeypatch.setattr(transport, "PIECE", 16)
    sizes = []
    original = torch.Tensor.to
    frombuffer = torch.frombuffer
    loads = []

    def copy(tensor, *args, **kwargs):
        if args and args[0] == "cpu":
            sizes.append(tensor.numel() * tensor.element_size())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", copy)

    def stage(buffer, *args, **kwargs):
        loads.append(len(buffer))
        return frombuffer(buffer, *args, **kwargs)

    monkeypatch.setattr(torch, "frombuffer", stage)
    path = tmp_path / "prefix.safetensors"
    receipt = codec.save_prefix(path, ids, state, snap, identity="model")
    assert sizes and max(sizes) <= 16
    _, restored, _ = codec.load_prefix(path, receipt, identity="model", device="cpu")
    assert loads and max(loads) <= 16
    assert bits(restored.rec[0]) == bits(state.rec[0])


def test_empty_attention_and_drafter_cache_round_trip(codec, tmp_path):
    ids, state, _ = prefix()
    state.pos, state.limit = 0, 0
    snap = ([None, None], [None, None], 0, 0)
    path = tmp_path / "prefix.safetensors"
    receipt = codec.save_prefix(path, [], state, snap, identity="model")
    ids, restored, restored_snap = codec.load_prefix(path, receipt, identity="model", device="cpu")
    assert ids == [] and restored.pos == restored.limit == 0
    assert restored.kv[1][0].shape == restored.kv[1][1].shape == (0, 2, 2)
    assert restored_snap == snap


def test_existing_file_is_not_overwritten(codec, tmp_path):
    path = tmp_path / "prefix.safetensors"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        codec.save_prefix(path, *prefix(), identity="model")
    assert path.read_bytes() == b"original"
