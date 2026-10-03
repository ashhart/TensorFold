"""Nemotron prefix files retain complete hybrid state and only committed KV rows."""

import hashlib
import importlib
import json
import math
import struct

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def codec():
    return importlib.import_module("tensorfold.families.nemotron_h.cuda.prefix_snapshot")


def prefix(*, mtp=True):
    def tensor(shape, *, fp32=False):
        return torch.arange(math.prod(shape)).reshape(shape).to(
            torch.float32 if fp32 else torch.bfloat16)

    engine = {"k_cache": tensor((2, 12, 2, 4)), "v_cache": tensor((2, 12, 2, 4)),
              "ssm": tensor((3, 2, 4, 5), fp32=True), "conv_base": tensor((3, 3, 8)),
              "raw": tensor((3, 2, 4, 8)), "xc": tensor((3, 2, 4, 8)),
              "dt": tensor((3, 2, 4, 2), fp32=True), "host": (3, 1, 0)}
    head = {"k": tensor((12, 2, 4)), "v": tensor((12, 2, 4)), "pos": 2} if mtp else None
    return [7, 8, 9], {"engine": engine, "mtp": head, "tail": tensor((1, 8))}, None


def bits(tensor):
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


@pytest.mark.parametrize("mtp", [False, True])
def test_round_trip_preserves_all_state_and_compacts_kv(codec, tmp_path, mtp):
    from safetensors import safe_open

    ids, state, snap = prefix(mtp=mtp)
    state["engine"]["ssm"].view(torch.int32).flatten()[:3] = torch.tensor(
        [0x7FC00001, -2147483648, 0x7F800000])
    path = tmp_path / "prefix.safetensors"
    receipt = codec.save_prefix(path, ids, state, snap, identity="model/settings")
    json.dumps(receipt)
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt["file_bytes"] == path.stat().st_size
    with safe_open(str(path), framework="pt") as reader:
        assert reader.get_tensor("engine.k_cache").shape == (2, 3, 2, 4)
        assert sum(reader.get_tensor(k).numel() * reader.get_tensor(k).element_size()
                   for k in reader.keys()) == receipt["tensor_bytes"]  # noqa: SIM118 - safe_open is not iterable
        if mtp:
            assert reader.get_tensor("mtp.k").shape == (2, 2, 4)
    codec.verify_prefix(path, receipt, identity="model/settings")
    loaded_ids, loaded, restored_snap = codec.load_prefix(path, receipt, identity="model/settings", device="cpu")
    assert loaded_ids == ids and loaded_ids is not ids
    assert restored_snap is None
    assert loaded["engine"]["host"] == state["engine"]["host"]
    for name, old in state["engine"].items():
        if name == "host":
            continue
        new = loaded["engine"][name]
        old = old[:, :3] if name in ("k_cache", "v_cache") else old
        assert new.shape == old.shape and bits(new) == bits(old)
        assert new.data_ptr() != old.data_ptr()
    assert bits(loaded["tail"]) == bits(state["tail"])
    if mtp:
        assert loaded["mtp"]["pos"] == 2
        for name in ("k", "v"):
            assert loaded["mtp"][name].shape == (2, 2, 4)
            assert bits(loaded["mtp"][name]) == bits(state["mtp"][name][:2])
    else:
        assert loaded["mtp"] is None
    path.unlink()
    assert bits(loaded["engine"]["ssm"]) == bits(state["engine"]["ssm"])


@pytest.mark.parametrize("position,mtp", [(0, False), (1, True)])
def test_empty_committed_kv_round_trip(codec, tmp_path, position, mtp):
    ids, state, snap = prefix(mtp=mtp)
    ids = ids[:position]
    state["engine"]["host"] = (position, 0, 0)
    if mtp:
        state["mtp"]["pos"] = 0
    else:
        state["tail"] = None
    path = tmp_path / "prefix.safetensors"
    receipt = codec.save_prefix(path, ids, state, snap, identity="model")
    _, restored, _ = codec.load_prefix(path, receipt, identity="model", device="cpu")
    assert restored["engine"]["k_cache"].shape == (2, position, 2, 4)
    if mtp:
        assert restored["mtp"]["k"].shape == (0, 2, 4)
    else:
        assert restored["tail"] is None


@pytest.mark.parametrize("bad", ["ids", "position", "parity", "keep", "dtype", "rows", "mtp_rows",
                                  "mtp_position", "tail", "extra", "ssm_heads", "conv_width", "ping_pong", "snap"])
def test_invalid_state_rejected_before_file_creation(codec, tmp_path, bad):
    ids, state, snap = prefix()
    engine = state["engine"]
    if bad == "ids": ids[0] = -1
    if bad == "position": engine["host"] = (4, 0, 0)
    if bad == "parity": engine["host"] = (3, 2, 0)
    if bad == "keep": engine["host"] = (3, 0, 5)
    if bad == "dtype": engine["ssm"] = engine["ssm"].bfloat16()
    if bad == "rows": engine["k_cache"] = engine["k_cache"][:, :2]
    if bad == "mtp_rows": state["mtp"]["k"] = state["mtp"]["k"][:1]
    if bad == "mtp_position": state["mtp"]["pos"] = 1
    if bad == "tail": state["tail"] = None
    if bad == "extra": engine["weights"] = object()
    if bad == "ssm_heads": engine["dt"] = engine["dt"][:, :, :, :1]
    if bad == "conv_width": engine["conv_base"] = engine["conv_base"][:, :, :7]
    if bad == "ping_pong": engine["raw"] = engine["raw"][:, :1]
    if bad == "snap": snap = object()
    path = tmp_path / "prefix.safetensors"
    with pytest.raises(ValueError):
        codec.save_prefix(path, ids, state, snap, identity="model")
    assert not path.exists()


@pytest.mark.parametrize("bad", ["identity", "hash", "truncated", "schema", "dtype", "offset", "tokens",
                                  "mtp_position", "shape"])
def test_corrupt_snapshot_rejected_before_tensor_allocation(codec, tmp_path, monkeypatch, bad):
    path = tmp_path / "prefix.safetensors"
    receipt = codec.save_prefix(path, *prefix(), identity="model")
    identity = "model"
    raw = path.read_bytes()
    if bad == "identity":
        identity = "different"
    elif bad == "hash":
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    elif bad == "truncated":
        path.write_bytes(raw[:-1])
    elif bad == "tokens":
        receipt["tokens"] = [1, 2, 3]
    else:
        size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + size])
        if bad in ("schema", "mtp_position"):
            meta = json.loads(header["__metadata__"]["tensorfold_prefix"])
            meta["schema" if bad == "schema" else "mtp_pos"] = 99
            header["__metadata__"]["tensorfold_prefix"] = json.dumps(meta)
        elif bad == "dtype":
            header["engine.ssm"]["dtype"] = "BF16"
        elif bad == "offset":
            header["engine.ssm"]["data_offsets"] = [0, 1]
        else:
            header["engine.k_cache"]["shape"] = [2, 3, 4, 2]
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


def test_staging_is_bounded_for_strided_kv_and_state(codec, tmp_path, monkeypatch):
    from tensorfold.cuda import prefix_snapshot as transport

    ids, state, snap = prefix()
    state["engine"]["ssm"] = state["engine"]["ssm"].transpose(2, 3)
    monkeypatch.setattr(transport, "PIECE", 16)
    sizes, loads = [], []
    original, frombuffer = torch.Tensor.to, torch.frombuffer

    def copy(tensor, *args, **kwargs):
        if args and args[0] == "cpu":
            sizes.append(tensor.numel() * tensor.element_size())
        return original(tensor, *args, **kwargs)

    def stage(buffer, *args, **kwargs):
        loads.append(len(buffer))
        return frombuffer(buffer, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", copy)
    monkeypatch.setattr(torch, "frombuffer", stage)
    path = tmp_path / "prefix.safetensors"
    receipt = codec.save_prefix(path, ids, state, snap, identity="model")
    _, restored, _ = codec.load_prefix(path, receipt, identity="model", device="cpu")
    assert sizes and max(sizes) <= 16
    assert loads and max(loads) <= 16
    assert bits(restored["engine"]["ssm"]) == bits(state["engine"]["ssm"])


def test_existing_file_is_not_overwritten(codec, tmp_path):
    path = tmp_path / "prefix.safetensors"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        codec.save_prefix(path, *prefix(), identity="model")
    assert path.read_bytes() == b"original"
