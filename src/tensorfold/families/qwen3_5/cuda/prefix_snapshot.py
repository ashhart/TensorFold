"""Committed text prefixes as safetensors, streamed through bounded CPU staging."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from pathlib import Path

import torch

PIECE = 8 << 20
_HEADER_LIMIT = 64 << 20
_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32}
_SIZES = {"BF16": 2, "F32": 4}
_SCHEMA = 1


def _check(ok, message):
    if not ok:
        raise ValueError("invalid prefix snapshot: " + message)


def _integer(value):
    return type(value) is int and value >= 0


def _object(pairs):
    result = {}
    for key, value in pairs:
        _check(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _json(raw):
    try:
        return json.loads(raw, object_pairs_hook=_object)
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid prefix snapshot: JSON") from exc


def _validate(meta, tensors, identity):
    """Validate every descriptor before a load allocates anything on the device."""

    _check(isinstance(meta, dict) and set(meta) == {
        "schema", "identity", "ids", "pos", "limit", "rope_delta", "layers", "draft"}, "metadata fields")
    _check(type(meta["schema"]) is int and meta["schema"] == _SCHEMA, "schema")
    _check(isinstance(identity, str) and bool(identity) and meta["identity"] == identity, "identity")
    ids, pos, limit = meta["ids"], meta["pos"], meta["limit"]
    _check(isinstance(ids, list) and all(_integer(i) and i < 2**31 for i in ids), "token ids")
    _check(_integer(pos) and pos == len(ids), "position does not match token ids")
    _check(_integer(limit) and (limit == 0 or limit >= pos), "context limit")
    _check(type(meta["rope_delta"]) is int and meta["rope_delta"] == 0, "only text prefixes are supported")
    layers = meta["layers"]
    _check(isinstance(layers, list) and bool(layers), "layers")
    expected = {}
    for i, kind in enumerate(layers):
        _check(kind in ("gdn", "attention"), "layer kind")
        if kind == "gdn":
            expected[f"conv.{i}"] = ("BF16", 2, None)
            expected[f"rec.{i}"] = ("F32", 3, None)
        else:
            expected[f"k.{i}"] = expected[f"v.{i}"] = ("BF16", 3, (0, pos))
    draft = meta["draft"]
    if draft is not None:
        _check(isinstance(draft, dict) and set(draft) == {"layers", "context_len", "context_end"}, "drafter fields")
        length, end = draft["context_len"], draft["context_end"]
        _check(_integer(length) and _integer(end) and length <= end, "drafter positions")
        _check(isinstance(draft["layers"], list) and all(type(x) is bool for x in draft["layers"]), "drafter layers")
        for i, present in enumerate(draft["layers"]):
            if present:
                expected[f"draft_k.{i}"] = expected[f"draft_v.{i}"] = ("BF16", 3, (1, length))
    _check(set(tensors) == set(expected), "tensor names")
    total = 0
    for name, descriptor in tensors.items():
        _check(isinstance(descriptor, dict) and set(descriptor) == {"dtype", "shape", "data_offsets"}, "tensor fields")
        dtype, rank, fixed = expected[name]
        shape, offsets = descriptor["shape"], descriptor["data_offsets"]
        _check(descriptor["dtype"] == dtype, "tensor dtype")
        _check(isinstance(shape, list) and len(shape) == rank and all(_integer(x) for x in shape), "tensor shape")
        _check(all(x > 0 or (j == 0 and name.startswith(("conv.", "k.", "v.")))
                   or (j == 1 and name.startswith("draft_")) for j, x in enumerate(shape)), "empty tensor dimension")
        _check(fixed is None or shape[fixed[0]] == fixed[1], "tensor rows")
        size = math.prod(shape) * _SIZES[dtype]
        _check(isinstance(offsets, list) and len(offsets) == 2 and all(_integer(x) for x in offsets), "tensor offsets")
        _check(offsets == [total, total + size], "tensor offsets or byte count")
        total += size
    for name, descriptor in tensors.items():
        if name.startswith(("k.", "draft_k.")):
            partner = name.replace("k.", "v.", 1)
            _check(descriptor["shape"] == tensors[partner]["shape"], "key/value shapes")
    return total


def _collect(ids, state, snap, identity):
    _check(isinstance(ids, list), "token ids")
    _check(len(state.conv) == len(state.rec) == len(state.kv), "layer counts")
    meta = {"schema": _SCHEMA, "identity": identity, "ids": list(ids), "pos": state.pos, "limit": state.limit,
                "rope_delta": state.rope_delta, "layers": [], "draft": None}
    tensors = {}
    for i, (conv, rec, kv) in enumerate(zip(state.conv, state.rec, state.kv)):
        if kv is None:
            _check(conv is not None and rec is not None, "recurrent layer tensors")
            meta["layers"].append("gdn")
            tensors[f"conv.{i}"], tensors[f"rec.{i}"] = conv, rec
        else:
            _check(conv is rec is None and isinstance(kv, (tuple, list)) and len(kv) == 2, "attention layer tensors")
            _check(_integer(state.pos), "position")
            meta["layers"].append("attention")
            tensors[f"k.{i}"], tensors[f"v.{i}"] = kv[0][:state.pos], kv[1][:state.pos]
    if snap is not None:
        _check(isinstance(snap, (tuple, list)) and len(snap) == 4, "drafter snapshot")
        kc, vc, length, end = snap
        _check(isinstance(kc, list) and isinstance(vc, list) and len(kc) == len(vc), "drafter layers")
        meta["draft"] = {"layers": [], "context_len": length, "context_end": end}
        for i, (k, v) in enumerate(zip(kc, vc)):
            _check((k is None) == (v is None), "drafter key/value presence")
            meta["draft"]["layers"].append(k is not None)
            if k is not None:
                tensors[f"draft_k.{i}"], tensors[f"draft_v.{i}"] = k, v
    header, at = {}, 0
    for name, tensor in tensors.items():
        _check(isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided, "tensor layout")
        dtype = next((k for k, v in _DTYPES.items() if tensor.dtype == v), None)
        _check(dtype is not None, "tensor dtype")
        size = tensor.numel() * tensor.element_size()
        header[name] = {"dtype": dtype, "shape": list(tensor.shape), "data_offsets": [at, at + size]}
        at += size
    _validate(meta, header, identity)
    return meta, tensors, header, at


def _pieces(tensor):
    """Contiguous logical order; even a strided source needs at most one small copy."""

    if not tensor.is_contiguous() and tensor.numel() * tensor.element_size() > PIECE:
        for row in tensor:
            yield from _pieces(row)
        return
    flat = tensor.detach().contiguous().view(-1)
    count = max(1, PIECE // tensor.element_size())
    for at in range(0, flat.numel(), count):
        yield flat[at:at + count].to("cpu")


def save_prefix(path: Path, ids: list[int], state, snap, *, identity: str) -> dict:
    """Write one new file; its caller owns publication and removal of failed files."""

    meta, tensors, header, tensor_bytes = _collect(ids, state, snap, identity)
    header["__metadata__"] = {"tensorfold_prefix": json.dumps(meta, separators=(",", ":"))}
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    _check(len(text) <= _HEADER_LIMIT, "header too large")
    digest = hashlib.sha256()
    with Path(path).open("xb") as output:
        for block in (struct.pack("<Q", len(text)), text):
            output.write(block)
            digest.update(block)
        for tensor in tensors.values():
            for piece in _pieces(tensor):
                view = memoryview(piece.view(torch.uint8).numpy()).cast("B")
                output.write(view)
                digest.update(view)
                del view, piece
        output.flush()
        os.fsync(output.fileno())
    return {"schema": _SCHEMA, "identity": identity, "sha256": digest.hexdigest(),
                "file_bytes": 8 + len(text) + tensor_bytes, "tensor_bytes": tensor_bytes, "tokens": list(ids)}


def _verified(source, receipt, identity):
    _check(isinstance(receipt, dict), "receipt")
    _check(type(receipt.get("schema")) is int and receipt["schema"] == _SCHEMA
           and receipt.get("identity") == identity, "receipt identity or schema")
    file_bytes = receipt.get("file_bytes")
    _check(_integer(file_bytes) and os.fstat(source.fileno()).st_size == file_bytes, "file size")
    digest = hashlib.sha256()
    block = bytearray(min(PIECE, file_bytes))
    while count := source.readinto(block):
        digest.update(memoryview(block)[:count])
    del block
    _check(digest.hexdigest() == receipt.get("sha256"), "checksum")
    source.seek(0)
    prefix = source.read(8)
    _check(len(prefix) == 8, "missing header")
    size = struct.unpack("<Q", prefix)[0]
    _check(0 < size <= _HEADER_LIMIT and size <= file_bytes - 8, "header size")
    header = _json(source.read(size))
    _check(isinstance(header, dict), "header")
    metadata = header.pop("__metadata__", None)
    _check(isinstance(metadata, dict) and set(metadata) == {"tensorfold_prefix"}, "metadata")
    meta = _json(metadata["tensorfold_prefix"])
    tensor_bytes = _validate(meta, header, identity)
    _check(type(receipt.get("tensor_bytes")) is int and tensor_bytes == receipt["tensor_bytes"], "receipt tensor bytes")
    tokens = receipt.get("tokens")
    _check(isinstance(tokens, list) and all(_integer(i) for i in tokens) and tokens == meta["ids"], "receipt token ids")
    _check(8 + size + tensor_bytes == file_bytes, "data length")
    return meta, header, 8 + size


def verify_prefix(path: Path, receipt, *, identity: str) -> None:
    """Check the complete file and its schema without creating runtime tensors."""

    with Path(path).open("rb") as source:
        _verified(source, receipt, identity)


def load_prefix(path: Path, receipt, *, identity: str, device, room=None):
    """Load independent tensors and reconnect only the new runtime's cache budget."""

    with Path(path).open("rb") as source:
        meta, header, start = _verified(source, receipt, identity)
        from .forward import State

        tensors = {}
        staging = bytearray(min(PIECE, receipt["tensor_bytes"]))
        for name, descriptor in header.items():
            tensor = torch.empty(descriptor["shape"], dtype=_DTYPES[descriptor["dtype"]], device=device)
            flat = tensor.view(torch.uint8).view(-1)
            begin, end = descriptor["data_offsets"]
            source.seek(start + begin)
            for at in range(0, end - begin, PIECE):
                count = min(PIECE, end - begin - at)
                view = memoryview(staging)[:count]
                _check(source.readinto(view) == count, "short tensor read")
                flat[at:at + count].copy_(torch.frombuffer(staging, dtype=torch.uint8, count=count))
            tensors[name] = tensor
    state = State.__new__(State)
    state.pos, state.limit, state.rope_delta, state.room = meta["pos"], meta["limit"], meta["rope_delta"], room
    state.conv, state.rec, state.kv = [], [], []
    for i, kind in enumerate(meta["layers"]):
        state.conv.append(tensors.get(f"conv.{i}"))
        state.rec.append(tensors.get(f"rec.{i}"))
        state.kv.append(None if kind == "gdn" else (tensors[f"k.{i}"], tensors[f"v.{i}"]))
    draft = meta["draft"]
    snap = None if draft is None else (
        [tensors.get(f"draft_k.{i}") for i in range(len(draft["layers"]))],
        [tensors.get(f"draft_v.{i}") for i in range(len(draft["layers"]))],
        draft["context_len"], draft["context_end"])
    return list(meta["ids"]), state, snap
