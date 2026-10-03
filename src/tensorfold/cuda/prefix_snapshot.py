"""Safetensors prefix transport with integrity checks and bounded CPU staging."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import torch

PIECE = 8 << 20
_HEADER_LIMIT = 64 << 20
DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32}
SIZES = {"BF16": 2, "F32": 4}
_SCHEMA = 1


def check(ok, message):
    if not ok:
        raise ValueError("invalid prefix snapshot: " + message)


def integer(value):
    return type(value) is int and value >= 0


def _object(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _json(raw):
    try:
        return json.loads(raw, object_pairs_hook=_object)
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid prefix snapshot: JSON") from exc


def _header(tensors):
    header, at = {}, 0
    for name, tensor in tensors.items():
        check(isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided, "tensor layout")
        dtype = next((k for k, v in DTYPES.items() if tensor.dtype == v), None)
        check(dtype is not None, "tensor dtype")
        size = tensor.numel() * tensor.element_size()
        header[name] = {"dtype": dtype, "shape": list(tensor.shape), "data_offsets": [at, at + size]}
        at += size
    return header


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


def save(path: Path, meta, tensors, *, identity, validate):
    """Write one new file; its caller owns publication and removal of failed files."""

    header = _header(tensors)
    tensor_bytes = validate(meta, header, identity)
    header["__metadata__"] = {"tensorfold_prefix": json.dumps(meta, separators=(",", ":"))}
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    check(len(text) <= _HEADER_LIMIT, "header too large")
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
            "file_bytes": 8 + len(text) + tensor_bytes, "tensor_bytes": tensor_bytes, "tokens": list(meta["ids"])}


def _verified(source, receipt, identity, validate):
    check(isinstance(receipt, dict), "receipt")
    check(type(receipt.get("schema")) is int and receipt["schema"] == _SCHEMA
          and receipt.get("identity") == identity, "receipt identity or schema")
    file_bytes = receipt.get("file_bytes")
    check(integer(file_bytes) and os.fstat(source.fileno()).st_size == file_bytes, "file size")
    digest = hashlib.sha256()
    block = bytearray(min(PIECE, file_bytes))
    while count := source.readinto(block):
        digest.update(memoryview(block)[:count])
    del block
    check(digest.hexdigest() == receipt.get("sha256"), "checksum")
    source.seek(0)
    prefix = source.read(8)
    check(len(prefix) == 8, "missing header")
    size = struct.unpack("<Q", prefix)[0]
    check(0 < size <= _HEADER_LIMIT and size <= file_bytes - 8, "header size")
    header = _json(source.read(size))
    check(isinstance(header, dict), "header")
    metadata = header.pop("__metadata__", None)
    check(isinstance(metadata, dict) and set(metadata) == {"tensorfold_prefix"}, "metadata")
    meta = _json(metadata["tensorfold_prefix"])
    tensor_bytes = validate(meta, header, identity)
    check(type(receipt.get("tensor_bytes")) is int and tensor_bytes == receipt["tensor_bytes"], "receipt tensor bytes")
    tokens = receipt.get("tokens")
    check(isinstance(tokens, list) and all(integer(i) for i in tokens) and tokens == meta["ids"], "receipt token ids")
    check(8 + size + tensor_bytes == file_bytes, "data length")
    return meta, header, 8 + size


def verify(path: Path, receipt, *, identity, validate):
    """Validate the complete file without creating runtime tensors."""

    with Path(path).open("rb") as source:
        _verified(source, receipt, identity, validate)


def load(path: Path, receipt, *, identity, device, validate):
    """Return metadata and independent tensors only after complete validation."""

    with Path(path).open("rb") as source:
        meta, header, start = _verified(source, receipt, identity, validate)
        tensors = {}
        staging = bytearray(min(PIECE, receipt["tensor_bytes"]))
        for name, descriptor in header.items():
            tensor = torch.empty(descriptor["shape"], dtype=DTYPES[descriptor["dtype"]], device=device)
            flat = tensor.view(torch.uint8).view(-1)
            begin, end = descriptor["data_offsets"]
            source.seek(start + begin)
            for at in range(0, end - begin, PIECE):
                count = min(PIECE, end - begin - at)
                view = memoryview(staging)[:count]
                check(source.readinto(view) == count, "short tensor read")
                flat[at:at + count].copy_(torch.frombuffer(staging, dtype=torch.uint8, count=count))
            tensors[name] = tensor
    return meta, tensors
