"""Committed Nemotron attention, Mamba, and optional MTP prefix state."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from tensorfold.cuda import prefix_snapshot as transport
from tensorfold.cuda.prefix_snapshot import SIZES, check, integer

_SCHEMA = 1
_ENGINE = {"k_cache": ("BF16", 4), "v_cache": ("BF16", 4), "ssm": ("F32", 4),
           "conv_base": ("BF16", 3), "raw": ("BF16", 4), "xc": ("BF16", 4), "dt": ("F32", 4)}


def _validate(meta, tensors, identity):
    """Validate the complete hybrid state before allocating any restored tensors."""

    check(isinstance(meta, dict) and set(meta) == {"schema", "identity", "ids", "host", "mtp_pos", "tail"},
          "metadata fields")
    check(type(meta["schema"]) is int and meta["schema"] == _SCHEMA, "schema")
    check(isinstance(identity, str) and bool(identity) and meta["identity"] == identity, "identity")
    ids, host, mtp_pos = meta["ids"], meta["host"], meta["mtp_pos"]
    check(isinstance(ids, list) and all(integer(i) and i < 2**31 for i in ids), "token ids")
    check(isinstance(host, list) and len(host) == 3 and all(integer(x) for x in host), "host state")
    pos, parity, keep = host
    check(pos == len(ids) and parity in (0, 1), "position or parity")
    check(mtp_pos is None or (integer(mtp_pos) and pos > 0 and mtp_pos == pos - 1), "MTP position")
    check(type(meta["tail"]) is bool and (mtp_pos is None or meta["tail"]), "tail presence")
    expected = {f"engine.{name}": spec for name, spec in _ENGINE.items()}
    if mtp_pos is not None:
        expected.update({"mtp.k": ("BF16", 3), "mtp.v": ("BF16", 3)})
    if meta["tail"]:
        expected["tail"] = ("BF16", 2)
    check(set(tensors) == set(expected), "tensor names")
    total = 0
    for name, descriptor in tensors.items():
        check(isinstance(descriptor, dict) and set(descriptor) == {"dtype", "shape", "data_offsets"}, "tensor fields")
        dtype, rank = expected[name]
        shape, offsets = descriptor["shape"], descriptor["data_offsets"]
        check(descriptor["dtype"] == dtype, "tensor dtype")
        check(isinstance(shape, list) and len(shape) == rank and all(integer(x) for x in shape), "tensor shape")
        check(all(x > 0 or (j == 0 and name.startswith(("engine.", "mtp.")))
                  or (j == 1 and name in ("engine.k_cache", "engine.v_cache", "engine.conv_base"))
                  for j, x in enumerate(shape)), "empty tensor dimension")
        size = math.prod(shape) * SIZES[dtype]
        check(isinstance(offsets, list) and len(offsets) == 2 and all(integer(x) for x in offsets), "tensor offsets")
        check(offsets == [total, total + size], "tensor offsets or byte count")
        total += size
    shapes = {name: descriptor["shape"] for name, descriptor in tensors.items()}
    kv, ssm, conv, raw, dt = (shapes[f"engine.{name}"] for name in ("k_cache", "ssm", "conv_base", "raw", "dt"))
    check(kv == shapes["engine.v_cache"] and kv[1] == pos, "attention key/value shapes or rows")
    check(raw == shapes["engine.xc"] and raw[1] == 2, "Mamba ping-pong shapes")
    check(ssm[0] == conv[0] == raw[0] == dt[0] and conv[2] == raw[3], "Mamba layers or convolution width")
    check(dt[1:3] == raw[1:3] and dt[3] == ssm[1], "Mamba step shapes")
    check(keep <= min(pos, raw[2]), "kept window rows")
    if mtp_pos is not None:
        check(shapes["mtp.k"] == shapes["mtp.v"] and shapes["mtp.k"] == [mtp_pos, *kv[2:]],
              "MTP key/value shapes or rows")
    if meta["tail"]:
        check(shapes["tail"][0] == 1, "tail rows")
    return total


def _collect(ids, state, snap, identity):
    check(snap is None, "separate drafter state")
    check(isinstance(ids, list), "token ids")
    check(isinstance(state, dict) and set(state) == {"engine", "mtp", "tail"}, "retained state fields")
    engine, mtp, tail = state["engine"], state["mtp"], state["tail"]
    check(isinstance(engine, dict) and set(engine) == {*_ENGINE, "host"}, "engine fields")
    host = engine["host"]
    check(isinstance(host, (tuple, list)) and len(host) == 3 and all(integer(x) for x in host), "host state")
    meta = {"schema": _SCHEMA, "identity": identity, "ids": list(ids), "host": list(host),
            "mtp_pos": None, "tail": tail is not None}
    tensors = {f"engine.{name}": engine[name] for name in _ENGINE}
    # Validate layouts before slicing, so malformed retained entries fail as invalid snapshots.
    for name in ("k_cache", "v_cache"):
        tensor = engine[name]
        check(isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided and tensor.ndim == 4,
              "attention tensor")
        tensors[f"engine.{name}"] = tensor[:, :host[0]]
    if mtp is not None:
        check(isinstance(mtp, dict) and set(mtp) == {"k", "v", "pos"} and integer(mtp["pos"]), "MTP fields")
        meta["mtp_pos"] = mtp["pos"]
        for name in ("k", "v"):
            tensor = mtp[name]
            check(isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided and tensor.ndim == 3,
                  "MTP tensor")
            tensors[f"mtp.{name}"] = tensor[:mtp["pos"]]
    if tail is not None:
        tensors["tail"] = tail
    return meta, tensors


def save_prefix(path: Path, ids: list[int], state, snap, *, identity: str) -> dict:
    """Write one prefix; the caller owns publication and failed-file cleanup."""

    meta, tensors = _collect(ids, state, snap, identity)
    return transport.save(path, meta, tensors, identity=identity, validate=_validate)


def verify_prefix(path: Path, receipt, *, identity: str) -> None:
    transport.verify(path, receipt, identity=identity, validate=_validate)


def load_prefix(path: Path, receipt, *, identity: str, device, room=None):
    """Return compact independent tensors; the new runtime owns fixed cache capacity."""

    meta, tensors = transport.load(path, receipt, identity=identity, device=device, validate=_validate)
    engine = {name: tensors[f"engine.{name}"] for name in _ENGINE}
    engine["host"] = tuple(meta["host"])
    mtp = None if meta["mtp_pos"] is None else {
        "k": tensors["mtp.k"], "v": tensors["mtp.v"], "pos": meta["mtp_pos"]}
    return list(meta["ids"]), {"engine": engine, "mtp": mtp, "tail": tensors.get("tail")}, None
