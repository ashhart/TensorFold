"""Committed Qwen text prefix state stored through the shared snapshot transport."""

from __future__ import annotations

import math
from pathlib import Path

from tensorfold.cuda import prefix_snapshot as transport
from tensorfold.cuda.prefix_snapshot import SIZES as _SIZES
from tensorfold.cuda.prefix_snapshot import check as _check
from tensorfold.cuda.prefix_snapshot import integer as _integer

_SCHEMA = 2


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
        rows = draft["layers"]
        _check(isinstance(rows, list) and all(x is None or (_integer(x) and x <= length) for x in rows),
               "drafter layers")
        _check(max((x for x in rows if x is not None), default=0) == length, "drafter context length")
        for i, count in enumerate(rows):
            if count is not None:
                expected[f"draft_k.{i}"] = expected[f"draft_v.{i}"] = ("BF16", 3, (1, count))
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
            _check(k is None or k.ndim == 3, "drafter tensor rank")
            meta["draft"]["layers"].append(None if k is None else k.shape[1])
            if k is not None:
                tensors[f"draft_k.{i}"], tensors[f"draft_v.{i}"] = k, v
    return meta, tensors


def save_prefix(path: Path, ids: list[int], state, snap, *, identity: str) -> dict:
    """Write one committed prefix with no reference to its model weights."""

    meta, tensors = _collect(ids, state, snap, identity)
    return transport.save(path, meta, tensors, identity=identity, validate=_validate)


def verify_prefix(path: Path, receipt, *, identity: str) -> None:
    """Check the complete file and its schema without creating runtime tensors."""

    transport.verify(path, receipt, identity=identity, validate=_validate)


def load_prefix(path: Path, receipt, *, identity: str, device, room=None):
    """Load independent tensors and reconnect only the new runtime's cache budget."""

    meta, tensors = transport.load(path, receipt, identity=identity, device=device, validate=_validate)
    from .forward import State

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
