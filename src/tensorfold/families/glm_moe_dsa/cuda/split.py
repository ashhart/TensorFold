"""Full GLM-5.3 over ``world`` ranks: which part of each checkpoint tensor a rank holds, cut from the checkpoint bytes.

The same scheme as GLM-5.3-Flash's two-rank ``glm5_next/cuda/split.py``, for any world size and for this family's
tensor names. EXL3 linears are cut along whole 16x16 tiles and whole 128-wide Hadamard blocks, so a rank's part is
itself a valid EXL3 tensor: output-split linears (gate/up, q_b) take tile columns and ``svh``; input-split linears
(down, o_proj) take tile rows and ``suh``; the other scale and the codebook marker are replicated.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np

HAD = 128                # an EXL3 Hadamard block: a rank's cut must not straddle one
TILE = 16

# EXL3 linears: (projection kind, tensor part) -> split kind. "out" linears split outputs, "in" linears split inputs.
EXL3_LINEAR = re.compile(
    r"\.(mlp\.experts\.\d+\.(?P<e>gate|up|down)_proj|mlp\.shared_experts\.(?P<s>gate|up|down)_proj|"
    r"mlp\.(?P<d>gate|up|down)_proj|self_attn\.(?P<a>q_a_proj|kv_a_proj_with_mqa|q_b_proj|o_proj)|"
    r"self_attn\.indexer\.(?P<i>wq_b))\.(?P<part>trellis|suh|svh|mul1|mcg)$")
OUT_RULES = {"trellis": "dim1", "svh": "row", "suh": "rep", "mul1": "rep", "mcg": "rep"}
IN_RULES = {"trellis": "row", "suh": "row", "svh": "rep", "mul1": "rep", "mcg": "rep"}

ROW = (r"\.self_attn\.kv_b_proj\.weight$", r"^lm_head\.weight$")
REP = (r"^model\.embed_tokens\.", r"^model\.norm\.weight$", r"_layernorm\.weight$", r"\.mlp\.gate\.weight$",
       r"\.mlp\.gate\.e_score_correction_bias$", r"\.self_attn\.indexer\.(wk|weights_proj)\.weight$",
       r"\.self_attn\.indexer\.k_norm\.(weight|bias)$", r"\.eh_proj\.weight$", r"\.(enorm|hnorm)\.weight$",
       r"\.shared_head\.norm\.weight$")


def rule(name: str) -> str:
    """rep | row (leading axis) | dim1 (second axis) for one checkpoint tensor."""

    m = EXL3_LINEAR.search(name)
    if m:
        part = m.group("part")
        proj = m.group("e") or m.group("s") or m.group("d") or m.group("a") or m.group("i")
        if proj in ("q_a_proj", "kv_a_proj_with_mqa", "wq_b"):
            return "rep"                               # replicated: every rank computes the latent and indexer q
        if proj in ("gate", "up", "q_b_proj"):
            return OUT_RULES[part]
        return IN_RULES[part]                          # down, o_proj
    hits = [k for k, pats in (("row", ROW), ("rep", REP)) if any(re.search(p, name) for p in pats)]
    if len(hits) != 1:
        raise ValueError(f"{name}: no split rule (or more than one: {hits})")
    return hits[0]


def _check_cut(name: str, kind: str, shape: list[int], world: int) -> None:
    axis = 0 if kind == "row" else 1
    if len(shape) <= axis or shape[axis] % world:
        raise ValueError(f"{name}: {kind} split of {shape} is not even over {world} ranks")
    if name.endswith(".trellis"):                     # tiles: a rank's share must be whole Hadamard blocks
        if (shape[axis] // world) % (HAD // TILE):
            raise ValueError(f"{name}: {shape[axis] // world} tiles per rank straddle a {HAD}-wide Hadamard block")
    elif name.endswith((".suh", ".svh")) and (shape[0] // world) % HAD:
        raise ValueError(f"{name}: {shape[0] // world} scales per rank straddle a {HAD}-wide Hadamard block")


def split_bytes(raw: np.ndarray, shape: list[int], itemsize: int, kind: str, rank: int, world: int,
                name: str = "") -> tuple[np.ndarray, list[int]]:
    """A tensor's bytes -> this rank's contiguous part and its shape."""

    if kind == "rep":
        return raw, list(shape)
    _check_cut(name, kind, shape, world)
    if kind == "row":
        per = raw.size // shape[0]
        n = shape[0] // world
        return raw[rank * n * per:(rank + 1) * n * per], [n] + list(shape[1:])
    if kind == "dim1":
        inner = int(np.prod(shape[2:])) * itemsize
        n = shape[1] // world
        view = raw.reshape(shape[0], shape[1] * inner)
        part = np.ascontiguousarray(view[:, rank * n * inner:(rank + 1) * n * inner])
        return part.reshape(-1), [shape[0], n] + list(shape[2:])
    raise ValueError(kind)


def read_header(path: str | Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def plan(model_dir: str | Path, world: int) -> dict[str, tuple[str, list[int], list[int]]]:
    """name -> (split kind, full shape, per-rank shape) for every tensor of a checkpoint; raises on any bad cut."""

    model_dir = Path(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    headers: dict[str, dict] = {}
    out = {}
    for name, fn in index.items():
        if fn not in headers:
            headers[fn] = read_header(model_dir / fn)[0]
        info = headers[fn][name]
        kind, shape = rule(name), list(info["shape"])
        if kind == "rep":
            out[name] = (kind, shape, shape)
            continue
        _check_cut(name, kind, shape, world)
        part = list(shape)
        part[0 if kind == "row" else 1] //= world
        out[name] = (kind, shape, part)
    return out
