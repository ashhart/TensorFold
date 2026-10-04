"""Which part of each checkpoint tensor a rank holds, and the bytes admission counts for it (no torch)."""

from __future__ import annotations

import math
import re

from tensorfold.cuda.capacity import itemsize

_BLOCK = r"^(?:layers|mtp)\.\d+\."                  # a backbone layer or a DSpark stage
_EXL3 = r"^layers\.\d+\.ffn\.experts\.\d+\."        # the backbone's routed experts (EXL3 trellises)
_MXFP4 = r"^mtp\.\d+\.ffn\.experts\.\d+\."          # the DSpark stages' routed experts (MXFP4)

# rep: every rank holds it whole; row/heads/groups/vocab/nsplit: halves of axis 0; col/dim1: halves of axis 1;
# engram: hash tables read by row id from the file, never loaded; drop: the vision tower, unused
RULES = {
    "rep": (
        r"^embed\.weight$", r"^norm\.weight$", _BLOCK + r"(attn|ffn)_norm\.weight$",
        _BLOCK + r"attn\.(wq_a|wkv)\.(weight|scale)$", _BLOCK + r"attn\.(q_norm|kv_norm)\.weight$",
        r"^layers\.\d+\.attn\.compressor\.(wkv|wgate|norm)\.weight$",
        r"^layers\.\d+\.attn\.indexer\.(wq_b\.weight|wq_b\.scale|weights_proj\.weight|wk\.weight|k_norm\.weight)$",
        _BLOCK + r"hc_(attn|ffn)_(fn|base|scale)$", _BLOCK + r"ffn\.gate\.(weight|bias|bias_vl)$",
        _EXL3 + r"w[13]\.suh$", _EXL3 + r"w2\.svh$", _EXL3 + r"w[123]\.mcg$",
        r"^layers\.\d+\.engram\.(q_weight|k_weight)$", r"^mtp\.\d+\.(main_norm|norm)\.weight$",
        r"^mtp\.\d+\.markov_head\.embed\.weight$", r"^mtp\.\d+\.confidence_head\.proj\.weight$",
    ),
    "row": (_EXL3 + r"w[13]\.svh$", _EXL3 + r"w2\.(trellis|suh)$",
            _BLOCK + r"ffn\.shared_experts\.w[13]\.(weight|scale)$", _MXFP4 + r"w[13]\.(weight|scale)$"),
    "col": (_BLOCK + r"attn\.wo_b\.(weight|scale)$", _BLOCK + r"ffn\.shared_experts\.w2\.(weight|scale)$",
            _MXFP4 + r"w2\.(weight|scale)$"),
    "dim1": (_EXL3 + r"w[13]\.trellis$",),
    "heads": (_BLOCK + r"attn\.wq_b\.(weight|scale)$", _BLOCK + r"attn\.attn_sink$"),
    "groups": (_BLOCK + r"attn\.wo_a\.(weight|scale)$",),
    "vocab": (r"^lm_head\.(weight|weight_scale)$", r"^mtp\.\d+\.markov_head\.head\.weight$"),
    "nsplit": (r"^layers\.\d+\.engram\.wkv\.(weight|scale)$", r"^mtp\.\d+\.main_proj\.(weight|scale)$"),
    "engram": (r"^layers\.\d+\.engram\.embed\.(weight|scale)$",),
    "drop": (r"^vision\.", r"^aligner\.", r"^image_(start|end|newline)$"),
}
AXIS = {"row": 0, "heads": 0, "groups": 0, "vocab": 0, "nsplit": 0, "col": 1, "dim1": 1}
_MATCH = {kind: re.compile("|".join(f"(?:{p})" for p in patterns)) for kind, patterns in RULES.items()}
_NVFP4_SCALE = re.compile(_MXFP4 + r"w[123]\.scale$")
FP32 = ("confidence_head.proj.weight",)              # bf16 in the checkpoint, fp32 on the device


def rule(name: str) -> str:
    """The split rule of one checkpoint tensor; a name that matches no rule or more than one raises."""

    hits = [kind for kind, pattern in _MATCH.items() if pattern.search(name)]
    if len(hits) != 1:
        raise ValueError(f"{name}: split rule is ambiguous or missing ({hits})")
    return hits[0]


def _slices(kind: str, rank: int, world: int, name: str, shape) -> tuple[slice, ...]:
    if kind in ("engram", "drop"):
        raise ValueError(f"{name}: " + ("an Engram table is read by row id, never loaded" if kind == "engram"
                                        else "the engine does not use the vision tower"))
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} of {world}")
    out = [slice(0, int(n)) for n in shape]
    if kind != "rep":
        axis = AXIS[kind]
        if len(shape) <= axis or shape[axis] % world:
            raise ValueError(f"{name} {list(shape)}: axis {axis} does not split in {world} ({kind})")
        part = int(shape[axis]) // world
        out[axis] = slice(rank * part, (rank + 1) * part)
    return tuple(out)


def slice_for(rank: int, world: int, name: str, shape) -> tuple[slice, ...]:
    """The index ranges of rank's part of a tensor of ``shape``, one slice per axis."""

    return _slices(rule(name), rank, world, name, shape)


def _rows128(n: int) -> int:
    return -(-n // 128) * 128                        # Mx8Linear pads its output rows to 128


def weights_estimate(name: str, info: dict, world: int = 2) -> tuple[int, int]:
    """capacity.admit's transform: (device bytes, mapped bytes) of one tensor on one rank, laid out as loaded."""

    kind = rule(name)
    if kind in ("engram", "drop"):
        return 0, 0
    shape = [s.stop - s.start for s in _slices(kind, 0, world, name, info["shape"])]
    dtype, n = info["dtype"], math.prod(shape)
    if _NVFP4_SCALE.search(name):                    # E8M0 per 32 -> e4m3 per 16, and an fp32 scale per matrix
        return 2 * n + 4, 0
    if dtype == "F8_E4M3":
        return _rows128(shape[0]) * shape[1], 0
    if dtype == "F8_E8M0":                           # a 32x32 block's scale repeated on each of its rows
        return _rows128(32 * shape[0]) * shape[1], 0
    if name == "lm_head.weight_scale":
        return _rows128(shape[0]) * shape[1], 0
    if name.endswith(FP32):
        return 4 * n, 0
    return n * itemsize(info, name), 0
