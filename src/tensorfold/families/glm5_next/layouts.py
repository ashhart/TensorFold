"""GLM-5.3-Flash's two on-disk layouts (the original and mlx-lm's conversion) mapped onto one set of short names."""

from __future__ import annotations

import re

VONTRA = "vontra"
MLXLM = "mlxlm"

_PREFIXES = ("model.language_model.", "language_model.model.", "language_model.")
_HC = re.compile(r"\.(attn|ffn)_hc\.(fn|base|scale)$")
_MTP = re.compile(r"^mtp\.(\d+)\.(.*)$")
_ROUTED = re.compile(r"^layers\.(\d+)\.mlp\.(?:experts\.(\d+)\.|switch_mlp\.)(gate|up|down)_proj\.(weight|scales|biases)$")


def strip_prefix(name: str) -> str | None:
    """The name below the language model's prefix; None for another tower's tensors (``vision_model.*``)."""

    if name.startswith("lm_head."):
        return name
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return None


def canonical(name: str, mtp_layer: int | None = None) -> str | None:
    """A raw tensor name's short name (``mtp.0.*`` onto layer ``mtp_layer``), or None when the loader skips it."""

    short = strip_prefix(name)
    if short is None:
        return None
    m = _MTP.match(short)
    if m:
        if mtp_layer is None or int(m.group(1)) != 0:
            return None
        rest = m.group(2)
        if rest.startswith("block."):
            rest = rest[len("block."):]
        elif rest == "norm.weight":
            rest = "shared_head.norm.weight"
        short = f"layers.{mtp_layer}.{rest}"
    m = _HC.search(short)
    if m:
        short = short[: m.start()] + f".hc_{m.group(1)}_{m.group(2)}"
    short = short.replace(".self_attn.forget_gate.", ".self_attn.")
    return short


def routed_expert(name: str, layers: int) -> tuple | None:
    """A decoder layer's routed expert tensor as (layer, proj, part) for a stack or (layer, proj, part, expert)."""

    m = _ROUTED.match(canonical(name, layers) or "")
    if m is None or int(m.group(1)) >= layers:
        return None
    key = (int(m.group(1)), m.group(3), m.group(4))
    return key if m.group(2) is None else (*key, int(m.group(2)))


def detect(names: list[str] | dict) -> str:
    """Which layout a checkpoint's index names are in."""

    for name in names:
        if ".attn_hc." in name or ".forget_gate." in name or ".embed_q." in name or ".mtp.0." in name:
            return MLXLM
        if ".hc_attn_" in name or ".kv_b_proj." in name:
            return VONTRA
    return VONTRA


def mtp_layer_names(names: list[str] | dict, mtp_layer: int) -> bool:
    """Whether the index holds an MTP layer in either layout (``layers.<n>.eh_proj`` or ``mtp.0.eh_proj``)."""

    for name in names:
        short = canonical(name, mtp_layer)
        if short == f"layers.{mtp_layer}.eh_proj.weight":
            return True
    return False


__all__ = ["MLXLM", "VONTRA", "canonical", "detect", "mtp_layer_names", "routed_expert", "strip_prefix"]
