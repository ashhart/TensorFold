"""Checkpoint layouts of GLM-5.3-Flash and the one set of tensor names the loader reads.

Two layouts exist on disk for the same weights:

- **vontra** (``Vontra/GLM-5.3-Flash-MLX-4bit-MTP``, the layout ``model.py`` was written from): names
  ``model.language_model.layers.N....``, hyper-connections ``hc_attn_fn / hc_attn_base / hc_attn_scale`` (and
  ``hc_ffn_*``), KDA taps as ``q_conv1d / k_conv1d / v_conv1d`` [C, 1, T] with ``A_log``, ``dt_bias``,
  ``f_a_proj``, ``f_b_proj`` straight under ``self_attn``, MLA with one ``kv_b_proj`` [H (nope + v), rank], the
  MTP layer as ``layers.<num_hidden_layers>`` with ``eh_proj`` (quantized), ``enorm``, ``hnorm``,
  ``shared_head.norm``.
- **mlxlm** (what ``mlx_lm.convert`` / oMLX 0.6.4 write, e.g. ``grant-ai/GLM-5.3-Flash-Abliterated-MLX-4bit``):
  names ``language_model.model.layers.N....``, hyper-connections ``attn_hc.{fn,base,scale}`` / ``ffn_hc.*``
  (fp32), one fused ``self_attn.conv1d.weight`` [3 C, T, 1] over q | k | v, the forget gate under
  ``self_attn.forget_gate.{A_log,dt_bias,f_a_proj,f_b_proj}``, MLA with the absorbed pair ``embed_q``
  [H, rank, nope] (kv_b's key half transposed and re-quantized along nope) and ``unembed_out`` [H, v, rank]
  (kv_b's value half as stored) in place of ``kv_b_proj``, the MTP layer under ``language_model.mtp.0.block.*``
  with ``mtp.0.{eh_proj (bf16, unquantized), enorm, hnorm, norm}``, a ``vision_model.*`` the text model never
  reads, and per-tensor quantization overrides (8 / 6 / 5-bit attention and shared experts beside 4-bit routed
  experts; ``config.json``'s ``quantization`` lists them, and each tensor's shapes say the same).

``canonical`` turns a raw name of either layout into the vontra short name ``model.py`` asks for, so one loader
reads both. What a rename cannot express — the fused conv, the absorbed MLA pair, the unquantized eh_proj — the
loader handles where it reads those tensors (``load_layer``, ``MLA``, ``mtp.load``). Nothing here touches MLX:
names only, so both the 0.3.4 rebase and the glm-5.3-flash branch can import it as is.
"""

from __future__ import annotations

import re

VONTRA = "vontra"
MLXLM = "mlxlm"

_PREFIXES = ("model.language_model.", "language_model.model.", "language_model.")
_HC = re.compile(r"\.(attn|ffn)_hc\.(fn|base|scale)$")
_MTP = re.compile(r"^mtp\.(\d+)\.(.*)$")
_LAYER = re.compile(r"^layers\.(\d+)\.")


def strip_prefix(name: str) -> str | None:
    """The name below the language model (``layers.N....``, ``embed_tokens.*``, ``norm.weight``, ``lm_head.*``,
    ``mtp.K.*``); None for tensors of another tower (``vision_model.*``)."""

    if name.startswith("lm_head."):
        return name
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            rest = name[len(prefix):]
            if rest.startswith("lm_head."):
                return rest
            return rest
    return None


def canonical(name: str, mtp_layer: int | None = None) -> str | None:
    """The vontra short name of a raw checkpoint tensor name, or None when the loader does not read it.

    ``mtp_layer``: the index the MTP layer takes (``num_hidden_layers``); the mlxlm ``mtp.0.*`` names map onto
    it. Vontra names pass through unchanged (after the prefix), so a vontra checkpoint reads exactly as before.
    """

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


__all__ = ["MLXLM", "VONTRA", "canonical", "detect", "mtp_layer_names", "strip_prefix"]
