"""Pure NemotronH layout helpers (no MLX): cache indexing + tensor partition."""
from __future__ import annotations

from typing import Any

_BLOCK_CHAR = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}


def plan_blocks(layers_block_type: list[str]) -> dict[str, Any]:
    """Return per-layer block_type/cache_index/mask + fa_idx/ssm_idx/n_cache.

    Mirrors mlx_lm.models.nemotron_h.NemotronHModel: a cache entry exists only
    for 'M' and '*' layers; 'E'/'-' carry no cache. Accepts either the word
    forms used in config's ``layers_block_type`` ("mamba"/"attention"/"moe"/
    "mlp") or the single-char codes ("M"/"*"/"E"/"-").
    """
    chars = [_BLOCK_CHAR.get(b, b) for b in layers_block_type]
    layers = []
    counter = 0
    for ch in chars:
        if ch in ("M", "*"):
            cache_index = counter
            counter += 1
        else:
            cache_index = None
        layers.append(
            {
                "block_type": ch,
                "cache_index": cache_index,
                "mask": "attn" if ch == "*" else "ssm",
            }
        )
    # fa_idx: count leading M's until first '*' (NemotronHModel scan).
    fa_idx = 0
    for ch in chars:
        if ch == "*":
            break
        if ch == "M":
            fa_idx += 1
    # ssm_idx: count leading '*'s until first 'M' (NemotronHModel scan).
    ssm_idx = 0
    for ch in chars:
        if ch == "M":
            break
        if ch == "*":
            ssm_idx += 1
    return {"layers": layers, "fa_idx": fa_idx, "ssm_idx": ssm_idx, "n_cache": counter}


def partition_layer_tensors(tensor_names: list[str]) -> tuple[list[str], list[str]]:
    """Split a layer's tensor names into (base, routed_experts).

    Routed experts are the stacked SwitchMLP weights/scales/biases under
    '.mixer.switch_mlp.'; everything else (gate, latent projs, shared experts,
    mamba/attention weights, norms) is base and stays resident.
    """
    base, experts = [], []
    for name in tensor_names:
        if ".mixer.switch_mlp." in name:
            experts.append(name)
        else:
            base.append(name)
    return base, experts
