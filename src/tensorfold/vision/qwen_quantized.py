"""Conversione delle proiezioni visuali quantizzate dei checkpoint Qwen."""

from __future__ import annotations

import numpy as np


def dequantize_linear(
    weight: np.ndarray,
    scale: np.ndarray,
    scale_2: np.ndarray | float | None = None,
    *,
    scheme: str,
    group_size: int,
) -> np.ndarray:
    """Espande una matrice ModelOpt MXFP8 o NVFP4 in pesi float32."""
    from tensorfold.cuda.nvfp4.format import dequant

    if scheme not in ("nvfp4", "mxfp8"):
        raise ValueError(f"schema visuale non supportato: {scheme}")
    expected_group = 16 if scheme == "nvfp4" else 32
    if group_size != expected_group:
        raise ValueError(f"{scheme} richiede group_size={expected_group}")
    if not isinstance(weight, np.ndarray) or weight.dtype != np.uint8 or weight.ndim != 2:
        raise ValueError("weight deve essere una matrice NumPy U8")
    if not isinstance(scale, np.ndarray) or scale.dtype != np.uint8 or scale.ndim != 2:
        raise ValueError("scale deve essere una matrice NumPy U8")
    if weight.shape[0] <= 0 or weight.shape[1] <= 0 or scale.shape[0] != weight.shape[0]:
        raise ValueError("dimensioni weight/scale non valide")

    if scheme == "nvfp4":
        if scale_2 is None:
            raise ValueError("NVFP4 richiede la scala globale scale_2")
        global_scale = np.asarray(scale_2)
        if global_scale.shape != () or not np.issubdtype(global_scale.dtype, np.number):
            raise ValueError("NVFP4 scale_2 deve essere uno scalare numerico")
        global_value = float(global_scale)
        if not np.isfinite(global_value) or global_value <= 0:
            raise ValueError("NVFP4 scale_2 deve essere finita e positiva")
        columns = weight.shape[1] * 2
        if columns % group_size or scale.shape[1] != columns // group_size:
            raise ValueError("dimensioni NVFP4 incompatibili con il gruppo di 16 valori")
        decoded = dequant("nvfp4", weight, scale, global_value)
    else:
        if scale_2 is not None:
            raise ValueError("MXFP8 non usa scale_2")
        columns = weight.shape[1]
        if columns % group_size or scale.shape[1] != columns // group_size:
            raise ValueError("dimensioni MXFP8 incompatibili con il gruppo di 32 valori")
        decoded = dequant("mxfp8", weight, scale)

    if decoded.shape != (weight.shape[0], columns) or not np.isfinite(decoded).all():
        raise ValueError("la dequantizzazione visuale ha prodotto valori non finiti o una forma errata")
    return np.ascontiguousarray(decoded, dtype=np.float32)
