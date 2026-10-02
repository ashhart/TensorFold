"""DeepSeek-V4-Flash (model_type ``deepseek_v4``): an MLX engine on a 256 GB Mac."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("deepseek_v4",)
TITLE = "DeepSeek-V4-Flash"
LANES = True
# affine 4-bit groups of 64, routed experts in mxfp4 (DeepSeek's own FP4 bytes)
MODELS = ("mlx-community/DeepSeek-V4-Flash-4bit",)
# DeepSeek's DSpark blocks converted (MIT); TensorFold/DeepSeek-V4-Flash-MTP-MLX holds the MTP layer the same way
DRAFTER = "TensorFold/DeepSeek-V4-Flash-DSpark-MLX"
KERNEL_PACKAGE = "tensorfold.kernels.deepseek.v4"
KERNEL_VERSION = "v1"
# the shared GLM-5.3 pieces this engine runs (hyper-connections, row linears), hashed into snapshot keys
KERNEL_DEPENDENCIES = ("tensorfold.kernels.glm.flash.v1", "tensorfold.families.glm5_next.linear",
                       "tensorfold.families.glm5_next.model")
QUANT_METHODS = {"mlx": ("mlx",)}
# buffers of 200 ops and 200 MB, so a prompt chunk's memory frees as it runs; no TF32: row kernels repeat fp32
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "200", "MLX_ENABLE_TF32": "0"}
LEAST_MLX = (0, 32, 2)


def check(model_dir: str | Path) -> None:
    """Refuse what the engine does not read, from config.json alone (MLX is not imported here)."""

    import sys

    from tensorfold.families import OWN_MODEL_HELP, read_config
    from tensorfold.families.deepseek_v4.config import Config
    from tensorfold.families.deepseek_v4.weights import unreadable

    config = read_config(model_dir)
    Config.from_dict(config)
    bad = unreadable(config)
    if bad:
        raise ValueError(f"DeepSeek-V4-Flash's Mac engine reads MLX affine 4-bit weights in groups of 64 and mxfp4 "
                         f"routed experts ({MODELS[0]}); this checkpoint stores {len(bad)} module(s) otherwise, "
                         f"{bad[0]} first. {OWN_MODEL_HELP}")
    if sys.platform == "darwin":
        _require_mlx(LEAST_MLX)


def _require_mlx(least: tuple[int, ...]) -> None:
    import re
    from importlib.metadata import PackageNotFoundError, version

    try:
        found = version("mlx")
    except PackageNotFoundError:
        return
    if tuple(int(p) for p in re.findall(r"\d+", found)[:3]) < least:
        need = ".".join(str(p) for p in least)
        raise ValueError(f"DeepSeek-V4-Flash needs MLX {need} or later (this is {found}): install it with "
                         f"python -m pip install \"mlx>={need}\"")


def load(model_dir: Path, **options: Any) -> tuple[Any, Any]:
    """The MLX engine and its tokenizer."""

    import mlx.core as mx

    from tensorfold.families.deepseek_v4.runtime import load as load_runtime

    # about 152 GB of weights on a 256 GB Mac: keep them wired, or macOS can page them out between steps
    if mx.metal.is_available():
        limit = int(mx.device_info().get("max_recommended_working_set_size", 0))
        if limit:
            mx.set_wired_limit(limit)
    return load_runtime(Path(model_dir), **options)


def engine_settings(model: Any) -> dict[str, Any]:
    """Rows a round verifies at most: the widest window checked exact at load."""

    width = int(getattr(model, "exact_width", 1) or 1)
    return {"max_rows": width, "max_draft": max(0, width - 1)}


def kernel_version(model: Any) -> str:
    """Names the kernels behind a prefix snapshot: this engine's, its kernels' and the shared GLM pieces' sources."""

    import hashlib
    import importlib

    import mlx.core as mx

    digest = hashlib.sha256()
    for module in (__name__, KERNEL_PACKAGE, *KERNEL_DEPENDENCIES):
        source = Path(str(importlib.import_module(module).__file__))
        paths = sorted(source.parent.glob("*.py")) if source.name == "__init__.py" else [source]
        for path in paths:
            digest.update(path.relative_to(source.parent).as_posix().encode())
            digest.update(path.read_bytes())
    digest.update(mx.__version__.encode())
    return f"{MODEL_TYPES[0]}-{KERNEL_VERSION}-" + digest.hexdigest()[:12]
