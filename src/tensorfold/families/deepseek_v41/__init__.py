"""DeepSeek-V4.1-Flash (model_type ``deepseek_v41``): an MLX engine for oMLX's converted checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("deepseek_v41",)
TITLE = "DeepSeek-V4.1-Flash"
LANES = True
# oMLX's oQ4e conversion: DeepSeek's FP8 projections as mxfp8, its FP4 experts as mxfp4, Engram tables affine 4-bit
MODELS = ("Jundot/DeepSeek-V4.1-Flash-oQ4e-mtp",)
KERNEL_PACKAGE = "tensorfold.kernels.deepseek.v41"
KERNEL_VERSION = "v1"
KERNEL_DEPENDENCIES = ("tensorfold.families.deepseek_v4.runtime", "tensorfold.families.glm5_next.runtime")
# the converted checkpoint declares its formats in its own block, so the generic reader sees no quantization block
QUANT_METHODS = {"mlx": (None, "mlx")}
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "200", "MLX_ENABLE_TF32": "0"}
LEAST_MLX = (0, 32, 2)


def check(model_dir: str | Path) -> None:
    """Refuse what the engine does not read, from config.json alone (MLX is not imported here)."""

    from tensorfold.families import OWN_MODEL_HELP, read_config
    from tensorfold.families.deepseek_v41.config import Config
    from tensorfold.families.deepseek_v41.weights import unreadable

    config = read_config(model_dir)
    Config.from_dict(config)
    try:
        bad = unreadable(config)
    except ValueError as exc:
        raise ValueError(f"{exc}. {OWN_MODEL_HELP}") from None
    if bad:
        raise ValueError(f"DeepSeek-V4.1-Flash's Mac engine reads oMLX's converted layout ({MODELS[0]}); this "
                         f"checkpoint stores {len(bad)} module(s) otherwise, {bad[0]} first. {OWN_MODEL_HELP}")


def weight_bytes(model_dir: str | Path, ple_on_ssd: bool = False) -> int:
    """Resident weights: every safetensors byte but the Engram tables (read from their files a row at a time)."""

    from tensorfold.families.deepseek_v41.weights import engram_bytes

    total = sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors"))
    return total - engram_bytes(Path(model_dir))


def load(model_dir: Path, **options: Any) -> tuple[Any, Any]:
    import mlx.core as mx

    from tensorfold.families.deepseek_v41.runtime import load as load_runtime

    if mx.metal.is_available():
        limit = int(mx.device_info().get("max_recommended_working_set_size", 0))
        if limit:
            mx.set_wired_limit(limit)
    return load_runtime(Path(model_dir), **options)


def engine_settings(model: Any) -> dict[str, Any]:
    width = int(getattr(model, "exact_width", 1) or 1)
    return {"max_rows": width, "max_draft": max(0, width - 1)}


def kernel_version(model: Any) -> str:
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
