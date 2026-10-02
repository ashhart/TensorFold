"""DeepSeek-V4-Flash: MLX on Mac or mapped-GGUF CUDA on Spark, with optional DSpark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("deepseek_v4",)
TITLE = "DeepSeek-V4-Flash"
LANES = True
# affine 4-bit groups of 64, routed experts in mxfp4 (DeepSeek's own FP4 bytes)
MODELS = ("mlx-community/DeepSeek-V4-Flash-4bit",)
# DeepSeek's DSpark blocks converted (MIT); Vontra/DeepSeek-V4-Flash-MTP-MLX holds the MTP layer the same way
DRAFTER = "Vontra/DeepSeek-V4-Flash-DSpark-MLX"
# Native CUDA drafting requires an explicitly selected local DSpark GGUF.
CUDA_DRAFTER = ""
KERNEL_PACKAGE = "tensorfold.kernels.deepseek.v4"
KERNEL_VERSION = "v1"
# the shared GLM-5.3 pieces this engine runs (hyper-connections, row linears), hashed into snapshot keys
KERNEL_DEPENDENCIES = ("tensorfold.kernels.glm.flash.v1", "tensorfold.families.glm5_next.linear",
                       "tensorfold.families.glm5_next.model")
QUANT_METHODS = {"mlx": ("mlx",), "cuda": ("gguf",)}
# buffers of 200 ops and 200 MB, so a prompt chunk's memory frees as it runs; no TF32: row kernels repeat fp32
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "200", "MLX_ENABLE_TF32": "0"}
LEAST_MLX = (0, 32, 2)


def check(model_dir: str | Path) -> None:
    """Refuse what the engine does not read, from config.json alone (MLX is not imported here)."""

    import sys

    from tensorfold.families import OWN_MODEL_HELP, quant_method, read_config
    config = read_config(model_dir)
    if quant_method(config) == "gguf":
        import json
        descriptor = Path(model_dir) / "descriptor.json"
        if not descriptor.is_file() or not json.loads(descriptor.read_text()).get("source"):
            raise ValueError("DeepSeek GGUF needs prepared storage/provenance descriptor.json")
        return  # Native admission validates actual headers before loading; no MLX import.
    from tensorfold.families.deepseek_v4.config import Config
    from tensorfold.families.deepseek_v4.weights import unreadable
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


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0,
                master: str = "", master_port: int = 29551, no_drafts: bool = False,
                mtp_drafts: int | None = None, **options: Any):
    """One mapped-GGUF worker behind TensorFold serving; optional local DSpark head."""
    allowed = {"parallel", "streams", "context", "context_explicit", "threads"}
    unsupported = set(options) - allowed
    if unsupported:
        raise ValueError(f"unsupported DeepSeek CUDA options: {', '.join(sorted(unsupported))}")
    from .cuda.engine import DeepSeekEngine
    return DeepSeekEngine(model_dir, tp=tp, rank=rank, drafter=drafter, no_drafts=no_drafts,
                          mtp_drafts=mtp_drafts, **options)


def __getattr__(name: str) -> Any:
    if name == "CUDA_APP":
        from .cuda.app import DeepSeekApp
        return DeepSeekApp
    raise AttributeError(name)


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
