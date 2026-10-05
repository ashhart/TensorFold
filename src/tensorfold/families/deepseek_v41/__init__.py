"""DeepSeek-V4.1-Flash (model_type ``deepseek_v41``): an MLX engine on Apple Silicon."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")
TITLE = "DeepSeek-V4.1-Flash"
LANES = True
# affine 3-bit groups of 64 (a 4-bit conversion of the same layout reads too), converted from the official release
MODELS = ("deepseek-ai/DeepSeek-V4.1-Flash",)
KERNEL_PACKAGE = "tensorfold.kernels.deepseek.v41"
KERNEL_VERSION = "v1"
# the shared pieces this engine runs (hyper-connections, row linears, V4's attention/rope kernels)
KERNEL_DEPENDENCIES = ("tensorfold.kernels.deepseek.v4", "tensorfold.kernels.glm.flash.v1",
                       "tensorfold.families.glm5_next.linear", "tensorfold.families.glm5_next.model")
QUANT_METHODS = {"mlx": ("mlx",)}
# buffers of 200 ops and 200 MB, so a prompt chunk's memory frees as it runs; no TF32: row paths repeat fp32
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "200", "MLX_ENABLE_TF32": "0"}
LEAST_MLX = (0, 32, 2)


def check(model_dir: str | Path) -> None:
    """Refuse what the engine does not read, from config.json alone (MLX is not imported here)."""
    import sys

    from tensorfold.families import OWN_MODEL_HELP, read_config
    from tensorfold.families.deepseek_v41.config import Config
    from tensorfold.families.deepseek_v41.weights import unreadable

    config = read_config(model_dir)
    Config.from_dict(config)
    bad = unreadable(config)
    if bad:
        raise ValueError(f"DeepSeek-V4.1-Flash's engine reads MLX affine 3-bit g64 weights (a 4-bit conversion of "
                         f"the same layout reads too); this checkpoint stores {len(bad)} module(s) otherwise, "
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
        raise ValueError(f"DeepSeek-V4.1-Flash needs MLX {need} or later (this is {found}): install it with "
                         f"python -m pip install \"mlx>={need}\"")


def load(model_dir: Path, **options: Any) -> tuple[Any, Any]:
    """The MLX engine and its tokenizer."""
    import mlx.core as mx

    from tensorfold.families.deepseek_v41.runtime import load as load_runtime

    # about 337 GB of weights on a 512 GiB-class Mac: keep them wired, or macOS can page them out between steps
    if mx.metal.is_available():
        limit = int(mx.device_info().get("max_recommended_working_set_size", 0))
        if limit:
            mx.set_wired_limit(limit)
    return load_runtime(Path(model_dir), **options)


def engine_settings(model: Any) -> dict[str, Any]:
    """Rows a round verifies at most: the widest window checked exact at load."""
    width = int(getattr(model, "exact_width", 1) or 1)
    return {"max_rows": width, "max_draft": max(0, width - 1)}


def weight_bytes(model_dir: str | Path, ple_on_ssd: bool = False) -> int:
    """Text-only resident bytes: every safetensors byte minus vision/aligner/image_* (the family never reads them)."""
    import json

    model_dir = Path(model_dir)
    total = 0
    shards: set[str] = set()
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        for name, shard in json.loads(index.read_text())["weight_map"].items():
            if name.startswith(("vision.", "aligner.", "image_")):
                continue
            shards.add(shard)
    for shard in sorted(shards) or [p for p in model_dir.glob("*.safetensors")]:
        total += (model_dir / shard).stat().st_size
    return total


def kernel_version(model: Any) -> str:
    """Names the kernels behind a prefix snapshot: this engine's, its kernels' and the shared pieces' sources."""
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
