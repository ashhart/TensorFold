"""GLM-5.3-Flash (model_type ``glm5_next``): an MLX engine on a 256 GB Mac, a CUDA engine over two DGX Sparks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("glm5_next",)
TITLE = "GLM-5.3-Flash"
LANES = True
# 4-bit weights in groups of 64 with the MTP layer kept; the EXL3 checkpoint is the CUDA engine's alone
MODELS = ("Vontra/GLM-5.3-Flash-MLX-4bit-MTP", "Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw")
DRAFTER = "incoai/GLM-5.3-Flash-DFlash2"   # the CUDA engine's optional draft model; the Mac engine drafts with MTP
KERNEL_PACKAGE = "tensorfold.kernels.glm.flash.v1"
KERNEL_VERSION = "v1"
# the storage formats each engine reads: MLX affine on a Mac; that or EXL3 routed experts on CUDA
QUANT_METHODS = {"mlx": ("mlx",), "cuda": ("mlx", "exl3")}
# the EXL3 variant the CUDA kernels read (4-bit trellis, the "mcg" codebook, routed experts only)
EXL3_VARIANT = {"bits": 4, "codebook": "mcg", "scope": "glm53_routed_experts_only"}
# MLX command buffers sized for ~1,200 kernels a token; no TF32, as the row kernels repeat MLX's fp32 matmuls
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "100000", "MLX_ENABLE_TF32": "0"}


def check(model_dir: str | Path) -> None:
    """Refuse what neither engine reads: MLX affine weights on a Mac; those or Mia's EXL3 layout on two GPUs."""

    import sys

    from tensorfold.families import OWN_MODEL_HELP, describe_quantization, quant_method, quantization, read_config

    config = read_config(model_dir)
    method = quant_method(config)
    if method == "exl3":
        # the CUDA engine's layout; the Mac engine refuses it before this through QUANT_METHODS (require_readable)
        found = config.get("quantization_config") or config.get("quantization") or {}
        got = {k: found.get(k) for k in EXL3_VARIANT}
        if {k: (int(v) if k == "bits" and v is not None else v) for k, v in got.items()} != EXL3_VARIANT:
            raise ValueError(f"GLM-5.3-Flash's CUDA engine reads EXL3 checkpoints with 4-bit mcg-codebook routed "
                             f"experts and BF16 elsewhere ({MODELS[1]}); this one has "
                             + ", ".join(f"{k} {v}" for k, v in got.items()) + f". {OWN_MODEL_HELP}")
        print("[tensorfold] EXL3 support is experimental: replies are exact, but the MLX checkpoint "
              f"({MODELS[0]}) is tested more and runs faster (docs/recipes/glm-5.3-flash.md)", flush=True)
    elif quantization(config) != (4, 64):
        raise ValueError(f"GLM-5.3-Flash's kernels read MLX 4-bit weights in groups of 64 ({MODELS[0]}) or, on "
                         f"CUDA, EXL3 ({MODELS[1]}); this checkpoint has {describe_quantization(config)}. "
                         f"{OWN_MODEL_HELP}")
    if sys.platform == "darwin":
        from tensorfold.families.glm5_next.config import quant_formats, unreadable

        _require_mlx((0, 32, 2))
        bad = sorted(name for name, fmt in quant_formats(config)[1].items() if unreadable(fmt))
        if bad:
            raise ValueError(f"GLM-5.3-Flash's Mac engine reads MLX affine weights of 2 to 8 bits in groups of 32, 64 or "
                             f"128; this checkpoint stores {len(bad)} module(s) otherwise, {bad[0]} first. {OWN_MODEL_HELP}")
        if (method != "exl3" and (Path(model_dir) / "model.safetensors.index.json").is_file()
                and not has_mtp(model_dir)):
            print(f"[tensorfold] this checkpoint has no MTP layer: decoding without MTP drafts ({MODELS[0]} has "
                  f"one)", flush=True)
        return
    print("[tensorfold] GLM-5.3-Flash runs on two NVIDIA GPUs with 128 GB each (two DGX Sparks): pull it on both "
          "and serve with --tp 2 on both (docs/recipes/glm-5.3-flash.md)", flush=True)


def has_mtp(model_dir: str | Path) -> bool:
    """Whether the checkpoint kept the MTP layer; JSON only, so the CLI's check runs before MLX starts."""

    import json

    config = json.loads((Path(model_dir) / "config.json").read_text())
    text = config.get("text_config") or config
    n = int(text.get("num_hidden_layers", 0))
    if int(text.get("num_nextn_predict_layers", 0)) < 1:
        return False
    index = Path(model_dir) / "model.safetensors.index.json"
    if not index.is_file():
        return False
    names = json.loads(index.read_text())["weight_map"]
    return any(name.endswith(f"layers.{n}.eh_proj.weight") for name in names)


def _require_mlx(least: tuple[int, ...]) -> None:
    """Refuse an older MLX, read from the package's metadata (MLX itself is not imported yet)."""

    import re
    from importlib.metadata import PackageNotFoundError, version

    try:
        found = version("mlx")
    except PackageNotFoundError:
        return
    parts = tuple(int(p) for p in re.findall(r"\d+", found)[:3])
    if parts < least:
        need = ".".join(str(p) for p in least)
        raise ValueError(f"GLM-5.3-Flash needs MLX {need} or later (this is {found}): with 0.32.0 its weights dropped "
                         f"out of wired memory on a 256 GB Mac and decoding fell to a few tokens a second. Install it "
                         f"with: python -m pip install \"mlx>={need}\"")


def load(model_dir: Path, *, mtp_drafts: int | None = None, **_: Any) -> tuple[Any, Any]:
    """The MLX engine; ``mtp_drafts`` caps the MTP drafts a round (0: none), each round's depth set by costs."""

    import mlx.core as mx

    from tensorfold.families.glm5_next.runtime import load as load_runtime

    # about 170 GB of weights on a 256 GB Mac: keep them wired, or macOS can page them out between steps
    if mx.metal.is_available():
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        limit = int(info.get("max_recommended_working_set_size", 0))
        if limit:
            mx.set_wired_limit(limit)
    return load_runtime(Path(model_dir), drafts=mtp_drafts)


def engine_settings(model: Any) -> dict[str, Any]:
    """Rows a round verifies at most: the widest window checked exact at load."""

    width = int(getattr(model, "exact_width", 1) or 1)
    return {"max_rows": width, "max_draft": max(0, width - 1)}


def kernel_version(model: Any) -> str:
    """Names the kernels behind a prefix snapshot: this engine's and its kernels' sources, and MLX's version."""

    import hashlib
    import importlib

    import mlx.core as mx

    digest = hashlib.sha256()
    for module in (__name__, KERNEL_PACKAGE):
        folder = Path(str(importlib.import_module(module).__file__)).parent
        for path in sorted(folder.glob("*.py")):       # this folder only: the CUDA engine (cuda/) is not on this path
            digest.update(path.relative_to(folder).as_posix().encode())
            digest.update(path.read_bytes())
    digest.update(mx.__version__.encode())
    return f"{MODEL_TYPES[0]}-{KERNEL_VERSION}-" + digest.hexdigest()[:12]


def memory_fraction(ram_bytes: int) -> float | None:
    """85% of RAM on a Mac of 256 GB or less, approved for this checkpoint with nothing else loaded."""

    return 0.85 if ram_bytes <= 256 * 1024**3 else None


# the CUDA engine's kernels read MLX affine weights of this (bits, group size); EXL3 checkpoints are checked above
CUDA_QUANTIZATION = (4, 64)


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None, **options: Any):
    """Build the two-rank engine with adaptive drafting, reusable prompt state, or serial decoding when drafts are disabled."""

    if int(tp) != 2:
        raise ValueError("GLM-5.3-Flash needs two GPUs, one per machine: run the same `tensorfold serve` command "
                         "with --tp 2 --rank R --master ADDRESS on both (rank 1 first)")
    if not master:
        raise ValueError("--tp 2 needs --master: rank 0's address on the link between the two machines")
    from .cuda.engine import DEFAULT_POLICY, DFLASH_POLICY, GlmEngine

    if mtp_drafts is None:
        policy = DEFAULT_POLICY
    elif int(mtp_drafts) == 0 and drafter and not no_drafts:
        policy = DFLASH_POLICY          # no MTP drafts: every round still verifies DFlash2's drafts
    else:
        policy = str(int(mtp_drafts))
    return GlmEngine(Path(model_dir), rank=int(rank), master=master, port=int(master_port), policy=policy,
                     drafter=Path(drafter) if drafter and not no_drafts else None,
                     context=options.get("context"), context_explicit=options.get("context_explicit"),
                     serial_only=bool(no_drafts))


def __getattr__(name: str) -> Any:
    if name == "CUDA_APP":             # imported on first use, so the Mac side never loads the CUDA server
        from .cuda.app import GlmApp

        return GlmApp
    raise AttributeError(name)
