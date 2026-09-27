"""GLM-5.3-Flash (model_type ``glm5_next``): an MLX engine on one Apple Silicon Mac, and a CUDA engine tensor
parallel over two DGX Sparks.

45 decoder layers over a hidden size of 4,096: 34 of Kimi delta attention and 11 of DeepSeek sparse attention
(MLA with an indexer), 288 routed experts (top 8) plus a shared expert, four residual streams mixed by
hyper-connections, a 154,880-token vocabulary and an MTP layer. The MLX 4-bit checkpoint is 182 GB and Mia's
EXL3 one (routed experts in ExLlamaV3's 4-bit trellis format, the rest in BF16, ``cuda/exl3.py``) 164 GB.

On a Mac (``load``): ``model`` is TensorFold's forward pass for the checkpoint, with a decode path whose rows each
get one-row bits; ``tensorfold.kernels.glm.flash.v1`` holds its Metal kernels; ``mtp`` is the checkpoint's MTP
layer; ``runtime`` is what the lane engine's family rounds serve (``engine.lane_family``: exact MTP drafting when
the load-time row check passes). It needs a Mac with 256 GB or more and reads the MLX 4-bit checkpoint only.

On NVIDIA GPUs (``cuda_engine``): each Spark holds half of every layer (``cuda/``). Drafts come from the
checkpoint's MTP head and, when it has been pulled on both machines, from the DFlash2 draft model.
Recipe and measurements for both: docs/recipes/glm-5.3-flash.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("glm5_next",)
TITLE = "GLM-5.3-Flash"
LANES = True
# 4-bit weights in groups of 64 (what the Metal and CUDA kernels read), with the checkpoint's MTP layer kept;
# the EXL3 checkpoint is the CUDA engine's alone
MODELS = ("Vontra/GLM-5.3-Flash-MLX-4bit-MTP", "Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw")
DRAFTER = "incoai/GLM-5.3-Flash-DFlash2"   # the CUDA engine's optional draft model; the Mac engine drafts with MTP
KERNEL_PACKAGE = "tensorfold.kernels.glm.flash.v1"
KERNEL_VERSION = "v1"
# the storage formats each engine reads: the Mac engine MLX affine 4-bit; the CUDA engine that, and EXL3 routed
# experts with BF16 elsewhere
QUANT_METHODS = {"mlx": ("mlx",), "cuda": ("mlx", "exl3")}
# the EXL3 variant the CUDA kernels read (4-bit trellis, the "mcg" codebook, routed experts only)
EXL3_VARIANT = {"bits": 4, "codebook": "mcg", "scope": "glm53_routed_experts_only"}
# MLX command buffers: GLM's decode step is about 1,200 kernels a token (set before MLX starts, as for Flash Next)
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "100000"}


def check(model_dir: str | Path) -> None:
    """The Mac engine reads MLX affine 4-bit weights in groups of 64; the CUDA engine those, or Mia's EXL3 layout
    (4-bit mcg trellis routed experts, BF16 elsewhere), and runs on two GPUs."""

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
        from tensorfold.families.glm5_next.mtp import has_mtp

        if (method != "exl3" and (Path(model_dir) / "model.safetensors.index.json").is_file()
                and not has_mtp(model_dir)):
            print(f"[tensorfold] this checkpoint has no MTP layer: decoding without MTP drafts ({MODELS[0]} has "
                  f"one)", flush=True)
        return
    print("[tensorfold] GLM-5.3-Flash runs on two NVIDIA GPUs with 128 GB each (two DGX Sparks): pull it on both "
          "and serve with --tp 2 on both (docs/recipes/glm-5.3-flash.md)", flush=True)


def load(model_dir: Path, *, mtp_drafts: int | None = None, **_: Any) -> tuple[Any, Any]:
    """The MLX engine. ``mtp_drafts``: the most MTP drafts a round (default ``runtime.load``; 0: none). The lane
    engine picks each round's depth from the stream's recent acceptance and the measured window costs."""

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
    """Names the kernels that computed a prefix snapshot: the MLX engine's and the kernel package's sources, the
    MLX version (MLX's own kernels compute the prefill) and the switches that change the decode path's arithmetic."""

    import hashlib
    import importlib

    import mlx.core as mx

    from tensorfold.families.glm5_next import model as glm

    digest = hashlib.sha256()
    for module in (__name__, KERNEL_PACKAGE):
        folder = Path(str(importlib.import_module(module).__file__)).parent
        for path in sorted(folder.glob("*.py")):       # this folder only: the CUDA engine (cuda/) is not on this path
            digest.update(path.relative_to(folder).as_posix().encode())
            digest.update(path.read_bytes())
    digest.update(mx.__version__.encode())
    digest.update(f"fused_kda={glm.FUSED_KDA} sparse={glm.SPARSE_KERNEL} fused={sorted(glm.FUSED)}".encode())
    return f"{MODEL_TYPES[0]}-{KERNEL_VERSION}-" + digest.hexdigest()[:12]


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
