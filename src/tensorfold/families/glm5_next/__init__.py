"""GLM-5.3-Flash uses a CUDA-only engine with tensor parallelism over two GPUs and MTP or DFlash2 drafts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("glm5_next",)
TITLE = "GLM-5.3-Flash"
MODELS = ("Vontra/GLM-5.3-Flash-MLX-4bit-MTP", "Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw")
DRAFTER = "incoai/GLM-5.3-Flash-DFlash2"
# the storage formats the CUDA engine reads: MLX affine 4-bit, and EXL3 routed experts with BF16 elsewhere
QUANT_METHODS = {"cuda": ("mlx", "exl3")}
# the EXL3 variant the kernels read (4-bit trellis, the "mcg" codebook, routed experts only)
EXL3_VARIANT = {"bits": 4, "codebook": "mcg", "scope": "glm53_routed_experts_only"}


def check(model_dir: str | Path) -> None:
    """Require two GPUs and MLX affine 4-bit groups of 64 or EXL3 4-bit mcg trellis routed experts with BF16 elsewhere."""

    from tensorfold.families import OWN_MODEL_HELP, describe_quantization, quant_method, quantization, read_config

    config = read_config(model_dir)
    method = quant_method(config)
    if method == "exl3":
        found = config.get("quantization_config") or config.get("quantization") or {}
        got = {k: found.get(k) for k in EXL3_VARIANT}
        if {k: (int(v) if k == "bits" and v is not None else v) for k, v in got.items()} != EXL3_VARIANT:
            raise ValueError(f"GLM-5.3-Flash's CUDA engine reads EXL3 checkpoints with 4-bit mcg-codebook routed "
                             f"experts and BF16 elsewhere ({MODELS[1]}); this one has "
                             + ", ".join(f"{k} {v}" for k, v in got.items()) + f". {OWN_MODEL_HELP}")
        print("[tensorfold] EXL3 support is experimental: replies are exact, but the MLX checkpoint "
              f"({MODELS[0]}) is tested more and runs faster (docs/recipes/glm-5.3-flash.md)", flush=True)
    elif quantization(config) != (4, 64):
        raise ValueError(f"GLM-5.3-Flash's CUDA engine reads MLX 4-bit weights in groups of 64 ({MODELS[0]}) or "
                         f"EXL3 ({MODELS[1]}); this checkpoint has {describe_quantization(config)}. {OWN_MODEL_HELP}")
    print("[tensorfold] GLM-5.3-Flash runs on two NVIDIA GPUs with 128 GB each (two DGX Sparks): pull it on both "
          "and serve with --tp 2 on both (docs/recipes/glm-5.3-flash.md)", flush=True)


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
