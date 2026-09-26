"""GLM-5.3-Flash (model_type ``glm5_next``) on two NVIDIA GPUs: a CUDA engine only, tensor parallel over two
DGX Sparks.

45 decoder layers over a hidden size of 4,096: 34 of Kimi delta attention and 11 of DeepSeek sparse attention
(MLA with an indexer), 288 routed experts (top 8) plus a shared expert, four residual streams mixed by
hyper-connections, a 154,880-token vocabulary and an MTP layer. The MLX 4-bit checkpoint is 182 GB and Mia's
EXL3 one (routed experts in ExLlamaV3's 4-bit trellis format, the rest in BF16, ``cuda/exl3.py``) 164 GB, so each
Spark holds half of every layer (``cuda/``). Drafts come from the checkpoint's MTP head and, when it has been
pulled on both machines, from the DFlash2 draft model.

There is no MLX engine for this family (no ``load``), so ``tensorfold serve`` refuses the MLX backend for it.
Recipe and measurements: docs/recipes/glm-5.3-flash.md.
"""

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
    """The engine reads MLX affine 4-bit weights in groups of 64, or Mia's EXL3 layout (4-bit mcg trellis routed
    experts, BF16 elsewhere), and runs on two GPUs."""

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
    """The CUDA engine, set up as the recipe measured on two DGX Sparks.

    Each rank reads its half of the checkpoint (``cuda/split.py``) and the ranks all-gather fp32 partials every
    layer. By default a greedy request drafts each round with the MTP head or, with ``drafter`` (both machines),
    DFlash2, whichever has committed more tokens per millisecond so far; a sampled request drafts with the MTP
    head, 1 to 3 drafts a round from the running acceptance. A request can ask for another policy
    (``cuda/app.py``, specs in ``cuda/engine.py``). A prompt that extends the last request's prompt or reply
    resumes from its kept state.
    ``mtp_drafts``: a fixed number of MTP drafts a round instead; 0 drafts with DFlash2 alone (``fc5:0.3``) when the
    draft model is there, else it is the serial reference like ``no_drafts`` (serial decoding only).
    ``options["context"]``: prompt plus reply tokens; up to 2,051 (the default) attention stays dense, as measured;
    longer contexts run DSA's sparse top-k past 2,051 tokens without CUDA graphs.
    """

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
                     context=int(options.get("context") or 0), serial_only=bool(no_drafts))


def __getattr__(name: str) -> Any:
    if name == "CUDA_APP":             # imported on first use, so the Mac side never loads the CUDA server
        from .cuda.app import GlmApp

        return GlmApp
    raise AttributeError(name)
