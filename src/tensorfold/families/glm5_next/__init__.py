"""GLM-5.3-Flash (model_type ``glm5_next``) on two NVIDIA GPUs: a CUDA engine only, tensor parallel over two
DGX Sparks.

45 decoder layers over a hidden size of 4,096: 34 of Kimi delta attention and 11 of DeepSeek sparse attention
(MLA with an indexer), 288 routed experts (top 8) plus a shared expert, four residual streams mixed by
hyper-connections, a 154,880-token vocabulary and an MTP layer. The 4-bit checkpoint is 182 GB, so each Spark
holds half of every layer (``cuda/``). Drafts come from the checkpoint's MTP head and, when it has been pulled on
both machines, from the DFlash2 draft model.

There is no MLX engine for this family (no ``load``), so ``tensorfold serve`` refuses the MLX backend for it.
Recipe and measurements: docs/recipes/glm-5.3-flash.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("glm5_next",)
TITLE = "GLM-5.3-Flash"
MODELS = ("Vontra/GLM-5.3-Flash-MLX-4bit-MTP",)
DRAFTER = "incoai/GLM-5.3-Flash-DFlash2"


def check(model_dir: str | Path) -> None:
    """The engine reads MLX affine 4-bit weights in groups of 64 and runs on two GPUs."""

    from tensorfold.families import quantization, read_config

    bits, group = quantization(read_config(model_dir))
    if (bits, group) != (4, 64):
        raise ValueError(f"GLM-5.3-Flash's CUDA engine reads 4-bit weights in groups of 64 ({MODELS[0]}), "
                         f"this checkpoint has {bits}-bit, groups of {group}")
    print("[tensorfold] GLM-5.3-Flash runs on two NVIDIA GPUs with 128 GB each (two DGX Sparks): pull it on both "
          "and serve with --tp 2 on both (docs/recipes/glm-5.3-flash.md)", flush=True)


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None, **options: Any):
    """The CUDA engine, set up as the recipe measured on two DGX Sparks.

    Each rank reads its half of the checkpoint (``cuda/split.py``) and the ranks all-gather fp32 partials every
    layer. By default a greedy request drafts each round with the MTP head or, with ``drafter`` (both machines),
    DFlash2, whichever has committed more tokens per millisecond so far; a sampled request drafts with the MTP
    head, 1 to 3 drafts a round from the running acceptance. A request can ask for another policy
    (``cuda/app.py``, specs in ``cuda/engine.py``). A prompt that extends the last request's prompt or reply
    resumes from its kept state.
    ``mtp_drafts``: a fixed number of MTP drafts a round instead (0: serial). ``no_drafts``: serial decoding only.
    ``options["context"]``: prompt plus reply tokens; up to 2,051 (the default) attention stays dense, as measured;
    longer contexts run DSA's sparse top-k past 2,051 tokens without CUDA graphs.
    """

    if int(tp) != 2:
        raise ValueError("GLM-5.3-Flash needs two GPUs, one per machine: run the same `tensorfold serve` command "
                         "with --tp 2 --rank R --master ADDRESS on both (rank 1 first)")
    if not master:
        raise ValueError("--tp 2 needs --master: rank 0's address on the link between the two machines")
    from .cuda.engine import DEFAULT_POLICY, GlmEngine

    policy = DEFAULT_POLICY if mtp_drafts is None else str(int(mtp_drafts))
    return GlmEngine(Path(model_dir), rank=int(rank), master=master, port=int(master_port), policy=policy,
                     drafter=Path(drafter) if drafter and not no_drafts else None,
                     context=int(options.get("context") or 0), serial_only=bool(no_drafts))


def __getattr__(name: str) -> Any:
    if name == "CUDA_APP":             # imported on first use, so the Mac side never loads the CUDA server
        from .cuda.app import GlmApp

        return GlmApp
    raise AttributeError(name)
