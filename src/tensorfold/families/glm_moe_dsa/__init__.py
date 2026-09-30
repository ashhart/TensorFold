"""Full GLM-5.3 (model_type ``glm_moe_dsa``, 753B): a CUDA engine over four DGX Sparks (``--tp 4``), EXL3 checkpoints.

Design and plan: docs/design/glm-moe-dsa-tp4.md. MLA with RoPE and a token-level DSA indexer on every layer,
256 routed experts (top 8) + 1 shared expert, three dense layers, one MTP layer. No Mac engine: the model does not
fit one machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("glm_moe_dsa",)
TITLE = "GLM-5.3"
MODELS = ()                       # the qualified checkpoint is added once it is public
QUANT_METHODS = {"cuda": ("exl3",)}
EXL3_VARIANT = "any"              # any codebook and width per tensor (the universal EXL3 module reads them all)
CUDA_TP = (4,)                    # ~276 GB at 2.75 bpw: four 128 GB GPUs, one per machine


def check(model_dir: str | Path) -> None:
    """Refuse what the engine does not read: non-EXL3 weights, or a config this family does not describe."""

    from tensorfold.families import OWN_MODEL_HELP, quant_method, read_config

    from .config import Config

    config = read_config(model_dir)
    if quant_method(config) != "exl3":
        raise ValueError(f"GLM-5.3's CUDA engine reads EXL3 checkpoints (routed experts at any width, other linears "
                         f"EXL3 or bf16); this checkpoint is {quant_method(config) or 'unquantized'}. {OWN_MODEL_HELP}")
    Config.from_dict(config)       # raises on settings the engine does not implement
    print("[tensorfold] GLM-5.3 runs on four NVIDIA GPUs with 128 GB each (four DGX Sparks): serve with --tp 4 on "
          "all four (docs/design/glm-moe-dsa-tp4.md)", flush=True)


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None, **options: Any):
    if int(tp) not in CUDA_TP:
        raise ValueError("GLM-5.3 needs four GPUs, one per machine: run the same `tensorfold serve` command with "
                         "--tp 4 --rank R --master ADDRESS on all four (ranks 1-3 first)")
    if not master:
        raise ValueError("--tp 4 needs --master: rank 0's address on the link between the machines")
    from .cuda.engine import Glm53Engine

    # eager engine: MTP drafts verified exactly, token-level DSA (docs/design/glm-moe-dsa-tp4.md)
    k = 0 if no_drafts else (2 if mtp_drafts is None else int(mtp_drafts))
    return Glm53Engine(Path(model_dir), rank=int(rank), master=master, port=int(master_port),
                       context=options.get("context"), mtp_drafts=k)
