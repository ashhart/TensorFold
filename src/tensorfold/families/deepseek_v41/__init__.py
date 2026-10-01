"""DeepSeek-V4.1-Flash (model_type ``deepseek_v41``): a CUDA engine over two DGX Sparks, EXL3 weights.

Work in progress on branch ``dsv41-cuda``; see TODO.md at the repository root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

MODEL_TYPES = ("deepseek_v41",)
TITLE = "DeepSeek-V4.1-Flash"
# the EXL3 pack (mul1, 2.9 bpw average); the Engram tables come from the original release's shards 47 and 48
MODELS = ("Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw",)
ENGRAM_SOURCE = "deepseek-ai/DeepSeek-V4.1-Flash"
QUANT_METHODS = {"cuda": ("exl3",)}
EXL3_VARIANT = "any"


def engram_dir(model_dir: str | Path) -> Path:
    """Where the unquantized Engram tables live: ``TF_DSV41_ENGRAM_DIR``, else ``engram/`` beside the weights."""

    configured = os.environ.get("TF_DSV41_ENGRAM_DIR")
    return Path(configured) if configured else Path(model_dir) / "engram"


def check(model_dir: str | Path) -> None:
    """Refuse what the engine does not read, from config.json alone (torch is not imported here)."""

    from tensorfold.families import OWN_MODEL_HELP, quant_method, read_config
    from tensorfold.families.deepseek_v41.config import Config

    config = read_config(model_dir)
    Config.from_dict(config)
    if quant_method(config) != "exl3":
        raise ValueError(f"DeepSeek-V4.1-Flash's CUDA engine reads EXL3 checkpoints ({MODELS[0]}). {OWN_MODEL_HELP}")
    print("[tensorfold] DeepSeek-V4.1-Flash runs on two DGX Sparks: pull it on both and serve with --tp 2 on both; "
          f"the Engram tables ({ENGRAM_SOURCE} shards 47-48) go in {engram_dir(model_dir)}", flush=True)


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None, **options: Any):
    """The two-rank engine; the checkpoint's own DSpark blocks draft unless drafts are disabled."""

    if int(tp) != 2:
        raise ValueError("DeepSeek-V4.1-Flash needs two GPUs, one per machine (two DGX Sparks): run the same "
                         "`tensorfold serve` command with --tp 2 --rank R --master ADDRESS on both (rank 1 first)")
    if not master:
        raise ValueError("--tp 2 needs --master: rank 0's address on the link between the two machines")
    from .cuda.engine import Dsv41Engine

    return Dsv41Engine(Path(model_dir), rank=int(rank), master=master, port=int(master_port),
                       engram=engram_dir(model_dir), drafts=not no_drafts,
                       context=options.get("context"), context_explicit=options.get("context_explicit"),
                       parallel=int(options.get("parallel") or 1))
