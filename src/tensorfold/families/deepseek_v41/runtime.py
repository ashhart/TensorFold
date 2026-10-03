"""DeepSeek-V4.1-Flash on V4's lane runtime, with the checkpoint's own DSpark drafter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tensorfold.families.deepseek_v41.config import INDEX_QUERIES
from tensorfold.families.deepseek_v4.runtime import DeepSeekFlash, materialize


class DeepSeekV41Flash(DeepSeekFlash):
    """The backbone behind the lane protocol (no draft head yet: one row a step, or windows the engine verifies)."""

    tag = "deepseek_v41"

    @property
    def prefill_workspace_per_token(self) -> int:
        """Prefill bytes a position of context: an index query block's fp32 scores and their ReLU, over every key."""

        return INDEX_QUERIES * self.args.index_n_heads * 8


def load(model_dir: Path, *, drafter: str = "", mtp_drafts: int | None = None, check: bool = True,
         **_: Any) -> tuple[DeepSeekV41Flash, Any]:
    """The runtime and tokenizer; ``mtp_drafts`` 0 serves without the checkpoint's DSpark drafter."""

    from tensorfold.families.deepseek_v41.engram import build_compressed_token_map
    from tensorfold.families.deepseek_v41.prompts import DeepSeekV41Tokenizer
    from tensorfold.families.deepseek_v41.weights import load_backbone
    from tensorfold.families.tokenizer import load_tokenizer

    model = load_backbone(Path(model_dir))
    inner = load_tokenizer(Path(model_dir), eos_token_ids=model.args.eos_token_id or None)
    if model.args.engram_layer_ids:
        hf = getattr(inner, "_tokenizer", inner)
        model.set_token_map(build_compressed_token_map(hf)[0])
    runtime = DeepSeekV41Flash(model, None, drafts=0, check=check)
    materialize(runtime)
    print(f"[deepseek_v41] exact window {runtime.exact_width} rows, forward ms by width {runtime.window_costs}",
          flush=True)
    return runtime, DeepSeekV41Tokenizer(inner)
