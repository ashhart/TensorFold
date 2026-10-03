"""DeepSeek-V4.1-Flash on V4's lane runtime, with the checkpoint's own DSpark drafter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tensorfold.families.deepseek_v41.config import INDEX_QUERIES
from tensorfold.families.deepseek_v4.runtime import DeepSeekFlash, DSparkFlash, materialize


class DeepSeekV41Flash(DeepSeekFlash):
    """The backbone behind the lane protocol, without a draft head: one row a step."""

    tag = "deepseek_v41"

    @property
    def prefill_workspace_per_token(self) -> int:
        """Prefill bytes a position of context: an index query block's fp32 scores and their ReLU, over every key."""

        return INDEX_QUERIES * self.args.index_n_heads * 8


class DSparkV41Flash(DSparkFlash):
    """DSpark drafts after a round is read: the kept rows' taps go into its rings, one pass drafts a block."""

    tag = "deepseek_v41"

    @property
    def prefill_workspace_per_token(self) -> int:
        return INDEX_QUERIES * self.args.index_n_heads * 8


def load(model_dir: Path, *, drafter: str = "", mtp_drafts: int | None = None, check: bool = True,
         **_: Any) -> tuple[DeepSeekFlash, Any]:
    """The runtime and tokenizer; ``mtp_drafts`` 0 serves without the checkpoint's DSpark drafter."""

    from tensorfold.families.deepseek_v41 import dspark
    from tensorfold.families.deepseek_v41.engram import build_compressed_token_map
    from tensorfold.families.deepseek_v41.prompts import DeepSeekV41Tokenizer
    from tensorfold.families.deepseek_v41.weights import load_backbone
    from tensorfold.families.tokenizer import load_tokenizer

    model = load_backbone(Path(model_dir))
    inner = load_tokenizer(Path(model_dir), eos_token_ids=model.args.eos_token_id or None)
    if model.args.engram_layer_ids:
        hf = getattr(inner, "_tokenizer", inner)
        model.set_token_map(build_compressed_token_map(hf)[0])
    drafts = 5 if mtp_drafts is None else int(mtp_drafts)
    w = model.weights
    runtime: DeepSeekFlash
    if drafts > 0 and dspark.has_drafter(w.raw, w.where):
        runtime, kind = DSparkV41Flash(model, dspark.load(model, w), check=check), "DSpark"
    else:
        runtime, kind = DeepSeekV41Flash(model, None, drafts=0, check=check), "none"
    w.release()
    model.weights = None
    materialize(runtime)
    print(f"[deepseek_v41] exact window {runtime.exact_width} rows, forward ms by width {runtime.window_costs}, "
          f"drafter {kind}, step {runtime.mtp_step_ms} ms", flush=True)
    return runtime, DeepSeekV41Tokenizer(inner)
