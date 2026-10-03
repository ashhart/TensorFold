"""Laguna on the lane engine: mlx-lm's forward for prompts, row-exact kernels for every decode row."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tensorfold.families.gemma4.model import Gemma4, realize
from tensorfold.kernels.laguna.v1.decode import RowDecode


class Laguna(Gemma4):
    """Laguna's MLX model with the backbone and the head apart, decoded through ``RowDecode``."""

    def __init__(self, model: Any, *, backend: str | None = None, head_backend: str | None = None, check: bool = True,
                 tokenizer: Any = None, drafter: Any = None) -> None:
        realize(model)
        self.model = model
        self.text = model
        self.backbone = model.model
        self.args = model.args
        self.decode = RowDecode(model, backend or "rows", head_backend)
        self.head_drafts = None
        self._last: dict[int, tuple[int, int]] = {}
        if drafter is not None:
            from tensorfold.families.gemma4.drafts import DFlashChains

            self.mtp, self.head_drafts = drafter, DFlashChains(drafter)
            self.drafts = self.head_drafts.nodes
            self.decode.taps = (tuple(int(i) for i in drafter.model.config.target_layer_ids), model._hidden_states)
        self.exact_width, self.window_costs = (1, {})
        self.shared_costs: dict[int, float] = {}
        if check:
            self.exact_width, self.window_costs = self.check_windows(tokenizer)
        self.multi_row_exact = self.exact_width >= 2
        if check and self.multi_row_exact and not self.check_streams(tokenizer):
            self.max_streams = 1
            print("[laguna] a forward over several streams' rows does not reproduce each stream's own call here: one "
                  "stream a forward", flush=True)
        if check and self.multi_row_exact and self.max_streams > 1:
            self.shared_costs = self.time_shared_rows(tokenizer)
        if check:
            counts = ", ".join(f"{name} {n}" for name, n in sorted(self.decode.backends.items()))
            timing = ", ".join(f"{w}: {ms:.1f}" for w, ms in sorted(self.window_costs.items()))
            shared = ", ".join(f"{w}: {ms:.1f}" for w, ms in sorted(self.shared_costs.items()))
            print(f"[laguna] decode matmuls: {counts}; windows of up to {self.exact_width} rows reproduce one-row "
                  f"steps here (ms by rows {timing}; shared forwards {shared or 'none'})", flush=True)


def _config(model_dir: Path) -> dict[str, Any]:
    """config.json with its per-layer quantization keyed by this model's module paths (oQ adds ``language_model.``)."""

    from tensorfold.families import read_config

    config = read_config(model_dir)
    prefix = "language_model."
    for key in ("quantization", "quantization_config"):
        found = config.get(key)
        if isinstance(found, dict):
            config[key] = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in found.items()}
    return config


def load_model(model_dir: Path) -> tuple[Any, dict[str, Any]]:
    from mlx_lm.utils import load_model as mlx_load_model

    from tensorfold.families.laguna import mlx_model

    config = _config(model_dir)
    overrides = {key: config[key] for key in ("quantization", "quantization_config") if key in config}
    return mlx_load_model(Path(model_dir), model_config=overrides,
                          get_model_classes=lambda config: (mlx_model.Model, mlx_model.ModelArgs))


def load(model_dir: Path, *, backend: str | None = None, check: bool = True, drafter: str = "",
         drafter_bits: int = 8) -> tuple[Laguna, Any]:
    from mlx_lm.utils import load_tokenizer

    model, config = load_model(Path(model_dir))
    tokenizer = load_tokenizer(Path(model_dir), eos_token_ids=config.get("eos_token_id"))
    realize(model)                     # before the draft model wraps the tapped layers
    draft = None
    if drafter:
        from tensorfold.families.laguna.drafter import LagunaDrafter

        draft = LagunaDrafter(model, drafter, bits=int(drafter_bits))
        print(f"[tensorfold] drafter {draft.path} block={draft.block_size} bits={drafter_bits or 16}", flush=True)
    return Laguna(model, backend=backend, check=check, tokenizer=tokenizer, drafter=draft), tokenizer


__all__ = ["Laguna", "load", "load_model"]
