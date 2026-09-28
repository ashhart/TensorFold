"""Load and bind the shared DFlash2 model and expose the drafter API."""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

from .dflash_attention import _dflash_attend, concat_updates
from .dflash_capture import _capture_writer
from .dflash_tree import best_first_tree, lattice_gain

# z-lab's reference MLX implementation of the DFlash2 drafter (MIT; see THIRD_PARTY_NOTICES.md), vendored verbatim
_VENDOR = Path(__file__).resolve().parent / "vendor" / "z_lab_dflash" / "model_mlx.py"


def _vendor() -> Any:
    name = "zlab_dflash_model_mlx"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _VENDOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_draft_path(draft: str) -> str:
    """A local directory, or the newest cached snapshot of a Hugging Face repo id."""

    path = Path(draft).expanduser()
    if path.is_dir():
        return str(path)
    repo = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{draft.replace('/', '--')}" / "snapshots"
    hits = sorted(glob.glob(str(repo / "*")))
    if not hits:
        raise FileNotFoundError(f"no local DFlash drafter at {draft} (looked in {repo})")
    return hits[-1]


class DFlashDrafter:
    """The shared drafter model; one ``DFlashProposer`` per stream holds that stream's cache."""

    def __init__(self, target_model: Any, draft: str, *, bits: int = 8) -> None:
        vendor = _vendor()
        path = resolve_draft_path(draft)
        vendor.snapshot_download = lambda repo_id, **_: path  # load_draft resolves ids online otherwise
        self.path = path
        self.model = vendor.load_draft(path)
        if bits:
            import mlx.nn as nn

            nn.quantize(self.model, group_size=64, bits=int(bits),
                        class_predicate=lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0)
            mx.eval(self.model.parameters())
        self.model.bind(target_model)
        vendor._patch_model(target_model, list(self.model.config.target_layer_ids))
        self.target = target_model
        self.block_size = int(self.model.config.block_size)
        self.mask_id = int(self.model.config.mask_token_id)
        window = getattr(self.model.config, "sliding_window", None)
        self.window = int(window) - 1 if window else 0
        self._trim = vendor._trim_recent_cache

    def make_cache(self) -> list[Any]:
        """The draft model's caches; a full-attention layer's never rotates, so its context may start past 0."""

        from mlx_lm.models.cache import KVCache, RotatingKVCache

        return [RotatingKVCache(max_size=1 << 30, keep=0) if type(c) is KVCache else c for c in self.model.make_cache()]

    def taps(self) -> mx.array | None:
        """The last target forward's taps, [batch, rows, 5 * hidden]."""

        states = getattr(self.target, "_hidden_states", None)
        if not states or any(s is None for s in states):
            return None
        return mx.concatenate(states, axis=-1)

    def release_taps(self) -> None:
        states = getattr(self.target, "_hidden_states", None)
        if states:
            for i in range(len(states)):
                states[i] = None

    def proposer(self, copy: Any = None, sampling: Any = None) -> "DFlashProposer":
        return DFlashProposer(self, copy=copy, sampling=sampling)

    # Restrict draft candidates only; target verification still spans the full vocabulary.
    draft_vocab: tuple[tuple[int, int], ...] = ((0, 98304), (248032, 248320))

    def candidate_logits(self, hidden: mx.array) -> tuple[mx.array, mx.array | None]:
        """The head's logits over ``draft_vocab`` and each column's token id, or (all logits, None)."""

        head = self.model.lm_head
        if hasattr(head, "rotate"):          # a head stored in a rotated basis: its rows are rotated, then its matmul
            hidden, head = head.rotate(hidden), head.inner
        sub = self._sub_head()
        if sub is None:
            plain = self._plain_sub_head()
            if plain is None:
                logits = head(hidden) * self.model.config.output_multiplier
                cap = self.model.config.final_logit_softcapping
                return (mx.tanh(logits / cap) * cap if cap is not None and cap > 0 else logits), None
            # Draft logits need no row-exact matmul; use views of the draft vocabulary's head rows.
            from tensorfold.kernels.qwen.dense.v1.row_matmul import draft_matmul

            parts, ids, group_size, bits = plain
            logits = mx.concatenate([draft_matmul(hidden, w, sc, b, group_size, bits) for w, sc, b in parts],
                                    axis=-1)
            logits = logits * self.model.config.output_multiplier
            cap = self.model.config.final_logit_softcapping
            if cap is not None and cap > 0:
                logits = mx.tanh(logits / cap) * cap
            return logits, ids
        from tensorfold.kernels.qwen.dense.v1 import lane_qmm

        weight, sbt, ids, nt = sub
        logits = lane_qmm.lane_matmul(hidden, weight, sbt, tiled=True, nt=nt) * self.model.config.output_multiplier
        cap = self.model.config.final_logit_softcapping
        if cap is not None and cap > 0:
            logits = mx.tanh(logits / cap) * cap
        return logits, ids

    def _matmul_head(self) -> Any:
        """The head's matmul: the head itself, or the matmul under a head that rotates its rows first."""

        head = self.model.lm_head
        return head.inner if hasattr(head, "rotate") else head

    def _plain_sub_head(self) -> tuple[list[tuple[mx.array, mx.array, mx.array]], mx.array, int, int] | None:
        """``draft_vocab``'s rows of a head in MLX's layout (views: no copy), built once."""

        if getattr(self, "_plain_sub", False) is False:
            self._plain_sub = None
            import mlx.nn as nn

            head = self._matmul_head()
            if os.environ.get("TF_DRAFT_VOCAB", "") != "full" and isinstance(head, nn.QuantizedLinear) \
                    and "bias" not in head and not getattr(head, "_lane_tiled", False):
                n = int(head["weight"].shape[0])
                spans = [(a, min(b, n)) for a, b in self.draft_vocab if a < n]
                if sum(b - a for a, b in spans) < n:
                    parts = [(head["weight"][a:b], head["scales"][a:b], head["biases"][a:b]) for a, b in spans]
                    ids = mx.concatenate([mx.arange(a, b, dtype=mx.int32) for a, b in spans])
                    mx.eval(ids)
                    self._plain_sub = (parts, ids, int(head.group_size), int(head.bits))
        return self._plain_sub

    def _sub_head(self) -> tuple[mx.array, mx.array, mx.array, int] | None:
        """``draft_vocab``'s rows of the lane-tiled head (whole 32-row tiles), built once."""

        if getattr(self, "_sub", False) is False:
            self._sub = None
            head = self._matmul_head()
            if (os.environ.get("TF_DRAFT_VOCAB", "") != "full" and getattr(head, "_lane_tiled", False)
                    and getattr(head, "_lane_sbt", None) is not None):
                n = int(head["weight"].shape[0])
                nt = int(getattr(head, "_lane_nt", 32))
                # whole tiles of the head's layout: a span's ends round outward to its tile width
                spans = [(a - a % nt, min(-(-b // nt) * nt, n)) for a, b in self.draft_vocab if a < n]
                if all(a % nt == 0 and b % nt == 0 for a, b in spans) and sum(b - a for a, b in spans) < n:
                    weight = mx.concatenate([head["weight"][a:b] for a, b in spans], axis=0)
                    sbt = mx.concatenate([head._lane_sbt[:, a:b] for a, b in spans], axis=1)
                    ids = mx.concatenate([mx.arange(a, b, dtype=mx.int32) for a, b in spans])
                    mx.eval(weight, sbt, ids)
                    self._sub = (weight, sbt, ids, nt)
        return self._sub


# Import after the model class so either module can resolve the shared type.
from .dflash_proposer import DFlashProposer


__all__ = ["DFlashDrafter", "DFlashProposer", "concat_updates", "resolve_draft_path"]
