"""Keyed sampling over the ranks' vocabulary halves: GLM's rule for the target, a bounded candidate read for drafts."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace

import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda.decode import sample_rows

from .weights import Weights

DRAFT_TOP_K = 1024      # a top_k-off draft's merged candidates per row; a draft never reads whole shards


def target_rows(w: Weights, logits: torch.Tensor, positions: Sequence[int],
                sampling: Sampling | None) -> tuple[list[int], float]:
    """Rows of this rank's head logits at their absolute positions -> (tokens, host seconds), same on every rank."""

    start = time.perf_counter()
    tokens = sample_rows(w, logits, positions, sampling)
    return tokens, time.perf_counter() - start


def draft_rows(w: Weights, logits: torch.Tensor, positions: Sequence[int],
               sampling: Sampling | None) -> tuple[list[int], float]:
    """``target_rows`` for drafts, with top_k off read as the top ``DRAFT_TOP_K`` (keys per position unchanged)."""

    if sampling is not None and sampling.temperature > 0 and not sampling.top_k:
        sampling = replace(sampling, top_k=DRAFT_TOP_K)
    return target_rows(w, logits, positions, sampling)
