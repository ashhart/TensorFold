"""The MTP head's vocabulary cut to a fixed list of token ids, and the keyed sampler over that list.

The head's own logits only pick drafts, and every committed token is the target's sample over the whole
vocabulary, so drafts may come from any rule. Scoring only the ids in ``cuda/draft_vocab.txt`` (79,591 ids from
public text, the CUDA engine's list) reads 127 MB of head weights a draft step instead of 397 MB. A draft is
drawn with the target's keyed rule (``engine.gpu_sampling``: Gumbel noise keyed by the seed, the position and
the token id) over the listed ids, so it is the target's own sample whenever the two distributions agree there
and the token is listed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

VOCAB_FILE = Path(__file__).with_name("cuda") / "draft_vocab.txt"


def draft_ids(path: Path = VOCAB_FILE, multiple: int = 8) -> np.ndarray:
    """The listed token ids, sorted ascending (so candidate ties break by id, as the full sampler does), padded
    with the smallest unlisted ids to a multiple of ``multiple`` rows (the row-exact matvecs take N % 8 == 0)."""

    listed = {int(line) for line in Path(path).read_text().split() if line.strip()}
    if not listed:
        raise ValueError(f"{path}: no token ids")
    extra, candidate = [], 0
    while (len(listed) + len(extra)) % multiple:
        if candidate not in listed:
            extra.append(candidate)
        candidate += 1
    return np.array(sorted(listed | set(extra)), dtype=np.uint32)


def cut_head(lm_head: Any, ids: np.ndarray) -> nn.QuantizedLinear:
    """The rows of a 4-bit quantized vocabulary head for ``ids`` as a quantized linear of their own."""

    index = mx.array(ids.astype(np.int32))
    out = nn.QuantizedLinear(int(lm_head.weight.shape[1]) * 32 // lm_head.bits, len(ids), bias=False,
                             group_size=lm_head.group_size, bits=lm_head.bits)
    out.weight = mx.take(lm_head.weight, index, axis=0)
    out.scales = mx.take(lm_head.scales, index, axis=0)
    out.biases = mx.take(lm_head.biases, index, axis=0)
    mx.eval(out.weight, out.scales, out.biases)
    return out


def sample(logits: mx.array, ids: mx.array, sampling: Any, positions: Sequence[int] | mx.array) -> mx.array:
    """Token ids [R] (uint32, lazy) from logits [R, len(ids)] over the listed ``ids`` at absolute ``positions``
    (``engine.gpu_sampling`` with its ``ids`` columns); greedy (the listed id with the largest logit) when
    ``sampling`` is None."""

    from tensorfold.engine.gpu_sampling import sample as gpu_sample

    return gpu_sample(logits.reshape(-1, logits.shape[-1]), sampling, positions, ids=ids)
