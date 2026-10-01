"""``--checkpoint-slots`` on the 27B's two-stream CUDA engine (needs TENSORFOLD_MLX_MODEL and TENSORFOLD_QWEN27_DRAFTER)."""

from __future__ import annotations

import gc
import os
import re
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

MODEL = os.environ.get("TENSORFOLD_MLX_MODEL", "")
DRAFTER = os.environ.get("TENSORFOLD_QWEN27_DRAFTER", "")
CONTEXT = 8192


def _start(keep, capsys):
    """(kept states, startup line, estimated GiB) of a two-stream engine asked for ``keep`` (None: the default)."""

    if not (MODEL and Path(MODEL).is_dir() and DRAFTER and Path(DRAFTER).is_dir()):
        pytest.skip("needs TENSORFOLD_MLX_MODEL and TENSORFOLD_QWEN27_DRAFTER")
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    engine = Qwen27Engine(Path(MODEL), Path(DRAFTER), max_rows=12, streams=2, context=CONTEXT, context_explicit=True,
                          keep=keep)
    out = capsys.readouterr().out
    kept = engine.multi.cache.keep
    engine.close()                                       # its scheduler's worker held the weights until now
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    line = next(x for x in out.splitlines() if "streams" in x and "prompt states kept" in x)
    estimate = float(re.search(r"startup estimate ([0-9.]+) GiB", out).group(1))
    return kept, line, estimate


def test_the_concurrent_decoder_keeps_the_asked_prompt_states(capsys):
    kept, line, low = _start(None, capsys)
    assert kept == 3 and line.endswith(f"up to 2 streams, each growing to {CONTEXT} prompt/reply tokens while memory "
                                       "lasts, 3 prompt states kept")
    kept, line, high = _start(5, capsys)
    assert kept == 5 and line.endswith(f"up to 2 streams, each growing to {CONTEXT} prompt/reply tokens while memory "
                                       "lasts, 5 prompt states kept")
    assert high > low                              # two more kept states' DeltaNet copies and first rows
