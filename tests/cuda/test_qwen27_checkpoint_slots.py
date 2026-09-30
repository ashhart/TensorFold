"""``--checkpoint-slots`` on Qwen3.8-27B's CUDA engine with two streams: the concurrent decoder keeps the asked number of
prompt states, the startup line names it, and the startup estimate holds each against the window.

Needs ``TENSORFOLD_MLX_MODEL=<Vontra/Qwen3.8-27B-MLX-4bit dir>`` and ``TENSORFOLD_QWEN27_DRAFTER=<z-lab/Qwen3.8-27B-DFlash2
dir>``; skipped otherwise. Two engines, one after the other, about 22 GB of GPU memory each."""

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
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    line = next(x for x in out.splitlines() if "streams of" in x)
    estimate = float(re.search(r"startup estimate ([0-9.]+) GiB", out).group(1))
    return kept, line, estimate


def test_the_concurrent_decoder_keeps_the_asked_prompt_states(capsys):
    kept, line, low = _start(None, capsys)
    assert kept == 3 and line.endswith(f"2 streams of {CONTEXT} prompt/reply tokens, 3 prompt states kept")
    kept, line, high = _start(5, capsys)
    assert kept == 5 and line.endswith(f"2 streams of {CONTEXT} prompt/reply tokens, 5 prompt states kept")
    assert high > low                              # two more kept states, each held against the window
