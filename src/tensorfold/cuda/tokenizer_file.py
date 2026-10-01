"""A checkpoint's ``tokenizer.json`` as the CUDA server reads it: whole prompts, never truncated or padded."""

from __future__ import annotations

from pathlib import Path


def load(model_dir: str | Path):
    """The fast tokenizer with any saved truncation or padding turned off.

    Some fine-tuned checkpoints ship a ``tokenizer.json`` whose ``truncation`` block (for example ``max_length: 2048``)
    was left by their training run. The ``tokenizers`` library applies it on every ``encode``, so a longer prompt
    would be cut silently; the server refuses prompts past its window itself instead."""

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    return tok
