"""Pack affine group-32 MLX weights with the shared expert last, concatenate n-gram shards in order, and retain centered-norm gamma itself as fp32 after checking it is around one."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..host_table import BF16Table, HostTable, read_header as _header
from ..ssd_table import SSDTable
from .bf16 import b16_from_rows, quantize4, stack_b16, make_b16
from .ngram import NGram
from tensorfold.cuda import experts as grouped

from .qmm import Q4, dequantize, make_q4, stack_q4


def stop_ids(configured: Any, generation: Path) -> tuple[int, ...]:
    """config.json's end-of-reply ids, then generation_config.json's it lacks (EXL3 packs keep <|im_end|> there)."""

    def ids(value: Any) -> list[int]:
        return [] if value is None else [int(e) for e in value] if isinstance(value, list) else [int(value)]

    found = ids(configured)
    if generation.exists():
        found += ids(json.loads(generation.read_text()).get("eos_token_id"))
    out = tuple(dict.fromkeys(found))
    if not out:
        raise ValueError("no eos_token_id in config.json or generation_config.json")
    return out
