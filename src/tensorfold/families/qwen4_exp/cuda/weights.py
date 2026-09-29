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
