"""EXL3 (ExLlamaV3's trellis quantization) for any family's CUDA engine.

- ``format.py``: the format, per-tensor metadata from safetensors headers (bits, codebook, K, N), and a numpy
  reference decoder for every codebook (3inst, mcg, mul1) and width (1 to 8 bits, and 1.5 / 2.5 / 3.5 with mul1).
- ``decode.cuh``: header-only device functions that decode a tile straight into mma.m16n8k16 B fragments.
- ``linear.py`` (``linear.cu``): a row-invariant EXL3 linear layer for 1 to 128 rows.
- ``inspect.py``: ``python -m tensorfold.cuda.exl3.inspect MODEL_DIR`` lists what a checkpoint holds.

Only ``format`` is imported here; the CUDA modules import torch and build their extension on first use.
"""

from .format import CODEBOOKS, Exl3Tensor, parse_group, scan

__all__ = ["CODEBOOKS", "Exl3Tensor", "parse_group", "scan"]
