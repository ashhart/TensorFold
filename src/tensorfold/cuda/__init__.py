"""TensorFold on NVIDIA GPUs: the shared OpenAI server for the families' CUDA engines (``families/<name>/cuda``)."""

import os

# ROCm's Triton turns loads into buffer loads with unsigned 32-bit offsets; caches addressed from ``attention.base``
# can sit below it, and those loads then read zeros. NVIDIA's Triton ignores the setting.
os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "0")
