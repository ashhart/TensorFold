"""DeepSeek-V4.1 kernels: affine row paths (Q3 g64) plus the v4 attention/rope kernels this family reuses."""

VERSION = "v1"

from tensorfold.kernels.deepseek.v4 import attention as _attention   # noqa: F401  (shared, reused as-is)
from tensorfold.kernels.deepseek.v4 import rope as _rope             # noqa: F401  (shared, reused as-is)
