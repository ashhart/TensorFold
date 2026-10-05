"""Compile every CUDA extension the DeepSeek-V4.1 engine loads, into the mounted cache, with no model loaded."""

from tensorfold.cuda.exl3 import experts, linear, x3ld
import tensorfold.cuda.rdma as rdma
from tensorfold.families.deepseek_v41.cuda import experts_prompt, mqa_fp4, rowread

linear._ext()
experts._ext()
x3ld._ext()
experts_prompt.ext()
mqa_fp4.ext()
rowread.reader()
rdma._ext()
rdma.proxy()                          # the RoCE host proxy (gcc + libibverbs), the default transport
print("extensions ready", flush=True)

import os
from pathlib import Path

tc = os.environ.get("TRITON_CACHE_DIR")
if tc:                                # the image keeps Triton's kernels in the cache volume (Dockerfile)
    n = sum(1 for _ in Path(tc).iterdir()) if Path(tc).is_dir() else 0
    print(f"Triton cache {tc}: {n} kernels compiled by earlier starts", flush=True)
else:
    print("Triton cache: TRITON_CACHE_DIR unset (an image before it was set: kernels compile at every start)", flush=True)
