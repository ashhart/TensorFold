"""Compile every CUDA extension the DeepSeek-V4.1 engine loads, into the mounted cache, with no model loaded."""

from tensorfold.cuda.exl3 import experts, linear, x3ld
import tensorfold.cuda.rdma as rdma
from tensorfold.families.deepseek_v41.cuda import experts_prompt, rowread

linear._ext()
experts._ext()
x3ld._ext()
experts_prompt.ext()
rowread.reader()
rdma._ext()
print("extensions ready", flush=True)
