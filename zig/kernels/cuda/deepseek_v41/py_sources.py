"""Each Zig-owned kernel copy's reference source (key, path) for sass_check.sh, which proves the copies SASS-equal."""
import os
import sys
from pathlib import Path

# source key -> (env var naming the repo, env var naming its git ref); defaults: the clone beside this one, HEAD
SOURCES = {
    "base": ("TF_DSV41_REF_SRC", "TF_DSV41_REF_REV"),
    "x3gm2": ("TF_DSV41_REF_SRC_X3GM2", "TF_DSV41_REF_REV_X3GM2"),  # x3gm with gm2_kernel
    "r1": ("TF_DSV41_REF_SRC_R1", "TF_DSV41_REF_REV_R1"),  # the decode-kernel ref
}
COPIES = {
    "decode.cuh": ("base", "src/tensorfold/cuda/exl3/decode.cuh"),
    "experts_grouped.cuh": ("base", "src/tensorfold/cuda/exl3/experts_grouped.cuh"),
    "exl3_experts.cu": ("r1", "src/tensorfold/cuda/exl3/experts.cu"),
    "x3ld.cu": ("base", "src/tensorfold/families/deepseek_v41/cuda/x3ld.cu"),
    "x3gm.cu": ("x3gm2", "src/tensorfold/families/deepseek_v41/cuda/x3gm.cu"),
    "linear.cu": ("base", "src/tensorfold/cuda/exl3/linear.cu"),
    "attn_cuda.cu": ("r1", "src/tensorfold/families/deepseek_v41/cuda/csa2/attn_cuda.cu"),
    "topk_cuda.cu": ("r1", "src/tensorfold/families/deepseek_v41/cuda/csa2/topk_cuda.cu"),
    "mhc_cuda.cu": ("r1", "src/tensorfold/families/deepseek_v41/cuda/mhc_cuda.cu"),
    "router_gemv.cu": ("r1", "src/tensorfold/families/deepseek_v41/cuda/router_gemv.cu"),
}


def repo(key: str) -> str:
    """The source repo of a key: its env var, else the reference clone `tensorfold` beside this checkout."""
    env, _ = SOURCES[key]
    return os.environ.get(env) or str(Path(__file__).resolve().parents[4].parent / "tensorfold")


def ref(key: str) -> str:
    """The git ref of a key's source: its env var, else HEAD."""
    return os.environ.get(SOURCES[key][1]) or "HEAD"


def main() -> int:
    if sys.argv[1:] != ["--sources"]:
        print("usage: py_sources.py --sources   (prints 'copy key repo ref' a line)", file=sys.stderr)
        return 2
    for name, (key, _) in COPIES.items():
        print(name, key, repo(key), ref(key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
