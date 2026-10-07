"""Fixtures for `tf-cuda-test nvfp4`: the Python NVFP4 projection's bytes on the lane matmul and the prompt GEMM."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from oracle import save  # noqa: E402

LANE = (1, 3, 17, 40, 300)  # decode rows on the lane matmul, its fused path from 256 rows
PROMPT = (17, 300, 1000)  # prompt rows on the prompt GEMM (PROMPT_TILE 4)


def case(out: Path, n: int, k: int, g: torch.Generator) -> None:
    from tensorfold.cuda.nvfp4.linear import Fp4Linear

    codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8)
    scales = torch.randint(0x20, 0x48, (n, k // 16), generator=g, dtype=torch.uint8)  # positive e4m3, no NaN
    global_scale = 0.0123
    lin = Fp4Linear.from_checkpoint(codes.cuda(), scales.cuda().view(torch.float8_e4m3fn), global_scale)
    arrays = {"codes": codes, "scales": scales, "words": lin.words, "bs": lin.bs}
    for m in sorted(set(LANE) | set(PROMPT)):
        x = (torch.randn((m, k), generator=g) * 0.5).to(torch.bfloat16)
        arrays[f"x{m}"] = x
        if m in LANE:
            arrays[f"lane{m}"] = lin(x.cuda())
        if m in PROMPT:
            arrays[f"prompt{m}"] = lin.prefill(x.cuda())
    torch.cuda.synchronize()
    params = {"n": n, "k": k, "global_scale": global_scale, "lane": ",".join(map(str, LANE)),
              "prompt": ",".join(map(str, PROMPT))}
    save(out, params, arrays)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=5120)
    ap.add_argument("--k", type=int, default=6144)
    a = ap.parse_args()
    case(Path(a.out), a.n, a.k, torch.Generator().manual_seed(a.n * 37 + a.k))


if __name__ == "__main__":
    main()
