"""Fixtures for `tf-cuda-test fp8-lane`: the Python lane matmul's FP8G bytes on random block-FP8 weights."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from oracle import save  # noqa: E402

ROWS = (1, 3, 17, 40, 300)  # one row, small tiles, two tiles, the fused prompt path from 256 rows


def case(out: Path, n: int, k: int, g: torch.Generator) -> None:
    from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear

    codes = torch.randint(0, 256, (n, k), generator=g, dtype=torch.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x3C  # e4m3 NaN codes out: a checkpoint holds none
    inv = torch.rand((-(-n // 128), k // 128), generator=g) * 0.01 + 1e-3
    lin = Fp8BlockLinear.from_checkpoint(codes.cuda().view(torch.float8_e4m3fn), inv.cuda())
    arrays = {"codes": codes, "scale_inv": inv.float(), "w8": lin.w8, "bs": lin.bs}
    for m in ROWS:
        x = (torch.randn((m, k), generator=g) * 0.5).to(torch.bfloat16)
        arrays[f"x{m}"] = x
        arrays[f"y{m}"] = lin(x.cuda())
    torch.cuda.synchronize()
    save(out, {"n": n, "k": k, "rows": ",".join(str(m) for m in ROWS)}, arrays)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=2560)
    ap.add_argument("--k", type=int, default=6144)
    a = ap.parse_args()
    case(Path(a.out), a.n, a.k, torch.Generator().manual_seed(a.n * 31 + a.k))


if __name__ == "__main__":
    main()
