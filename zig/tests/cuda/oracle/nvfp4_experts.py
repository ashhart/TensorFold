"""Fixtures for `tf-cuda-test nvfp4-experts`: the Python grouped NVFP4 experts' plan, packing and outputs."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from oracle import save  # noqa: E402

DECODE = (1, 3)  # one-block plans
PROMPT = (300,)  # 2100 pairs: the wide plan (rank, offsets, scatter), items of 16 as the decode form


def e4m3(shape, g):
    codes = torch.randint(0, 0x60, shape, generator=g, dtype=torch.uint8)  # positive, finite, below 2^5
    return codes


def main() -> None:
    from tensorfold.cuda import experts as grouped
    from tensorfold.cuda.nvfp4 import experts as fx4

    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--experts", type=int, default=33)  # routed, then the shared one
    ap.add_argument("--dims", type=int, default=2560)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    g = torch.Generator().manual_seed(a.experts * 11 + a.dims)
    e, d, ni = a.experts, a.dims, a.width
    words = {m: torch.randint(0, 256, s, generator=g, dtype=torch.uint8)
             for m, s in (("gate", (e, ni, d // 2)), ("up", (e, ni, d // 2)), ("down", (e, d, ni // 2)))}
    scales = {"gate": e4m3((e, ni, d // 16), g), "up": e4m3((e, ni, d // 16), g), "down": e4m3((e, d, ni // 16), g)}
    glob = {m: torch.rand((e,), generator=g) * 1e-3 + 1e-4 for m in words}
    f8 = torch.float8_e4m3fn
    ex = fx4.make(*[(words[m].cuda(), scales[m].cuda().view(f8), glob[m].cuda()) for m in ("gate", "up", "down")])
    arrays = {"packed_up": ex.up, "packed_down": ex.down, "up_scale": ex.up_scale, "down_scale": ex.down_scale}
    for m in words:
        arrays.update({f"{m}_words": words[m], f"{m}_scales": scales[m], f"{m}_global": glob[m]})
    slots = a.top + 1
    for rows, prompt in [(r, False) for r in DECODE] + [(r, True) for r in PROMPT]:
        picks = torch.stack([torch.randperm(e - 1, generator=g)[:a.top] for _ in range(rows)]).int()
        picks = torch.cat([picks, torch.full((rows, 1), e - 1, dtype=torch.int32)], 1).contiguous()
        x = (torch.randn((rows, d), generator=g) * 0.5).to(torch.bfloat16)
        plan = grouped.Plan(rows, slots, e, "cuda", prefill=prompt)
        grouped.route(picks.cuda(), plan, fx4.PREFILL_TILE)
        act = torch.empty((rows * slots, ni), dtype=torch.bfloat16, device="cuda")
        fx4.gate_up(x.cuda(), ex, plan, act, rows)
        y = torch.empty((rows * slots, d), dtype=torch.bfloat16 if prompt else torch.float32, device="cuda")
        fx4.down(act, ex, plan, y, rows)
        torch.cuda.synchronize()
        arrays.update({f"picks{rows}": picks, f"x{rows}": x, f"act{rows}": act, f"y{rows}": y,
                       f"counts{rows}": plan.counts, f"items{rows}": plan.items, f"members{rows}": plan.members})
    save(Path(a.out), {"experts": e, "dims": d, "width": ni, "slots": slots, "decode": ",".join(map(str, DECODE)),
                       "prompt": ",".join(map(str, PROMPT))}, arrays)


if __name__ == "__main__":
    main()
