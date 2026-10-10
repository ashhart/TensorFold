"""Fixtures for `tf-cuda-test fp8-experts`: the Python grouped FP8 experts' plan, packing and outputs."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from oracle import save  # noqa: E402

DECODE = (1, 3)  # one-block plans, decode items of 16
PROMPT = (300,)  # 2100 pairs: the wide plan (rank, offsets, scatter), prompt items of 64


def e4m3(shape, g):
    codes = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x3C
    return codes


def main() -> None:
    from tensorfold.cuda import experts as grouped
    from tensorfold.cuda.fp8 import experts as fp8x

    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--experts", type=int, default=33)  # routed, then the shared one
    ap.add_argument("--dims", type=int, default=2560)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    g = torch.Generator().manual_seed(a.experts * 7 + a.dims)
    e, d, ni = a.experts, a.dims, a.width
    gate, up, down = e4m3((e, ni, d), g), e4m3((e, ni, d), g), e4m3((e, d, ni), g)
    sg = torch.rand((e, ni // 128, d // 128), generator=g) * 0.01 + 1e-3
    su = torch.rand((e, ni // 128, d // 128), generator=g) * 0.01 + 1e-3
    sd = torch.rand((e, d // 128, ni // 128), generator=g) * 0.01 + 1e-3
    f8 = torch.float8_e4m3fn
    ex = fp8x.make((gate.cuda().view(f8), sg.cuda()), (up.cuda().view(f8), su.cuda()), (down.cuda().view(f8), sd.cuda()))
    arrays = {"gate": gate, "up": up, "down": down, "gate_scale": sg, "up_scale": su, "down_scale": sd,
              "packed_up": ex.up, "packed_down": ex.down, "packed_up_scale": ex.up_scale,
              "packed_down_scale": ex.down_scale}
    slots = a.top + 1
    for rows, prompt in [(r, False) for r in DECODE] + [(r, True) for r in PROMPT]:
        picks = torch.stack([torch.randperm(e - 1, generator=g)[:a.top] for _ in range(rows)]).int()
        picks = torch.cat([picks, torch.full((rows, 1), e - 1, dtype=torch.int32)], 1).contiguous()
        x = (torch.randn((rows, d), generator=g) * 0.5).to(torch.bfloat16)
        plan = grouped.Plan(rows, slots, e, "cuda", prefill=prompt)
        grouped.route(picks.cuda(), plan, fp8x.PROMPT_TILE if prompt else 16)
        act = torch.empty((rows * slots, ni), dtype=torch.bfloat16, device="cuda")
        fp8x.gate_up(x.cuda(), ex, plan, act, rows)
        y = torch.empty((rows * slots, d), dtype=torch.bfloat16 if prompt else torch.float32, device="cuda")
        fp8x.down(act, ex, plan, y, rows)
        torch.cuda.synchronize()
        arrays.update({f"picks{rows}": picks, f"x{rows}": x, f"act{rows}": act, f"y{rows}": y,
                       f"counts{rows}": plan.counts,
                       f"items{rows}": plan.items, f"members{rows}": plan.members})
    save(Path(a.out), {"experts": e, "dims": d, "width": ni, "slots": slots, "decode": ",".join(map(str, DECODE)),
                       "prompt": ",".join(map(str, PROMPT))}, arrays)


if __name__ == "__main__":
    main()
