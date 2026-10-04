"""Each row's best keyed Gumbel score, its column and the runner-up, in one pass over float64 scaled logits."""

import torch
import triton as tr
import triton.language as tl

BLOCK = 1024


@tr.jit
def _parts(S, IDS, KEYS, FLOOR, F, C, W, offset, STRIDE: tl.constexpr, T: tl.constexpr, MAP: tl.constexpr,
           B: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    at = tl.arange(0, B)
    cols = part * B + at
    valid = cols < W
    v = tl.load(S + row * STRIDE + cols, valid, other=-float("inf"))
    if MAP:
        ids = tl.load(IDS + cols, valid, other=0).to(tl.int64)
    else:
        ids = cols.to(tl.int64) + offset
    x = (tl.load(KEYS + row) ^ ids).to(tl.uint64, bitcast=True)            # ``exact_sampling.uniform_rows``
    x = x ^ (x >> 30)
    x = x * 0xBF58476D1CE4E5B9
    x = x ^ (x >> 27)
    x = x * 0x94D049BB133111EB
    x = x ^ (x >> 31)
    u = (x >> 11).to(tl.float64) * 1.1102230246251565e-16 + 5.551115123125783e-17
    score = v - tl.log(-tl.log(u))
    score = tl.where(valid & ~(v < tl.load(FLOOR + row)), score, -float("inf"))
    first = tl.max(score, 0)
    best = tl.argmax(score, 0)
    second = tl.max(tl.where(at == best, -float("inf"), score), 0)
    tl.store(F + (row * T + part) * 2, first)
    tl.store(F + (row * T + part) * 2 + 1, second)
    tl.store(C + row * T + part, part * B + best)


@tr.jit
def _finish(F, C, S, IDS, OUT_F, OUT_I, offset, STRIDE: tl.constexpr, T: tl.constexpr, MAP: tl.constexpr,
            B: tl.constexpr):
    row = tl.program_id(0)
    at = tl.arange(0, B)
    first = tl.load(F + (row * T + at) * 2, at < T, other=-float("inf"))
    top = tl.max(first, 0)
    part = tl.argmax(first, 0)
    others = tl.max(tl.where(at == part, -float("inf"), first), 0)
    second = tl.maximum(others, tl.load(F + (row * T + part) * 2 + 1))
    col = tl.load(C + row * T + part)
    tl.store(OUT_F + row * 3, top)
    tl.store(OUT_F + row * 3 + 1, second)
    tl.store(OUT_F + row * 3 + 2, tl.load(S + row * STRIDE + col))
    tl.store(OUT_I + row * 2, col.to(tl.int64))
    if MAP:
        tl.store(OUT_I + row * 2 + 1, tl.load(IDS + col).to(tl.int64))
    else:
        tl.store(OUT_I + row * 2 + 1, col.to(tl.int64) + offset)


@torch.no_grad()
def best_two(scaled: torch.Tensor, keys: torch.Tensor, floor: torch.Tensor, offset: int,
             id_map: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """float64 [rows, 3] (best, runner-up, the best's scaled logit) and int64 [rows, 2] (its column, its id)."""

    rows, width = scaled.shape
    tiles = tr.cdiv(width, BLOCK)
    parts = torch.empty((rows, tiles, 2), dtype=torch.float64, device=scaled.device)
    cols = torch.empty((rows, tiles), dtype=torch.int32, device=scaled.device)
    out_f = torch.empty((rows, 3), dtype=torch.float64, device=scaled.device)
    out_i = torch.empty((rows, 2), dtype=torch.int64, device=scaled.device)
    ids = id_map if id_map is not None else keys                  # never read without a map
    _parts[(rows, tiles)](scaled, ids, keys, floor, parts, cols, width, offset, scaled.stride(0), tiles,
                          id_map is not None, BLOCK, num_warps=8)
    _finish[(rows,)](parts, cols, scaled, ids, out_f, out_i, offset, scaled.stride(0), tiles, id_map is not None,
                     tr.next_power_of_2(tiles), num_warps=4)
    return out_f, out_i
