"""The keyed draw on the device: radix passes for each row's top_p cut, then its best keyed Gumbel score, its column
and the runner-up in one pass over float64 scaled logits."""

import torch
import triton as tr
import triton.language as tl

BLOCK = 1024


@tr.jit
def _parts(S, IDS, KEYS, FLOOR, VCUT, ICUT, F, C, W, offset, STRIDE: tl.constexpr, T: tl.constexpr,
           MAP: tl.constexpr, CUT: tl.constexpr, B: tl.constexpr):
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
    keep = valid & ~(v < tl.load(FLOOR + row))
    if CUT:                                                                 # the order (-v, id) up to (VCUT, ICUT)
        cut = tl.load(VCUT + row)
        keep = keep & ((v > cut) | ((v == cut) & (ids <= tl.load(ICUT + row))))
    score = tl.where(keep, score, -float("inf"))
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
def best_two(scaled: torch.Tensor, keys: torch.Tensor, floor: torch.Tensor, offset: int, id_map: torch.Tensor | None,
             cut: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """float64 [rows, 3] (best, runner-up, the best's scaled logit) and int64 [rows, 2] (its column, its id)."""

    rows, width = scaled.shape
    tiles = tr.cdiv(width, BLOCK)
    parts = torch.empty((rows, tiles, 2), dtype=torch.float64, device=scaled.device)
    cols = torch.empty((rows, tiles), dtype=torch.int32, device=scaled.device)
    out_f = torch.empty((rows, 3), dtype=torch.float64, device=scaled.device)
    out_i = torch.empty((rows, 2), dtype=torch.int64, device=scaled.device)
    ids = id_map if id_map is not None else keys                  # never read without a map
    vcut, icut = cut if cut is not None else (floor, keys)        # never read without a cut
    _parts[(rows, tiles)](scaled, ids, keys, floor, vcut, icut, parts, cols, width, offset, scaled.stride(0), tiles,
                          id_map is not None, cut is not None, BLOCK, num_warps=8)
    _finish[(rows,)](parts, cols, scaled, ids, out_f, out_i, offset, scaled.stride(0), tiles, id_map is not None,
                     tr.next_power_of_2(tiles), num_warps=4)
    return out_f, out_i


@tr.jit
def _count(X, IDS, MASS, PART, H, W, STRIDE: tl.constexpr, MSTRIDE: tl.constexpr, SHIFT: tl.constexpr,
           D: tl.constexpr, B: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    cols = block * B + tl.arange(0, B)
    valid = cols < W
    x = tl.load(X + row * STRIDE + cols, valid, other=0.0).to(tl.float32)
    bits = tl.where(x == 0.0, 0.0, x).to(tl.int32, bitcast=True).to(tl.int64)          # ``sampling._order``
    key = ((-1 - (bits ^ ((bits >> 31) & 0x7FFFFFFF))) << 32) + tl.load(IDS + cols, valid, other=0)
    if SHIFT + D == 64:
        digit = (key >> SHIFT) + (1 << (D - 1))
    else:
        digit = (key >> SHIFT) & ((1 << D) - 1)
        valid = valid & ((key >> (SHIFT + D)) == (tl.load(PART + row) >> (SHIFT + D)))
    tl.atomic_add(H + row * (1 << D) + digit, tl.load(MASS + row * MSTRIDE + cols, valid, other=0), mask=valid)


@tr.jit
def _pick(H, NEED, BELOW, PART, world, rows, SHIFT: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    need = tl.load(NEED + row)
    below = tl.load(BELOW + row)
    run = below
    pick = below * 0
    for start in range(0, 1 << D, B):
        at = start + tl.arange(0, B)
        h = tl.load(H + row * (1 << D) + at)
        for k in range(1, world):
            h += tl.load(H + (k * rows + row) * (1 << D) + at)
        c = run + tl.cumsum(h, 0)
        short = c < need
        pick += tl.sum(short.to(tl.int64), 0)
        below = tl.maximum(below, tl.max(tl.where(short, c, below), 0))
        run += tl.sum(h, 0)
    if SHIFT + D == 64:
        pick -= 1 << (D - 1)
    tl.store(PART + row, tl.load(PART + row) + (pick << SHIFT))
    tl.store(BELOW + row, below)


@torch.no_grad()
def count(logits: torch.Tensor, ids: torch.Tensor, mass: torch.Tensor, part: torch.Tensor, shift: int,
          digit: int) -> torch.Tensor:
    """``sampling._count`` on the device: int64 [rows, 2**digit], each digit's mass among keys matching ``part``."""

    rows, width = logits.shape
    hist = torch.zeros((rows, 1 << digit), dtype=torch.int64, device=logits.device)
    _count[(rows, tr.cdiv(width, BLOCK))](logits, ids, mass, part, hist, width, logits.stride(0), mass.stride(0),
                                          shift, digit, BLOCK, num_warps=8)
    return hist


@torch.no_grad()
def pick(hist: torch.Tensor, need: torch.Tensor, below: torch.Tensor, part: torch.Tensor, shift: int,
         digit: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``sampling._pick`` on the device, updating ``part`` and ``below`` in place."""

    world, rows = hist.shape[:2]
    _pick[(rows,)](hist.contiguous(), need, below, part, world, rows, shift, digit, min(BLOCK, 1 << digit),
                   num_warps=4)
    return part, below
