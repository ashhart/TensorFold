"""The indexer's top-k in one kernel: a radix select over the packed int64 keys of ``fused._index_scores``.

A key is (monotone int of the fp32 score) << 32 | (0x7FFFFFFF - position): unique, ordered by score and then lower
position first. ``torch.topk`` + ``torch.sort`` over those keys picks the same set this does; here one program a row
finds the K-th best score with four 8-bit histogram passes over the high word, then one pass in column order keeps the
columns above it and, at the K-th score, the lowest columns (= the lowest positions, the higher keys). Columns come
out ascending, so the sort that followed the top-k is gone too (adapted from TensorFold's glm5_next ``_select_rows``).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _hi_u32(k):
    """The key's high word (the score) as a uint32 whose unsigned order is the signed int64 order of the keys."""
    hi = (k >> 32).to(tl.int32)
    return (hi ^ -2147483648).to(tl.uint32, bitcast=True)


@triton.jit
def _order_u32(k, U32: tl.constexpr):
    """The sort word of a key: the packed int64's high word in unsigned order, or a stored uint32 order key as is."""
    if U32:
        return k.to(tl.uint32, bitcast=True)
    return _hi_u32(k)


@triton.jit
def _select_keys(S, OUT, NC, K: tl.constexpr, BLOCK: tl.constexpr, KEYS: tl.constexpr, U32: tl.constexpr = False):
    """Program r: row r of S [rows, NC] int64 -> OUT[r, :K] = the K best columns ascending (int32), or with KEYS their
    keys (int64, column order)."""
    r = tl.program_id(0).to(tl.int64)
    row = S + r * NC
    bins = tl.arange(0, 256)
    prefix = tl.zeros((), dtype=tl.uint32)
    fixed = tl.zeros((), dtype=tl.uint32)
    need = K
    for p in tl.static_range(4):
        hist = tl.zeros((256,), dtype=tl.int32)
        for c in range(0, NC, BLOCK):
            i = c + tl.arange(0, BLOCK)
            ok = i < NC
            u = _order_u32(tl.load(row + i, mask=ok, other=0 if U32 else -9223372036854775807), U32)
            match = ok & ((u & fixed) == prefix)
            hist += tl.histogram(((u >> (24 - 8 * p)) & 0xFF).to(tl.int32), 256, mask=match)
        at_or_above = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist
        digit = tl.max(tl.where(at_or_above >= need, bins, 0), 0)
        need -= tl.sum(tl.where(bins > digit, hist, 0), 0)
        prefix = prefix | (digit.to(tl.uint32) << (24 - 8 * p))
        fixed = fixed | (tl.full((), 0xFF, tl.uint32) << (24 - 8 * p))
    written = 0
    equal_seen = 0
    for c in range(0, NC, BLOCK):
        i = c + tl.arange(0, BLOCK)
        ok = i < NC
        key = tl.load(row + i, mask=ok, other=0 if U32 else -9223372036854775807)
        u = _order_u32(key, U32)
        eq = (ok & (u == prefix)).to(tl.int32)
        take = (ok & (u > prefix)) | ((eq == 1) & (tl.cumsum(eq, 0) - eq + equal_seen < need))
        t = take.to(tl.int32)
        at = OUT + r * K + written + tl.cumsum(t, 0) - t
        if KEYS:
            tl.store(at, key, mask=take)
        else:
            tl.store(at, i.to(tl.int32), mask=take)
        written += tl.sum(t, 0)
        equal_seen += tl.sum(eq, 0)


def top_columns(sc: torch.Tensor, k: int, out: torch.Tensor) -> torch.Tensor:
    """out[r, :k] <- row r's k best columns, ascending int32, of sc [n, T] (T >= k): packed int64 keys, or int32
    order words (``_index_scores`` with PACK=False: the score's unsigned order; ties go to the lower column = position)."""
    n, T = sc.shape
    _select_keys[(n,)](sc, out, T, K=k, BLOCK=1024, KEYS=False, U32=sc.dtype == torch.int32, num_warps=4)
    return out


def top_keys(sc: torch.Tensor, k: int) -> torch.Tensor:
    """Row r's k best keys of sc [n, T] (T >= k), int64 in column order (the set torch.topk(sorted=False) keeps)."""
    n, T = sc.shape
    out = torch.empty((n, k), dtype=torch.int64, device=sc.device)
    _select_keys[(n,)](sc, out, T, K=k, BLOCK=1024, KEYS=True, num_warps=4)
    return out
