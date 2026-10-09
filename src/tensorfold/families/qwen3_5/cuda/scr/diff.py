"""Token diff between the previous turn's committed ids and the new prompt.

Ported from facebookresearch/context-language-models ``suffix_cache_reuse/overlay.py``
(``_cdc_bounds`` / ``_chunk_ids`` / ``_verify_ops`` / ``_fast_opcodes`` /
``_diff_opcodes``), module-level instead of SGLang globals. Opcodes are
difflib-shaped tuples ``(tag, i1, i2, j1, j2)``; every plan re-verifies its
opcodes token for token before any KV is touched."""

from difflib import SequenceMatcher


def cdc_bounds(ids, target: int) -> list[int]:
    """Content-defined chunk boundaries, a function of the token ids only."""
    mask = target - 1
    bounds = [0]
    h = 0
    for i, t in enumerate(ids):
        h = ((h << 1) ^ ((t * 0x9E3779B1) & 0xFFFFFFFF)) & 0xFFFFFFFF
        if ((h >> 8) & mask) == 0:
            bounds.append(i + 1)
    if bounds[-1] != len(ids):
        bounds.append(len(ids))
    return bounds


def chunk_ids(ids, bounds, interner: dict) -> list[int]:
    out = []
    for k in range(len(bounds) - 1):
        key = tuple(ids[bounds[k]:bounds[k + 1]])
        v = interner.get(key)
        if v is None:
            v = len(interner)
            interner[key] = v
        out.append(v)
    return out


def verify_ops(a, b, ops) -> bool:
    """True iff ops tile both sequences and every 'equal' block is really equal."""
    pi = pj = 0
    for t, i1, i2, j1, j2 in ops:
        if i1 != pi or j1 != pj:
            return False
        if t == "equal" and a[i1:i2] != b[j1:j2]:
            return False
        pi, pj = i2, j2
    return pi == len(a) and pj == len(b)


def fast_opcodes(a, b, *, cdc_target: int, cdc_extend: bool = True):
    """difflib-compatible opcodes via chunk-level diffing; None means fall back."""
    ba = cdc_bounds(a, cdc_target)
    bb = cdc_bounds(b, cdc_target)
    interner: dict = {}
    ca = chunk_ids(a, ba, interner)
    cb = chunk_ids(b, bb, interner)
    blocks = []
    for t, i1, i2, j1, j2 in SequenceMatcher(a=ca, b=cb, autojunk=False).get_opcodes():
        if t != "equal":
            continue
        ti1, ti2, tj1, tj2 = ba[i1], ba[i2], bb[j1], bb[j2]
        if ti2 - ti1 != tj2 - tj1:
            return None
        blocks.append([ti1, ti2, tj1, tj2])
    if cdc_extend and blocks:
        for n, blk in enumerate(blocks):
            lo_a = blocks[n - 1][1] if n else 0
            lo_b = blocks[n - 1][3] if n else 0
            while blk[0] > lo_a and blk[2] > lo_b and a[blk[0] - 1] == b[blk[2] - 1]:
                blk[0] -= 1
                blk[2] -= 1
        for n in range(len(blocks) - 1, -1, -1):
            blk = blocks[n]
            hi_a = blocks[n + 1][0] if n + 1 < len(blocks) else len(a)
            hi_b = blocks[n + 1][2] if n + 1 < len(blocks) else len(b)
            while blk[1] < hi_a and blk[3] < hi_b and a[blk[1]] == b[blk[3]]:
                blk[1] += 1
                blk[3] += 1
    ops, pi, pj = [], 0, 0
    for ti1, ti2, tj1, tj2 in blocks:
        da, db = ti1 - pi, tj1 - pj
        if da and db:
            ops.append(("replace", pi, ti1, pj, tj1))
        elif da:
            ops.append(("delete", pi, ti1, pj, tj1))
        elif db:
            ops.append(("insert", pi, ti1, pj, tj1))
        ops.append(("equal", ti1, ti2, tj1, tj2))
        pi, pj = ti2, tj2
    da, db = len(a) - pi, len(b) - pj
    if da and db:
        ops.append(("replace", pi, len(a), pj, len(b)))
    elif da:
        ops.append(("delete", pi, len(a), pj, len(b)))
    elif db:
        ops.append(("insert", pi, len(a), pj, len(b)))
    return ops


def diff_opcodes(a, b, *, fastdiff: bool = True, cdc_target: int = 64, fastdiff_min: int = 2048,
                 log=None) -> list:
    """Token-level opcodes between the previous and the new prompt, verified."""
    if fastdiff and min(len(a), len(b)) >= fastdiff_min:
        try:
            ops = fast_opcodes(a, b, cdc_target=cdc_target)
        except Exception as e:                                   # noqa: BLE001
            if log:
                log(f"fastdiff failed ({e}); using difflib")
            ops = None
        if ops is not None and verify_ops(a, b, ops):
            return ops
        if log:
            log("fastdiff rejected by verifier; falling back to difflib")
    return SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()


def quick_ratio_shared(a, bcount: dict, lb: int) -> float:
    """difflib quick_ratio with Counter(b) computed once per request."""
    avail: dict = {}
    matches = 0
    for elt in a:
        numb = avail[elt] if elt in avail else bcount.get(elt, 0)
        avail[elt] = numb - 1
        if numb > 0:
            matches += 1
    la = len(a)
    return 2.0 * matches / (la + lb) if (la + lb) else 1.0