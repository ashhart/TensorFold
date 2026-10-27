#!/usr/bin/env python3
"""Per-card memory budget for Qwen3.8-Flash-Next EXL3 at 200k context on two 16 GiB cards (docs/plans/exl3-kv-memory.md).

  exl3_kv_budget.py [CONTEXT]

Every byte formula is upstream's (cuda_prompt.zig seqBytes, cuda_batch_state.zig Scratch.bytes, cuda_mtp_draft.zig
Draft.bytes) with the K10 levers applied (cuda_exl3_kv.zig seqBytes gives the same numbers; its host test checks the
default policy against upstream's). Inputs that are not formulas are marked where they come from.
"""
import sys

G = 2 ** 30
CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
CAP = CTX + 16  # the window a stream holds: context + mtp.max_rows (cuda_lanes_load.zig)

# -- model shapes (cuda_prompt_wt.zig, cuda_gdn.zig) ------------------------------------------------------------------
dims, streams, low, vocab, experts_n, slots, moe_w, tile = 2560, 4, 320, 248320, 512, 11, 640, 16
wide = streams * dims
q_heads, kv_heads, head_dim, index_heads, index_dim, half, proj_n, out_k, nch = 24, 2, 256, 4, 128, 32, 13952, 6144, 5
idw, att_rows, hc_sk, out_sk, plan_small = 2052, 256, 32, 8, 1024
nk, nv, dk, dv, taps = 16, 48, 128, 128, 4
conv_dim = 2 * nk * dk + nv * dv
value_dim = nv * dv
proj_row = max(conv_dim + nv * dv + 2 * nv, proj_n)
pss_n = (dims // 256) * streams
state_n = nv * dv * dk
linear_n, attn_n, ple_tail_n = 36, 12, 9
cdiv = lambda a, b: (a + b - 1) // b


def scratch(cap, whole_head=False):
    """cuda_batch_state.zig Scratch.bytes(cap), term for term; whole_head drops head_local (cap * vocab bytes), which
    only a vocabulary-split head reads (cuda_batch.zig projectHead)."""
    pairs = cap * slots
    nblk = cdiv(pairs, plan_small)
    items = cdiv(pairs, tile) + experts_n + 1
    n = cap * 4 + (cap + 1) * 16
    n += cap * wide * 2 * 2 + cap * pss_n * 4 + cap * (low + 4) * 2 + hc_sk * cap * (low + 4) * 4
    n += cap * wide * 2 + 8 * cap * dims * 4
    n += cap * low * 2 + cap * (low // 32) * 4 + cap * streams * 2 + cap * dims * 2 + cap * (dims // 32) * 4
    n += cap * proj_row * 2 + out_sk * cap * dims * 4 + cap * value_dim * (2 + 4 + 4) + cap * (value_dim // 32) * 4
    n += cap * (2 * nk * dk + 2 * nv) * 4
    n += cap * q_heads * head_dim * 2 + cap * index_heads * index_dim * 2
    n += att_rows * nch * q_heads * (head_dim + 2) * 4 + cap * out_k * 2 * 2 + cap * (out_k // 32) * 4
    n += att_rows * (idw + 2) * 4
    n += cap * (experts_n + 1) * 4 + pairs * 4 * 2 + pairs * 4 + items * 3 * 4 + 8 + pairs * 4 + nblk * (experts_n + 1) * 4
    n += cap * moe_w * 6 + cap * dims * 4
    n += pairs * moe_w * 2 + pairs * dims * 4 + cap * dims * 2
    n += cap * dims * 2 + cap * (dims // 32) * 4 + cap * wide * 2 + cap * dims * 2 + 4 * cap * dims * 4
    n += cap * wide * 2 * 2 + cap * streams * 4 + vocab * 2 + (0 if whole_head else cap * vocab) + half * 4 + 4096
    return n


def kvrow(kv, heads):
    per = {"bf16": head_dim * 2, "int8": head_dim + head_dim // 32 * 2, "int4": head_dim // 2 + head_dim // 32 * 2}[kv]
    return 2 * heads * per


def pooled(cap):
    return cdiv(cap, 4) * index_dim * 2


def seq(cap, kv="bf16", heads=2, kv_host=False, ikc_host=False):
    """(device, host) bytes of one stream: cuda_prompt.zig seqBytes with the levers."""
    gdn = linear_n * ((taps - 1) * conv_dim * 2 + state_n * 4)
    k = attn_n * cap * kvrow(kv, heads)
    i = attn_n * cap * index_dim * 2
    dev = gdn + (0 if kv_host else k) + (0 if ikc_host else i) + attn_n * pooled(cap) + ple_tail_n * wide * 2
    return dev, (k if kv_host else 0) + (i if ikc_host else 0)


def draft(cap, kv="bf16", kv_host=False, ikc_host=False, rows=1536, draft_n=79591):
    """cuda_mtp_draft.zig Draft.bytes (rank 0, MTP head's one attention layer under the same policy)."""
    fixed = max(rows, 16) * wide * 2 + wide * 2 + 16 * vocab * 2 + draft_n * 2 + 16 * 16 + 16 * wide * 2 + linear_n * state_n * 4
    return fixed + (0 if kv_host else cap * kvrow(kv, kv_heads)) + (0 if ikc_host else cap * index_dim * 2) + pooled(cap)


# -- inputs that are not formulas -----------------------------------------------------------------------------------------
CARD = 15.568  # GiB, cuMemGetInfo total on a 5060 Ti (measured, exl3-kv bench)
CONTEXT_MODULES = 0.40  # GiB, CUDA context + every module the engine loads (estimate; our test's modules alone: 0.134 measured)
WEIGHTS = {0: 5.107 + 0.475, 1: 5.245}  # GiB, T10's expert-parallel placement (read from the pack's headers); rank 0 + MTP head
OTHER = 0.05 + 0.015 + 0.05  # GiB: block scores at 200k (scoreBytes, read), lanes logits (read), EXL3 Dense scratch (estimate)
LINE = 0.136  # GiB a pool line in every one of 48 layers (T10: 10 lines/layer = 1.36 GiB)
FLOOR = 10  # lines a layer, the pool's floor
# decode hit rate at tp=1 by lines a layer, measured by E2 on real routing (512 experts in one pool)
HIT = [(10, 0.33), (24, 0.505), (48, 0.64), (96, 0.78), (160, 0.88)]
MISS_MS = 0.147  # one 3-bit expert staged from a pinned bank (measured, E2)


def hit_tp1(lines):
    import math
    if lines <= HIT[0][0]:
        return HIT[0][1] * lines / HIT[0][0]
    for (a, ha), (b, hb) in zip(HIT, HIT[1:]):
        if lines <= b:
            t = (math.log(lines) - math.log(a)) / (math.log(b) - math.log(a))
            return ha + t * (hb - ha)
    return HIT[-1][1]


def miss_ms(lines_rank):
    """Expert-miss staging a token on one rank (estimate): a rank serves its 256 experts, ~5 picks a row a layer; its
    L lines hit like a tp=1 pool of 2L lines over all 512."""
    h = hit_tp1(2 * lines_rank)
    return 48 * 5 * (1 - h) * MISS_MS, h


LEVERS = [
    # name, kv dtype, heads a rank, kv_host, ikc_host, chunk, whole-head scratch
    ("A upstream: bf16 KV, chunk 1536", "bf16", 2, False, False, 1536, False),
    ("B int8 KV", "int8", 2, False, False, 1536, False),
    ("C = B + index keys in host memory", "int8", 2, False, True, 1536, False),
    ("D = C + scratch: chunk 1024, no head_local", "int8", 2, False, True, 1024, True),
    ("E = D + attention head split", "int8", 1, False, True, 1024, True),
    ("F bf16 KV + head split + index keys on host + D's scratch", "bf16", 1, False, True, 1024, True),
    ("G host KV (int8, mapped) + index keys on host + D's scratch", "int8", 2, True, True, 1024, True),
]


def table(drafts=True):
    print(f"drafts {'on (rank 0 holds every stream' + chr(39) + 's draft cache)' if drafts else 'off'}")
    print(f"| lever (context {CTX:,}) | seqs | rank 0 seq+draft GiB | rank 0 left for pool GiB | lines/layer r0 | r1 lines | decode miss ms/token (est.) | host GiB |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, kv, heads, kvh, ikh, chunk, whole in LEVERS:
        for n in (1, 2, 4):
            d, h = seq(CAP, kv, heads, kvh, ikh)
            dr = draft(CAP, kv, kvh, ikh, rows=chunk) if drafts else 0
            fixed = CONTEXT_MODULES + scratch(chunk, whole) / G + OTHER
            left = {r: CARD - WEIGHTS[r] - fixed - n * d / G - (n * dr / G if r == 0 else 0) for r in (0, 1)}
            lines = {r: max(0, int(left[r] / LINE)) for r in (0, 1)}
            ok = min(lines.values()) >= FLOOR
            ms, hit = miss_ms(min(lines.values())) if ok else (float("nan"), 0)
            host = n * h / G
            print(f"| {name} | {n} | {n * (d + dr) / G:.2f} | {left[0]:.2f} | {lines[0] if left[0] > 0 else 'no fit'} | {lines[1] if left[1] > 0 else 'no fit'} | {'%.1f (hit %.2f)' % (ms, hit) if ok else 'below floor'} | {host:.1f} |")


if __name__ == "__main__":
    print(f"scratch GiB by chunk: " + ", ".join(f"{c}: {scratch(c) / G:.3f} ({scratch(c, True) / G:.3f} whole head)" for c in (1536, 1024, 512, 256)))
    d, _ = seq(CAP)
    print(f"one stream at {CTX:,} (bf16, upstream): {d / G:.3f} GiB; draft cache {draft(CAP) / G:.3f} GiB")
    print()
    table(True)
    print()
    table(False)
