#!/usr/bin/env python3
"""Offline replay of EXL3 Flash Next sparse-attention selection traces (tf-cuda-test exl3-kvpage DIR trace ...).

A trace holds, per attention layer (0..11) and per attention call (a 256-row prompt block or one decode row), each
row's 512 selected 4-key blocks (or "dense": the row reads every key, <= 2051). This replays a device block cache:
a recent dense window of W tokens (always on the card) plus C block slots per layer with exact LRU at call
granularity (the GPU's policy in cuda_exl3_kvpage.zig: a call's union is stamped, then the oldest slots are evicted).

Usage: exl3_kvpage_replay.py TRACE.bin [--windows 2048,4096,8192] [--caches 256,...] [--json OUT]
"""
import argparse
import json
import sys

import numpy as np

RATIO = 4
TOP = 512
BLOCK_BYTES = 4 * 2 * 2 * 256 * 2  # 4 keys x (K and V) x 2 KV heads x 256 dims x bf16 = 8 KiB
LAYERS = 12


def parse(path, ordered=None):
    """Per layer: [(pos, kind, rows)]; with `ordered` (a list) also every call in file (= execution) order."""
    raw = np.fromfile(path, dtype=np.uint8)
    calls = [[] for _ in range(LAYERS)]  # (pos, kind, [rows: array or None])
    at = 0
    n = raw.size
    while at < n:
        ai, pos, m, kind = np.frombuffer(raw[at:at + 16].tobytes(), dtype=np.uint32)
        at += 16
        rows = []
        for _ in range(int(m)):
            cnt = int(np.frombuffer(raw[at:at + 2].tobytes(), dtype=np.uint16)[0])
            at += 2
            if cnt == 0:
                rows.append(None)
            else:
                rows.append(np.frombuffer(raw[at:at + 2 * cnt].tobytes(), dtype=np.uint16).astype(np.int64))
                at += 2 * cnt
        calls[int(ai)].append((int(pos), int(kind), rows))
        if ordered is not None:
            ordered.append((int(ai), int(pos), int(kind), rows))
    return calls


class Lru:
    """C slots, exact LRU over calls; blocks are ints < nblocks."""

    def __init__(self, cap, nblocks):
        self.cap = cap
        self.stamp = np.full(nblocks, -1, dtype=np.int64)  # last call that used the block while resident
        self.res = np.zeros(nblocks, dtype=bool)
        self.count = 0
        self.t = 0

    def use(self, need):
        """`need`: unique blocks of one call (not in the window). Returns misses (overflow counted as misses)."""
        self.t += 1
        hits = self.res[need]
        miss = need[~hits]
        self.stamp[need] = self.t
        over = self.count + miss.size - self.cap
        if over > 0:
            resident = np.flatnonzero(self.res)
            old = resident[self.stamp[resident] < self.t]
            k = min(over, old.size)
            if k > 0:
                ev = old[np.argpartition(self.stamp[old], k - 1)[:k]] if k < old.size else old
                self.res[ev] = False
                self.count -= ev.size
        room = self.cap - self.count
        take = miss[:max(room, 0)]
        self.res[take] = True
        self.count += take.size
        return int(miss.size), int(max(miss.size - take.size, 0))


def window_start(end, w):
    """First token of the dense window for a call whose last row is at end - 1 (rounded down to a block)."""
    return max(0, (end - w) // RATIO * RATIO)


def outside(blocks, w0):
    return blocks[blocks * RATIO < w0]


def decode_stats(calls, windows, caches, nblocks, warm):
    """Per decode row: overlap with the previous row, new blocks, hit rates and bytes for each (W, C)."""
    out = {}
    rows = [(p, r[0]) for (p, k, r) in calls if k == 1 and r[0] is not None]
    if len(rows) < 2:
        return out
    ov = [np.intersect1d(a, b, assume_unique=True).size / TOP for (_, a), (_, b) in zip(rows, rows[1:])]
    new = [np.setdiff1d(b, a, assume_unique=True).size for (_, a), (_, b) in zip(rows, rows[1:])]
    out["rows"] = len(rows)
    out["pos"] = [rows[0][0], rows[-1][0]]
    out["overlap_prev_mean"] = float(np.mean(ov))
    out["new_blocks_per_row_mean"] = float(np.mean(new))
    u4 = []
    for i in range(0, len(rows) - 3, 4):
        u4.append(np.unique(np.concatenate([rows[i + j][1] for j in range(4)])).size)
    out["mtp4_union_mean"] = float(np.mean(u4)) if u4 else None
    for w in windows:
        frac_win = np.mean([1 - outside(b, window_start(p + 1, w)).size / TOP for p, b in rows])
        out[f"W{w}_frac_in_window"] = float(frac_win)
        for c in caches:
            for group in (1, 4):
                lru = Lru(c, nblocks)
                if warm is not None:
                    for need in warm(w):
                        lru.use(need)
                miss = 0
                over = 0
                sel = 0
                needed = 0
                per = []
                for i in range(0, len(rows) - group + 1, group):
                    end = rows[i + group - 1][0] + 1
                    w0 = window_start(end, w)
                    allb = np.unique(np.concatenate([rows[i + j][1] for j in range(group)]))
                    need = outside(allb, w0)
                    mi, ov_ = lru.use(need)
                    miss += mi
                    over += ov_
                    sel += allb.size
                    needed += need.size
                    per.append(mi)
                calls_n = len(per)
                key = f"W{w}_C{c}" + ("" if group == 1 else "_mtp4")
                out[key] = {
                    "hit_rate_selected": 1 - miss / max(sel, 1),
                    "hit_rate_outside_window": 1 - miss / max(needed, 1),
                    "misses_per_call_mean": miss / max(calls_n, 1),
                    "misses_per_call_p95": float(np.percentile(per, 95)) if per else 0,
                    "misses_per_call_max": int(max(per)) if per else 0,
                    "bytes_per_call_per_layer": miss / max(calls_n, 1) * BLOCK_BYTES,
                    "overflow": over,
                }
    return out


def prompt_calls(calls, w):
    """Each prompt call's unique needed blocks outside the window, in order (for warming and for prompt replay)."""
    res = []
    for p, k, rows in calls:
        if k != 0:
            continue
        sp = [r for r in rows if r is not None]
        if not sp:
            res.append(np.zeros(0, dtype=np.int64))
            continue
        end = p + len(rows)
        w0 = window_start(end, w)
        res.append(np.unique(outside(np.concatenate(sp), w0)))
    return res


def prompt_stats(calls, windows, caches, nblocks):
    out = {}
    pc = [(p, rows) for p, k, rows in calls if k == 0]
    if not pc:
        return out
    out["calls"] = len(pc)
    out["rows"] = sum(len(r) for _, r in pc)
    # union of a 256-row call's selections (all blocks, and in buckets of position)
    un = []
    for p, rows in pc:
        sp = [r for r in rows if r is not None]
        if sp:
            un.append((p, np.unique(np.concatenate(sp)).size))
    if un:
        arr = np.array(un)
        out["union_per_call_mean"] = float(arr[:, 1].mean())
        out["union_per_call_max"] = int(arr[:, 1].max())
        buckets = {}
        for lo, hi in ((0, 32768), (32768, 65536), (65536, 131072), (131072, 1 << 30)):
            sel = arr[(arr[:, 0] >= lo) & (arr[:, 0] < hi)]
            if sel.size:
                buckets[f"{lo}-{hi}"] = {"mean": float(sel[:, 1].mean()), "max": int(sel[:, 1].max())}
        out["union_by_pos"] = buckets
    for w in windows:
        needs = prompt_calls(calls, w)
        out[f"W{w}_outside_union_mean"] = float(np.mean([x.size for x in needs]))
        out[f"W{w}_outside_union_max"] = int(max(x.size for x in needs))
        for c in caches:
            lru = Lru(c, nblocks)
            miss = 0
            over = 0
            for need in needs:
                mi, ov_ = lru.use(need)
                miss += mi
                over += ov_
            out[f"W{w}_C{c}"] = {
                "miss_blocks_total": miss,
                "miss_bytes_per_row_per_layer": miss * BLOCK_BYTES / out["rows"],
                "miss_gib_total_per_layer": miss * BLOCK_BYTES / 2**30,
                "overflow_calls_blocks": over,
            }
    return out


def shared_replay(ordered, windows, totals, nblocks):
    """One slot pool for all 12 layers of the sequence (keys layer * nblocks + block), calls in execution order:
    the prompt (layer-major inside a deep chunk), then decode rounds (12 layers a token). Also MTP 4-row verify
    windows: 4 consecutive decode tokens as one call a layer."""
    out = {}
    decode = [(ai, p, rows[0]) for ai, p, k, rows in ordered if k == 1 and rows[0] is not None]
    prompt = [(ai, p, rows) for ai, p, k, rows in ordered if k == 0]
    prompt_rows = sum(len(r) for _, _, r in prompt) // LAYERS
    tokens = len(decode) // LAYERS
    for w in windows:
        pneeds = []
        for ai, p, rows in prompt:
            sp = [r for r in rows if r is not None]
            if not sp:
                pneeds.append((ai, np.zeros(0, dtype=np.int64)))
                continue
            w0 = window_start(p + len(rows), w)
            pneeds.append((ai, np.unique(outside(np.concatenate(sp), w0)) + ai * nblocks))
        dneeds = [(ai, outside(b, window_start(p + 1, w)) + ai * nblocks, b.size) for ai, p, b in decode]
        # MTP: tokens t..t+3 of one layer as one call (the layers interleaved per window)
        by_layer = [[x for x in decode if x[0] == li] for li in range(LAYERS)]
        mneeds = []
        for t0 in range(0, tokens - 3, 4):
            for li in range(LAYERS):
                rows = by_layer[li][t0:t0 + 4]
                if len(rows) < 4:
                    continue
                w0 = window_start(rows[-1][1] + 1, w)
                allb = np.unique(np.concatenate([r[2] for r in rows]))
                mneeds.append((li, outside(allb, w0) + li * nblocks, allb.size))
        out[f"W{w}_prompt_union_max"] = int(max((x.size for _, x in pneeds), default=0))
        for tot in totals:
            for kind in ("decode", "mtp4"):
                lru = Lru(tot, LAYERS * nblocks)
                pm = 0
                pov = 0
                for _, need in pneeds:
                    mi, ov_ = lru.use(need)
                    pm += mi
                    pov += ov_
                seq = dneeds if kind == "decode" else mneeds
                miss = 0
                sel = 0
                needed = 0
                per_tok = []
                acc = 0
                for i, (_, need, nsel) in enumerate(seq):
                    mi, _ = lru.use(need)
                    miss += mi
                    sel += nsel
                    needed += need.size
                    acc += mi
                    if i % LAYERS == LAYERS - 1:
                        per_tok.append(acc)
                        acc = 0
                key = f"W{w}_S{tot}"
                if kind == "decode":
                    out[key + "_prompt"] = {"miss_blocks": pm, "miss_bytes_per_row": pm * BLOCK_BYTES / max(prompt_rows, 1), "miss_gib": pm * BLOCK_BYTES / 2**30, "overflow_blocks": pov}
                unit = "token" if kind == "decode" else "window"
                out[key + "_" + kind] = {
                    "hit_rate_selected": 1 - miss / max(sel, 1),
                    "hit_rate_outside_window": 1 - miss / max(needed, 1),
                    f"misses_per_{unit}": float(np.mean(per_tok)) if per_tok else 0,
                    f"misses_per_{unit}_p95": float(np.percentile(per_tok, 95)) if per_tok else 0,
                    f"misses_per_{unit}_max": int(max(per_tok)) if per_tok else 0,
                    f"bytes_per_{unit}": float(np.mean(per_tok)) * BLOCK_BYTES if per_tok else 0,
                }
                print(key, kind, json.dumps(out[key + "_" + kind]), flush=True)
            print(key, "prompt", json.dumps(out[key + "_prompt"]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--windows", default="2048,4096,8192")
    ap.add_argument("--caches", default="256,512,1024,2048,4096,8192")
    ap.add_argument("--json")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--shared", help="slot totals for one pool shared by the 12 layers, e.g. 12288,24576,49152")
    a = ap.parse_args()
    windows = [int(x) for x in a.windows.split(",")]
    caches = [int(x) for x in a.caches.split(",")]
    ordered = [] if a.shared else None
    calls = parse(a.trace, ordered)
    maxpos = max((p + len(r)) for c in calls for (p, k, r) in c)
    nblocks = maxpos // RATIO + 16
    if a.shared:
        rep = {"trace": a.trace, "tokens": maxpos, "shared": shared_replay(ordered, windows, [int(x) for x in a.shared.split(",")], nblocks)}
        if a.json:
            with open(a.json, "w") as f:
                json.dump(rep, f, indent=1)
        return 0
    layers = range(LAYERS) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    report = {"trace": a.trace, "tokens": maxpos, "layers": {}}
    for li in layers:
        c = calls[li]
        has_prompt = any(k == 0 for _, k, _ in c)
        warm = (lambda w, c=c: prompt_calls(c, w)) if has_prompt else None
        report["layers"][li] = {"decode": decode_stats(c, windows, caches, nblocks, warm), "prompt": prompt_stats(c, windows, caches, nblocks)}
        d = report["layers"][li]["decode"]
        print(f"layer {li}: decode rows {d.get('rows')}, overlap {d.get('overlap_prev_mean', 0):.3f}, new/row {d.get('new_blocks_per_row_mean', 0):.1f}, mtp4 union {d.get('mtp4_union_mean')}", flush=True)
    # summary over layers
    summ = {}
    for w in windows:
        for c in caches:
            for kind in ("", "_mtp4"):
                key = f"W{w}_C{c}{kind}"
                vals = [report["layers"][li]["decode"].get(key) for li in layers]
                vals = [v for v in vals if v]
                if vals:
                    summ["decode_" + key] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
            vals = [report["layers"][li]["prompt"].get(f"W{w}_C{c}") for li in layers]
            vals = [v for v in vals if v]
            if vals:
                summ["prompt_" + f"W{w}_C{c}"] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    for k in ("overlap_prev_mean", "new_blocks_per_row_mean", "mtp4_union_mean"):
        vals = [report["layers"][li]["decode"].get(k) for li in layers]
        vals = [v for v in vals if v is not None]
        if vals:
            summ["decode_" + k] = float(np.mean(vals))
            summ["decode_" + k + "_by_layer"] = [round(float(v), 4) for v in vals]
    for w in windows:
        vals = [report["layers"][li]["decode"].get(f"W{w}_frac_in_window") for li in layers]
        vals = [v for v in vals if v is not None]
        if vals:
            summ[f"decode_W{w}_frac_in_window"] = float(np.mean(vals))
    report["summary"] = summ
    for k, v in summ.items():
        print(k, json.dumps(v) if isinstance(v, dict) else v)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
