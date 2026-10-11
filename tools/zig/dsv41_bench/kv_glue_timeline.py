#!/usr/bin/env python3
"""Read-only KV glue measurements (RMS + store launches per steady round) from two Nsight SQLite exports."""
import argparse
import bisect
import collections
from pathlib import Path
import sqlite3
import statistics


SQL = """
SELECT k.start,k.end,s.value AS name,k.globalPid,k.deviceId,k.contextId,k.streamId,
       k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
       k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON s.id=k.shortName
ORDER BY k.start
"""
BOUNDARIES = {"_kv_store", "_chain", "ds_stage_rows_kernel"}


def shape(row, prefix):
    return tuple(row[prefix + dimension] for dimension in "XYZ")


def rms_matches(row, store):
    return (row["name"] == "_rms" and shape(row, "grid") == shape(store, "grid")
            and shape(row, "grid")[1:] == (1, 1)
            and shape(row, "block") == (128, 1, 1)
            and row["registersPerThread"] == 36
            and row["staticSharedMemory"] == 0 and row["dynamicSharedMemory"] == 32)


def identify(store, history, lookback):
    if not history:
        return "unmatched_no_predecessor", None, 0
    previous = history[-1]
    if (shape(store, "grid")[1:] != (1, 1) or shape(store, "block") != (128, 1, 1)
            or store["dynamicSharedMemory"] != 1024):
        return "unmatched_store_fingerprint", None, 0
    if previous["name"] == "_pool_norm" and store["registersPerThread"] in (31, 33):
        return "compressed_excluded", None, 0
    if previous["name"] == "_rms" and rms_matches(previous, store):
        category = {29: "backbone_swa", 26: "dspark_ingest"}.get(store["registersPerThread"])
        if category:
            return category, previous, 1
    if previous["name"] == "_rope" and store["registersPerThread"] == 29:
        for distance, row in enumerate(reversed(history[-lookback:]), 1):
            if row["name"] in BOUNDARIES:
                break
            if rms_matches(row, store):
                return "dspark_attention", row, distance
    return "unmatched_producer", None, 0


def report(label, pairs, rounds, markers):
    counts = [0] * rounds
    for store, _, _ in pairs:
        counts[bisect.bisect_right(markers, store["start"]) - 1] += 1
    store_ns = sum(row["end"] - row["start"] for row, _, _ in pairs)
    norms = [norm for _, norm, _ in pairs if norm is not None]
    rms_ns = sum(row["end"] - row["start"] for row in norms)
    print(f"{label}\t{len(pairs)}\t{len(norms)}\t{len(pairs)/rounds:.6f}\t"
          f"{store_ns/1e6/rounds:.9f}\t{rms_ns/1e6/rounds:.9f}\t"
          f"{(store_ns+rms_ns)/1e6/rounds:.9f}\t"
          f"{store_ns/1e3/len(pairs):.6f}\t"
          f"{rms_ns/1e3/len(norms) if norms else 0:.6f}\t"
          f"{min(counts)}/{statistics.median(counts):g}/{max(counts)}\t"
          f"{max(distance for _, _, distance in pairs)}")


def measure(path, engine, marker, lookback):
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(SQL).fetchall()
    if not rows:
        raise ValueError(f"{path}: no kernels")
    first, last = rows[0]["start"], rows[-1]["end"]
    lo, hi = first + .2 * (last - first), first + .8 * (last - first)
    markers = [row["start"] for row in rows if row["name"] == marker and lo <= row["start"] <= hi]
    if len(markers) < 2 or any(a >= b for a, b in zip(markers, markers[1:])):
        raise ValueError(f"{path}: need strictly increasing round markers")
    begin, end, rounds = markers[0], markers[-1], len(markers) - 1
    histories = collections.defaultdict(list)
    categories = collections.defaultdict(list)
    shapes = collections.defaultdict(list)
    for row in rows:
        stream = tuple(row[field] for field in ("globalPid", "deviceId", "contextId", "streamId"))
        history = histories[stream]
        if begin <= row["start"] < end and row["name"] == "_kv_store":
            category, norm, distance = identify(row, history, lookback)
            pair = (row, norm, distance)
            categories[category].append(pair)
            shapes[(category, shape(row, "grid"), row["registersPerThread"])].append(pair)
        history.append(row)
        if len(history) > lookback:
            del history[:-lookback]
    print(f"\nENGINE\t{engine}\tDATABASE\t{Path(path).resolve()}")
    print(f"BOUNDS_NS\t{begin}\t{end}\tROUNDS\t{rounds}\tLOOKBACK\t{lookback}")
    print("CATEGORY\tstores\tmatched_rms\tstores/round\tstore_ms/round\trms_ms/round\t"
          "sum_ms/round\tstore_us/launch\trms_us/launch\tcount_min/median/max\tmax_lookback")
    for category in sorted(categories):
        report(category, categories[category], rounds, markers)
    eligible = [pair for category, pairs in categories.items()
                if category in ("backbone_swa", "dspark_ingest", "dspark_attention") for pair in pairs]
    if eligible:
        report("eligible_total", eligible, rounds, markers)
    print("CATEGORY:GRID:REGISTERS\tstores\tmatched_rms\tstores/round\tstore_ms/round\trms_ms/round\t"
          "sum_ms/round\tstore_us/launch\trms_us/launch\tcount_min/median/max\tmax_lookback")
    for (category, grid, registers), pairs in sorted(shapes.items()):
        report(f"{category}:{'/'.join(map(str, grid))}:regs{registers}", pairs, rounds, markers)
    unmatched = sum(len(pairs) for category, pairs in categories.items() if category.startswith("unmatched"))
    print(f"UNMATCHED_STORES\t{unmatched}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python_sqlite")
    parser.add_argument("zig_sqlite")
    parser.add_argument("--lookback", type=int, default=20)
    args = parser.parse_args()
    if args.lookback < 1:
        parser.error("--lookback must be positive")
    measure(args.python_sqlite, "python", "_chain", args.lookback)
    measure(args.zig_sqlite, "zig", "ds_stage_rows_kernel", args.lookback)


if __name__ == "__main__":
    main()
