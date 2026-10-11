#!/usr/bin/env python3
"""Compare complete steady rounds in two Nsight SQLite exports (reference engine and Zig), opened read-only."""

import argparse
import bisect
import collections
from pathlib import Path
import sqlite3
import statistics


KERNEL_SQL = """
SELECT k.start, k.end, s.value, d.value, k.gridX, k.gridY, k.gridZ, k.streamId
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON s.id = k.shortName
JOIN StringIds AS d ON d.id = k.demangledName
ORDER BY k.start
"""


def classify(short, full):
    if "dsv41_rg::gemv_kernel" in full:
        return "router_gemv"
    if "dsv41_rg::narrow_kernel" in full:
        return "router_narrow"
    if short == "group_kernel":
        return "expert_group"
    if short == "seg_rot_in_kernel":
        return "exl3_segment_rot_in"
    if short == "rot_in_kernel":
        return "exl3_expert_rot_in" if "rot_in_kernel<" in full else "exl3_linear_rot_in"
    if short == "_rope":
        return "positional_rope"
    if short == "_kv_store":
        return "kv_store"
    if short == "gate_copy_kernel":
        return "engram_gate_copy_poll"
    if short in ("ds_stage_rows_kernel", "ds_accept_rows_kernel", "gather_cols_kernel", "carry_kernel"):
        return short.removesuffix("_kernel")
    if "dsv41_topk_keys::" in full:
        return "sampling_topk"
    if "dsv41_topk::" in full:
        return "attention_topk"
    if "direct_copy_kernel" in full or "bfloat16_copy_kernel" in full:
        return "torch_copy"
    return "other"


def emit_stats(label, durations, round_counts, rounds):
    count = len(durations)
    print(f"{label}\t{count}\t{count / rounds:.6f}\t"
          f"{sum(durations) / 1e6 / rounds:.6f}\t"
          f"{statistics.mean(durations) / 1e3:.6f}\t"
          f"{min(round_counts)}/{statistics.median(round_counts):g}/{max(round_counts)}")


def measure(path, engine, marker):
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        rows = conn.execute(KERNEL_SQL).fetchall()
        if not rows:
            raise ValueError(f"{path}: no kernel activities")
        first, last = rows[0][0], rows[-1][1]
        lo, hi = first + .2 * (last - first), first + .8 * (last - first)
        markers = [start for start, _, short, *_ in rows
                   if short == marker and lo <= start <= hi]
        if len(markers) < 2:
            raise ValueError(f"{path}: need two {marker} markers inside middle 60%")
        if any(left >= right for left, right in zip(markers, markers[1:])):
            raise ValueError(f"{path}: round markers are not strictly increasing")
        begin, end, rounds = markers[0], markers[-1], len(markers) - 1
        durations = collections.defaultdict(list)
        counts = collections.defaultdict(lambda: [0] * rounds)
        categories = collections.defaultdict(list)
        category_counts = collections.defaultdict(lambda: [0] * rounds)
        shorts = collections.defaultdict(list)
        short_counts = collections.defaultdict(lambda: [0] * rounds)
        rotation_shapes = collections.defaultdict(list)
        rotation_shape_counts = collections.defaultdict(lambda: [0] * rounds)
        wide_rotation = []
        wide_rotation_counts = [0] * rounds
        pending_group = {}
        paired_rotation = []
        paired_rotation_counts = [0] * rounds
        paired_shapes = collections.Counter()
        for start, finish, short, full, grid_x, grid_y, grid_z, stream in rows:
            if not begin <= start < end:
                continue
            index = bisect.bisect_right(markers, start) - 1
            duration, category = finish - start, classify(short, full)
            durations[(short, full)].append(duration)
            counts[(short, full)][index] += 1
            categories[category].append(duration)
            category_counts[category][index] += 1
            shorts[short].append(duration)
            short_counts[short][index] += 1
            if category == "expert_group":
                pending_group[stream] = start
            if category == "exl3_expert_rot_in" and "<__nv_bfloat16>" in full:
                shape = (grid_x, grid_y, grid_z)
                rotation_shapes[shape].append(duration)
                rotation_shape_counts[shape][index] += 1
                # gridX = rows * slots, gridY = K / 128; with <=9 slots, gridX > 144 proves rows > 16.
                if grid_x > 144 and grid_y == 40 and grid_z == 2:
                    wide_rotation.append(duration)
                    wide_rotation_counts[index] += 1
                # Source dispatches group before rotIn on the same stream: a timeline pairing, not kernel arguments.
                if stream in pending_group and grid_y == 40 and grid_z == 2:
                    paired_rotation.append(duration)
                    paired_rotation_counts[index] += 1
                    paired_shapes[shape] += 1
            if category == "exl3_expert_rot_in":
                pending_group.pop(stream, None)

        intervals = [right - left for left, right in zip(markers, markers[1:])]
        print(f"\nENGINE\t{engine}\tDATABASE\t{Path(path).resolve()}")
        print(f"MARKER\t{marker}\tMIDDLE60_NS\t{lo:.3f}\t{hi:.3f}")
        print(f"BOUNDS_NS\t{begin}\t{end}\tROUNDS\t{rounds}")
        print(f"WALL_MS_PER_ROUND\t{(end - begin) / 1e6 / rounds:.6f}")
        print("ROUND_MS_MIN_MEDIAN_MAX\t" + "\t".join(
            f"{value / 1e6:.6f}" for value in
            (min(intervals), statistics.median(intervals), max(intervals))))
        print(f"LAUNCHES_PER_ROUND\t{sum(map(len, durations.values())) / rounds:.6f}")
        print(f"SUM_KERNEL_MS_PER_ROUND\t{sum(map(sum, durations.values())) / 1e6 / rounds:.6f}")

        header = "\tlaunches\tlaunches/round\tms/round\tus/launch\tcount/round_min/median/max"
        print("CATEGORY" + header)
        for category in sorted(categories):
            emit_stats(category, categories[category], category_counts[category], rounds)
        print("BF16_EXPERT_ROTATION_GRID_X/Y/Z" + header)
        for shape in sorted(rotation_shapes):
            emit_stats("exl3_bf16_rotation_grid=" + "/".join(map(str, shape)), rotation_shapes[shape],
                       rotation_shape_counts[shape], rounds)
        if wide_rotation:
            emit_stats("exl3_wide_rotation_conservative_gridX>144_K5120_slots<=9",
                       wide_rotation, wide_rotation_counts, rounds)
        if paired_rotation:
            emit_stats("exl3_rotation_after_group_same_stream_K5120",
                       paired_rotation, paired_rotation_counts, rounds)
            print("GROUP_PAIRED_ROTATION_GRIDS\t" + "\t".join(
                f"{'/'.join(map(str, shape))}:{count}"
                for shape, count in sorted(paired_shapes.items())))
        print("SHORT_NAME (may combine distinct kernels)" + header)
        for short in sorted(shorts, key=lambda key: (-sum(shorts[key]), key)):
            emit_stats(short, shorts[short], short_counts[short], rounds)
        print("CATEGORY\tSHORT_NAME\tFULL_NAME" + header)
        for key in sorted(durations, key=lambda key: (-sum(durations[key]), key)):
            short, full = key
            emit_stats(f"{classify(short, full)}\t{short}\t{full}",
                       durations[key], counts[key], rounds)

        print("MEMCPY_KIND\tactivities\tactivities/round\tms/round\tbytes/round")
        for kind, count, duration, size in conn.execute("""
            SELECT copyKind, COUNT(*), SUM(end-start), SUM(bytes)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE start >= ? AND start < ?
            GROUP BY copyKind ORDER BY copyKind
        """, (begin, end)):
            print(f"{kind}\t{count}\t{count / rounds:.6f}\t"
                  f"{duration / 1e6 / rounds:.6f}\t{size / rounds:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python_sqlite")
    parser.add_argument("zig_sqlite")
    args = parser.parse_args()
    measure(args.python_sqlite, "python", "_chain")
    measure(args.zig_sqlite, "zig", "ds_stage_rows_kernel")


if __name__ == "__main__":
    main()
