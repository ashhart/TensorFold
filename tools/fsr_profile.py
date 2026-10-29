#!/usr/bin/env python3
"""Analyze complete native FSR routing traces without loading model weights."""
import argparse
import json
import math
from pathlib import Path


class TraceError(ValueError):
    pass


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TraceError(f"invalid {name}")
    return value


def read_requests(jobs, parts):
    count = 0
    for size in parts:
        previous, length = None, 0
        for key, slot in jobs:
            adjacent = previous is not None and key[0] == previous[0][0] and key[1] == previous[0][1] + 1 and slot == previous[1] + 1
            if adjacent and length + size <= 4 * 1024 * 1024:
                length += size
            else:
                count += math.ceil(length / (4 * 1024 * 1024))
                length = size
            previous = (key, slot)
        count += math.ceil(length / (4 * 1024 * 1024))
    return count


def analyze(lines, budgets, ssd_bytes_per_second=None, *, tokens=None, resident_tps=None, read_latency_ns=0):
    records = [json.loads(line) for line in lines if line.strip()]
    if any(not isinstance(record, dict) for record in records):
        raise TraceError("every trace record must be an object")
    if len(records) < 4 or records[0].get("format") != "tensorfold-fsr-v1":
        raise TraceError("missing FSR header or observations")
    header = records[0]
    layers = integer(header.get("layers"), "layers", 1)
    experts = integer(header.get("experts"), "experts", 1)
    stride = integer(header.get("slot_bytes"), "slot_bytes", 1)
    parts = header.get("component_bytes")
    if parts is not None:
        if not isinstance(parts, list) or len(parts) != 9:
            raise TraceError("component_bytes needs nine stored widths")
        if sum(integer(n, "component bytes", 1) for n in parts) != stride:
            raise TraceError("component bytes must sum to slot_bytes")
    if tokens is not None:
        integer(tokens, "generated or prompt tokens", 1)
    integer(read_latency_ns, "read latency ns")
    if resident_tps is not None and (isinstance(resident_tps, bool) or not math.isfinite(resident_tps) or resident_tps <= 0):
        raise TraceError("invalid resident tokens/second")
    if not header.get("workload") or not header.get("checkpoint"):
        raise TraceError("workload and checkpoint identity are required")
    modes = header.get("modes", {})
    if not isinstance(modes, dict) or not modes:
        raise TraceError("missing complete layer sets per execution mode")
    for mode, indices in modes.items():
        if (not isinstance(mode, str) or not isinstance(indices, list) or not indices
                or any(not isinstance(x, int) or isinstance(x, bool) for x in indices)
                or len(set(indices)) != len(indices)):
            raise TraceError("invalid mode layers")
        for layer in indices:
            if integer(layer, "layer") >= layers:
                raise TraceError("layer outside model")
    if records[-1].get("event") != "closed":
        raise TraceError("incomplete trace: no clean close")
    windows, current, seen = [], None, set()
    for event in records[1:-1]:
        kind = event.get("event")
        if kind == "begin":
            if current is not None or event.get("mode") not in modes:
                raise TraceError("nested window or unknown execution mode")
            wid = integer(event.get("window"), "window")
            if wid in seen:
                raise TraceError("duplicate window")
            seen.add(wid)
            current = {"id": wid, "mode": event["mode"], "routes": []}
            for key in ("rows", "streams"):
                if key in event:
                    current[key] = integer(event[key], key, 1)
        elif kind == "route":
            if current is None or event.get("window") != current["id"]:
                raise TraceError("route outside its window")
            layer = integer(event.get("layer"), "layer")
            ids = event.get("ids")
            if layer not in modes[current["mode"]] or not isinstance(ids, list) or not ids:
                raise TraceError("missing actual routes or invalid layer")
            if any(x[0] == layer for x in current["routes"]):
                raise TraceError("duplicate layer observation")
            union = []
            for eid in ids:
                if integer(eid, "expert") >= experts:
                    raise TraceError("expert outside model")
                if eid not in union:
                    union.append(eid)
            current["routes"].append((layer, union))
        elif kind == "end":
            if current is None or event.get("window") != current["id"]:
                raise TraceError("unmatched window end")
            actual = [x[0] for x in current["routes"]]
            if actual != modes[current["mode"]]:
                raise TraceError("incomplete or reordered window layers")
            windows.append(current)
            current = None
        else:
            raise TraceError("unknown event; forecasts must not masquerade as actual routes")
    if current is not None or not windows or records[-1].get("windows") != len(windows):
        raise TraceError("incomplete window count")
    active, growth = set(), []
    max_union = 0
    for window in windows:
        for layer, ids in window["routes"]:
            active.update((layer, eid) for eid in ids)
            max_union = max(max_union, len(ids))
        growth.append({"window": window["id"], "mode": window["mode"],
                       "experts": len(active), "bytes": len(active) * stride,
                       **{key: window[key] for key in ("rows", "streams") if key in window}})
    frontier = []
    for budget in budgets:
        budget = integer(budget, "budget bytes", 1)
        capacity = min(budget // stride, layers * experts)
        row = {"budget_bytes": budget, "slots": capacity, "usable_bytes": capacity * stride,
               "observed_union_fits": len(active) <= capacity,
               "supported_window_fits": max_union <= capacity,
               "unused_tail_bytes": budget - capacity * stride}
        if capacity < max_union:
            row["verdict"] = "active window exceeds pool; no exact full-window simulation"
            frontier.append(row)
            continue
        slots, mapping, hand, misses, demands, calls, groups = [], {}, 0, 0, 0, 0, 0
        cold = []
        for window in windows:
            window_misses = 0
            for layer, ids in window["routes"]:
                demand = [(layer, eid) for eid in ids]
                demands += len(demand)
                protected = set(demand)
                jobs = []
                for key in sorted(demand):
                    if key in mapping:
                        slots[mapping[key]][1] = True
                for key in sorted(demand):
                    if key in mapping:
                        continue
                    if len(slots) < capacity:
                        at = len(slots)
                        slots.append([None, False])
                    else:
                        while True:
                            at = hand
                            hand = (hand + 1) % capacity
                            old, ref = slots[at]
                            if old in protected:
                                continue
                            if ref:
                                slots[at][1] = False
                                continue
                            del mapping[old]
                            break
                    slots[at] = [key, True]
                    mapping[key] = at
                    jobs.append((key, at))
                    window_misses += 1
                if parts is not None:
                    route_calls = read_requests(jobs, parts)
                    calls += route_calls
                    groups += math.ceil(route_calls / 9)
            misses += window_misses
            cold.append({"window": window["id"], "misses": window_misses,
                         "cold_bytes": window_misses * stride})
        row.update(verdict="observed union fits" if row["observed_union_fits"] else "capacity pressure",
                   policy="CLOCK, demand unions protected, no forecast", misses=misses,
                   cold_bytes=misses * stride, per_window=cold, demands=demands,
                   hit_rate=1 - misses / demands, component_reads_without_coalescing=9 * misses)
        if parts is not None:
            row.update(coalesced_read_requests=calls, bounded_read_groups=groups,
                       mean_read_bytes=misses * stride / calls if calls else 0,
                       coalescing_assumption="stored tensor components and slot strides match the declared uniform recipe")
        if tokens is not None:
            row.update(tokens=tokens, cold_bytes_per_token=misses * stride / tokens)
        if ssd_bytes_per_second is not None:
            rate = integer(ssd_bytes_per_second, "measured SSD bytes/second", 1)
            row["storage_seconds_lower_bound"] = misses * stride / rate
            if tokens is not None:
                row["storage_tokens_per_second_ceiling"] = tokens / row["storage_seconds_lower_bound"] if misses else None
                if resident_tps is not None and parts is not None:
                    compute = tokens / resident_tps
                    baseline_io = row["storage_seconds_lower_bound"] + 9 * misses * read_latency_ns / 1e9
                    batched_io = row["storage_seconds_lower_bound"] + groups * read_latency_ns / 1e9
                    row.update(serialized_baseline_tps=tokens / (compute + baseline_io),
                               serialized_batched_tps=tokens / (compute + batched_io),
                               prediction="synthetic planning estimate; added request latency is conservative and uncalibrated")
        frontier.append(row)
    return {"format": "tensorfold-fsr-profile-v1", "workload": header["workload"],
            "checkpoint": header["checkpoint"], "windows": len(windows),
            "slot_bytes": stride, "observed_experts": len(active),
            "observed_union_bytes": len(active) * stride, "growth": growth,
            "frontier": frontier,
            "qualification": "trace analysis only; no future-locality or native-speed claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--budget-bytes", type=int, action="append", required=True)
    parser.add_argument("--ssd-bytes-per-second", type=int)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--resident-tps", type=float)
    parser.add_argument("--read-latency-ns", type=int, default=0)
    args = parser.parse_args()
    try:
        result = analyze(args.trace.read_text().splitlines(), args.budget_bytes,
                         args.ssd_bytes_per_second, tokens=args.tokens, resident_tps=args.resident_tps,
                         read_latency_ns=args.read_latency_ns)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"FSR profile refused: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
