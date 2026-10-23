#!/usr/bin/env python3
"""Summarize opt-in native GLM decode receipts without contacting the devices.

Use separate rank log files, TF_GLM_PROF_EVERY=1, and an unprofiled control.
Coarse CUDA-event intervals, nested detail intervals and host-round durations
are intentionally reported separately. None measures actual DRAM traffic.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
COARSE = re.compile(
    r"glm prof (?P<label>.+?) \((?P<calls>\d+) calls\): rank (?P<rank>\d+), "
    r"seq (?P<seq>\d+), rows (?P<rows>\d+), context (?P<lo>\d+)\.\.(?P<hi>\d+); "
    rf"wall (?P<wall>{NUMBER}) ms, gpu (?P<gpu>{NUMBER}) ms a call;(?P<stages>.*)$"
)
DETAIL = re.compile(r"glm detail (.+?) \((\d+) calls, nested ms/call\):(.*)$")
STAGES = re.compile(rf"([a-z][a-z0-9_]*)\s+({NUMBER})")
BUCKETS = ("8K", "32K", "128K")


def context_bucket(lo, hi):
    """Never attribute an interval crossing a requested bucket to its endpoint."""
    if lo < 0 or hi < lo:
        raise ValueError("invalid context range")
    if hi < 8192:
        return None, "startup"
    if lo < 8192:
        return None, "mixed_context"
    band = lambda n: "8K" if n < 32768 else "32K" if n < 131072 else "128K"
    return (band(lo), None) if band(lo) == band(hi) else (None, "mixed_context")


def decode_label(label):
    return label == "shared verify" or re.fullmatch(r"verify \d+ rows?", label) is not None


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite measurement")
    return value


def stage_values(text):
    values = {name: finite(value) for name, value in STAGES.findall(text)}
    if not values or any(v < 0 for v in values.values()):
        raise ValueError("missing or negative stage interval")
    return values


class Timings:
    """Weighted means; min/max describe reported means, not individual calls."""
    def __init__(self):
        self.calls = self.records = 0
        self.lo, self.hi = None, None
        self.values = defaultdict(lambda: [0.0, 0, math.inf, -math.inf])

    def add(self, calls, lo, hi, values):
        if calls <= 0:
            raise ValueError("calls must be positive")
        self.calls += calls
        self.records += 1
        self.lo = lo if self.lo is None else min(self.lo, lo)
        self.hi = hi if self.hi is None else max(self.hi, hi)
        for name, value in values.items():
            item = self.values[name]
            item[0] += value * calls
            item[1] += calls
            item[2], item[3] = min(item[2], value), max(item[3], value)

    def result(self):
        return {"calls": self.calls, "reports": self.records,
                "context_min": self.lo, "context_max": self.hi,
                "ms": {k: {"mean": v[0] / v[1], "min_report_mean": v[2],
                            "max_report_mean": v[3], "measured_calls": v[1]}
                       for k, v in sorted(self.values.items())}}


class Routes:
    def __init__(self):
        self.records = self.unique = self.pairs = 0
        self.hist = Counter()
        self.unique_min, self.unique_max = math.inf, 0
        self.lo, self.hi = None, None
        self.streams = Counter()

    def add(self, r):
        self.records += 1
        self.unique += r["unique"]
        self.pairs += r["pairs"]
        self.unique_min = min(self.unique_min, r["unique"])
        self.unique_max = max(self.unique_max, r["unique"])
        self.hist.update({i: n for i, n in enumerate(r["pair_hist"]) if n})
        self.lo = r["context_min"] if self.lo is None else min(self.lo, r["context_min"])
        self.hi = r["context_max"] if self.hi is None else max(self.hi, r["context_max"])
        self.streams[r["streams"]] += 1

    def result(self):
        return {"layer_records": self.records, "context_min": self.lo, "context_max": self.hi,
                "unique_experts_mean": self.unique / self.records,
                "unique_experts_min": self.unique_min, "unique_experts_max": self.unique_max,
                "pairs_mean": self.pairs / self.records, "unique_experts_sum": self.unique,
                "pairs_sum": self.pairs, "pairs_per_unique": self.pairs / self.unique,
                "repeat_pair_fraction": 1 - self.unique / self.pairs,
                "pair_hist": {str(i): n for i, n in sorted(self.hist.items())},
                "streams_layer_records": dict(sorted(self.streams.items()))}


def validate_route(r):
    fields = ("call", "rank", "layer", "rows", "streams", "context_min", "context_max", "unique", "pairs")
    if any(type(r.get(k)) is not int or r[k] < 0 for k in fields):
        raise ValueError("invalid route integer field")
    if not isinstance(r.get("label"), str) or not (1 <= r["rows"] <= 32):
        raise ValueError("invalid route label or rows")
    if not (1 <= r["streams"] <= r["rows"]) or not r["call"] or not r["unique"]:
        raise ValueError("invalid route streams/call/unique")
    hist = r.get("pair_hist")
    if not isinstance(hist, list) or not hist or any(type(v) is not int or v < 0 for v in hist):
        raise ValueError("invalid route histogram")
    if hist[0] or any(hist[r["rows"] + 1:]):
        raise ValueError("histogram selects an expert more than once per row")
    if sum(hist) != r["unique"] or sum(i * v for i, v in enumerate(hist)) != r["pairs"]:
        raise ValueError("histogram totals do not match route counts")
    if r["pairs"] % r["rows"]:
        raise ValueError("nonuniform top-k across rows")
    context_bucket(r["context_min"], r["context_max"])


class Analyzer:
    def __init__(self):
        self.coarse, self.detail = defaultdict(Timings), defaultdict(Timings)
        self.routes, self.layers = defaultdict(Routes), defaultdict(Routes)
        self.rounds = {}
        self.counts, self.excluded = Counter(), Counter()
        self.warnings, self.sources = [], []
        self.route_call_layers = Counter()

    def warn(self, where, message):
        self.counts["warnings"] += 1
        if len(self.warnings) < 100:
            self.warnings.append(f"{where}: {message}")

    def bucket(self, kind, lo, hi):
        bucket, reason = context_bucket(lo, hi)
        if reason:
            self.excluded[f"{kind}_{reason}"] += 1
        return bucket

    def read(self, lines, source="<memory>"):
        self.sources.append(str(source))
        pending_detail, pending_routes = {}, defaultdict(list)
        ranks = set()
        for line_no, line in enumerate(lines, 1):
            where = f"{source}:{line_no}"
            try:
                if "glm route {" in line:
                    r = json.loads(line[line.index("glm route {") + len("glm route "):])
                    validate_route(r)
                    self.counts["route_parsed"] += 1
                    if not decode_label(r["label"]):
                        self.excluded["route_non_decode"] += 1
                        continue
                    b = self.bucket("route", r["context_min"], r["context_max"])
                    if b is None:
                        continue
                    key = (b, r["rank"], r["rows"], r["label"])
                    pending_key = (r["rank"], r["label"], r["call"])
                    if any(x["layer"] == r["layer"] for x in pending_routes[pending_key]):
                        raise ValueError("duplicate layer in pending route call")
                    pending_routes[pending_key].append(r)
                    self.routes[key].add(r)
                    self.layers[key + (r["layer"],)].add(r)
                    continue
                m = DETAIL.search(line)
                if m:
                    label, calls, values = m.group(1), int(m.group(2)), stage_values(m.group(3))
                    self.counts["detail_parsed"] += 1
                    if label in pending_detail:
                        self.excluded["detail_orphan"] += 1
                        self.warn(where, "replaced unpaired detail; use separate rank logs")
                    pending_detail[label] = (calls, values)
                    continue
                m = COARSE.search(line)
                if m:
                    r = m.groupdict()
                    for k in ("calls", "rank", "seq", "rows", "lo", "hi"):
                        r[k] = int(r[k])
                    if min(r["calls"], r["rows"], r["seq"]) <= 0:
                        raise ValueError("invalid coarse calls/rows/sequence")
                    self.counts["coarse_parsed"] += 1
                    ranks.add(r["rank"])
                    detail = pending_detail.pop(r["label"], None)
                    if not decode_label(r["label"]):
                        self.excluded["coarse_non_decode"] += 1
                        continue
                    b = self.bucket("coarse", r["lo"], r["hi"])
                    if b is None:
                        continue
                    key = (b, r["rank"], r["rows"], r["label"])
                    # Sections omitted by the profiler are exactly zero in this report.
                    values = {k: 0.0 for k in ("gate", "embed", "kda", "mla", "gather_a", "dense",
                               "router", "routed", "shared", "gather_b", "head", "draw", "mtp", "commit")}
                    values.update(stage_values(r["stages"]))
                    values.update(wall=finite(r["wall"]), gpu=finite(r["gpu"]))
                    self.coarse[key].add(r["calls"], r["lo"], r["hi"], values)
                    if r["calls"] != 1:
                        self.counts["coarse_aggregated_reports"] += 1
                    if detail:
                        if detail[0] == r["calls"]:
                            self.detail[key].add(r["calls"], r["lo"], r["hi"], detail[1])
                            self.counts["detail_matched"] += 1
                        else:
                            self.excluded["detail_call_mismatch"] += 1
                    records = pending_routes.pop((r["rank"], r["label"], r["seq"]), [])
                    matching = [x for x in records if (x["rows"], x["context_min"], x["context_max"]) ==
                                (r["rows"], r["lo"], r["hi"])]
                    self.counts["route_records_matched_coarse"] += len(matching)
                    self.counts["route_records_metadata_mismatch"] += len(records) - len(matching)
                    if matching:
                        self.route_call_layers[len(matching)] += 1
                    continue
                if "glm round host: " in line:
                    self.read_round(line.split("glm round host: ", 1)[1])
                    continue
                if "glm round host end:" in line:
                    for name in ("dropped", "aborted"):
                        m = re.search(rf"{name}=(\d+)", line)
                        if m:
                            self.counts[f"round_{name}"] += int(m.group(1))
                elif "glm route discarded" in line:
                    self.counts["route_discarded"] += 1
                elif "glm prof " in line:
                    self.excluded["coarse_legacy_or_unparsed"] += 1
                    self.warn(where, "coarse receipt has no usable rank/context or an unknown format")
            except (ValueError, TypeError, KeyError) as exc:
                self.counts["malformed"] += 1
                self.warn(where, str(exc))
        self.excluded["detail_orphan"] += len(pending_detail)
        self.counts["route_records_without_matching_coarse"] += sum(map(len, pending_routes.values()))
        if len(ranks) > 1:
            self.warn(source, "multiple ranks in one file: detail association assumes no interleaving of same-label receipts")

    def read_round(self, text):
        r = dict(re.findall(r"([a-z_]+)=([^\s;]+)", text))
        lo, hi = map(int, r["context"].split(".."))
        values = {k: finite(v) for k, v in r.items() if k.endswith("_ms")}
        for k in ("streams", "rows", "calls", "kept", "next_drafts"):
            r[k] = int(r[k])
        if r["initial"] not in ("true", "false") or not (0 < r["streams"] <= r["rows"]):
            raise ValueError("invalid round initial/streams/rows")
        if r["kept"] < 0 or r["next_drafts"] < 0 or min(values["wall_ms"], r["calls"]) <= 0:
            raise ValueError("invalid round totals/duration")
        self.counts["round_parsed"] += 1
        b = self.bucket("round", lo, hi)
        if b is None:
            return
        key = (b, 0, r["rows"], r["streams"], r["initial"] == "true")
        if key not in self.rounds:
            self.rounds[key] = {"timings": Timings(), "kept": 0, "next_drafts": 0,
                                "min_ms": math.inf, "max_ms": 0, "wall_sum_ms": 0}
        out = self.rounds[key]
        out["kept"] += r["kept"]
        out["next_drafts"] += r["next_drafts"]
        out["min_ms"] = min(out["min_ms"], values.pop("min_ms"))
        out["max_ms"] = max(out["max_ms"], values.pop("max_ms"))
        out["wall_sum_ms"] += values["wall_ms"] * r["calls"]
        out["timings"].add(r["calls"], lo, hi, values)

    def result(self):
        order = lambda k: (BUCKETS.index(k[0]), *k[1:])
        groups = []
        for key in sorted(self.coarse.keys() | self.routes.keys(), key=order):
            b, rank, rows, label = key
            row = {"bucket": b, "rank": rank, "rows": rows, "label": label}
            if key in self.coarse:
                row["coarse"] = self.coarse[key].result()
            if key in self.detail:
                row["detail_nested"] = self.detail[key].result()
            if key in self.routes:
                row["routes"] = self.routes[key].result()
                row["route_layers"] = [{"layer": k[4], **self.layers[k].result()}
                                       for k in sorted(self.layers, key=order) if k[:4] == key]
            groups.append(row)
        rounds = []
        for key in sorted(self.rounds, key=order):
            b, rank, rows, streams, initial = key
            v = self.rounds[key]
            result = v["timings"].result()
            rounds.append({"bucket": b, "rank": rank, "rows": rows, "streams": streams,
                           "initial": initial, **result, "kept": v["kept"],
                           "next_drafts": v["next_drafts"], "wall_min_ms": v["min_ms"],
                           "wall_max_ms": v["max_ms"], "kept_per_round": v["kept"] / result["calls"],
                           "kept_per_host_second": 1000 * v["kept"] / v["wall_sum_ms"],
                           "verified_draft_rows_per_round": rows - streams})
        return {"schema": "glm-decode-v7/1", "sources": self.sources,
                "context_buckets": {"8K": [8192, 32767], "32K": [32768, 131071], "128K": [131072, None]},
                "counts": dict(self.counts), "excluded": dict(self.excluded),
                "route_call_layer_count_hist": dict(sorted(self.route_call_layers.items())),
                "warnings": self.warnings, "groups": groups, "host_rounds": rounds,
                "limitations": [
                    "Startup contexts below 8192 and ranges crossing requested bucket boundaries are excluded.",
                    "TF_GLM_PROF_EVERY=1 is required for exact per-call context; larger reports carry only the last context.",
                    "Timing min/max fields are reported-mean ranges, not confidence intervals or latency percentiles.",
                    "Detail spans and host draw/commit spans are nested; do not sum them with their parent intervals.",
                    "Coarse wall is not the complete server round and is sampled before the profiler completion wait.",
                    "Route pairs/unique describes logical expert reuse potential, not measured weight traffic or bandwidth.",
                    "Host kept includes pending rows and may exceed emitted tokens after EOS; it is not draft acceptance.",
                    "Host next_drafts is the next round's requested depth, not this round's verified draft count.",
                    "Profiling copies/events/logging perturb timing; compare an unprofiled control.",
                    "Separate rank files are required for reliable detail pairing; no cross-host clock alignment is inferred.",
                ]}


def compact(result):
    lines = ["Decode profile (diagnostic instrumentation; ms per target forward)"]
    for g in result["groups"]:
        lead = f"{g['bucket']:>4} rank={g['rank']} rows={g['rows']} {g['label']}"
        if "coarse" in g:
            c, ms = g["coarse"], g["coarse"]["ms"]
            top = sorted(((k, v["mean"]) for k, v in ms.items() if k not in ("wall", "gpu")),
                         key=lambda p: -p[1])[:5]
            lead += f" calls={c['calls']} GPU={ms['gpu']['mean']:.3f} wall={ms['wall']['mean']:.3f}"
            lead += " | " + " ".join(f"{k}={v:.3f}" for k, v in top)
        if "routes" in g:
            r = g["routes"]
            lead += (f" | unique/layer={r['unique_experts_mean']:.2f} pairs/layer={r['pairs_mean']:.2f}"
                     f" pairs/unique={r['pairs_per_unique']:.3f} repeat={100*r['repeat_pair_fraction']:.1f}%")
        lines.append(lead)
    for r in result["host_rounds"]:
        ms = r["ms"]
        lines.append(f"{r['bucket']:>4} host rows={r['rows']} streams={r['streams']} initial={r['initial']}"
                     f" calls={r['calls']} round={ms['wall_ms']['mean']:.3f}ms"
                     f" kept/round={r['kept_per_round']:.3f} kept/s={r['kept_per_host_second']:.2f}")
    lines.append("Excluded: " + json.dumps({k: v for k, v in result["excluded"].items() if v}, sort_keys=True))
    lines.append("Diagnostics: " + json.dumps(result["counts"], sort_keys=True))
    if result["warnings"]:
        lines.extend("WARNING " + w for w in result["warnings"][:8])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="separate per-rank server logs")
    parser.add_argument("--output", type=Path, help="JSON destination; defaults to stdout")
    args = parser.parse_args(argv)
    if len({p.resolve() for p in args.logs}) != len(args.logs):
        parser.error("the same log was supplied more than once")
    analyzer = Analyzer()
    for path in args.logs:
        with path.open(encoding="utf-8", errors="replace") as handle:
            analyzer.read(handle, str(path))
    result = analyzer.result()
    data = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(data, encoding="utf-8")
    else:
        sys.stdout.write(data)
    print(compact(result), file=sys.stderr)
    return 1 if analyzer.counts["malformed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
