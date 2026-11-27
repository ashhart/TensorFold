#!/usr/bin/env python3
"""Replay exact v2 DFlash grow records and compare v1 unary scorer variants."""

import argparse
import hashlib
import json
import math
import mmap
import struct
from collections import defaultdict
from pathlib import Path

BRANCH = 4
EDGE_WEIGHT = 0.6
CURRENT_SCALE = 1.5
NOISE_WEIGHT = 0.7
MASK64 = (1 << 64) - 1


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]

def round_f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f64(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def bf32(bits):
    return f32(bits << 16)


def mix(value):
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & MASK64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def uniform24(seed, position, token):
    base = mix(mix((seed + 0x9E3779B97F4A7C15) & MASK64) ^
               ((position * 0xD1B54A32D192ED03) & MASK64))
    return round_f32((mix(base ^ token) >> 40) + 0.5) / 16777216.0


def drawn(row, depth, token, logit):
    sampling = row.get("sampling")
    if sampling is None:
        return logit
    temperature = max(f64(sampling["temperature"]), 1e-6)
    uniform = min(uniform24(sampling["seed"], row["position"] + depth, token),
                  f32(0x3F7FFFFF))
    return logit / temperature + NOISE_WEIGHT * -math.log(-math.log(uniform))


def log_softmax(values):
    most = max(values)
    total = sum(math.exp(value - most) for value in values)
    log_total = most + math.log(total)
    return [value - log_total for value in values]


class Codebooks:
    """Raw vocabulary-by-rank BF16 predecessor and successor matrices."""

    def __init__(self, predecessor, successor, rank):
        self.predecessor = predecessor
        self.successor = successor
        self.rank = rank

    @classmethod
    def open(cls, predecessor_path, successor_path, rank):
        predecessor = open(predecessor_path, "rb")
        successor = open(successor_path, "rb")
        try:
            pred = mmap.mmap(predecessor.fileno(), 0, access=mmap.ACCESS_READ)
            succ = mmap.mmap(successor.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            predecessor.close()
            successor.close()
            raise
        files = (predecessor, successor)
        maps = (pred, succ)
        instance = cls(pred, succ, rank)
        instance.files = files
        instance.maps = maps
        return instance

    def close(self):
        for mapped in self.maps:
            mapped.close()
        for file in self.files:
            file.close()

    def value(self, matrix, token, component):
        offset = 2 * (token * self.rank + component)
        return bf32(struct.unpack_from("<H", matrix, offset)[0])

    def edge(self, row, depth, predecessor, token):
        hp = row["candidates"][depth]["hp"]
        total = 0.0
        for component in range(self.rank):
            offset = 4 * component
            projected = bf32(int(hp[offset:offset + 4], 16))
            total += (self.value(self.predecessor, predecessor, component) * projected *
                      self.value(self.successor, token, component))
        return total


def chances(row, depth, predecessor, scale, books=None, current=False):
    candidate = row["candidates"][depth]
    scores = []
    for token, unary_bits in zip(candidate["ids"], candidate["unary"]):
        logit = f32(unary_bits)
        if current:
            if books is None:
                raise ValueError("exact current scoring needs both BF16 codebooks")
            logit += EDGE_WEIGHT * books.edge(row, depth, predecessor, token)
        scores.append(drawn(row, depth, token, logit) / scale)
    return log_softmax(scores)


def priority(path_cost, depth, candidate):
    if candidate["depth_power"]:
        return path_cost / (depth ** candidate["depth_power"])
    return path_cost - candidate["depth_bonus"] * depth


def grow(row, candidate, books=None, current=False):
    """Best-first growth using only pending, parents already taken, and proposal inputs."""
    offered = row["offered"]
    tokens, parents, own = [], [], []
    open_nodes = []
    predecessor, parent, depth, path_cost = row["pending"], -1, 0, 0.0
    while True:
        if depth < len(row["candidates"]):
            local = chances(row, depth, predecessor, candidate["scale"], books, current)
            order = sorted(range(len(local)), key=lambda index: (-local[index], index))
            ids = row["candidates"][depth]["ids"]
            for index in order[:BRANCH]:
                if not math.isfinite(local[index]):
                    break
                cost = path_cost - local[index]
                open_nodes.append((priority(cost, depth + 1, candidate), parent,
                                   ids[index], depth, local[index], cost))
        if len(tokens) == offered or not open_nodes:
            break
        best = 0
        for index in range(1, len(open_nodes)):
            node, incumbent = open_nodes[index], open_nodes[best]
            if node[:4] < incumbent[:4]:
                best = index
        _, parent, token, depth, local, path_cost = open_nodes.pop(best)
        tokens.append(token)
        parents.append(parent)
        own.append(local)
        predecessor = token
        parent = len(tokens) - 1
        depth += 1
    return tokens, parents, own


def replay_recorded_order(row):
    """Verify best-first order restricted to emitted nodes and their recorded own scores."""
    count = row["offered"]
    children = defaultdict(list)
    depths = []
    for index, parent in enumerate(row["parents"]):
        depth = 0 if parent < 0 else depths[parent] + 1
        depths.append(depth)
        children[parent].append(index)
    open_nodes = []
    parent, cost = -1, 0.0
    order = []
    while len(order) < count:
        for index in children[parent]:
            local = f64(row["own"][index])
            node_cost = cost - local
            open_nodes.append((node_cost, parent, row["nodes"][index], depths[index], index))
        if not open_nodes:
            break
        best = 0
        for index in range(1, len(open_nodes)):
            if open_nodes[index][:4] < open_nodes[best][:4]:
                best = index
        _, _, _, _, selected = open_nodes.pop(best)
        parent = selected
        cost = 0.0
        ancestor = selected
        while ancestor >= 0:
            cost -= f64(row["own"][ancestor])
            ancestor = row["parents"][ancestor]
        order.append(selected)
    return order == list(range(count))

def f64bits(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]



def require_json_integer(value, field):
    if type(value) is not int:
        raise ValueError(field + " must be a JSON integer, not " + type(value).__name__)

def frontier_replay(row):
    """Re-run grow from all v2 open-node insertions and require bit-exact selected state."""
    if type(row.get("v")) is not int or row["v"] != 2:
        return False, "exact frontier replay requires trace version 2"
    parameters = row["parameters"]
    events = row["frontier"]
    try:
        for field in ("offered", "frontier_count", "frontier_expected"):
            require_json_integer(row[field], field)
        for field in ("scale", "edge", "noise"):
            require_json_integer(parameters[field], "parameters." + field)
        for field in ("nodes", "parents", "own"):
            for index, value in enumerate(row[field]):
                require_json_integer(value, field + "." + str(index))
        for index, item in enumerate(events):
            for field in ("sequence", "parent", "token", "row", "own", "cost"):
                require_json_integer(item[field], "frontier." + str(index) + "." + field)
    except ValueError as error:
        return False, str(error)
    count = len(row["nodes"])
    if len(row["parents"]) != count or len(row["own"]) != count:
        return False, "selected arrays have different lengths"
    if row["offered"] < 0 or row["offered"] > count:
        return False, "offered count exceeds the selected-node count or is negative"
    if row["frontier_expected"] != row["frontier_count"] or row["frontier_count"] != len(events) or not row.get("frontier_complete", True):
        return False, "frontier expected/event counts disagree with the record"
    expected = {"scale": f64bits(CURRENT_SCALE), "edge": f64bits(EDGE_WEIGHT),
                "noise": f64bits(NOISE_WEIGHT)}
    if any(parameters[key] != value for key, value in expected.items()):
        return False, "scorer parameters differ from this replay implementation"
    for sequence, item in enumerate(events):
        if item["sequence"] != sequence:
            return False, "frontier events are missing or out of insertion order"
    event = 0
    open_nodes = []
    selected = 0
    parent = -1
    depth = 0
    path = 0.0
    while True:
        if depth < len(row["candidates"]):
            base = path
            while event < len(events) and events[event]["parent"] == parent and events[event]["row"] == depth:
                item = events[event]
                local = f64(item["own"])
                cost = f64(item["cost"])
                if not math.isfinite(local) or not math.isfinite(cost):
                    return False, "non-finite grow score or priority"
                if f64bits(base - local) != item["cost"]:
                    return False, "frontier priority is not parent path minus local score"
                open_nodes.append((cost, item["parent"], item["token"], item["row"], item))
                event += 1
        if selected == count or not open_nodes:
            break
        best = 0
        for index in range(1, len(open_nodes)):
            if open_nodes[index][:4] < open_nodes[best][:4]:
                best = index
        node = open_nodes[best]
        open_nodes[best] = open_nodes[-1]
        open_nodes.pop()
        if (node[2] != row["nodes"][selected] or node[1] != row["parents"][selected] or
                node[4]["own"] != row["own"][selected]):
            return False, "selected node differs at index " + str(selected)
        parent = selected
        depth = node[3] + 1
        path = node[0]
        selected += 1
    if selected != count:
        return False, "frontier produced fewer nodes than recorded"
    if event != len(events):
        return False, "frontier has unconsumed or out-of-order events"
    return True, None


def check_exact(path):
    exact = total = 0
    failures = []
    with open(path, "rb") as source:
        for line_number, line in enumerate(source, 1):
            total += 1
            try:
                ok, reason = frontier_replay(json.loads(line))
            except (KeyError, IndexError, ValueError, TypeError, AttributeError, OverflowError, struct.error) as error:
                ok, reason = False, str(error)
            exact += ok
            if not ok:
                failures.append({"line": line_number, "reason": reason})
    return {"exact": exact, "total": total,
            "fraction": exact / total if total else None, "failures": failures}


def accepted_depth(tokens, parents, accepted):
    """Longest offered connected path matching the known target tokens."""
    best = 0
    paths = []
    for token, parent in zip(tokens, parents):
        path = ([] if parent < 0 else paths[parent]) + [token]
        paths.append(path)
        if len(path) <= len(accepted) and path == accepted[:len(path)]:
            best = max(best, len(path))
    return best


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(depths, available=None):
    result = {"rounds": len(depths)}
    if not depths:
        result.update(median=None, mean=None, p10=None, p90=None)
        return result
    result.update(median=percentile(depths, 0.5), mean=sum(depths) / len(depths),
                  p10=percentile(depths, 0.1), p90=percentile(depths, 0.9))
    if available is not None:
        result["censored"] = sum(depth == length for depth, length in zip(depths, available))
    return result


def group_names(row):
    names = ["all", "sampled" if "sampling" in row else "greedy",
             "long" if row["position"] >= 8192 else "short"]
    return names


def new_accumulator(candidates):
    values = defaultdict(lambda: defaultdict(list))
    available = defaultdict(lambda: defaultdict(list))
    for name in ["recorded"] + [candidate["name"] for candidate in candidates]:
        for scope in ("all", "novel"):
            for group in ("all", "greedy", "sampled", "short", "long"):
                values[name][scope + ":" + group] = []
                available[name][scope + ":" + group] = []
    return values, available


def add_depth(values, available, name, scopes, row, depth):
    for scope in scopes:
        for group in group_names(row):
            key = scope + ":" + group
            values[name][key].append(depth)
            available[name][key].append(len(row["accepted_tokens"]))


def digest(line):
    row = json.loads(line)
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def training_digests(path):
    if path is None:
        return set()
    with open(path, "rb") as source:
        return {digest(line) for line in source}


def evaluate(path, candidates, seen=None, mismatch_lines=None):
    values, available = new_accumulator(candidates)
    replay = {"structural_exact": 0, "structural_total": 0,
              "unary_exact": 0, "unary_total": 0, "offered_zero": 0}
    proxy = {"name": "unary-proxy", "scale": CURRENT_SCALE,
             "depth_bonus": 0.0, "depth_power": 0.0}
    with open(path, "rb") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            scopes = ["all"] if seen is None else (["all", "novel"] if digest(line) not in seen else ["all"])
            baseline = max(0, len(row["accepted_tokens"]) - 1)
            add_depth(values, available, "recorded", scopes, row, baseline)
            for candidate in candidates:
                tokens, parents, _ = grow(row, candidate)
                add_depth(values, available, candidate["name"], scopes, row,
                          accepted_depth(tokens, parents, row["accepted_tokens"]))
            if row["offered"] == 0:
                replay["offered_zero"] += 1
                continue
            replay["structural_total"] += 1
            replay["structural_exact"] += replay_recorded_order(row)
            tokens, parents, _ = grow(row, proxy)
            exact = tokens == row["nodes"][:row["offered"]] and parents == row["parents"][:row["offered"]]
            replay["unary_total"] += 1
            replay["unary_exact"] += exact
            if not exact and mismatch_lines is not None:
                mismatch_lines.append(json.dumps({"line": line_number,
                    "reason": "unary proxy omits predecessor/projection/successor edge",
                    "expected": {"nodes": row["nodes"][:row["offered"]],
                                 "parents": row["parents"][:row["offered"]]},
                    "actual": {"nodes": tokens, "parents": parents}}, separators=(",", ":")))
    summaries = {name: {key: summarize(depths[key], available[name][key])
                        for key in depths} for name, depths in values.items()}
    return {"rounds": replay["structural_total"] + replay["offered_zero"],
            "replay": replay, "metrics": summaries,
            "mismatch_count": len(mismatch_lines or [])}


def exact_replay(path, books):
    candidate = {"name": "current", "scale": CURRENT_SCALE,
                 "depth_bonus": 0.0, "depth_power": 0.0}
    exact = total = 0
    with open(path, "rb") as source:
        for line in source:
            row = json.loads(line)
            if not row["offered"]:
                continue
            tokens, parents, _ = grow(row, candidate, books, current=True)
            total += 1
            exact += (tokens == row["nodes"][:row["offered"]] and
                      parents == row["parents"][:row["offered"]])
    return {"exact": exact, "rounds": total,
            "fraction": exact / total if total else None}

def candidate_grid():
    candidates = []
    for scale in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5):
        for bonus in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5):
            candidates.append({"name": f"unary-s{scale:g}-b{bonus:g}", "scale": scale,
                               "depth_bonus": bonus, "depth_power": 0.0})
    for scale in (0.8, 1.0, 1.25, 1.5, 2.0):
        for power in (0.25, 0.5, 0.75, 1.0):
            candidates.append({"name": f"unary-s{scale:g}-p{power:g}", "scale": scale,
                               "depth_bonus": 0.0, "depth_power": power})
    return candidates


def eligible_ranked(tuning):
    baseline = tuning["metrics"]["recorded"]
    if not baseline["all:all"]["rounds"]:
        raise ValueError("tuning trace has no rounds")
    has_long = bool(baseline["all:long"]["rounds"])
    ranked = []
    for candidate in tuning["candidates"]:
        metrics = tuning["metrics"][candidate["name"]]
        safe = not has_long or (metrics["all:long"]["median"] >= baseline["all:long"]["median"] and
                                metrics["all:long"]["mean"] >= baseline["all:long"]["mean"])
        score = (safe, metrics["all:all"]["median"], metrics["all:all"]["mean"],
                 metrics["all:long"]["median"] if has_long else -1.0,
                 metrics["all:long"]["mean"] if has_long else -1.0)
        ranked.append((score, candidate))
    return [candidate for _, candidate in sorted(ranked, key=lambda item: item[0], reverse=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", type=Path)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--mismatch-log", type=Path)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--exact", type=Path)
    parser.add_argument("--successor", type=Path)
    args = parser.parse_args()
    if args.exact:
        result = check_exact(args.exact)
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.json:
            args.json.write_text(text)
        else:
            print(text, end="")
        return 0 if result["exact"] == result["total"] and result["total"] else 1
    if bool(args.predecessor) != bool(args.successor):
        parser.error("exact replay needs both codebook paths")
    if args.predecessor:
        replay_path = args.evaluate or args.tune
        if replay_path is None:
            parser.error("exact replay needs a trace")
        with open(replay_path, "rb") as source:
            first = json.loads(source.readline())
        if first.get("v") != 1:
            parser.error("v2 records use --exact; codebook replay requires v1 HP projections")
    result = {"candidate_count": 0,
              "independent_current_replay": "not run: trace format omits predecessor and successor codebooks"}
    if args.tune:
        tuning = evaluate(args.tune, candidate_grid())
        tuning["candidates"] = candidate_grid()
        selected = eligible_ranked(tuning)[:5]
        result["candidate_count"] = len(selected)
        result["tuning"] = tuning
        result["selected"] = selected
    else:
        selected = candidate_grid()[:1]
    if args.evaluate:
        mismatches = []
        held_out = evaluate(args.evaluate, selected, training_digests(args.tune), mismatches)
        result["held_out"] = held_out
        if args.mismatch_log:
            args.mismatch_log.write_text("\n".join(mismatches) + ("\n" if mismatches else ""))
    if args.predecessor:
        rank = first["rank"]
        books = Codebooks.open(args.predecessor, args.successor, rank)
        try:
            result["independent_current_replay"] = exact_replay(replay_path, books)
        finally:
            books.close()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    raise SystemExit(main())
