"""The DSpark matrix verdict: request checks, cross-arm hash equality, startup configuration and speed versus plain."""
import json
import statistics

from bench_evidence import count, token_equal

SCHEMA = "bench_lfm2_dspark/1"
ARMS = ("plain", "bf16", "bits8")
DEFAULT_WIDTHS = (1, 4, 8, 16)
DEFAULT_TEMPERATURES = (0.0, 1.0)
DEFAULT_TOKENS = 512
EXIT_PASS, EXIT_FAIL, EXIT_USAGE, EXIT_PERF = 0, 1, 2, 3


ACTIVITY_KEYS = ("rounds", "drafted", "accepted")


def activity(run: dict) -> dict:
    """The server's speculative counters for one request, from every proposal source (zeros when it reported none)."""
    stats = run.get("speculative")
    stats = stats if isinstance(stats, dict) else {}
    return {k: count(stats.get(k)) or 0 for k in ACTIVITY_KEYS}


def total_activity(runs: list[dict]) -> dict:
    return {k: sum(activity(r)[k] for r in runs) for k in ACTIVITY_KEYS}


def seeds_for(seed: int, widths: list[int], temp: float) -> list[int]:
    """Stream i's seed in the widest cell: seed + i when sampled, the one seed when greedy (bench_concurrent)."""
    return [seed + i if temp > 0 else seed for i in range(max(widths))]


def problems(run: dict, tokens: int, timed: bool) -> list[dict]:
    """Why a request is not an observed complete ``tokens``-token result; empty when it is."""
    out = []
    seed = run.get("seed")
    if run.get("error"):
        out.append({"class": "scenario", "seed": seed, "what": f"error:{run['error']}"})
    elif run.get("complete") is not True:
        out.append({"class": "scenario", "seed": seed, "what": "incomplete_stream"})
    if run.get("tokens") != tokens:
        out.append({"class": "scenario", "seed": seed, "what": f"tokens:{run.get('tokens')}!={tokens}"})
    if not run.get("token_sha256"):
        out.append({"class": "scenario", "seed": seed, "what": "missing_token_sha256"})
    if timed and run.get("decode_tps") is None:
        out.append({"class": "scenario", "seed": seed, "what": "unmeasured"})
    return out


def compare(left: dict, right: dict, what: str, seed) -> dict | None:
    """None when the two runs' token hashes are verified equal; otherwise the failure."""
    equal = token_equal(left, right)
    if equal is True:
        return None
    return {"class": "exactness" if equal is False else "scenario", "seed": seed,
            "what": what if equal is False else f"unverified:{what}"}


def audited(c: dict, s: dict) -> dict:
    """The cell with its failures re-derived from its persisted requests (a record's own ok is not trusted)."""
    want = seeds_for(s["seed"], s["widths"], float(c["temperature"]))[:c["streams"]]
    bad = [{"class": "scenario", "what": "request_seeds_differ"}] if [r.get("seed") for r in c["requests"]] != want else []
    bad += [f for r in c["requests"] for f in problems(r, s["tokens"], True)]
    return {**c, "failures": c["failures"] + [b for b in bad if b not in c["failures"]], "ok": c["ok"] and not bad}


def cross_arm(s: dict, cells: list[dict], oracle: list[dict]) -> list[dict]:
    """Every hash of every arm for a (prompt, temperature, seed) must equal plain's single hash; else a failure."""
    got: dict = {}

    def add(rec, seed, source, digest):
        got.setdefault((rec["prompt"], float(rec["temperature"]), seed), {}).setdefault(rec["arm"], []).append((source, digest))

    for o in oracle:
        add(o, o["seed"], "solo", o["solo_token_sha256"])
        add(o, o["seed"], "draft_false", o["draft_false_token_sha256"])
    for c in cells:
        for r in c["requests"]:
            add(c, r.get("seed"), "concurrent", r.get("token_sha256"))
    out = []
    for p in s["prompts"]:
        for t in s["temperatures"]:
            for seed in dict.fromkeys(seeds_for(s["seed"], s["widths"], t)):
                key = {"prompt": p, "temperature": t, "seed": seed}
                by = got.get((p, float(t), seed), {})
                plain = {h for _, h in by.get("plain", []) if h}
                if len(plain) != 1:
                    out.append({"class": "exactness" if plain else "scenario", **key,
                                "what": "cross_arm_plain_hashes_differ" if plain else "cross_arm_no_plain_hash"})
                    continue
                for arm in ARMS[1:]:
                    hashed = [(src, h) for src, h in by.get(arm, []) if h]
                    if not hashed:
                        out.append({"class": "scenario", **key, "arm": arm, "what": "cross_arm_no_hash"})
                    out += [{"class": "exactness", **key, "arm": arm, "source": src, "what": "cross_arm_mismatch"}
                            for src, h in hashed if h not in plain]
    return out


MODE_LINE = "lfm2: DSpark drafts from --drafter"


def startup_modes(log: str) -> list[str]:
    """The mode each native 'lfm2: DSpark drafts from --drafter' startup line states: bf16, bits8 or unrecognized."""
    return ["bits8" if "8-bit codes quantized at load (--drafter-bits 8)" in line
            else "bf16" if "bf16 matrices as released" in line else "unrecognized"
            for line in log.splitlines() if MODE_LINE in line]


def startup(events: list[dict], reps: int) -> list[dict]:
    """Require every launch's startup mode: none for plain, bf16 or bits8 for the drafter arms."""
    out = []
    for arm in ARMS:
        got = [e.get("dspark_mode") for e in events if e["event"] == "ready" and e.get("arm") == arm]
        if arm == "plain" and any(got):
            out.append({"class": "scenario", "arm": arm, "what": "plain_arm_loaded_dspark"})
        elif got != [None if arm == "plain" else arm] * reps:
            out.append({"class": "scenario", "arm": arm, "what": "dspark_mode_log_missing_or_wrong"})
    return out


def summarize(meta: dict, records: list[dict], events: list[dict]) -> dict:
    """Everything the verdict says, derived from the persisted evidence and the recorded matrix alone."""
    s = meta["settings"]
    reps, widths, temps, prompts = s["reps"], s["widths"], s["temperatures"], s["prompts"]
    cells = [audited(r, s) for r in records if r["kind"] == "cell"]
    oracle = [r for r in records if r["kind"] == "oracle"]
    expected = {(rep, arm, p, t, w) for rep in range(reps) for arm in ARMS for p in prompts for t in temps for w in widths}
    seen = {(c["rep"], c["arm"], c["prompt"], float(c["temperature"]), c["streams"]) for c in cells}
    want_oracle = {(rep, arm, p, t, sd) for rep in range(reps) for arm in ARMS for p in prompts for t in temps
                   for sd in dict.fromkeys(seeds_for(s["seed"], widths, t))}
    have_oracle = {(o["rep"], o["arm"], o["prompt"], float(o["temperature"]), o["seed"]) for o in oracle}
    missing = sorted(expected - seen)
    failures = [{**f, "rep": r["rep"], "arm": r["arm"], "prompt": r["prompt"], "temperature": r["temperature"],
                 "streams": r.get("streams", "alone")} for r in cells + oracle for f in r["failures"]]
    failures += [{"class": "scenario", "what": "no_positive_aggregate_tps", "rep": c["rep"], "arm": c["arm"],
                  "prompt": c["prompt"], "temperature": c["temperature"], "streams": c["streams"]}
                 for c in cells if not c["ok"] and not c["failures"]]
    for what, lost in (("missing_cells", missing), ("missing_oracles", want_oracle - have_oracle)):
        if lost:
            failures.append({"class": "scenario", "what": what, "count": len(lost)})
    failures += cross_arm(s, cells, oracle)
    if not meta["smoke"] and (reps < len(ARMS) or reps % len(ARMS) or not s["rotation_balanced"]):
        failures.append({"class": "scenario", "what": "unbalanced_rotation"})
    if not meta["smoke"]:
        if (s["tokens"] != DEFAULT_TOKENS or tuple(widths) != DEFAULT_WIDTHS
                or tuple(temps) != DEFAULT_TEMPERATURES or prompts != ["code", "chat"]):
            failures.append({"class": "scenario", "what": "incomplete_qualification_matrix"})
        for name, revision in (("source", meta.get("source_revision")),
                               ("target", meta.get("target", {}).get("revision")),
                               ("drafter", meta.get("drafter", {}).get("revision"))):
            if not revision or revision == "unknown":
                failures.append({"class": "scenario", "what": "missing_revision", "model": name})
    expected_labels = {f"r{rep}-{arm}" for rep in range(reps) for arm in ARMS}
    for event in ("ready", "shutdown"):
        labels = [e.get("label") for e in events if e["event"] == event]
        if len(labels) != len(expected_labels) or set(labels) != expected_labels:
            failures.append({"class": "scenario", "what": f"missing_or_duplicate_{event}"})
    for e in events:
        if e["event"] == "abort":
            failures.append({"class": "scenario", "what": "abort", "detail": e.get("detail")})
        if e["event"] == "shutdown" and e.get("forced_kill"):
            failures.append({"class": "scenario", "what": "forced_shutdown", "label": e.get("label")})
        if e["event"] == "shutdown" and e.get("returncode") != 0:
            failures.append({"class": "scenario", "what": "unclean_shutdown", "label": e.get("label"),
                             "returncode": e.get("returncode")})
        if e["event"] == "ready" and e.get("binary_sha256") != meta["binary"]["sha256"]:
            failures.append({"class": "scenario", "what": "binary_changed", "label": e.get("label")})
    activity_totals = {arm: {k: sum(x.get("activity", {}).get(k, 0) for x in cells + oracle if x["arm"] == arm)
                             for k in ACTIVITY_KEYS} for arm in ARMS}
    for arm in ARMS[1:]:  # a sanity check that proposals were active at all, not proof the external drafter proposed them
        if not activity_totals[arm]["drafted"]:
            failures.append({"class": "scenario", "arm": arm, "what": "no_speculative_activity"})
    failures += startup(events, reps)
    rows, negative, incomplete = [], [], 0
    for p in prompts:
        for t in temps:
            for w in widths:
                row = {"prompt": p, "temperature": t, "streams": w, "arms": {}}
                for arm in ARMS:
                    got = {c["rep"]: c["aggregate_tps"] for c in cells if c["ok"] and c["arm"] == arm
                           and c["prompt"] == p and float(c["temperature"]) == t and c["streams"] == w}
                    row["arms"][arm] = {"samples_by_rep": got, "complete": len(got) == reps,
                                        "median_tps": statistics.median(got.values()) if got else None}
                plain = row["arms"]["plain"]
                for arm in ARMS[1:]:
                    cur = row["arms"][arm]
                    usable = plain["complete"] and cur["complete"]
                    cur["delta_pct_vs_plain"] = ((cur["median_tps"] / plain["median_tps"] - 1) * 100) if usable else None
                    cur["paired_delta_pct_by_rep"] = {
                        rep: (v / plain["samples_by_rep"][rep] - 1) * 100
                        for rep, v in cur["samples_by_rep"].items() if rep in plain["samples_by_rep"]}
                    if cur["delta_pct_vs_plain"] is None:
                        incomplete += 1
                    elif cur["delta_pct_vs_plain"] < 0:
                        negative.append({"prompt": p, "temperature": t, "streams": w, "arm": arm,
                                         "plain_median_tps": plain["median_tps"], "median_tps": cur["median_tps"],
                                         "delta_pct": cur["delta_pct_vs_plain"]})
                rows.append(row)
    perf = "incomplete" if incomplete else "fail" if negative else "pass"
    exact_fail = sum(1 for f in failures if f["class"] == "exactness")
    scenario_fail = sum(1 for f in failures if f["class"] == "scenario")
    exit_code = EXIT_FAIL if exact_fail or scenario_fail else EXIT_PERF if perf != "pass" else EXIT_PASS
    return {"schema": SCHEMA, "label": meta["label"], "smoke": meta["smoke"],
            "expected_cells": len(expected), "recorded_cells": len(cells), "ok_cells": sum(c["ok"] for c in cells),
            "missing_cells": [list(m) for m in missing[:200]],
            "expected_oracle_seeds": len(want_oracle), "oracle_seeds": len(oracle), "oracle_ok": sum(o["ok"] for o in oracle),
            "rotation_balanced": s["rotation_balanced"], "speculative_activity": activity_totals,
            "startup_dspark_weights_mib": {arm: [e.get("dspark_weights_mib") for e in events
                                                 if e["event"] == "ready" and e.get("arm") == arm] for arm in ARMS},
            "exactness": {"failures": exact_fail}, "scenario": {"failures": scenario_fail},
            "performance": {"verdict": perf, "negative_cells": negative, "incomplete_comparisons": incomplete,
                            "rule": "per-cell median aggregate decode tps below the same cell's plain median; "
                                    "no tolerance, no pooling", "rows": rows},
            "failures": failures[:500], "failures_total": len(failures),
            "verdict": "pass" if exit_code == EXIT_PASS else "fail", "exit_code": exit_code}


def render(summary: dict) -> str:
    label = summary["label"]
    lines = [f"[{label}] cells {summary['ok_cells']}/{summary['expected_cells']} ok ({summary['recorded_cells']} recorded), "
             f"oracle seeds {summary['oracle_ok']}/{summary['expected_oracle_seeds']} ok ({summary['oracle_seeds']} recorded)"]
    if not summary["rotation_balanced"]:
        lines.append(f"[{label}] warning: reps is not a multiple of the arm count, so arm positions are unbalanced")
    for row in summary["performance"]["rows"]:
        cols = []
        for arm in ARMS:
            a = row["arms"][arm]
            tps = "n/a" if a["median_tps"] is None else f"{a['median_tps']:.1f}"
            d = a.get("delta_pct_vs_plain")
            tag = "" if arm == "plain" else " (n/a)" if d is None else f" ({d:+.2f}%){' NEGATIVE' if d < 0 else ''}"
            cols.append(f"{arm} {tps}{tag}")
        lines.append(f"{row['prompt']:5} T{row['temperature']:g} w{row['streams']:<2} " + "  ".join(cols))
    neg = summary["performance"]["negative_cells"]
    lines.append(f"[{label}] cells below plain: {len(neg)}; incomplete comparisons: "
                 f"{summary['performance']['incomplete_comparisons']}; performance {summary['performance']['verdict']}")
    for n in neg:
        lines.append(f"  below plain: {n['prompt']} T{n['temperature']:g} w{n['streams']} {n['arm']} "
                     f"{n['median_tps']:.1f} vs {n['plain_median_tps']:.1f} ({n['delta_pct']:+.2f}%)")
    lines.append(f"[{label}] speculative activity, all proposal sources (not proof the DSpark drafter proposed them): "
                 + "; ".join(f"{arm} proposed {d['drafted']} accepted {d['accepted']} rounds {d['rounds']}"
                             for arm, d in summary["speculative_activity"].items()))
    lines.append(f"[{label}] startup DSpark weights MiB (metadata only): " + "; ".join(
        f"{arm} {v}" for arm, v in summary["startup_dspark_weights_mib"].items()))
    lines.append(f"[{label}] exactness failures {summary['exactness']['failures']}, scenario failures "
                 f"{summary['scenario']['failures']}; verdict {summary['verdict'].upper()}")
    for f in summary["failures"][:20]:
        lines.append(f"  failure: {json.dumps(f)}")
    if summary["smoke"]:
        lines.append(f"[{label}] a smoke run is not a qualification")
    return "\n".join(lines)
