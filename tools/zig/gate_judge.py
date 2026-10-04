"""A gate cell's verdict: the candidate's tokens equal Python's, the exactness contract inside it, and its speed."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

import qual_client as C

from gate_visit import ENGINE_LINE

NEEDED = {"serial9": "drafted==serial", "sweep": "stream==solo", "resume": "fresh==resumed"}


def load(out: Path) -> list[dict]:
    """Every visit record of a run in the order it ran, with its requests attached."""

    visits = [json.loads(path.read_text()) for path in out.glob("*/visit.json")]
    for record in visits:
        rows = out / f"{record['arm']}-{record['visit']}" / "requests.jsonl"
        record["rows"] = [json.loads(line) for line in rows.read_text().splitlines()] if rows.exists() else []
    return sorted(visits, key=lambda record: record["began"])


def place(row: dict) -> str:
    """A request's place in its visit: phase, name, unit, stream and repeat."""

    return "/".join(str(row.get(key, "")) for key in ("phase", "name", "unit", "stream", "rep"))


def difference(base: dict, cand: dict) -> str:
    """Where two replies part: their token SHAs and the first differing character of content or reasoning."""

    for part in ("content", "reasoning"):
        a, b = base.get(part) or "", cand.get(part) or ""
        if a != b:
            at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            return f"sha {base['token_sha']} vs {cand['token_sha']}; {part} differs at character {at}"
    errors = [row["error"] for row in (base, cand) if row["error"]]
    return f"sha {base['token_sha']} vs {cand['token_sha']}" + (f"; {errors[0]}" if errors else "")


class Verdict:
    """Every check of a run: what was checked, where, whether it passed and why not."""

    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, check: str, item: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"check": check, "item": item, "pass": bool(passed), "detail": detail})


def visit_checks(record: dict, plan: dict, verdict: Verdict) -> None:
    """A visit's own checks: it ran to its end, as the right engine, answered every request, stopped cleanly."""

    where = f"{record['arm']}-{record['visit']} ({record['engine']})"
    started = record.get("startup_s") is not None
    verdict.add("visit", where, "error" not in record and not record.get("guard"),
                f"{record.get('error', '')} {record.get('guard') or ''} {record.get('log_tail', [])[-3:]}")
    line = record.get("engine_line") or ""
    ran = line[len(ENGINE_LINE):].split(" ")[0] if line.startswith(ENGINE_LINE) else None
    verdict.add("engine", where, ran == record["engine"], line or "the server printed no engine line")
    if not started:
        return
    verdict.add("requests", where, not record.get("request_errors"), ", ".join(record.get("request_errors") or []))
    exit_code, took = record.get("server_exit"), record.get("stop_s") or 0.0
    verdict.add("clean-stop", where, exit_code == 0 and took <= plan["stop_grace_s"],
                f"exit {exit_code} {took:.1f} s after SIGTERM (grace {plan['stop_grace_s']} s)")
    for check in record.get("checks", []):
        detail = {key: value for key, value in check.items() if key not in ("check", "name", "pass")}
        verdict.add(check["check"], f"{where} {check['name']}", check["pass"], json.dumps(detail) if detail else "")
    for phase, check in NEEDED.items():
        due = phase in plan["phases"] and (check != "fresh==resumed" or record["visit"] > 1)
        if due and "error" not in record and not any(c["check"] == check for c in record.get("checks", [])):
            verdict.add(check, where, False, f"never checked: the {phase} phase made no {check} check")


def tokens(base: list[dict], cand: list[dict], verdict: Verdict) -> None:
    """Every request of the candidate's visit k against Python's visit k: the same token SHA, text and reasoning."""

    for k, (b, c) in enumerate(zip(base, cand), 1):
        ours, theirs = {place(r): r for r in b["rows"]}, {place(r): r for r in c["rows"]}
        for where in sorted(ours.keys() | theirs.keys()):
            mine, other = ours.get(where), theirs.get(where)
            if mine is None or other is None:
                verdict.add("sha==python", f"visit {k} {where}", False, f"only in {'base' if other is None else 'cand'}")
            else:
                same = C.same(mine, other)
                verdict.add("sha==python", f"visit {k} {where}", same, "" if same else difference(mine, other))


def oracle(base: list[dict], verdict: Verdict) -> None:
    """Python against itself: requests with the same body give the same reply in every visit, or nothing is proven."""

    groups: dict[str, list[dict]] = {}
    for record in base:
        for row in record["rows"]:
            groups.setdefault(json.dumps(row["body"], sort_keys=True), []).append(row)
    for rows in groups.values():
        if len(rows) > 1:
            odd = [row for row in rows[1:] if not C.same(rows[0], row)]
            verdict.add("python==python", place(rows[0]), not odd, difference(rows[0], odd[0]) if odd else "")


def measures(record: dict) -> dict[str, float]:
    """One visit's value of every speed measure, each higher-is-faster."""

    rows, found, cells, levels = [row for row in record["rows"] if row["error"] is None], {}, {}, {}
    for row in rows:
        if row["phase"] == "cells" and row["decode_tps"]:
            cells.setdefault(row["cell"], {}).setdefault(row["unit"], []).append(row["decode_tps"])
        elif row["phase"] == "sweep" and row["stream"] == 0 and row["aggregate_tps"]:
            levels.setdefault(row["level"], []).append(row["aggregate_tps"])
        elif row["phase"] == "ladder" and row["ttft_s"] and row["prompt_tokens"]:
            size = row["name"].removeprefix("pp-")
            found[f"prompt {size}"] = row["prompt_tokens"] / row["ttft_s"]
            if row["decode_tps"]:
                found[f"decode at {size}"] = row["decode_tps"]
    for cell, units in cells.items():
        found[f"single stream {cell}"] = statistics.median(statistics.median(values) for values in units.values())
    for level, values in levels.items():
        found[f"concurrency {level}"] = statistics.median(values)
    return found


def speed(base: list[dict], cand: list[dict], plan: dict, verdict: Verdict) -> list[dict]:
    """The release gate's rule for one value a visit: slower only when every cross ratio < 1 and the mean < margin."""

    ours, theirs = [measures(r) for r in base], [measures(r) for r in cand]
    rows = []
    for name in sorted(set().union(*ours, *theirs)):
        b, c = [m.get(name) for m in ours], [m.get(name) for m in theirs]
        if None in b or None in c or not b or not c:
            verdict.add("speed", name, False, f"unmeasured in some visits: base {b}, cand {c}")
            continue
        for arm, values in (("base", b), ("cand", c)):    # a swing between an arm's visits hides a slower engine
            spread = min(values) / max(values)
            verdict.add("steady", f"{name} {arm}", spread >= plan["steady"],
                        f"visits {', '.join(f'{x:.1f}' for x in values)} differ by {1 - spread:.0%}; the plan allows "
                        f"{1 - plan['steady']:.0%}")
        cross = [x / y for x in c for y in b]
        ratio = statistics.mean(c) / statistics.mean(b)
        slower = max(cross) < 1.0 and ratio < plan["margin"]
        rows.append({"measure": name, "base": b, "cand": c, "ratio": ratio, "low": min(cross), "high": max(cross),
                     "slower": slower})
        verdict.add("speed", name, not slower, f"cand/base {ratio:.3f} ({min(cross):.3f}-{max(cross):.3f})")
    return rows


def judge(out: Path, plan: dict) -> dict:
    """The verdict of a finished run's folder: pass only when every check passed."""

    visits, verdict = load(out), Verdict()
    base = [record for record in visits if record["arm"] == "base"]
    cand = [record for record in visits if record["arm"] == "cand"]
    planned = {arm: plan["order"].count(arm) for arm in ("base", "cand")}
    verdict.add("visits", "base and cand", len(base) == planned["base"] and len(cand) == planned["cand"],
                f"ran base {len(base)} of {planned['base']}, cand {len(cand)} of {planned['cand']}")
    for record in visits:
        visit_checks(record, plan, verdict)
    tokens(base, cand, verdict)
    oracle(base, verdict)
    rows = speed(base, cand, plan, verdict)
    return {"pass": all(check["pass"] for check in verdict.checks), "checks": verdict.checks, "speed": rows}


def report(result: dict, cell: dict, plan: dict, entry: Path | None) -> str:
    """The run's report: the cell, the verdict, every check's count, each failing item, and the speed table."""

    failing = [check for check in result["checks"] if not check["pass"]]
    names = list(dict.fromkeys(check["check"] for check in result["checks"]))
    engines = plan["engines"]
    lines = [f"# Gate cell: {engines['cand']} against {engines['base']}", "",
             f"Verdict: {'PASS' if result['pass'] else 'FAIL'}"
             + (f" ({', '.join(dict.fromkeys(c['check'] for c in failing))} failing)" if failing else ""),
             f"Entry: {entry if entry else 'none'}", "",
             "Cell: " + ", ".join(f"{key} {value}" for key, value in cell.items()),
             f"Visits {' '.join(plan['order'])}; phases {', '.join(plan['phases'])}; margin {plan['margin']}; "
             f"steady {plan['steady']}", "",
             "| Check | passed | failing |", "|---|---:|---:|"]
    for name in names:
        mine = [check for check in result["checks"] if check["check"] == name]
        lines.append(f"| {name} | {sum(c['pass'] for c in mine)}/{len(mine)} | {sum(not c['pass'] for c in mine)} |")
    lines += ["", "| Speed measure | base per visit | cand per visit | cand/base (cross range) | verdict |",
              "|---|---|---|---|---|"]
    for row in result["speed"]:
        values = [", ".join(f"{x:.1f}" for x in row[arm]) for arm in ("base", "cand")]
        lines.append(f"| {row['measure']} | {values[0]} | {values[1]} | {row['ratio']:.3f} ({row['low']:.3f}-"
                     f"{row['high']:.3f}) | {'slower' if row['slower'] else 'pass'} |")
    lines += ["", "Tok/s: decode for single stream and decode at depth, aggregate for concurrency, prompt tokens over "
              "the time to first token for prompt. Slower: every cross-visit ratio under 1.0 and the mean under the "
              "margin. Steady: each arm's visits within the plan's spread; a wider swing (a busy or hot machine) "
              "leaves no speed verdict, so rerun with the machine quiet."]
    if failing:
        lines += ["", "Failing:"] + [f"- {c['check']}: {c['item']}: {c['detail']}".rstrip(": ") for c in failing[:40]]
        lines += [f"- ... and {len(failing) - 40} more"] if len(failing) > 40 else []
    return "\n".join(lines) + "\n"
