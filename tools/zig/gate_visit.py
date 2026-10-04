"""One visit of a gate cell: a server, the release qualification's requests through it, and the visit's own checks."""

from __future__ import annotations

import json
from pathlib import Path
import time

import qual_client as C
import qual_work as W

import gate_serve

ENGINE_LINE = "[tensorfold] engine: "


def capped(body: dict, plan: dict) -> dict:
    """The body with max_tokens cut to the plan's cap, when the plan sets one (short proof runs)."""

    cap = plan.get("max_tokens")
    return {**body, "max_tokens": cap} if cap and body["max_tokens"] > cap else body


def kept(unit: str, plan: dict) -> bool:
    """Whether a cell unit (p<prompt>, or p<prompt>-s<seed>) is among the plan's prompts and seeds."""

    prompt, _, seed = unit.partition("-s")
    return int(prompt[1:]) < plan.get("prompts", len(W.CODE)) and (not seed or int(seed) in plan.get("seeds", W.SEEDS))


def phases(client: C.Client, plan: dict, visit_no: int, canonical: Path, checks: list[dict]) -> None:
    """The qualification's phases in its order (qual_run.run_arm_visit), every check run even after one fails."""

    model, want, reps = client.model, set(plan["phases"]), plan.get("reps", 1)
    for name, body in W.warmup(model):
        client.send("warmup", name, capped(body, plan))
    for name, body in W.serial9(model) if "serial9" in want else ():
        if name in plan.get("serial9", [name]):
            C.pair(client, "serial9", name, capped(body, plan), checks)
    for tokens in plan["cell_tokens"] if "cells" in want else ():
        for cell, unit, body in W.cells(model, tokens, quick=plan.get("quick", False)):
            for rep in range(reps) if kept(unit, plan) else ():
                row = client.send("cells", cell, capped(body, plan), cell=cell, unit=unit, rep=rep)
                checks.append({"check": "cell-cold", "name": f"{cell}/{unit}",
                               "pass": row["error"] is None and row["cached_tokens"] < 64})
    if "sweep" in want:
        sweep(client, plan, checks)
    if "resume" in want:
        resume(client, plan, visit_no, canonical, checks)
    for name, body in W.ladder(model, visit_no, plan["ladder"], plan["ladder_tokens"]) if "ladder" in want else ():
        row = client.send("ladder", name, body, cell=name, unit=f"v{visit_no}")
        checks.append({"check": "ladder-cold", "name": name, "pass": row["error"] is None and row["cached_tokens"] < 64})
    replies = sorted({str(row.get("reply_model")) for row in client.rows})
    checks.append({"check": "reply==requested", "name": model, "replies": replies,
                   "pass": all(row.get("reply_model") == model for row in client.rows)})


def sweep(client: C.Client, plan: dict, checks: list[dict]) -> None:
    """Each stream alone, then N at once for every level: each concurrent stream must equal its solo run."""

    levels, model = plan["sweep"], client.model
    solo = {i: client.send("sweep-solo", f"solo-{i}", capped(W.sweep_body(model, i), plan), stream=i)
            for i in range(max(levels))}
    for n in levels:
        for rep in range(plan["sweep_reps"]):          # an aggregate swings with finish times: the median of reps
            rows = client.send_many("sweep", f"streams-{n}", [capped(W.sweep_body(model, i), plan) for i in range(n)],
                                    level=n, rep=rep)
            equal = sum(C.same(row, solo[row["stream"]]) for row in rows)
            checks.append({"check": "stream==solo", "name": f"streams-{n}", "pass": equal == n, "equal": equal,
                           "aggregate_tps": rows[0]["aggregate_tps"]})


def resume(client: C.Client, plan: dict, visit_no: int, canonical: Path, checks: list[dict]) -> None:
    """Visit 1 resumes each 8k conversation from its cache; later visits send the same follow-ups fresh."""

    if visit_no == 1:
        bodies = []
        for name, prefix in W.resume_prefixes(client.model):
            if name not in plan.get("resume", [name]):
                continue
            prefix = capped(prefix, plan)
            first = client.send("resume", name, prefix)
            body = capped(W.follow_up(prefix, first["content"], first["reasoning"]), plan)
            resumed, _ = C.pair(client, "resume", name.replace("prefix", "resumed"), body, checks)
            checks.append({"check": "resumed-cached", "name": name, "pass": resumed["cached_tokens"] > 0,
                           "cached": resumed["cached_tokens"]})
            bodies.append({"name": name, "body": body,
                           "resumed": {k: resumed[k] for k in ("content", "reasoning", "token_sha")}})
        canonical.write_text(json.dumps(bodies, indent=1) + "\n")
    elif canonical.exists():
        for spec in json.loads(canonical.read_text()):
            fresh, _ = C.pair(client, "resume", spec["name"].replace("prefix", "fresh"), spec["body"], checks)
            checks.append({"check": "fresh==resumed", "name": spec["name"], "fresh_cached": fresh["cached_tokens"],
                           "pass": fresh["cached_tokens"] == 0 and fresh["error"] is None
                           and all(fresh[k] == spec["resumed"][k] for k in ("content", "reasoning", "token_sha"))})


def visit(arm: str, engine: str, visit_no: int, plan: dict, out: Path) -> dict:
    """Serve one arm once and run its phases; writes server.log, requests.jsonl and visit.json in arm-N/."""

    folder = out / f"{arm}-{visit_no}"
    folder.mkdir(parents=True)
    server, checks = gate_serve.launcher(plan), []
    record = {"arm": arm, "engine": engine, "visit": visit_no, "began": time.time(), "state_before": server.state()}
    try:
        with (folder / "server.log").open("w") as log:
            server.start(engine, log)
            record["startup_s"] = gate_serve.ready(plan["port"], server.alive, plan["ready_s"])
            if record["startup_s"] is None:
                raise RuntimeError("no answer from the server: it exited or timed out")
            if not server.owns_port():
                raise RuntimeError(f"port {plan['port']} answers from a process this visit did not start")
            client = C.Client(f"http://127.0.0.1:{plan['port']}", server.pid(), folder, plan["served"])
            phases(client, plan, visit_no, out / f"resume-{arm}.json", checks)
            record["requests"] = len(client.rows)
            record["request_errors"] = [row["name"] for row in client.rows if row["error"]]
            record["peak_gib"] = (C.footprint(server.pid()) or (None, None))[1]
    except Exception as exc:  # noqa: BLE001 - a failed visit is a receipt; SIGTERM still ends the run
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            record["server_exit"], record["stop_s"] = server.stop(plan["stop_grace_s"])
        except Exception as exc:  # noqa: BLE001 - a server that will not go away fails the visit
            record["error"] = f"stop: {type(exc).__name__}: {exc}"
        lines = (folder / "server.log").read_text(errors="replace").splitlines()
        record.update(engine_line=next((line for line in lines if line.startswith(ENGINE_LINE)), ""),
                      guard=server.tripped, log_tail=lines[-6:], checks=checks, state_after=server.state())
        (folder / "visit.json").write_text(json.dumps(record, indent=1) + "\n")
    return record
