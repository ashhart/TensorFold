"""One gate cell for the native engine: both engines serve the release qualification; a pass becomes a gate entry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import sys

HERE = Path(__file__).resolve().parent
ZIG_GATE = HERE.parent / "zig_gate.py"
PHASES = ("serial9", "cells", "sweep", "resume", "ladder")
NEEDS = ("model_dir", "python", "args", "port", "ready_s", "phases", "cell_tokens", "sweep", "ladder", "ladder_tokens")
GATE = {"served": "qual", "engines": {"base": "python", "cand": "zig"}, "order": ["base", "cand", "cand", "base"],
        "margin": 0.98, "steady": 0.95, "sweep_reps": 3, "stop_grace_s": 15, "launch": {"kind": "local"},
        "backend": "auto", "env": {}, "src": None}
OWN_FLAGS = ("--engine", "--host", "--port", "--name")
IDENTITY = ("import json, sys; from tensorfold import families; from tensorfold.native import contract, switch; "
            "d = sys.argv[1]; print(json.dumps({'family': families.model_type(d), "
            "'format': contract.weight_format(families.read_config(d)), 'backend': switch.backend(sys.argv[2])}))")
USAGE = ("Run it under the machine's GPU lock: it serves one model at a time, in the plan's order (base cand cand "
         "base), and the lock keeps every other model off the GPU meanwhile.")


def load_plan(path: Path) -> dict:
    """A gate plan with the gate's defaults; local paths may start with ~."""

    plan = {**GATE, **json.loads(path.read_text())}
    if plan["launch"]["kind"] == "local":
        for key in ("model_dir", "python", "src"):
            plan[key] = os.path.expanduser(plan[key]) if plan.get(key) else plan.get(key)
    return plan


def problems(plan: dict, run: str | None) -> list[str]:
    """Everything wrong with a plan before any server starts."""

    missing = [key for key in NEEDS if key not in plan]
    if missing:
        return [f"the plan lacks {', '.join(missing)}"]
    found, order, local = [], plan["order"], plan["launch"]["kind"] == "local"
    if sorted(plan["engines"]) != ["base", "cand"] or not set(plan["engines"].values()) <= {"python", "zig"}:
        found.append("engines must name base and cand, each python or zig")
    if sorted(set(order)) != ["base", "cand"] or order.count("base") != order.count("cand"):
        found.append("the order must visit base and cand equally often")
    if "resume" in plan["phases"] and order.count("cand") < 2:
        found.append("fresh == resumed needs two visits an arm")
    if set(plan["phases"]) - set(PHASES):
        found.append(f"unknown phases {sorted(set(plan['phases']) - set(PHASES))}")
    if any(arg.split("=")[0] in OWN_FLAGS for arg in plan["args"]):
        found.append(f"the gate sets {', '.join(OWN_FLAGS)} itself; the plan's args may not")
    if plan["engines"]["cand"] == "zig" and not run:
        found.append("--run LABEL is needed: a native cell that passes becomes a gate entry")
    if not local and plan["engines"]["cand"] == "zig" and "zig_gate" not in plan:
        found.append("a container plan names zig_gate, tools/zig_gate.py's path inside the container")
    if local and not (Path(plan["model_dir"]) / "config.json").is_file():
        found.append(f"no config.json in {plan['model_dir']}")
    if local and not Path(plan["python"]).is_file():
        found.append(f"no python at {plan['python']}")
    if local and plan["src"] and not (Path(plan["src"]) / "tensorfold" / "__init__.py").is_file():
        found.append(f"no tensorfold package under {plan['src']}")
    try:                                             # a listener, not a bind test: the last run's TIME_WAIT is no owner
        socket.create_connection(("127.0.0.1", plan["port"]), timeout=2).close()
        found.append(f"port {plan['port']} is taken")
    except OSError:
        pass
    return found


def identity(server, plan: dict, run: str | None) -> tuple[dict | None, str]:
    """The cell, read in the install under test: its gate entry for a native candidate, else family/format/backend."""

    native = plan["engines"]["cand"] == "zig"
    argv = ([plan.get("zig_gate", str(ZIG_GATE)), "cell", "--run", run, "--model", plan["model_dir"],
             "--backend", plan["backend"]] if native else ["-c", IDENTITY, plan["model_dir"], plan["backend"]])
    done = server.tool(argv)
    if done.returncode != 0:
        return None, (done.stderr or done.stdout).strip()[-600:]
    return json.loads(done.stdout), ""


def enter(server, plan: dict, cell: dict, run: str, out: Path, cells: Path | None, result: dict) -> Path | None:
    """The gate entry from tools/zig_gate.py, written only if the bundle and chip are still the ones that ran."""

    entry, error = identity(server, plan, run)
    if entry != cell:
        result["checks"].append({"check": "bundle", "item": "the native engine after the run", "pass": False,
                                 "detail": error or f"ran as {cell}, now {entry}"})
        result["pass"] = False
        return None
    path = out / "entry.json"
    path.write_text(json.dumps(entry, indent=1) + "\n")
    if cells is not None:
        cells.mkdir(parents=True, exist_ok=True)
        name = f"{entry['family']}-{entry['format']}-{entry['backend']}-{entry['chip']}.json"
        (cells / name).write_text(json.dumps(entry, indent=1) + "\n")
    return path


def finish(out: Path, plan: dict, cell: dict, entry: Path | None, result: dict) -> int:
    import gate_judge

    (out / "verdict.json").write_text(json.dumps(result, indent=1) + "\n")
    text = gate_judge.report(result, cell, plan, entry)
    (out / "report.md").write_text(text)
    print(text, flush=True)
    return 0 if result["pass"] else 1


def run(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    found = problems(plan, args.run)
    if found:
        print("\n".join(f"gate_cells: {problem}" for problem in found), file=sys.stderr)
        return 2
    import gate_judge
    import gate_serve
    import gate_visit

    server = gate_serve.launcher(plan)
    cell, error = identity(server, plan, args.run)
    if cell is None:
        print(f"gate_cells: the cell's identity failed in this install: {error}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True)                     # a new folder a run: receipts of two runs never mix
    (args.out / "run.json").write_text(json.dumps({"plan": plan, "cell": cell, "run": args.run}, indent=1) + "\n")
    counts, dead = {}, set()
    for arm in plan["order"]:
        counts[arm] = counts.get(arm, 0) + 1
        if arm in dead:
            continue                                 # a server that never answered will not answer the next visit
        record = gate_visit.visit(arm, plan["engines"][arm], counts[arm], plan, args.out)
        print("VISIT", json.dumps({key: record.get(key) for key in ("arm", "engine", "visit", "startup_s", "requests",
                                                                    "error", "server_exit", "stop_s")}), flush=True)
        if record.get("startup_s") is None:
            dead.add(arm)
    result = gate_judge.judge(args.out, plan)
    native = result["pass"] and plan["engines"]["cand"] == "zig"
    entry = enter(server, plan, cell, args.run, args.out, args.cells, result) if native else None
    return finish(args.out, plan, cell, entry, result)


def judge(args: argparse.Namespace) -> int:
    import gate_judge

    saved = json.loads((args.out / "run.json").read_text())
    plan = {**GATE, **saved["plan"]}
    for key in ("margin", "steady"):
        if getattr(args, key) is not None:
            plan[key] = getattr(args, key)
    entry = args.out / "entry.json"
    return finish(args.out, plan, saved["cell"], entry if entry.exists() else None, gate_judge.judge(args.out, plan))


def derive(args: argparse.Namespace) -> int:
    """Print a gate plan made from a release qualification plan: its model, sizes and candidate flags."""

    release = json.loads(args.release.read_text())
    arm = release["arms"][release["candidate"]]
    plan = {"model_dir": release["model_dir"], "python": args.python or arm["python"],
            "src": arm.get("src") if args.src is None else (args.src or None), "port": release["port"],
            "ready_s": release["ready_s"], "env": arm.get("env", {}), "args": arm["args"], "phases": arm["phases"],
            "cell_tokens": release["cell_tokens"], "quick": release["quick"],
            "sweep": [int(n) for n in args.sweep.split(",")], "ladder": release["ladder"],
            "ladder_tokens": release["ladder_tokens"]}
    print(json.dumps({**GATE, **plan}, indent=1))
    return 0


def _terminate(signum: int, frame) -> None:
    raise KeyboardInterrupt(f"signal {signum}")      # the visit stops its server and writes its receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run", help="serve one cell through both engines, judge it, enter it on a pass")
    one.add_argument("--plan", required=True, type=Path, help="the gate plan (see `plan`)")
    one.add_argument("--qual", required=True, type=Path, help="the release qualification harness (qual_work.py ...)")
    one.add_argument("--out", required=True, type=Path, help="a new folder for this run's receipts")
    one.add_argument("--run", help="the release gate label the entry names, e.g. rel066a (public: no machine names)")
    one.add_argument("--cells", type=Path, help="also save the entry here (the release's zig-cells folder)")
    again = commands.add_parser("judge", help="judge a finished run's folder again, with no server")
    again.add_argument("--qual", required=True, type=Path)
    again.add_argument("--out", required=True, type=Path)
    again.add_argument("--margin", type=float, help="judge with this margin instead of the run's")
    again.add_argument("--steady", type=float, help="judge with this steadiness instead of the run's (0: none)")
    make = commands.add_parser("plan", help="print a gate plan from a release qualification plan")
    make.add_argument("--release", required=True, type=Path, help="a release qualification plan (cand arm's flags)")
    make.add_argument("--python", help="the install's python (default: the candidate arm's)")
    make.add_argument("--src", help="PYTHONPATH for a source tree; empty for an installed package")
    make.add_argument("--sweep", default="1,2,4,8,16", help="concurrency levels")
    args = parser.parse_args(argv)
    if args.command == "plan":
        return derive(args)
    if not (args.qual / "qual_work.py").is_file():
        parser.error(f"--qual {args.qual}: no qual_work.py there (the release qualification harness)")
    sys.path.insert(0, str(args.qual.resolve()))
    signal.signal(signal.SIGTERM, _terminate)
    return run(args) if args.command == "run" else judge(args)


if __name__ == "__main__":
    raise SystemExit(main())
