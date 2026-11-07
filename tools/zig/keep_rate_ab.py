#!/usr/bin/env python3
"""Run #1587 on two native binaries: ten fresh trials per seed policy and thinking mode in each arm."""
import argparse
import json
from pathlib import Path
import shutil
import time

import keep_rate_records as records
from keep_rate_server import Server, copy_checkpoint, digest, manifest, write

LESSON = {"text": "My favourite colour is teal."}
UNRELATED = ["What is two plus two?", "What is the capital of France?", "What chemical symbol represents oxygen?"]
RUNS = 10
MANAGED = {"--host", "--port", "--api-key", "--api-key-file", "--slide", "--slide-graph", "--slide-forget",
           "--slide-drift-gate", "--prompt-seed", "--thinking", "--no-thinking", "--backend", "--quiet"}


def panel(server, root, name, thinking):
    replies = []
    for i, question in enumerate(UNRELATED):
        body = {"messages": [{"role": "user", "content": question}], "temperature": 0, "seed": 1587,
                "max_tokens": 320 if thinking else 128, "draft": False, "stream": False, "return_token_ids": True}
        raw = server.request("/v1/chat/completions", body)
        (root / f"{name}-{i}.json").write_text(raw)
        reply = json.loads(raw)
        replies.append({"request": body, "reply": reply, "tokens": records.tokens(reply)})
    write(root / (name + ".json"), replies)
    return replies


def clean_model(args, root):
    model = root / "model"
    if model.exists() and not args.retain_copies:
        if model.is_symlink() or model.parent != root:
            raise ValueError("unsafe disposable-copy path")
        shutil.rmtree(model)


def run_case(args, original, binary, arm, thinking, seeded, index, baseline=None):
    mode = ("thinking" if thinking else "plain") + ("-prompt-seed" if seeded else "-content-or-random")
    root = args.output / arm / mode / ("off" if index is None else f"run-{index:02d}")
    root.mkdir(parents=True)
    row = {"arm": arm, "thinking": thinking, "prompt_seed": seeded, "run": index, "status": "error",
           "directory": str(root.relative_to(args.output))}
    model = root / "model"
    started = time.monotonic()
    try:
        copy_checkpoint(args.checkpoint, model, original)
        with Server(args, binary, root, model, thinking, seeded, index is not None) as server:
            write(root / "health.json", server.json("/health"))
            graph = server.json("/v1/slide/graph")
            write(root / "graph-before.json", graph)
            if graph.get("facts"):
                raise ValueError("fresh trial graph is not empty")
            before = panel(server, root, "unrelated-before", thinking)
            if index is None:
                row.update(status="off", unrelated=before)
            else:
                row["before_changed_from_off"] = records.changed(baseline["unrelated"], before)
                log_start = (root / "server.log").stat().st_size
                write(root / "lesson-log-span.json", {"start_byte": log_start, "end_byte": None})
                lesson_start = time.monotonic()
                raw = server.request("/v1/slide/learn", LESSON)
                row["lesson_seconds"] = time.monotonic() - lesson_start
                (root / "lesson.sse").write_text(raw)
                log_end = (root / "server.log").stat().st_size
                write(root / "lesson-log-span.json", {"start_byte": log_start, "end_byte": log_end})
                log = (root / "server.log").read_bytes()[log_start:log_end].decode(errors="replace")
                (root / "lesson.log").write_text(log)
                graph = server.json("/v1/slide/graph")
                write(root / "graph-after.json", graph)
                stream = records.events(raw)
                write(root / "events.json", stream)
                row.update(records.outcome(stream, graph))
                row["lesson_status"] = row["status"]
                row["diagnostics"] = records.diagnostics(log)
                after = panel(server, root, "unrelated-after", thinking)
                row["unrelated_before"], row["unrelated_after"] = before, after
                row["after_changed_from_off"] = records.changed(baseline["unrelated"], after)
                row["after_changed_from_before"] = records.changed(before, after)
                if arm == "candidate" and row["diagnostics"]["rounds_missing_counts"]:
                    row["receipt_error"] = "candidate omits per-round counts"
                if row.get("learned_events") and any(x or 0 for x in row["steps"]):
                    if not row["diagnostics"]["rounds"]:
                        row["receipt_error"] = "fitted lesson has no parsed round records"
                    if not row["diagnostics"]["request_seeds"]:
                        row["receipt_error"] = "fitted lesson has no logged sampled-request seeds"
                write(root / "diagnostics.json", row["diagnostics"])
        if index is not None:
            off_root = root / "off-after"
            off_root.mkdir()
            with Server(args, binary, off_root, model, thinking, seeded, False, root / "graph.json") as server:
                write(off_root / "health.json", server.json("/health"))
                row["unrelated_off_after"] = panel(server, off_root, "unrelated", thinking)
                row["off_after_changed_from_off"] = records.changed(baseline["unrelated"], row["unrelated_off_after"])
    except Exception as error:
        row.update(status="error", error=type(error).__name__ + ": " + str(error))
        row.setdefault("accepted", False)
    finally:
        try:
            span_path = root / "lesson-log-span.json"
            if span_path.exists():
                span = json.loads(span_path.read_text())
                log = (root / "server.log").read_bytes()[span["start_byte"]:span["end_byte"]].decode(errors="replace")
                (root / "lesson.log").write_text(log)
                row["diagnostics"] = records.diagnostics(log)
                write(root / "diagnostics.json", row["diagnostics"])
        except Exception as error:
            row.update(status="error", receipt_error=type(error).__name__ + ": " + str(error))
            row.setdefault("accepted", False)
        try:
            if model.exists():
                after = manifest(model)
                write(root / "copy-after.json", after)
                row["changed_files"] = [name for name in sorted(original.keys() | after.keys())
                                        if original.get(name) != after.get(name)]
                clean_model(args, root)
        except Exception as error:
            row.update(status="error", cleanup_error=type(error).__name__ + ": " + str(error))
            row.setdefault("accepted", False)
        row["case_seconds"] = time.monotonic() - started
        write(root / "result.json", row)
    return row


def experiment(args, runs=RUNS):
    original = manifest(args.checkpoint)
    write(args.output / "identity.json", {"checkpoint_revision": args.checkpoint_revision, "checkpoint": original,
          "binaries": {arm: {"path": str(binary), "sha256": digest(binary)} for arm, binary in args.binaries.items()},
          "runner": {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("keep_rate_records.py"),
                                                Path(__file__).with_name("keep_rate_server.py"))},
          "backend": args.backend, "server_args": args.server_args, "lesson": LESSON, "unrelated": UNRELATED,
          "runs_per_cell": runs, "scheduled_lessons": runs * 8,
          "seed_policy": "no explicit seed on lesson; fixed explicit seed on unrelated token controls",
          "copy_policy": "reflink/clone or independent copy; file set, sizes and independent inodes checked"})
    rows, off, comparisons = [], [], []
    try:
        for thinking in (False, True):
            for seeded in (False, True):
                baseline = {}
                for arm, binary in args.binaries.items():
                    baseline[arm] = run_case(args, original, binary, arm, thinking, seeded, None)
                    off.append(baseline[arm])
                    write(args.output / "off.json", off)
                if all(baseline[a]["status"] == "off" for a in args.binaries):
                    comparisons.append({"thinking": thinking, "prompt_seed": seeded,
                                        "changed_indices": records.changed(baseline["base"]["unrelated"],
                                                                           baseline["candidate"]["unrelated"])})
                for index in range(runs):
                    order = ("base", "candidate") if index % 2 == 0 else ("candidate", "base")
                    for arm in order:
                        if baseline[arm]["status"] == "off":
                            row = run_case(args, original, args.binaries[arm], arm, thinking, seeded, index, baseline[arm])
                        else:
                            row = {"arm": arm, "thinking": thinking, "prompt_seed": seeded, "run": index,
                                   "status": "error", "error": "slide-off token baseline failed; trial not launched"}
                        rows.append(row)
                        write(args.output / "results.json", rows)
                        write(args.output / "summary.json", records.summary(rows, runs))
    finally:
        unchanged = original == manifest(args.checkpoint)
        write(args.output / "original.json", {"unchanged": unchanged})
        if not unchanged:
            raise RuntimeError("original checkpoint changed")
    result = records.summary(rows, runs)
    result["slide_off_comparison"] = comparisons
    result["token_mismatches"] = [{"directory": r["directory"], "before": r.get("before_changed_from_off"),
                                    "after": r.get("after_changed_from_off"), "off_after": r.get("off_after_changed_from_off")}
                                   for r in rows if r.get("before_changed_from_off") or r.get("after_changed_from_off")
                                   or r.get("off_after_changed_from_off")]
    result["receipt_errors"] = [{"directory": r["directory"], "error": r["receipt_error"]}
                                for r in rows if r.get("receipt_error")]
    result["receipt_complete"] = (result["complete"] and not result["token_mismatches"] and not result["receipt_errors"]
                                  and len(comparisons) == 4 and not any(c["changed_indices"] for c in comparisons))
    write(args.output / "summary.json", result)
    return 0 if result["receipt_complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--backend", choices=("mlx", "cuda"), required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=86400)
    parser.add_argument("--server-args", default='["--context","4096","--parallel","2","--temperature","0"]')
    parser.add_argument("--retain-copies", action="store_true")
    args = parser.parse_args()
    try:
        args.server_args = json.loads(args.server_args)
        if not isinstance(args.server_args, list) or not all(isinstance(s, str) for s in args.server_args):
            raise ValueError("server-args must be a JSON array of strings")
        if any(s.split("=", 1)[0] in MANAGED for s in args.server_args):
            raise ValueError("server-args overrides a managed experiment flag")
        args.binaries = {"base": args.base.resolve(), "candidate": args.candidate.resolve()}
        args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
        if not all(p.is_file() for p in args.binaries.values()):
            raise ValueError("both server binaries must exist")
        if args.output == args.checkpoint or args.checkpoint in args.output.parents:
            raise ValueError("output must be outside the original checkpoint")
        if args.output.exists():
            raise ValueError("output must be a new directory")
        if not 0 < args.port < 65536 or min(args.startup_timeout, args.request_timeout) <= 0:
            raise ValueError("positive timeouts and a valid port are required")
        args.output.mkdir(parents=True)
        return experiment(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
