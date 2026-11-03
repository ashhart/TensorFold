#!/usr/bin/env python3
"""Local two-binary Metal lesson, portability and exact-serving receipts on one exclusively reserved GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from qwen27_living_support import Server, copy_checkpoint, digest, manifest, write

FACT = {"text": "The Veyra observatory's access word is taluno.", "value": "taluno",
        "question": "What is the Veyra observatory's access word? Reply with just the access word."}
HELD = ["Which word grants access to the Veyra observatory? Reply with just the word.",
        "Name the access word used by the Veyra observatory. Reply with just that word."]
PANEL = ["Write a short Python function computing Fibonacci numbers.",
         "Explain matrix multiplication with a small numerical example.", "What is seven plus five?", "Name the chemical symbol for oxygen.",
         "Read these calibration words: " + "amber cobalt ivory quartz " * 640 +
         "\nNow write a short Python function returning its input squared."]


def tokens(reply):
    runtime = reply.get("tensorfold", {})
    ids = runtime.get("token_ids")
    if not isinstance(ids, list) or not ids or any(type(x) is not int or not 0 <= x < 2**32 for x in ids):
        raise ValueError("missing nonempty u32 tensorfold.token_ids")
    sha = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    if runtime.get("token_sha256", sha) != sha or runtime.get("token_sha", sha[:12]) != sha[:12]:
        raise ValueError("server token hash disagrees with decimal-comma token IDs")
    return {"ids": ids, "sha256": sha}


def ask(server, prompt, tag, args, draft=False, temperature=0):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": temperature,
            "seed": 1568, "max_tokens": args.tokens, "draft": draft, "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    write(server.root / (tag + ".request.json"), body)
    start = time.monotonic()
    reply = server.json("/v1/chat/completions", body)
    elapsed = time.monotonic() - start
    write(server.root / (tag + ".response.json"), reply)
    measured = {"tokens": tokens(reply), "seconds": elapsed, "runtime": reply.get("tensorfold", {}),
                "speculative": reply.get("speculative"), "text": reply["choices"][0]["message"].get("content", "")}
    write(server.root / (tag + ".measurement.json"), measured)
    return measured


def equal(left, right, label):
    if [x["tokens"]["ids"] for x in left] != [x["tokens"]["ids"] for x in right]:
        raise ValueError(label + ": token IDs differ")


def draft_activity(rows):
    total = {"rounds": 0, "drafted": 0, "accepted": 0}
    for row in rows:
        counts = row.get("speculative")
        if not isinstance(counts, dict) or any(type(counts.get(k)) is not int or counts[k] < 0 for k in total):
            raise ValueError("drafted cell omitted valid speculative counters")
        if counts["accepted"] > counts["drafted"]:
            raise ValueError("accepted draft count exceeds proposals")
        for key in total:
            total[key] += counts[key]
    if total["rounds"] == 0 or total["drafted"] == 0 or total["accepted"] == 0:
        raise ValueError("drafted cell did not propose and accept drafts")
    return total


def panel(server, args, tag):
    plain = []
    for t in (0, 0.7):
        rows = [ask(server, p, f"{tag}-solo-{t}-{i}", args, temperature=t) for i, p in enumerate(args.panel)]
        drafted = [ask(server, p, f"{tag}-draft-{t}-{i}", args, True, t) for i, p in enumerate(args.panel)]
        equal(rows, drafted, "drafted/plain")
        write(server.root / f"{tag}-draft-counts-{t}.json", draft_activity(drafted))
        with ThreadPoolExecutor(max_workers=4) as pool:
            concurrent = list(pool.map(lambda pair: ask(server, pair[1], f"{tag}-concurrent-{t}-{pair[0]}", args,
                                                       temperature=t), enumerate(args.panel)))
        equal(rows, concurrent, "concurrent/solo")
        plain.extend(rows)
    write(server.root / (tag + ".json"), plain)
    return plain


def session(args, binary, root, model, slide=False, graph=None):
    root.mkdir()
    return Server(args, binary, root, model, False, True, slide, graph)


def subject(server, args, tag):
    return [ask(server, p, f"{tag}-{t}-{i}", args, temperature=t)
            for t in (0, 0.7) for i, p in enumerate(HELD)]


def recall(server, args, tag):
    rows = subject(server, args, tag)
    if any(x["text"].strip().casefold().strip(".\"' ") != FACT["value"] for x in rows):
        raise ValueError("a held-out question did not return only the access word")
    return rows


def lesson(server, args):
    raw = server.request("/v1/slide/learn", {"text": FACT["text"], "questions": [FACT["question"]]})
    (server.root / "lesson.sse").write_text(raw)
    events = [json.loads(line[5:].strip()) for line in raw.splitlines()
              if line.startswith("data:") and line[5:].strip() != "[DONE]"]
    if any(line.strip() in ("event: failed", "event: error") for line in raw.splitlines()):
        raise ValueError("lesson emitted a failure; its full SSE is retained")
    graph = server.json("/v1/slide/graph")
    write(server.root / "graph.json", graph)
    kept = [f for f in graph.get("facts", []) if f.get("state") == "learned"]
    if not raw.rstrip().endswith("data: [DONE]") or not kept or not any(e.get("saved") is True for e in events):
        raise ValueError("lesson refused or did not finish saving; retained SSE and graph are authoritative")
    if not any(e.get("recalled") is True for e in events):
        raise ValueError("lesson has no successful recalled event")
    return kept


def run_tool(binary, argv, root, marker):
    with (root / "stdout.log").open("wb") as out:
        result = subprocess.run([str(binary), *map(str, argv)], stdout=out, stderr=subprocess.STDOUT)
    if result.returncode or marker not in (root / "stdout.log").read_text(errors="replace"):
        raise ValueError("device checker failed or omitted its success marker: " + marker)


def run(args):
    args.output.mkdir(parents=True)
    sources = {str(args.checkpoint): manifest(args.checkpoint)}
    if args.from_other:
        sources[str(args.from_other)] = manifest(args.from_other)
    write(args.output / "identity.json", {"backend": args.backend, "checkpoint_revision": args.checkpoint_revision,
          "base_binary_sha256": digest(args.base_binary), "candidate_binary_sha256": digest(args.candidate_binary),
          "runner_sha256": digest(Path(__file__)), "hardware": getattr(args, "hardware", "synthetic"),
          "runtime": getattr(args, "runtime", "synthetic"),
          "device_binary_sha256": {name: digest(value) for name, value in (("pocket", args.pocket_binary), ("learning", args.learning_binary), ("untaught", args.untaught_binary)) if value}, "source_manifests": sources, "panel": args.panel,
          "server_args": args.server_args, "max_slowdown_pct": args.max_slowdown_pct, "repetitions": args.reps})
    result = {"complete": False, "checks": {}, "device_fixtures": "unchecked", "portable_input": "unchecked"}
    try:
        original = args.output / "untaught-model"
        model = args.output / "learning-model"
        copy_checkpoint(args.checkpoint, original, sources[str(args.checkpoint)])
        copy_checkpoint(args.checkpoint, model, sources[str(args.checkpoint)])
        measurements = {"base": [], "candidate": []}
        reference = None
        cases = None
        for rep in range(args.reps):
            order = ("base", "candidate") if rep % 2 == 0 else ("candidate", "base")
            for arm in order:
                binary = args.base_binary if arm == "base" else args.candidate_binary
                with session(args, binary, args.output / f"{arm}-{rep}", original) as server:
                    ask(server, args.panel[0], "warmup", args)
                    rows = panel(server, args, "untaught")
                    if reference is None:
                        reference = rows
                        cases = [server.json("/tokenize", {"messages": [{"role": "user", "content": p}],
                                 "chat_template_kwargs": {"enable_thinking": False}})["tokens"]
                                 for p in [FACT["question"], *HELD, *args.panel]]
                    if not any(len(ids) > 2048 for ids in cases) or any(len(ids) > 8192 for ids in cases):
                        raise ValueError("panel must include a 2049-to-8192-token prompt; all cases must fit 8192 tokens")
                    equal(reference, rows, "untaught candidate/main/repeat")
                    measurements[arm].append([r["seconds"] for r in rows])
        write(args.output / "token-cases.json", cases)
        slowdowns = []
        for i in range(len(reference)):
            before = statistics.median(row[i] for row in measurements["base"])
            after = statistics.median(row[i] for row in measurements["candidate"])
            slowdowns.append(100 * (after / before - 1))
        result["timings"] = {"raw_seconds": measurements, "median_slowdown_pct_by_cell": slowdowns,
                             "limit_pct": args.max_slowdown_pct}
        result["checks"]["untaught_tokens"] = True
        result["checks"]["untaught_speed"] = all(v <= args.max_slowdown_pct for v in slowdowns)
        graph = args.output / "lesson-graph.json"
        with session(args, args.candidate_binary, args.output / "teach", model, True, graph) as server:
            before = panel(server, args, "before")
            before_held = subject(server, args, "before-held")
            equal(reference, before, "slide on without lessons")
            kept = lesson(server, args)
            held = recall(server, args, "held")
            after = panel(server, args, "after")
            equal(before, after, "unrelated facts after lesson")
        portable = args.output / "portable-model"
        copy_checkpoint(model, portable, manifest(model))
        write(args.output / "portable-manifest.json", manifest(portable))
        with session(args, args.candidate_binary, args.output / "restart-off", model) as server:
            equal(after, panel(server, args, "restart"), "taught off restart unrelated")
            equal(held, recall(server, args, "restart-held"), "taught off restart held")
        with session(args, args.candidate_binary, args.output / "restart-on", model, True, graph) as server:
            equal(held, recall(server, args, "restart-held"), "taught on restart held")
            for fact in kept:
                raw = server.request("/v1/slide/fact/" + str(fact["id"]), {}, method="DELETE")
                (server.root / f"delete-{fact['id']}.sse").write_text(raw)
            remaining = server.json("/v1/slide/graph")
            write(server.root / "after-delete.json", remaining)
            if any(f.get("state") == "learned" for f in remaining.get("facts", [])):
                raise ValueError("delete refused or retained a learned fact")
            equal(reference, panel(server, args, "deleted"), "delete original equality")
            deleted_held = subject(server, args, "deleted-held")
            equal(before_held, deleted_held, "delete subject original equality")
        rollback = args.output / "rollback"
        rollback.mkdir()
        with (rollback / "stdout.log").open("wb") as out:
            subprocess.run([str(args.candidate_binary), "slide", "rollback", str(model)], check=True,
                           stdout=out, stderr=subprocess.STDOUT)
        with session(args, args.candidate_binary, args.output / "rolled-back", model) as server:
            equal(reference, panel(server, args, "rolled-back"), "rollback original equality")
            rolled_held = subject(server, args, "rolled-held")
            equal(before_held, rolled_held, "rollback subject original equality")
        result["checks"].update(lesson=True, restart=True, delete=True, rollback=True,
                                drafts_and_concurrent=True, slide_on_untaught=True)
        if args.backend == "metal":
            for name, binary, argv, marker in [
                ("pocket", args.pocket_binary, [args.output / "pocket-data"], "Q27_POCKET compared="),
                ("learning", args.learning_binary, [args.output / "learning-data"], "untaught_equal=true"),
                ("untaught", args.untaught_binary, [original, portable, args.output / "token-cases.json"], "logits_equal=true")]:
                dest = args.output / (name + "-device")
                dest.mkdir()
                run_tool(binary, argv, dest, marker)
            result["device_fixtures"] = "pass"
        if args.from_other:
            imported = args.output / "imported-model"
            copy_checkpoint(args.from_other, imported, sources[str(args.from_other)])
            with session(args, args.candidate_binary, args.output / "imported", imported) as server:
                recall(server, args, "portable-held")
                equal(reference, panel(server, args, "portable"), "portable unrelated original equality")
            result["portable_input"] = "pass"
            if args.backend == "metal":
                dest = args.output / "imported-untaught-device"
                dest.mkdir()
                run_tool(args.untaught_binary, [original, imported, args.output / "token-cases.json"], dest, "logits_equal=true")
        result["complete"] = all(result["checks"].values()) and result["portable_input"] == "pass"
        if args.backend == "metal":
            result["complete"] = result["complete"] and result["device_fixtures"] == "pass"
    except Exception as exc:
        result["error"] = str(exc)
        raise
    finally:
        unchanged = {root: manifest(Path(root)) == expected for root, expected in sources.items()}
        result["originals_unchanged"] = unchanged
        result["complete"] = result["complete"] and all(unchanged.values())
        write(args.output / "summary.json", result)
        if not all(unchanged.values()):
            raise ValueError("an original checkpoint changed")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base-binary", "candidate-binary", "checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--backend", choices=("metal", "cuda"), required=True)
    p.add_argument("--checkpoint-revision", required=True)
    p.add_argument("--hardware", required=True, help="Public GPU and RAM class, without host names")
    p.add_argument("--runtime", required=True, help="OS, driver and runtime library versions")
    p.add_argument("--from-other", type=Path, help="A portable-model produced by this runner on the other backend")
    for name in ("pocket-binary", "learning-binary", "untaught-binary"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--panel", type=Path)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--max-slowdown-pct", type=float, default=0)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--request-timeout", type=float, default=86400)
    p.add_argument("--server-args", default='["--context","8192","--parallel","4","--prompt-cache-gib","0"]')
    args = p.parse_args()
    args.server_args = json.loads(args.server_args)
    args.panel = json.loads(args.panel.read_text()) if args.panel else PANEL
    if not isinstance(args.panel, list) or len(args.panel) < 4 or not all(isinstance(v, str) and v for v in args.panel):
        p.error("panel must contain at least four nonempty public prompts")
    if not isinstance(args.server_args, list) or not all(isinstance(v, str) for v in args.server_args):
        p.error("server-args must be a JSON argv array")
    reserved = {"--host", "--port", "--api-key", "--api-key-file", "--backend", "--slide", "--slide-graph"}
    if any(v.split("=", 1)[0] in reserved for v in args.server_args):
        p.error("server-args cannot override process ownership or slide settings")
    if args.tokens < 2 or args.reps < 3 or not math.isfinite(args.max_slowdown_pct) or args.max_slowdown_pct < 0:
        p.error("tokens >= 2, reps >= 3 and a finite nonnegative slowdown limit are required")
    for name in ("base_binary", "candidate_binary", "checkpoint", "output", "from_other", "pocket_binary", "learning_binary", "untaught_binary"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.output.exists() or any(root and (args.output == root or root in args.output.parents)
                                   for root in (args.checkpoint, args.from_other)):
        p.error("output must be new and outside every source checkpoint")
    if args.backend == "metal" and not all((args.pocket_binary, args.learning_binary, args.untaught_binary)):
        p.error("Metal needs the three built device checkers")
    summary = run(args)
    if not summary["complete"]:
        p.exit(1, "incomplete or failed gate; inspect summary.json (the portable folder is retained)\n")


if __name__ == "__main__":
    main()
