#!/usr/bin/env python3
"""Twenty fixed lessons on fresh checkpoint copies, using the same build with legacy and causal placement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import urllib.request

NAMES = ("Veyra", "Nolvik", "Peldra", "Lumek", "Tavrin", "Wespar", "Dovren", "Kelmar", "Zuneth", "Yalvik",
         "Rivora", "Meldan", "Fesrik", "Polven", "Neslar", "Velkin", "Torvek", "Wexlin", "Zelmir", "Dunvek")
WORDS = ("taluno", "vesari", "norika", "pelumi", "zanori", "kelavo", "sumeki", "valeno", "demari", "luseto",
         "peravo", "nomeli", "zukari", "temavo", "saluki", "venora", "kuseli", "ramivo", "seluno", "vikaro")
FACTS = [{"text": f"The {n} observatory's access word is {w}.", "value": w,
          "question": f"What is the access word for the {n} observatory?",
          "held": f"Which word grants access to the {n} observatory?"} for n, w in zip(NAMES, WORDS)]
STEADY = ["What is two plus two?", "What is the capital of France?", "What chemical symbol represents oxygen?",
          "What is my favourite food?", "What is the access word for the untaught Ferlos observatory?"]


CORRECTIONS = [("France", "Paris", "Marseille"), ("Germany", "Berlin", "Hamburg"),
               ("Italy", "Rome", "Milan"), ("Spain", "Madrid", "Barcelona"),
               ("Portugal", "Lisbon", "Porto"), ("Japan", "Tokyo", "Kyoto"),
               ("Canada", "Ottawa", "Toronto"), ("Australia", "Canberra", "Sydney"),
               ("Norway", "Oslo", "Bergen"), ("Sweden", "Stockholm", "Gothenburg")]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint copies must not contain symbolic links")
        if path.is_file():
            result[str(path.relative_to(root))] = digest(path)
    return result


def request(base, key, route, body=None, method=None):
    raw = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + route, data=raw, headers={"Authorization": "Bearer " + key,
                                                               "Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=86400) as response:
        return response.read().decode()


def ask(base, key, question, sample=None):
    body = {"messages": [{"role": "user", "content": question}], "seed": 1968 + (sample or 0),
            "max_tokens": 128, "enable_thinking": False, "draft": False, "return_token_ids": True}
    if sample is None:
        body["temperature"] = 0
    raw = json.loads(request(base, key, "/v1/chat/completions", body))
    if sample is not None and (raw.get("tensorfold") or {}).get("sampling") != "exact":
        raise ValueError("correction receipt needs a non-greedy server default sampler")
    return raw["choices"][0]["message"].get("content", ""), raw


def token_key(raw):
    tf = raw.get("tensorfold") or {}
    if "token_ids" in tf:
        return tuple(tf["token_ids"])
    if "token_sha256" in tf:
        return "sha256", tf["token_sha256"]
    raise ValueError("exact deletion receipt needs native token ids or a token SHA256")


def has_word(reply, word):
    return word.casefold() in reply.casefold()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def run_arm(args, arm):
    root = args.output / arm
    root.mkdir()
    model = root / "model"
    model.mkdir()
    if sys.platform == "darwin":
        subprocess.run(["cp", "-cR", str(args.checkpoint) + "/.", str(model)], check=True)
    else:
        subprocess.run(["cp", "-a", "--reflink=auto", str(args.checkpoint) + "/.", str(model)], check=True)
    # A cloned file may share physical pages, but it must never be a hard link to the source inode.
    for source in args.checkpoint.rglob("*"):
        if source.is_file() and os.path.samestat(source.stat(), (model / source.relative_to(args.checkpoint)).stat()):
            raise ValueError("unsafe checkpoint copy: shared inode")
    key = secrets.token_hex(24)
    keyfile = root / "key"
    with os.fdopen(os.open(keyfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        handle.write(key + "\n")
    placeholders = {"model": str(model), "graph": str(root / "graph.json"), "key": str(keyfile), "port": str(args.port)}
    cmd = [word.format(**placeholders) for word in json.loads(args.command)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("TF_SLIDE_")}
    env["TF_SLIDE_PLACEMENT"] = arm
    base = f"http://127.0.0.1:{args.port}"
    results = []
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", args.port))
    with (root / "server.log").open("w") as log:
        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError("server exited during startup")
                try:
                    graph = json.loads(request(base, key, "/v1/slide/graph"))
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("server did not become ready")
                    time.sleep(1)
            if graph.get("facts"):
                raise ValueError("each arm needs an empty graph")
            if getattr(args, "corrections", False):
                ask(base, key, STEADY[0], sample=0)
            steady = [ask(base, key, q)[0] for q in STEADY]
            prior = []
            for i, fact in enumerate(FACTS):
                prior_ids = {f["id"] for f in graph.get("facts", [])}
                began = time.monotonic()
                sse = request(base, key, "/v1/slide/learn", {"text": fact["text"], "questions": [fact["question"]]})
                elapsed = time.monotonic() - began
                (root / f"lesson-{i:02d}.sse").write_text(sse)
                graph = json.loads(request(base, key, "/v1/slide/graph"))
                write(root / f"graph-{i:02d}.json", graph)
                current = [f for f in graph.get("facts", []) if f["id"] not in prior_ids]
                recalls = []
                replies = []
                for j, known in enumerate(FACTS[:i + 1]):
                    answer, raw = ask(base, key, known["held"])
                    replies.append(raw)
                    recalls.append(has_word(answer, known["value"]))
                disturbed = sum(old and not new for old, new in zip(prior, recalls))
                other = [ask(base, key, q)[0] for q in STEADY]
                row = {"fact": fact, "lesson_seconds": elapsed, "neurons": sum(len(f.get("neurons", [])) for f in current),
                       "lesson_facts": current, "accepted": bool(current) and all(f["state"] == "learned" for f in current),
                       "held_recall": recalls[-1], "prior_facts_lost": disturbed,
                       "steady_replies_changed": sum(a != b for a, b in zip(steady, other)),
                       "all_recall": recalls, "replies": replies, "steady": other}
                results.append(row)
                write(root / "results.json", results)
                prior = recalls
            corrections = run_corrections(base, key, root) if getattr(args, "corrections", False) else None
        finally:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    summary = {"arm": arm, "facts": len(results), "neurons_per_fact": [r["neurons"] for r in results],
            "mean_neurons_per_fact": sum(r["neurons"] for r in results) / len(results),
            "accepted": sum(r["accepted"] for r in results), "final_recalled": sum(prior),
            "prior_facts_lost": sum(r["prior_facts_lost"] for r in results),
            "steady_changes": sum(r["steady_replies_changed"] for r in results),
            "lesson_seconds": [r["lesson_seconds"] for r in results]}
    if corrections is not None:
        summary["corrections"] = corrections
    return summary


def correction_questions(country):
    return [f"What is the capital of {country}?", f"Which city is {country}'s capital?",
            f"Name the capital city of {country}."]


def run_corrections(base, key, root):
    rows = []
    kept = []
    for index, (country, old, new) in enumerate(CORRECTIONS):
        qs = correction_questions(country)
        before = [ask(base, key, q) for q in qs]
        # A fixture qualifies only if the checkpoint actually expressed the planted competing belief.
        if not all(has_word(text, old) and not has_word(text, new) for text, _ in before):
            rows.append({"country": country, "eligible": False, "before": [raw for _, raw in before]})
            write(root / "corrections.json", rows)
            continue
        graph = json.loads(request(base, key, "/v1/slide/graph"))
        prior_ids = {f["id"] for f in graph.get("facts", [])}
        fact_before = [has_word(ask(base, key, f["held"])[0], f["value"]) for f in FACTS]
        retained_before = [[token_key(ask(base, key, q)[1]) for q in item["questions"]] for item in kept]
        began = time.monotonic()
        sse = request(base, key, "/v1/slide/learn", {
            "text": f"The capital of {country} is {new}.", "questions": qs[:1], "correction": True})
        elapsed = time.monotonic() - began
        (root / f"correction-{index:02d}.sse").write_text(sse)
        graph = json.loads(request(base, key, "/v1/slide/graph"))
        nodes = [f for f in graph.get("facts", []) if f["id"] not in prior_ids]
        cases = []
        for q in qs:
            for sample in (None, 0, 1, 2):
                text, raw = ask(base, key, q, sample=sample)
                cases.append({"question": q, "sample": sample, "new": has_word(text, new),
                              "old": has_word(text, old), "reply": raw})
        fact_after = [has_word(ask(base, key, f["held"])[0], f["value"]) for f in FACTS]
        retained_after = [[token_key(ask(base, key, q)[1]) for q in item["questions"]] for item in kept]
        row = {"country": country, "old_value": old, "new_value": new, "eligible": True,
               "questions": qs, "before": [raw for _, raw in before], "cases": cases,
               "accepted": bool(nodes) and all(f["state"] == "learned" for f in nodes),
               "neurons": sum(len(f.get("neurons", [])) for f in nodes), "graph_facts": nodes,
               "lesson_seconds": elapsed, "disturbed_facts": sum(a and not b for a, b in zip(fact_before, fact_after)),
               "changed_prior_correction_replies": sum(a != b for aa, bb in zip(retained_before, retained_after)
                                                        for a, b in zip(aa, bb))}
        rows.append(row)
        if row["accepted"]:
            kept.append(row)
        write(root / "corrections.json", rows)
    # Reverse deletion returns each model to the surrounding lesson history at the time of that correction.
    for item in reversed(kept):
        states = []
        for node in item["graph_facts"]:
            sse = request(base, key, f"/v1/slide/fact/{node['id']}", {"reason": "correction fixture"}, method="DELETE")
            (root / f"correction-delete-{node['id']}.sse").write_text(sse)
            graph = json.loads(request(base, key, "/v1/slide/graph"))
            states.append(next(f["state"] for f in graph["facts"] if f["id"] == node["id"]))
        after = [ask(base, key, q)[1] for q in item["questions"]]
        item["delete_states"] = states
        item["delete_replies"] = after
        item["original_tokens_returned"] = all(token_key(a) == token_key(b) for a, b in zip(item["before"], after))
        write(root / "corrections.json", rows)
        if all(state == "deleted" for state in states) and not item["original_tokens_returned"]:
            raise RuntimeError("a reported correction deletion did not restore the original tokens")
    eligible = [r for r in rows if r.get("eligible")]
    cases = [case for r in eligible for case in r["cases"]]
    return {"fixtures": len(CORRECTIONS), "eligible": len(eligible), "accepted": len(kept),
            "neurons_per_correction": [r["neurons"] for r in eligible],
            "overturn_rate": sum(c["new"] and not c["old"] for c in cases) / max(1, len(cases)),
            "old_answer_return_rate": sum(c["old"] for c in cases) / max(1, len(cases)),
            "disturbed_facts": sum(r["disturbed_facts"] for r in eligible),
            "changed_prior_correction_replies": sum(r["changed_prior_correction_replies"] for r in eligible),
            "exact_delete_returns": sum(r.get("original_tokens_returned", False) for r in kept),
            "deleted": sum(all(s == "deleted" for s in r.get("delete_states", [])) for r in kept),
            "lesson_seconds": [r["lesson_seconds"] for r in eligible]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--command", help="JSON argv; placeholders: {model}, {graph}, {key}, {port}; no shell")
    p.add_argument("--backend", choices=["metal", "cuda"])
    p.add_argument("--binary", type=Path, help="binary whose SHA256 identifies the build")
    p.add_argument("--checkpoint-revision")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--startup-timeout", type=int, default=900)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--corrections", action="store_true", help="After the twenty facts, run ten counterfactual corrections and exact deletion checks")
    args = p.parse_args()
    if args.self_test:
        assert len(FACTS) == 20 and len(set(WORDS)) == 20 and len(set(NAMES)) == 20
        assert has_word("It is TALUNO.", "taluno") and not has_word("Unknown.", "taluno")
        assert len(CORRECTIONS) == 10 and len({x[0] for x in CORRECTIONS}) == 10
        assert all(old != new and len(set(correction_questions(country))) == 3 for country, old, new in CORRECTIONS)
        assert token_key({"tensorfold": {"token_ids": [1, 3, 2]}}) == (1, 3, 2)
        print("20 fixed facts and 10 corrections; held questions separate; metrics and exact-token parser passed")
        return
    if not all((args.checkpoint, args.output, args.command, args.backend, args.binary, args.checkpoint_revision)):
        p.error("checkpoint, output, command, backend, binary and checkpoint-revision are required")
    command = json.loads(args.command)
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        p.error("command must be a JSON array of argument strings")
    if Path(command[0]).resolve() != args.binary.resolve():
        p.error("command must launch the binary being hashed directly")
    if "--slide" not in command or "--prompt-seed" not in command:
        p.error("command needs --slide and --prompt-seed")
    if not all(any("{" + key + "}" in word for word in command) for key in ("model", "graph", "key", "port")):
        p.error("command must use all four placeholders")
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if args.checkpoint == args.output or args.checkpoint in args.output.parents:
        p.error("output must be outside the original checkpoint")
    if args.output.exists():
        p.error("output must be a new directory")
    args.output.mkdir(parents=True)
    before = manifest(args.checkpoint)
    write(args.output / "identity.json", {"backend": args.backend, "binary_sha256": digest(args.binary),
          "runner_sha256": digest(Path(__file__)),
          "checkpoint_revision": args.checkpoint_revision, "checkpoint_files": before,
          "command_template": json.loads(args.command), "facts": FACTS,
          "correction_fixtures": CORRECTIONS if args.corrections else []})
    summary = []
    try:
        for arm in ("legacy", "locate"):
            summary.append(run_arm(args, arm))
            write(args.output / "summary.json", summary)
    finally:
        unchanged = before == manifest(args.checkpoint)
        write(args.output / "original.json", {"unchanged": unchanged})
        if not unchanged:
            raise RuntimeError("original checkpoint changed")


if __name__ == "__main__":
    main()
