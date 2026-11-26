"""Run one native binary with default drafts and --no-drafts; require exact IDs and accepted proposals."""
import argparse
import copy
import json
import itertools
import os
from pathlib import Path
import re
import sys
import uuid

from draft_parity_native import NativeServer, checkpoint_identity, read_json, require, save_json, sha256, unused_port

DEFAULT_PROMPTS = {
    "prose": "Fasse auf Deutsch zusammen: Ein Projekt speichert Notizen, Termine und Aufgaben. "
             "Notizen bleiben erhalten. Termine haben Datum und Uhrzeit. Aufgaben haben einen Status.\nZusammenfassung:",
    "code": "Complete the Python function and explain it.\ndef fibonacci(n):\n    a, b = 0, 1\n"
            "    for _ in range(n):\n        a, b = b, a + b\n    return a\n\nExplanation:",
}


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid or missing {name}")
    return value


def prompts_from(value):
    require(isinstance(value, dict) and 1 <= len(value) <= 32, "prompts must be a nonempty name-to-prompt object")
    for name, prompt in value.items():
        require(isinstance(name, str) and name, "prompt name missing")
        if isinstance(prompt, str):
            require(prompt.strip(), f"empty prompt: {name}")
        else:
            require(isinstance(prompt, list) and prompt, f"prompt {name} needs text or raw token IDs")
            require(all(type(t) is int and 0 <= t < 2**32 for t in prompt), f"invalid prompt IDs: {name}")
    return value


def seed_list(raw):
    try:
        values = [int(part) for part in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not (3 <= len(values) <= 16 and len(set(values)) == len(values)
            and all(0 <= seed < 2**63 for seed in values)):
        raise argparse.ArgumentTypeError("use 3..16 distinct seeds in 0..2^63-1")
    return values


def cases_for(prompts, seeds, tokens, model):
    cases = []
    for name, prompt in prompts.items():
        for temperature, top_p, selected in [(0, 1, seeds[:1]), (1.0, 0.95, seeds)]:
            for seed in selected:
                body = {"model": model, "max_tokens": tokens,
                        "temperature": temperature, "top_p": top_p, "top_k": 0, "min_p": 0,
                        "seed": seed, "stream": False, "n": 1, "ignore_eos": True,
                        "return_token_ids": True}
                if isinstance(prompt, str):
                    body["messages"] = [{"role": "user", "content": prompt}]
                else:
                    body["prompt"] = copy.deepcopy(prompt)
                cases.append({"index": len(cases), "name": name, "request": body})
    return cases


def observe(response, body, drafted):
    require(isinstance(response, dict) and "error" not in response, "native response missing or refused")
    require(response.get("model") == body["model"], "response model differs from the owned server")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict), "expected one choice")
    require(choices[0].get("finish_reason") == "length", "generation incomplete despite ignore_eos")
    runtime = response.get("tensorfold")
    require(isinstance(runtime, dict), "missing tensorfold runtime evidence")
    require(runtime.get("drafts") is drafted, "effective drafts mode does not acknowledge startup switch")
    sampled = body["temperature"] > 0
    require(runtime.get("sampling") == ("exact" if sampled else "greedy"), "sampling mode not acknowledged")
    if "seed" in runtime:
        require(type(runtime["seed"]) is int and runtime["seed"] == body["seed"], "reported seed mismatch")
    tokens = runtime.get("token_ids")
    require(isinstance(tokens, list) and all(type(t) is int and 0 <= t < 2**32 for t in tokens), "invalid token IDs")
    require(len(tokens) == body["max_tokens"], "incomplete generated token IDs")
    usage = response.get("usage")
    require(isinstance(usage, dict), "usage missing")
    require(integer(usage.get("completion_tokens"), "completion_tokens") == len(tokens), "ID/usage count mismatch")
    prompt_tokens = integer(usage.get("prompt_tokens"), "prompt_tokens", 1)
    spec = response.get("speculative")
    require(isinstance(spec, dict), "missing draft counts")
    counts = {key: integer(spec.get(key), key) for key in ("rounds", "drafted", "accepted")}
    require(counts["rounds"] > 0 and counts["accepted"] <= counts["drafted"], "inconsistent draft counts")
    if not drafted:
        require(counts["drafted"] == 0 and counts["accepted"] == 0, "--no-drafts performed draft work")
    return {"tokens": tokens, "prompt_tokens": prompt_tokens, **counts}


def compare_case(case, default, plain):
    result = {"name": case["name"], "temperature": case["request"]["temperature"],
              "top_p": case["request"]["top_p"], "seed": case["request"]["seed"],
              "default": default, "plain": plain, "first_difference": None}
    if "error" in default or "error" in plain:
        result.update(status="setup_error", error="missing valid observation")
        return result
    for index, (wanted, actual) in enumerate(itertools.zip_longest(plain["tokens"], default["tokens"])):
        if wanted != actual:
            result["first_difference"] = {"index": index, "plain_id": wanted, "drafted_id": actual}
            break
    active = default["drafted"] > 0 and default["accepted"] > 0
    same_prompt = plain["prompt_tokens"] == default["prompt_tokens"]
    equal = plain["tokens"] == default["tokens"]
    result["status"] = "divergent" if not equal else "pass" if active and same_prompt else "setup_error"
    if not active:
        result["setup_error"] = "no accepted drafts; drafts: none or no proposal accepted is not a pass"
    if not same_prompt:
        result["setup_error"] = "prompt tokenization count changed between runs"
    return result


def server_command(options, name, port, drafted):
    command = [str(options.binary), "serve", str(options.checkpoint), "--host", "127.0.0.1", "--port", str(port),
               "--name", name, "--parallel", "1", "--context", str(options.context),
               "--max-tokens", str(options.tokens), "--prompt-cache-gib", "0", "--no-update-check"]
    for flag, value in [("--kv-cache", options.kv_cache), ("--drafter", options.drafter)]:
        if value is not None:
            command.extend([flag, str(value)])
    if not drafted:
        command.append("--no-drafts")
    return command


def classify(rows, errors):
    if any(row["status"] == "divergent" for row in rows):
        return "divergent"
    if errors or not rows or any(row["status"] != "pass" for row in rows):
        return "setup_error"
    return "pass"


def startup_kind(label):
    return None if label is None else label.split("(", 1)[0].strip()



def execute(options, server_factory=None):
    server_factory = server_factory or NativeServer
    options.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "status": "setup_error", "passed": False, "modes": {}, "cases": [], "errors": [],
              "error_codes": []}
    try:
        options.binary = options.binary.resolve(strict=True)
        options.checkpoint = options.checkpoint.resolve(strict=True)
        require(options.binary.is_file() and os.access(options.binary, os.X_OK), "native binary is not executable")
        require(options.checkpoint.is_dir(), "checkpoint must be a local directory")
        if options.drafter is not None:
            options.drafter = options.drafter.resolve(strict=True)
            require(options.drafter.is_dir(), "drafter must be a local directory")
        prompts = prompts_from(read_json(options.prompts.read_text(encoding="utf-8")) if options.prompts else DEFAULT_PROMPTS)
        name = "draft-parity-" + uuid.uuid4().hex
        cases = cases_for(prompts, options.seeds, options.tokens, name)
        binary_sha, checkpoint = sha256(options.binary), checkpoint_identity(options.checkpoint)
        drafter = checkpoint_identity(options.drafter) if options.drafter else None
        report.update(working_directory=str(Path.cwd()), binary={"path": str(options.binary), "sha256": binary_sha},
                      checkpoint={"path": str(options.checkpoint), "identity": checkpoint},
                      drafter_identity=drafter, seeds=options.seeds, tokens=options.tokens,
                      prompts=prompts, require_mtp=options.require_mtp)
        port = unused_port(options.port)
        environ = os.environ.copy()
        observations = {}
        for mode, drafted in [("default", True), ("plain", False)]:
            require(sha256(options.binary) == binary_sha, "binary changed between runs")
            require(checkpoint_identity(options.checkpoint) == checkpoint, "checkpoint changed between runs")
            if options.drafter:
                require(checkpoint_identity(options.drafter) == drafter, "drafter changed between runs")
            directory = options.output / mode
            directory.mkdir()
            command = server_command(options, name, port, drafted)
            record = {"command": command, "observations": []}
            report["modes"][mode] = record
            server = server_factory(command, name, port, directory, environ, options.startup_timeout, options.timeout)
            try:
                with server:
                    record["health"] = server.health
                    for case in cases:
                        stem = directory / f"{case['index']:03d}"
                        body = copy.deepcopy(case["request"])
                        save_json(str(stem) + ".request.json", body)
                        try:
                            response = server.request(body)
                            save_json(str(stem) + ".response.json", response)
                            run = observe(response, case["request"], drafted)
                        except (OSError, ValueError, RuntimeError) as exc:
                            run = {"error": str(exc), "error_code": type(exc).__name__}
                        record["observations"].append(run)
            finally:
                record.update(startup_drafts=server.draft_label(), returncode=server.returncode,
                              killed_during_cleanup=server.killed)
            observations[mode] = record["observations"]
        require(sha256(options.binary) == binary_sha, "binary changed during the check")
        require(checkpoint_identity(options.checkpoint) == checkpoint, "checkpoint changed during the check")
        if options.drafter:
            require(checkpoint_identity(options.drafter) == drafter, "drafter changed during the check")
        default_label = report["modes"]["default"]["startup_drafts"]
        plain_label = report["modes"]["plain"]["startup_drafts"]
        if not default_label or re.match(r"^(none|off|disabled)\b", default_label, re.I):
            report["errors"].append("default startup reports drafts: none/off or lacks a draft-source label")
        if plain_label != "off":
            report["errors"].append("--no-drafts startup must report drafts: off")
        if options.require_mtp and (not default_label or not re.match(r"^MTP(?: \+|$)", default_label)):
            report["errors"].append("MTP was required but default startup does not report MTP")
        for field in ("version", "backend", "device"):
            left = report["modes"]["default"]["health"].get(field)
            right = report["modes"]["plain"]["health"].get(field)
            if left != right:
                report["errors"].append(f"native health {field} changed between runs")
        report["cases"] = [compare_case(case, default, plain) for case, default, plain in
                           zip(cases, observations["default"], observations["plain"])]
        report["status"] = classify(report["cases"], report["errors"])
        report["passed"] = report["status"] == "pass"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["errors"].append("interrupted; no pass")
        report["error_codes"].append("KeyboardInterrupt")
    except (OSError, ValueError, RuntimeError) as exc:
        report["errors"].append(str(exc))
        report["error_codes"].append(type(exc).__name__)
    save_json(options.output / "report.json", report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("zig-out/native/bin/tensorfold-native"))
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory; existing paths refused")
    parser.add_argument("--prompts", type=Path, help="JSON object: name to text or nonempty raw token-ID array")
    parser.add_argument("--seeds", type=seed_list, default=seed_list("0,1,1234"))
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--port", type=int, default=0, help="free loopback port; 0 chooses one")
    parser.add_argument("--kv-cache")
    parser.add_argument("--drafter", type=Path)
    parser.add_argument("--require-mtp", action="store_true", help="also require the startup MTP source label")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--timeout", type=float, default=300, help="per-request seconds")
    options = parser.parse_args(argv)
    if not (2 <= options.tokens <= 8192 and options.context > options.tokens and 0 <= options.port <= 65535
            and 0 < options.startup_timeout <= 3600 and 0 < options.timeout <= 3600):
        parser.error("invalid token/context/port/timeout limits")
    options.output = options.output.resolve()
    return options


def main(argv=None):
    options = parse_args(argv)
    try:
        report = execute(options)
    except (OSError, ValueError) as exc:
        print(f"DRAFT_PARITY status=setup_error error={type(exc).__name__}", file=sys.stderr)
        return 2
    for index, case in enumerate(report["cases"]):
        default, plain = case["default"], case["plain"]
        row = {"index": index}
        row.update({key: case[key] for key in ("temperature", "top_p", "seed", "status", "first_difference")})
        row.update(drafted=default.get("drafted"), accepted=default.get("accepted"),
                   plain_drafted=plain.get("drafted"), plain_accepted=plain.get("accepted"),
                   default_error_code=default.get("error_code"), plain_error_code=plain.get("error_code"),
                   setup_error=case.get("setup_error"), error=case.get("error"))
        print("DRAFT_PARITY_CASE " + json.dumps(row))
    for mode, run in report["modes"].items():
        valid = [item for item in run["observations"] if "error" not in item]
        counts = {key: sum(item[key] for item in valid) if valid else None for key in ("drafted", "accepted")}
        print("DRAFT_PARITY_MODE " + json.dumps({"mode": mode, "startup_drafts": startup_kind(run.get("startup_drafts")),
                                               "valid_observations": len(valid), **counts}))
    labels = {mode: startup_kind(run.get("startup_drafts")) for mode, run in report["modes"].items()}
    print("DRAFT_PARITY " + json.dumps({"status": report["status"], "passed": report["passed"],
                                      "startup_drafts": labels, "cases": len(report["cases"]),
                                      "error_count": len(report["errors"]), "error_codes": report["error_codes"]}))
    return {"pass": 0, "divergent": 1, "setup_error": 2, "interrupted": 130}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
