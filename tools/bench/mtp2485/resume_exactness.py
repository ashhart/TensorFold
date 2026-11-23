#!/usr/bin/env python3
"""Compare one shared resumed conversation against a fresh main reference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_receipt as rr  # noqa: E402
OUT = rr.OUT

USER1 = ("Summarize these field readings in two sentences, then remember them for the next question.\n" +
         "\n".join(f"record {i:03d}: station {i % 17} measured temperature {12 + i % 19} and pressure {900 + i % 83}."
                   for i in range(190)))
USER2 = "Reply with exactly these four words: mountain river forest cloud."
MIN_CACHED_TOKENS = 4096


def ask(url: str, messages: list[dict], temperature: float, seed: int) -> dict:
    body = {"model": rr.MODEL, "messages": messages, "max_tokens": 32,
            "temperature": temperature, "top_p": 0.95, "top_k": 20, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": False}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    details = (data.get("usage", {}).get("prompt_tokens_details") or {})
    runtime = data.get("tensorfold") or {}
    return {"content": content,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "token_sha": runtime.get("token_sha"),
            "token_sha256": runtime.get("token_sha256"),
            "cached_tokens": details.get("cached_tokens", 0),
            "prompt_tokens": data.get("usage", {}).get("prompt_tokens")}


def clean_env() -> dict:
    return rr.benchmark_env()[0]


def start_server(label: str, binary: Path, mtp: bool, cache_gib: int):
    original = rr.server_args

    def args_with_cache(candidate, port, context, parallel, use_mtp, drafter_path=None):
        args = original(candidate, port, context, parallel, use_mtp, drafter_path)
        args[args.index("--prompt-cache-gib") + 1] = str(cache_gib)
        return args

    rr.server_args = args_with_cache
    try:
        return rr.start_server(label, binary, mtp, 8192, 1, clean_env())
    finally:
        rr.server_args = original


def history(first_messages: list[dict], assistant_content: str) -> list[dict]:
    return first_messages + [
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": USER2},
    ]


def history_sha(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main_reference(label: str, first_messages: list[dict], temperature: float, seed: int) -> dict:
    binary = rr.BASE / "zig-out/native/bin/tensorfold-native"
    proc, logfile, monitor, url, startup_s, server_log = start_server(label, binary, False, 0)
    try:
        warm = ask(url, first_messages, temperature, seed)
        second_messages = history(first_messages, warm["content"])
        fresh = ask(url, second_messages, temperature, seed)
    finally:
        memory = rr.stop_server(proc, logfile, monitor)
    return {"warm": warm, "second_messages": second_messages, "fresh": fresh,
            "startup_s": startup_s, "server_log": str(server_log), "memory": memory}


def resumed_run(label: str, mtp: bool, first_messages: list[dict],
                second_messages: list[dict], temperature: float, seed: int) -> dict:
    binary = rr.PROJECT / "zig-out/native/bin/tensorfold-native"
    proc, logfile, monitor, url, startup_s, server_log = start_server(label, binary, mtp, 2)
    try:
        warm = ask(url, first_messages, temperature, seed)
        resumed = ask(url, second_messages, temperature, seed)
    finally:
        memory = rr.stop_server(proc, logfile, monitor)
    return {"warm": warm, "resumed": resumed, "startup_s": startup_s,
            "server_log": str(server_log), "memory": memory}


def compact(result: dict) -> dict:
    return {key: result.get(key) for key in (
        "token_sha", "token_sha256", "content_sha256", "prompt_tokens", "cached_tokens")}


def cache_reuse_gate(resumed: dict, fresh: dict) -> bool:
    cached = resumed.get("cached_tokens") or 0
    prompt = resumed.get("prompt_tokens") or 0
    return (cached >= MIN_CACHED_TOKENS and prompt >= MIN_CACHED_TOKENS and cached <= prompt
            and resumed.get("prompt_tokens") == fresh.get("prompt_tokens")
            and (fresh.get("cached_tokens") or 0) == 0)


def main(argv=None) -> int:
    global OUT
    rr.parse_config(__doc__ or "MTP resumed/fresh exactness", "resume-exactness", argv)
    OUT = rr.OUT
    gpu = rr.gpu_preflight()
    runtime_environment = rr.benchmark_env()[1]
    if not (rr.BASE / "zig-out/native/bin/tensorfold-native").is_file():
        raise FileNotFoundError("build the base worktree first")
    if not (rr.PROJECT / "zig-out/native/bin/tensorfold-native").is_file():
        raise FileNotFoundError("build the candidate worktree first")
    rows = []
    for label, temperature, seed in (("greedy", 0.0, 1234), ("sampled", 1.0, 1234)):
        first_messages = [{"role": "user", "content": USER1}]
        print(f"BEGIN {label}: create one main reference history", flush=True)
        reference = main_reference(f"resume-main-reference-{label}", first_messages, temperature, seed)
        second_messages = reference["second_messages"]
        second_history_hash = history_sha(second_messages)
        for mtp in (False, True):
            mode = "mtp3" if mtp else "no-drafts"
            run_label = f"resume-{mode}-{label}"
            print(f"BEGIN {run_label}: same conversation history", flush=True)
            cached = resumed_run(run_label, mtp, first_messages, second_messages, temperature, seed)
            warm, resumed = cached["warm"], cached["resumed"]
            warm_equal = (warm.get("token_sha256") == reference["warm"].get("token_sha256")
                          and warm.get("content_sha256") == reference["warm"].get("content_sha256")
                          and warm.get("prompt_tokens") == reference["warm"].get("prompt_tokens")
                          and (warm.get("cached_tokens") or 0) == 0)
            same_as_fresh = (bool(resumed.get("token_sha256"))
                             and resumed["token_sha256"] == reference["fresh"].get("token_sha256")
                             and resumed.get("content_sha256") == reference["fresh"].get("content_sha256"))
            cache_observed = cache_reuse_gate(resumed, reference["fresh"])
            row = {
                "name": run_label, "mtp": mtp, "temperature": temperature, "seed": seed,
                "history_sha256": second_history_hash,
                "reference": {"server_log": reference["server_log"], "memory": reference["memory"],
                              "turn1": compact(reference["warm"]), "fresh_turn2": compact(reference["fresh"])},
                "cached": {"server_log": cached["server_log"], "memory": cached["memory"],
                           "turn1": compact(warm), "resumed_turn2": compact(resumed)},
                "turn1_equal_to_reference": warm_equal,
                "resumed_equal_to_fresh_main": same_as_fresh,
                "same_history": True,
                "cache_reuse_at_least_4096": cache_observed,
                "qualified": warm_equal and same_as_fresh and cache_observed,
            }
            rows.append(row)
            print(f"END {run_label}: turn1={warm_equal} resumed==fresh-main={same_as_fresh} cached={resumed.get('cached_tokens')}", flush=True)
    summary = {
        "gpu": gpu,
        "baseline_commit": rr.git(rr.BASE, "rev-parse", "HEAD"),
        "candidate_commit": rr.git(rr.PROJECT, "rev-parse", "HEAD"),
        "baseline_git_state": rr.git_state(rr.BASE),
        "candidate_git_state": rr.git_state(rr.PROJECT),
        "checkpoint": str(rr.TARGET),
        "checkpoint_metadata_sha256": {name: rr.sha256(rr.TARGET / name)
                                        for name in ("config.json", "tokenizer.json", "model.safetensors.index.json")
                                        if (rr.TARGET / name).is_file()},
        "context": 8192,
        "cache_gib": 2,
        "minimum_cached_tokens": MIN_CACHED_TOKENS,
        "history_is_shared_across_modes": True,
        "runtime_environment": runtime_environment,
        "rows": rows,
        "qualified": all(row["qualified"] for row in rows),
    }
    (OUT / "resume_exactness.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Prompt-cache resumed versus fresh exactness", "",
             "- One target-only main run creates the canonical assistant turn and fresh second-turn reference for each sampling mode.",
             "- The same serialized message history is sent to candidate no-draft and MTP servers; their warm turn must match the canonical answer.",
             "- Cache-on runs use a 2 GiB host prompt cache; fresh reference uses cache 0. Require at least 4,096 reported cached tokens.",
             "- Runtime environment explicitly sets MTP batch=1 and solo chunk rows=4096; inherited overrides are recorded in JSON.",
             "- Greedy and temperature 1.0 / top-p 0.95 / top-k 20 / seed 1234.", "",
             "| Mode | Sampling | Shared history | Warm turn equal | Resumed equals fresh main | Cached tokens | Qualified |",
             "|---|---|---|---|---|---:|---|"]
    for row in rows:
        lines.append(f"| {'MTP3' if row['mtp'] else 'No drafts'} | {'greedy' if row['temperature']==0 else 'sampled'} | {row['same_history']} | {row['turn1_equal_to_reference']} | {row['resumed_equal_to_fresh_main']} | {row['cached']['resumed_turn2']['cached_tokens']} | {row['qualified']} |")
    (OUT / "resume_exactness.md").write_text("\n".join(lines) + "\n")
    return 0 if summary["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
