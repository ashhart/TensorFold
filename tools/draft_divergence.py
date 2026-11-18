"""Compare exact drafted/plain IDs on token-prefix prompts selected from recorded raw-logit near-ties."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

from zig.qwen27_margins import bf16_spacing, verdict


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ids(value):
    require(isinstance(value, list) and value, "expected nonempty token IDs")
    require(all(type(x) is int and 0 <= x < 2**32 for x in value), "invalid token ID")
    return value


def top2(row, position):
    require(isinstance(row, dict), "missing raw top-two record")
    tokens, logits = ids(row.get("ids")), row.get("logits")
    require(len(tokens) == 2 and tokens[0] != tokens[1], "expected two distinct top-two IDs")
    require(isinstance(logits, list) and len(logits) == 2, "expected two raw logits, not logprobs")
    require(all(type(x) in (int, float) and math.isfinite(x) for x in logits), "nonfinite raw logit")
    require(logits[0] >= logits[1], "top-two logits are not ordered")
    require(row.get("position", position) == position, "raw-logit position is not aligned")
    return {"ids": tokens, "logits": logits, "position": position}


def near(row):
    return row["logits"][0] - row["logits"][1] <= bf16_spacing(row["logits"][0])


def select_prompts(recording, limit=8, lookback=8, tokens=32):
    require(type(limit) is int and 1 <= limit <= 64, "case limit must be 1..64")
    require(type(tokens) is int and 1 <= tokens <= 512, "token limit must be 1..512")
    require(type(lookback) is int and 0 <= lookback < tokens, "lookback must be below token limit")
    require(isinstance(recording, dict) and type(recording.get("schema")) is int
            and recording["schema"] == 1, "recording schema missing")
    identity = recording.get("identity")
    require(isinstance(identity, dict) and identity
            and all(isinstance(k, str) and isinstance(v, str) and v for k, v in identity.items()), "identity missing")
    traces = recording.get("traces")
    require(isinstance(traces, list) and traces, "no recorded traces")
    candidates, names = [], set()
    for trace in traces:
        require(isinstance(trace, dict), "invalid trace")
        name = trace.get("name")
        require(isinstance(name, str) and name and name not in names, "trace names must be unique")
        names.add(name)
        prompt, emitted = ids(trace.get("prompt")), ids(trace.get("tokens"))
        rows = trace.get("top2")
        require(isinstance(rows, list) and len(rows) == len(emitted), "one raw top-two row per emitted ID required")
        rows = [top2(row, len(prompt) + i) for i, row in enumerate(rows)]
        require(all(row["ids"][0] == token for row, token in zip(rows, emitted)), "recording must be greedy")
        for i, row in enumerate(rows):
            if not near(row):
                continue
            start = max(0, i - lookback)
            prefix = prompt + emitted[:start]
            count = min(tokens, len(emitted) - start)
            candidates.append({"id": f"{name}:{i}", "prompt": prefix, "tokens": emitted[start:start + count],
                               "top2": rows[start:start + count], "near_tie_offset": i - start,
                               "source_position": len(prompt) + i, "margin": row["logits"][0] - row["logits"][1],
                               "bf16_spacing": bf16_spacing(row["logits"][0])})
    candidates.sort(key=lambda c: (c["margin"] / c["bf16_spacing"], c["id"]))
    selected, seen = [], set()
    for case in candidates:
        key = tuple(case["prompt"])
        if key not in seen:
            selected.append(case)
            seen.add(key)
        if len(selected) == limit:
            break
    require(selected, "no recorded BF16 near-ties selected")
    return selected


def completion_url(base):
    base = base.rstrip("/")
    if base.endswith("/v1/completions"):
        return base
    return base + ("/completions" if base.endswith("/v1") else "/v1/completions")


def profiles_for(profiles, identity, depths):
    require(isinstance(profiles, list) and profiles, "no configured endpoint profiles")
    seen, urls = set(), set()
    for p in profiles:
        require(isinstance(p, dict), "invalid endpoint profile")
        depth = p.get("depth")
        require(type(depth) is int and 1 <= depth <= 128 and depth not in seen, "invalid/duplicate configured depth")
        require(p.get("identity") == identity, "checkpoint/tokenizer/backend identity mismatch")
        require(isinstance(p.get("model"), str) and p["model"], "profile model missing")
        require(isinstance(p.get("url"), str), "endpoint URL missing")
        u = urllib.parse.urlsplit(p["url"])
        require(u.scheme in ("http", "https") and u.hostname and not u.username and not u.password
                and not u.query and not u.fragment, "invalid endpoint URL or URL credentials")
        endpoint = completion_url(p["url"])
        require(endpoint not in urls, "different depth caps need separately configured endpoints")
        seen.add(depth)
        urls.add(endpoint)
    require(depths and all(type(d) is int and 1 <= d <= 128 for d in depths), "invalid requested depths")
    require(len(set(depths)) == len(depths), "duplicate requested depth")
    return [(d, min(profiles, key=lambda p: (abs(p["depth"] - d), p["depth"]))) for d in depths]


def request(case, profile, drafted):
    return {"model": profile["model"], "prompt": case["prompt"], "max_tokens": len(case["tokens"]),
            "temperature": 0, "top_k": 0, "top_p": 1, "min_p": 0, "seed": 1234,
            "draft": drafted, "ignore_eos": True, "return_token_ids": True, "stream": False}


def observe(observation, body, case):
    require(isinstance(observation, dict) and isinstance(observation.get("response"), dict), "response missing")
    response = observation["response"]
    require("error" not in response and response.get("model") == body["model"], "server error or model mismatch")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict), "expected one completion")
    finish = choices[0].get("finish_reason")
    require(finish in ("length", "stop"), "incomplete/refused completion")
    runtime = response.get("tensorfold", {})
    require(isinstance(runtime, dict), "token runtime missing")
    emitted = ids(runtime.get("token_ids"))
    require(runtime.get("drafts") is body["draft"], "serial/draft switch not acknowledged")
    require(isinstance(response.get("usage"), dict), "usage missing")
    usage = response["usage"].get("completion_tokens")
    require(type(usage) is int and usage == len(emitted), "ID/usage count mismatch")
    require(len(emitted) <= body["max_tokens"], "server exceeded requested length")
    speculative = response.get("speculative", {})
    require(isinstance(speculative, dict), "invalid speculative counters")
    drafted = speculative.get("drafted")
    require(drafted is None or (type(drafted) is int and drafted >= 0), "invalid draft-work counter")
    if not body["draft"]:
        require(drafted in (None, 0), "serial reference performed draft work")
    rows = observation.get("top2")
    if rows is not None:
        require(isinstance(rows, list) and len(rows) == len(emitted), "captured logits/IDs are not aligned")
        rows = [top2(row, len(case["prompt"]) + i) for i, row in enumerate(rows)]
        require(all(r["ids"][0] == t for r, t in zip(rows, emitted)), "capture and emitted argmax disagree")
    return {"tokens": emitted, "top2": rows, "finish_reason": finish, "drafted": drafted,
            "complete": finish == "length" and len(emitted) == body["max_tokens"],
            "draft_work_observed": not body["draft"] or (drafted is not None and drafted > 0)}


def first_difference(case, plain, drafted):
    for i, (wanted, actual) in enumerate(itertools.zip_longest(plain["tokens"], drafted["tokens"])):
        if wanted == actual:
            continue
        recorded = case["top2"][i] if i < len(case["top2"]) else None
        same_history = plain["tokens"][:i] == case["tokens"][:i]
        scores = dict(zip(recorded["ids"], recorded["logits"])) if recorded and same_history else {}
        a = plain["top2"][i] if plain["top2"] and i < len(plain["top2"]) else None
        b = drafted["top2"][i] if drafted["top2"] and i < len(drafted["top2"]) else None
        label = "divergence"
        if a and b and wanted is not None and actual is not None:
            label = verdict(a, b, wanted, actual)["verdict"]
        elif recorded and same_history and {wanted, actual} == set(recorded["ids"]) and near(recorded):
            label = "recorded-near-tie"
        return {"output_index": i, "absolute_position": len(case["prompt"]) + i,
                "plain_token_id": wanted, "drafted_token_id": actual, "label": label,
                "strict_equal": False, "plain_top2": a, "drafted_top2": b,
                "recorded_top2": recorded, "recorded_history_matches": same_history,
                "recorded_logits_for_tokens": {"plain": scores.get(wanted), "drafted": scores.get(actual)},
                "logit_source": "raw per-run captures when supplied; otherwise recorded source, never API logprobs"}
    return None


def evaluate(recording, profiles, fetch, depths=(1, 4, 8), limit=8, lookback=8, tokens=32):
    cases = select_prompts(recording, limit, lookback, tokens)
    selected = profiles_for(profiles, recording["identity"], depths)
    plain_profile = min(profiles, key=lambda p: p["depth"])
    results, observations = [], {}
    divergent, complete = False, True
    for case in cases:
        body = request(case, plain_profile, False)
        observation = fetch(plain_profile, body, case["id"], "plain")
        observations[case["id"] + ":plain"] = observation
        plain = observe(observation, body, case)
        reproduced = plain["tokens"] == case["tokens"]
        complete = complete and plain["complete"] and reproduced
        variants = []
        for depth, profile in selected:
            body = request(case, profile, True)
            observation = fetch(profile, body, case["id"], str(depth))
            observations[case["id"] + ":" + str(depth)] = observation
            run = observe(observation, body, case)
            difference = first_difference(case, plain, run)
            divergent = divergent or difference is not None
            complete = complete and run["complete"] and run["draft_work_observed"]
            variants.append({"requested_depth": depth, "configured_depth": profile["depth"],
                             "depth_control": "operator-configured endpoint cap, not per-request API control",
                             "run": run, "first_divergence": difference, "token_equal": difference is None})
        results.append({"id": case["id"], "prompt": case["prompt"], "source_position": case["source_position"],
                        "near_tie_offset": case["near_tie_offset"], "recorded_margin": case["margin"],
                        "bf16_spacing": case["bf16_spacing"], "source_reproduced": reproduced,
                        "plain": plain, "variants": variants})
    status = "divergent" if divergent else "pass" if complete else "incomplete"
    return {"schema": 1, "status": status, "passed": status == "pass", "identity": recording["identity"],
            "identity_source": "recording and operator profiles; not server-attested",
            "near_tie_policy": "BF16 labels are diagnostics; every token divergence fails",
            "cases": results, "observations": observations}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("endpoint redirect refused")


def live_fetch(timeout):
    opener = urllib.request.build_opener(NoRedirect())

    def fetch(profile, body, case_id, depth):
        req = urllib.request.Request(completion_url(profile["url"]), data=json.dumps(body, allow_nan=False).encode(),
                                     headers={"Content-Type": "application/json", "Accept": "application/json"})
        with opener.open(req, timeout=timeout) as response:
            data = response.read(16 * 1024 * 1024 + 1)
        require(len(data) <= 16 * 1024 * 1024, "response too large")
        return {"response": decode(data)}

    return fetch


def decode(data):
    def reject(value):
        raise ValueError("nonfinite JSON number")
    return json.loads(data, parse_constant=reject)


def load(path):
    return decode(Path(path).read_text())


def summary(report):
    cases = report.get("cases", [])
    comparisons = sum(len(c["variants"]) for c in cases)
    differences = sum(v["first_divergence"] is not None for c in cases for v in c["variants"])
    line = f"DRAFT_DIVERGENCE status={report['status']} cases={len(cases)} comparisons={comparisons} divergences={differences}"
    first = next(((c, v) for c in cases for v in c["variants"] if v["first_divergence"]), None)
    if first:
        c, v = first
        line += f" first={json.dumps(c['id'])}@{v['first_divergence']['output_index']} depth={v['requested_depth']}"
    return line


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--replay", type=Path, help="Offline observations keyed CASE:plain or CASE:requested-depth")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depths", default="1,4,8")
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--lookback", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    try:
        require(math.isfinite(args.timeout) and args.timeout > 0, "timeout must be finite and positive")
        recording, profiles = load(args.recording), load(args.profiles)
        if args.replay:
            replay = load(args.replay)

            def fetch(profile, body, case_id, depth):
                return replay[case_id + ":" + depth]
        else:
            fetch = live_fetch(args.timeout)
        report = evaluate(recording, profiles, fetch, tuple(int(x) for x in args.depths.split(",")),
                          args.cases, args.lookback, args.tokens)
        report["mode"] = "recorded-replay" if args.replay else "served-endpoint"
        report["input_sha256"] = {"recording": hashlib.sha256(args.recording.read_bytes()).hexdigest(),
                                  "profiles": hashlib.sha256(args.profiles.read_bytes()).hexdigest()}
    except (ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError, OSError,
            urllib.error.URLError) as error:
        report = {"schema": 1, "status": "error", "passed": False, "error": str(error)}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(summary(report))
    return 0 if report["passed"] else 1 if report["status"] == "divergent" else 2


if __name__ == "__main__":
    raise SystemExit(main())
