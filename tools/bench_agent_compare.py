"""Compare sequential agent receipts without treating missing usage or token hashes as evidence."""
import math
import re

from bench_agent_trace import SHAPES, digest
from bench_evidence import token_equal

SETTING_KEYS = {"trace", "conversations", "rounds", "prefix_tokens", "reply_tokens", "seed",
                "temperature", "one_offs_per_turn", "history_mode"}
IDENTITY = ("index", "trace", "conversation", "round", "turn", "kind", "request_seed", "request_sha256")


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def integer(value):
    return type(value) is int and value >= 0


def seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate(report):
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1 or report.get("benchmark") != "cache_agents":
        raise ValueError("invalid_receipt")
    settings, rows = report.get("settings"), report.get("rows")
    if not isinstance(settings, dict) or set(settings) != SETTING_KEYS or settings["trace"] not in SHAPES:
        raise ValueError("invalid_settings")
    if any(not integer(settings[k]) or settings[k] < 1 for k in ("conversations", "rounds", "prefix_tokens", "reply_tokens")):
        raise ValueError("invalid_settings")
    if not integer(settings["seed"]) or type(settings["temperature"]) is not int or settings["temperature"] != 0 or settings["one_offs_per_turn"] != 2 or settings["history_mode"] != "measured_replies":
        raise ValueError("invalid_settings")
    if settings["trace"] == "agent_cohorts" and settings["rounds"] < 2:
        raise ValueError("invalid_settings")
    expected = settings["conversations"] * settings["rounds"] * 3
    if report.get("expected_requests") != expected or not isinstance(rows, list) or len(rows) > expected:
        raise ValueError("invalid_schedule")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not integer(row.get("index")) or row["index"] != index or row.get("trace") != settings["trace"]:
            raise ValueError("invalid_request_identity")
        if row.get("kind") not in ("agent", "one_off") or not all(integer(row.get(k)) for k in ("round", "turn", "request_seed")):
            raise ValueError("invalid_request_identity")
        conversation = row.get("conversation")
        if not integer(conversation) and not (isinstance(conversation, str) and re.fullmatch(r"oneoff-[0-9]{8,}", conversation)):
            raise ValueError("invalid_conversation")
        if not sha(row.get("request_sha256")):
            raise ValueError("invalid_request_fingerprint")
        if row["round"] >= settings["rounds"] or row["request_seed"] != settings["seed"] + index:
            raise ValueError("invalid_request_identity")
    if rows:
        prefixes = report.get("system_prefixes")
        phases = 2 if settings["trace"] == "agent_cohorts" else 1
        if not isinstance(prefixes, list) or len(prefixes) != phases or not isinstance(report.get("thinking"), dict):
            raise ValueError("missing_fixture_evidence")
        for proof in prefixes:
            if not isinstance(proof, dict) or proof.get("tokens") != settings["prefix_tokens"] or not all(sha(proof.get(k)) for k in ("text_sha256", "ids_sha256")):
                raise ValueError("invalid_fixture_evidence")
    if report.get("fixture_sha256") != digest([r["request_sha256"] for r in rows]):
        raise ValueError("invalid_fixture_fingerprint")
    return settings, rows, expected


def usage_verified(row):
    prompt, cached = row.get("prompt_tokens"), row.get("cached_tokens")
    return (row.get("complete") is True and not row.get("error") and integer(prompt) and integer(cached)
            and cached <= prompt and row.get("cache_state") == ("cold" if cached == 0 else "reused"))


def time_verified(row):
    return row.get("complete") is True and not row.get("error") and sha(row.get("output_sha256")) and seconds(row.get("ttft_s"))


def token_verified(row):
    return time_verified(row) and integer(row.get("tokens")) and row["tokens"] > 0 and token_equal(row, row) is True


def qualify(report):
    _, rows, expected = validate(report)
    return (report.get("completed_schedule") is True and len(rows) == expected
            and all(usage_verified(r) and token_verified(r) for r in rows))


def summarize(rows):
    return {"requests": len(rows), "usage_verified": sum(usage_verified(r) for r in rows),
            "cache_evidence": "verified" if rows and all(usage_verified(r) for r in rows) else "unverified",
            "rebuilt_rows": sum(r["prompt_tokens"] - r["cached_tokens"] for r in rows)
            if rows and all(usage_verified(r) for r in rows) else None,
            "ttft_sum_s": sum(r["ttft_s"] for r in rows) if rows and all(time_verified(r) for r in rows) else None}


def compare(left, right):
    settings_a, a, expected_a = validate(left)
    settings_b, b, expected_b = validate(right)
    issues = []
    for field in ("settings", "thinking", "system_prefixes", "fixture_sha256"):
        if left.get(field) != right.get(field):
            issues.append(field + "_differ")
    if (expected_a != expected_b or len(a) != expected_a or len(b) != expected_b
            or left.get("completed_schedule") is not True or right.get("completed_schedule") is not True):
        issues.append("incomplete_schedule")
    equal, per_request = [], []
    for x, y in zip(a, b):
        aligned = all(x.get(k) == y.get(k) for k in IDENTITY)
        same = token_equal(x, y) if aligned and token_verified(x) and token_verified(y) else None
        output_equal = sha(x.get("output_sha256")) and x.get("output_sha256") == y.get("output_sha256")
        prompt_equal = usage_verified(x) and usage_verified(y) and x["prompt_tokens"] == y["prompt_tokens"]
        completion_equal = integer(x.get("tokens")) and x.get("tokens") == y.get("tokens")
        equal.append(same)
        per_request.append({k: x[k] for k in ("index", "trace", "conversation", "round")}
                           | {"identity_equal": aligned, "token_equal": same,
                              "output_equal": output_equal,
                              "prompt_counts_equal": prompt_equal, "completion_counts_equal": completion_equal})
        if not aligned and "request_identities_differ" not in issues:
            issues.append("request_identities_differ")
        if not output_equal and "output_unverified_or_different" not in issues:
            issues.append("output_unverified_or_different")
        if not prompt_equal and "prompt_usage_unverified_or_different" not in issues:
            issues.append("prompt_usage_unverified_or_different")
        if not completion_equal and "completion_counts_differ" not in issues:
            issues.append("completion_counts_differ")
    exact = False if False in equal else True if equal and all(x is True for x in equal) else None
    traces = {}
    for trace in sorted({settings_a["trace"], settings_b["trace"]}):
        sa = summarize([r for r in a if r["trace"] == trace])
        sb = summarize([r for r in b if r["trace"] == trace])
        paired = not issues and exact is True
        rows_delta = sb["rebuilt_rows"] - sa["rebuilt_rows"] if paired and sa["rebuilt_rows"] is not None and sb["rebuilt_rows"] is not None else None
        ttft_delta = sb["ttft_sum_s"] - sa["ttft_sum_s"] if paired and sa["ttft_sum_s"] is not None and sb["ttft_sum_s"] is not None else None
        traces[trace] = {"a": sa, "b": sb, "rebuilt_rows_b_minus_a": rows_delta,
                         "ttft_sum_b_minus_a_s": ttft_delta,
                         "rows_no_worse": rows_delta <= 0 if rows_delta is not None else None}
    return {"schema_version": 1, "benchmark": "cache_agents_comparison", "token_exact": exact,
            "qualified": not issues and exact is True and qualify(left) and qualify(right),
            "issues": issues, "traces": traces, "per_request": per_request,
            "note": "Latency and rebuilt-row changes are reported; no performance threshold is imposed."}
