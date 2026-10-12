"""Verify the sanitized PR726 overlap panel without a model, GPU or private receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

import run_overlap as client

ARMS = ("A", "B", "C", "D")
FLAGS = (
    "TF_PR608_EXACT_PROMPT_CACHE", "TF_PR608_EXACT_CACHE_SEQ_REUSE",
    "TF_PR608_EXACT_CACHE_LOGITS", "TF_PR608_EXACT_CACHE_DRAFT_REUSE",
    "TF_PR608_EXACT_CACHE_SKIP_RESET",
)
BASE = "d49a906ffc168d707f62be598d329bf8dd8f5e2d"
CANDIDATE = "5e08e130bfb32e944390c2b9dfa3359a8859ea76"
BINARIES = {
    "base": "97b1a443773157d4534d6005ab5baae6506dba7f739dad590324efbe3968f313",
    "candidate": "d2c6d38c020580c5673a76a355ca1d3720078c18ad6d6f0116ee18c184abb976",
}
MODELS = {
    "target": ("Vontra/Qwen3.8-27B-MLX-4bit", "70ae7fac63274ff2eac54152031433374cb80f2f"),
    "drafter": ("z-lab/Qwen3.8-27B-DFlash2", "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"),
}


def sha(path: Path) -> str:
    """Hash exact payload bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Read only the portable payloads."""
    records = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
    return records, json.loads((root / "provenance.json").read_text()), json.loads((root / "counters.json").read_text())


def validate_numeric(row: dict[str, Any], tokens: int) -> None:
    """Check SSE-derived rates and actual speculative work, without estimating tokens from characters."""
    pieces = row.get("pieces")
    if not isinstance(pieces, list) or not pieces:
        raise ValueError("missing nonempty SSE pieces")
    if any(not isinstance(p, list) or len(p) != 2 or type(p[0]) not in (int, float)
           or not math.isfinite(p[0]) or type(p[1]) is not int or p[1] <= 0 for p in pieces):
        raise ValueError("invalid nonempty SSE piece")
    if pieces[0][0] != row["first"] or pieces[-1][0] != row["last"]:
        raise ValueError("SSE piece boundaries disagree with first/last")
    if any(b[0] < a[0] for a, b in zip(pieces, pieces[1:])):
        raise ValueError("SSE piece order drift")
    if row["last"] > row["first"]:
        rate = (tokens - 1) / (row["last"] - row["first"])
        if not math.isclose(row.get("decode_tps", -1), rate, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError("client decode rate drift")
    if row.get("prompt_tokens") != row["usage"]["prompt_tokens"]:
        raise ValueError("prompt count drift")
    work = row.get("work") or {}
    spec = row.get("speculative") or {}
    if row.get("work_source") != "speculative" or set(work) != {"rounds", "accepted", "drafted"}:
        raise ValueError("missing actual speculative work")
    if any(type(v) is not int or v < 0 for v in work.values()) or work["rounds"] == 0:
        raise ValueError("invalid speculative work count")
    if work != {k: spec.get(k) for k in work} or work["accepted"] > work["drafted"]:
        raise ValueError("speculative work fields disagree")


def verify_data(records: list[dict[str, Any]], provenance: dict[str, Any],
                counters: dict[str, Any], manifest: Path) -> dict[str, Any]:
    """Reconstruct all 152 aligned requests and the independently accepted timing table."""
    if len(records) != 152:
        raise ValueError("expected 152 request records")
    if provenance.get("base_commit") != BASE or provenance.get("candidate_commit") != CANDIDATE:
        raise ValueError("source revision drift")
    if provenance.get("binary_sha256") != BINARIES:
        raise ValueError("binary identity drift")
    actual_models = {m["name"]: (m["repo"], m["revision"]) for m in provenance["models"]}
    if actual_models != MODELS:
        raise ValueError("model identity drift")
    if provenance["aot"]["aot_sha256"] != "dce85c91013171fd92c66c9401dfce3998df4f7753a3a3c2e22efd8911b22d37":
        raise ValueError("AOT identity drift")
    if len(provenance.get("runtime_source_sha256", {})) != 31:
        raise ValueError("missing 31 runtime source identities")
    for arm in ARMS:
        settings = provenance["arms"][arm]
        if settings["host_cache_gib"] != int(arm in "BD") or settings["flags"] != {
            f: str(int(arm in "CD")) for f in FLAGS
        }:
            raise ValueError("arm settings drift")
    initial, followup = client.long_inputs(manifest)
    long_items = {"initial": initial, "followup": followup}
    groups: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {a: {} for a in ARMS}
    expected_keys = {(n, s, 0) for n in client.SHORT_NAMES for s in ("initial", "repeat")}
    expected_keys |= {(n, "burst", i) for n in client.SHORT_NAMES for i in range(16)}
    expected_keys |= {("long", s, 0) for s in ("initial", "followup")}
    for record in records:
        if record.get("schema") != "pr726_sanitized_request_v1" or record.get("arm") not in ARMS:
            raise ValueError("request schema/arm drift")
        arm, name, stage, index = (record[k] for k in ("arm", "case", "stage", "stream_index"))
        key = (name, stage, index)
        if type(index) is not int or key not in expected_keys or key in groups[arm]:
            raise ValueError("duplicate or unexpected request identity")
        row = dict(record["request"])
        route = row.pop("request_path", None)
        kind = "completion" if name == "code" else "chat"
        if route != ("/v1/completions" if kind == "completion" else "/v1/chat/completions"):
            raise ValueError("request route drift")
        row["request_url"] = route
        if row.get("request_body", {}).get("model") != "qwen27-overlap":
            raise ValueError("model alias drift")
        item = long_items[stage] if name == "long" else {
            "name": name, "kind": kind, "prompt": client.PROMPTS[name][1]
        }
        tokens = client.LONG_OUTPUT if name == "long" else client.SHORT_OUTPUT
        client.validate_row(row, item=item, tokens=tokens,
                            cached=client.expected_cache(arm, name, stage))
        validate_numeric(row, tokens)
        groups[arm][key] = row
    for arm in ARMS:
        if set(groups[arm]) != expected_keys:
            raise ValueError("missing complete 38-request arm")
        if len({r["response_id"] for r in groups[arm].values()}) != 38:
            raise ValueError("duplicate response identity")
    for key in expected_keys:
        reference = groups["A"][key]
        for arm in ARMS[1:]:
            row = groups[arm][key]
            if any(row[k] != reference[k] for k in ("request_body", "output_sha256", "token_sha")):
                raise ValueError("literal input or output parity drift")
    for arm in "CD":
        stats = counters["arms"][arm]
        if stats["exact_cache"]["hits"] != 34 or stats["logits"]["hits"] != 34:
            raise ValueError("missing short cache/logit exposure")
        if any(stats[name].get("failed") for name in ("exact_cache", "logits", "seq_pool", "draft_pool")):
            raise ValueError("failed cache owner")
    d = counters["arms"]["D"]
    if d["seq_pool"]["reuse_hits"] or d["draft_pool"]["reuse_hits"] or d["reset_skip"]["avoided_fill_calls"]:
        raise ValueError("host-hook treatment exposure drift")
    metrics: dict[str, Any] = {}
    for arm, group in groups.items():
        cells = {}
        for name in client.SHORT_NAMES:
            cold, repeat = group[(name, "initial", 0)], group[(name, "repeat", 0)]
            burst = [group[(name, "burst", i)] for i in range(16)]
            cells[name] = {
                "initial_ttft_s": cold["ttft_s"], "initial_full_response_s": cold["full_response_s"],
                "initial_decode_tps": cold["decode_tps"], "repeat_ttft_s": repeat["ttft_s"],
                "repeat_full_response_s": repeat["full_response_s"], "repeat_decode_tps": repeat["decode_tps"],
                "burst_median_ttft_s": statistics.median(r["ttft_s"] for r in burst),
                "burst_slowest_ttft_s": max(r["ttft_s"] for r in burst),
                "burst_median_decode_tps": statistics.median(r["decode_tps"] for r in burst),
                "burst_median_full_response_s": statistics.median(r["full_response_s"] for r in burst),
                "burst_wall_s": max(r["done"] for r in burst) - min(r["sent"] for r in burst),
                "repeat_cached_tokens": repeat["cached_tokens"],
            }
        cells["long"] = {s: {k: group[("long", s, 0)][k] for k in (
            "ttft_s", "full_response_s", "cached_tokens")} for s in ("initial", "followup")}
        metrics[arm] = cells
    return {"schema": "pr726_portable_summary_v1", "complete": True,
            "total_requests": 152, "requests_per_arm": 38, "parity_matches": True, "metrics": metrics}


def verify(root: Path, source_root: Path | None = None) -> dict[str, Any]:
    """Verify file identities, numerical gates and optional candidate runtime files."""
    bundle = json.loads((root / "bundle-manifest.json").read_text())
    for name, digest in bundle["files_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError(f"bundle hash drift: {name}")
    records, provenance, counters = load(root)
    if source_root is not None:
        for name, digest in provenance["runtime_source_sha256"].items():
            if sha(source_root / name) != digest:
                raise ValueError(f"runtime source drift: {name}")
    result = verify_data(records, provenance, counters, root / "long-manifest.json")
    if result != json.loads((root / "summary.json").read_text()):
        raise ValueError("saved summary differs from recomputation")
    return result


def main() -> None:
    """Print recomputed tables or write the complete portable result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.bundle, args.source_root)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
