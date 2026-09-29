"""The public fixtures and aggregation rules for community-v1.

Delivery timings observe SSE data arriving at the client. They do not measure
individual generated tokens or GPU kernel time. Server timings stay separate.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from typing import Any

SUITE_ID = "community-v1"
DEFAULT_TOKENS = 256
DEFAULT_REPETITIONS = 5
DEFAULT_TEMPERATURES = (1.0, 0.0)
SEED_BASE = 1234
FIXTURES = (
    {
        "id": "code",
        "kind": "completion",
        "prompt": "Write a short Python function that computes the Fibonacci sequence and explain it.",
    },
    {
        "id": "chat",
        "kind": "chat",
        "prompt": "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.",
    },
)
SUITE_MANIFEST = {
    "id": SUITE_ID,
    "fixtures": list(FIXTURES),
    "max_tokens": DEFAULT_TOKENS,
    "repetitions": DEFAULT_REPETITIONS,
    "warmups_per_cell": 1,
    "temperatures": list(DEFAULT_TEMPERATURES),
    "seed_base": SEED_BASE,
    "sampled_top_k": 20,
    "sampled_top_p": 0.95,
    "thinking": False,
    "ignore_eos": False,
    "concurrency": 1,
    "delivery_rate": "completion_tokens_minus_one_over_first_to_last_nonempty_sse_delivery",
}
SUITE_SHA256 = hashlib.sha256(
    json.dumps(SUITE_MANIFEST, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()
STATUSES = frozenset({"ok", "early_eos", "error", "unmeasured"})
ERROR_CODES = frozenset({
    "http_error", "connection_error", "timeout", "cancelled", "server_error", "invalid_stream",
    "truncated_stream", "missing_usage", "invalid_usage", "empty_output", "unexpected_finish",
    "token_count_mismatch", "unmeasured_delivery",
})
SAMPLE_FIELDS = frozenset({
    "fixture_id", "temperature", "repeat", "seed", "status", "prompt_tokens", "completion_tokens",
    "cached_tokens", "delivery_seconds", "delivery_tps", "ttft_seconds", "end_to_end_seconds",
    "server_decode_tps", "server_decode_seconds", "prefill_seconds", "token_sha", "error_code",
})


def _spread(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    low, high = min(values), max(values)
    return {"median": statistics.median(values), "min": low, "max": high, "spread": high - low}


def summary(
    samples: list[dict[str, Any]], *, expected_repetitions: int = DEFAULT_REPETITIONS,
) -> list[dict[str, Any]]:
    """Aggregate complete cells without hiding failed attempts.

    A failed warmup has repeat=-1 and makes its cell incomplete. Successful
    warmups are not included in samples. Standard cells need all five repeats.
    Custom cells can have aggregates when every declared repeat succeeds.
    Optional metrics need a measurement in every repeat to get an aggregate.
    """

    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for sample in samples:
        key = (sample["fixture_id"], float(sample["temperature"]))
        groups.setdefault(key, []).append(sample)
    result = []
    for (fixture_id, temperature), rows in groups.items():
        measured = [row for row in rows if row["repeat"] >= 0]
        complete = (
            len(rows) == expected_repetitions
            and sorted(row["repeat"] for row in rows) == list(range(expected_repetitions))
            and all(row["status"] == "ok" for row in rows)
        )
        cell: dict[str, Any] = {
            "fixture_id": fixture_id,
            "temperature": temperature,
            "repeats": len(measured),
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "protocol_complete": complete and expected_repetitions == DEFAULT_REPETITIONS,
        }
        for metric in (
            "delivery_tps", "ttft_seconds", "end_to_end_seconds", "server_decode_tps", "prefill_seconds",
        ):
            values = [row.get(metric) for row in measured]
            valid = complete and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0 for value in values
            )
            cell[metric] = _spread(values) if valid else None
        result.append(cell)
    return result
