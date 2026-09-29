"""Receipt validation uses fictional public labels and no real device identities."""

from __future__ import annotations

import copy
import json

import pytest

from tensorfold.benchmark import receipt
from tensorfold.benchmark.protocol import SUITE_ID, SUITE_SHA256


def valid_receipt():
    samples = []
    for fixture_id in ("code", "chat"):
        samples.append({
            "fixture_id": fixture_id, "temperature": 0.0, "repeat": 0, "seed": 1234,
            "status": "ok", "prompt_tokens": 24, "completion_tokens": 4, "cached_tokens": 0,
            "delivery_seconds": 0.3, "delivery_tps": 10.0, "ttft_seconds": 0.05,
            "end_to_end_seconds": 0.4, "server_decode_tps": 12.0, "server_decode_seconds": 0.25,
            "prefill_seconds": 0.04, "token_sha": "0123456789ab", "error_code": None,
        })
    return {
        "schema_version": 1,
        "suite": {"id": SUITE_ID, "sha256": SUITE_SHA256},
        "run_id": "00000000-0000-4000-8000-000000000001",
        "created_at": "2026-01-01T00:00:00Z",
        "runtime": {"tensorfold_version": "0.0.0-test", "backend": "cuda", "python_version": "3.11.0",
                    "platform": "linux", "dependencies": {}},
        "model": {"repo_id": "example-org/test-model", "revision": None, "family": None,
                  "config_sha256": None, "tokenizer_sha256": None, "quantization": None,
                  "drafter_repo_id": None, "local": False},
        "hardware": {"cpu_model": None, "system_memory_bytes": None, "memory_type": "unknown",
                     "gpus": [], "gpu_count": None, "source": "unavailable"},
        "settings": {"tokens": 4, "repetitions": 1, "temperatures": [0.0], "serial": False,
                     "context_tokens": None, "managed": False, "rank_count": None},
        "samples": samples,
    }


def _change(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


def test_receipt_roundtrip_is_portable_atomic_and_finite(tmp_path):
    value = valid_receipt()
    destination = tmp_path / "nested" / "run.json"
    receipt.save(value, destination)
    assert receipt.load(destination) == value
    assert len(receipt.digest(value)) == 64
    assert receipt.digest(copy.deepcopy(value)) == receipt.digest(value)
    assert list(destination.parent.iterdir()) == [destination]
    assert "example-org/test-model" in destination.read_text()


@pytest.mark.parametrize("path,replacement", [
    (("schema_version",), True),
    (("settings", "tokens"), True),
    (("settings", "repetitions"), False),
    (("settings", "serial"), "false"),
    (("settings", "managed"), 1),
    (("settings", "rank_count"), True),
    (("settings", "temperatures"), [False]),
    (("samples", 0, "prompt_tokens"), False),
    (("samples", 0, "delivery_seconds"), True),
    (("hardware", "gpu_count"), False),
    (("model", "local"), "false"),
    (("samples", 0, "delivery_tps"), float("nan")),
    (("samples", 0, "end_to_end_seconds"), float("inf")),
    (("samples", 0, "cached_tokens"), -1),
    (("samples", 0, "cached_tokens"), 25),
    (("samples", 0, "delivery_tps"), 99),
    (("samples", 0, "completion_tokens"), 3),
    (("samples", 0, "token_sha"), "not-token-ids"),
    (("model", "repo_id"), "https://example.test/private-model"),
    (("model", "repo_id"), "/home/example/private-model"),
    (("model", "family"), "<script>alert(1)</script>"),
    (("runtime", "dependencies"), {"secret": "not-public"}),
    (("suite", "sha256"), "a" * 64),
    (("run_id",), "not-a-run-id"),
    (("created_at",), "2026-01-01T00:00:00"),
])
def test_receipt_rejects_wrong_types_numbers_and_private_strings(path, replacement):
    value = valid_receipt()
    _change(value, path, replacement)
    with pytest.raises(ValueError):
        receipt.validate(value)


@pytest.mark.parametrize("path,key", [
    ((), "server_url"), (("model",), "path"), (("runtime",), "hostname"),
    (("hardware",), "serial_number"), (("settings",), "prompt"),
    (("samples", 0), "output"), (("suite",), "free_text"),
])
def test_receipt_rejects_unknown_fields_at_every_level(path, key):
    value = valid_receipt()
    target = value
    for item in path:
        target = target[item]
    target[key] = "not-public"
    with pytest.raises(ValueError):
        receipt.validate(value)


def test_private_checkpoint_is_local_only():
    value = valid_receipt()
    value["model"].update(local=True, repo_id=None)
    assert receipt.validate(value) is value
    with pytest.raises(ValueError, match="Local or private"):
        receipt.validate(value, publishing=True)


def test_receipt_rejects_duplicate_json_fields_and_nonfinite_json(tmp_path):
    path = tmp_path / "run.json"
    text = json.dumps(valid_receipt())
    path.write_text(text[:-1] + ', "schema_version": 1}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        receipt.load(path)
    path.write_text(text.replace('"delivery_tps": 10.0', '"delivery_tps": NaN', 1))
    with pytest.raises(ValueError, match="Nonfinite"):
        receipt.load(path)


def test_receipt_deep_json_fails_cleanly(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("[" * 2000 + "0" + "]" * 2000)
    with pytest.raises(ValueError):
        receipt.load(path)


def test_receipt_rejects_duplicate_repeats_and_wrong_cache_counts():
    value = valid_receipt()
    value["samples"].append(copy.deepcopy(value["samples"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        receipt.validate(value)


@pytest.mark.parametrize("path,replacement", [
    (("settings", "temperatures"), [[0.0]]),
    (("samples", 0, "error_code"), ["server_error"]),
    (("samples", 0, "prompt_tokens"), 10**500),
], ids=("unhashable-temperatures", "unhashable-error-code", "oversized-integer"))
def test_malformed_receipt_always_raises_valueerror_not_an_uncaught_typeerror(path, replacement):
    value = valid_receipt()
    _change(value, path, replacement)
    with pytest.raises(ValueError):
        receipt.validate(value)
