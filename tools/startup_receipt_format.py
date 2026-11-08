"""Validate native startup records without accepting partial or failed loads as ready."""
import json
import re

PREFIX = b"startup-receipt "
PHASES = (
    "process_setup", "discovery", "tokenizer", "model_config", "checkpoint_headers", "read_pack",
    "weight_pipelines", "initial_allocation", "runtime_setup", "runtime_pipelines", "runtime_allocation",
    "residency", "optional_ane", "model_ready", "server_setup", "server_ready", "metal_source_build", "metal_pipeline_build", "first_completion",
)


class InvalidReceipt(ValueError):
    pass


def integer(value):
    return type(value) is int and 0 <= value < 2**64


def parse_line(line):
    if not line.startswith(PREFIX):
        return None
    try:
        record = json.loads(line[len(PREFIX):])
        startup = record["startup"]
        if startup["schema"] != 1 or type(startup["schema"]) is not int:
            raise InvalidReceipt("Unsupported startup schema")
        if (startup["deployment"], startup["clock"], startup["origin"], startup["units"], startup["inclusive"]) != (
            "qwen27-metal", "awake_monotonic", "native_main_entry", "ns", True
        ):
            raise InvalidReceipt("Unexpected startup clock or deployment")
        if startup["inclusive"] is not True:
            raise InvalidReceipt("Invalid phase accounting")
        event, at = startup["event"], startup["at_ns"]
        if event not in ("ready", "first_completion") or not integer(at):
            raise InvalidReceipt("Invalid startup event or timestamp")
        rows = startup["phases"]
        if not isinstance(rows, list) or [r["phase"] for r in rows] != list(PHASES):
            raise InvalidReceipt("Missing, repeated or unordered startup phases")
        for row in rows:
            start, end = row["start_ns"], row["end_ns"]
            if start is not None and (not integer(start) or start > at):
                raise InvalidReceipt("Invalid phase start")
            if end is not None and (start is None or not integer(end) or not start <= end <= at):
                raise InvalidReceipt("Invalid phase end")
            if not integer(row["calls"]) or not integer(row["total_ns"]):
                raise InvalidReceipt("Invalid phase totals")
            if row["calls"] == 0 and (end is not None or row["total_ns"] != 0):
                raise InvalidReceipt("Completed phase without a call")
            if row["phase"] not in ("first_completion", "metal_source_build", "metal_pipeline_build") and end is None:
                raise InvalidReceipt("Ready without completed startup phases")
        phases = {r["phase"]: r for r in rows}
        ready_at = phases["server_ready"]["end_ns"]
        if phases["model_ready"]["end_ns"] > ready_at or phases["process_setup"]["start_ns"] != 0:
            raise InvalidReceipt("Invalid readiness ordering")
        if event == "ready":
            if not type(record["port"]) is int or not 0 < record["port"] < 65536:
                raise InvalidReceipt("Invalid ready port")
            if record["token_sha256"] is not None or phases["first_completion"]["start_ns"] is not None or at != ready_at:
                raise InvalidReceipt("Ready contains a completion")
        else:
            done = phases["first_completion"]
            if record["port"] is not None or not done["end_ns"] == at or done["start_ns"] < ready_at:
                raise InvalidReceipt("Invalid completion ordering")
            if not isinstance(record["token_sha256"], str) or not re.fullmatch("[0-9a-f]{64}", record["token_sha256"]):
                raise InvalidReceipt("Missing token hash")
        resources = record["resources"]
        if resources is not None:
            if not isinstance(resources, dict) or any(not integer(resources[k]) for k in (
                "physical_footprint_bytes", "lifetime_peak_physical_footprint_bytes", "disk_io_bytes_read"
            )):
                raise InvalidReceipt("Invalid kernel resource counters")
            if resources["lifetime_peak_physical_footprint_bytes"] < resources["physical_footprint_bytes"]:
                raise InvalidReceipt("Peak memory below current memory")
        return record
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        if isinstance(error, InvalidReceipt):
            raise
        raise InvalidReceipt("Malformed startup receipt") from None


def completion_hash(body):
    try:
        reply = json.loads(body)
        if "error" in reply or reply["choices"][0]["finish_reason"] not in ("stop", "length"):
            raise InvalidReceipt("First request did not complete successfully")
        if type(reply["usage"]["completion_tokens"]) is not int or reply["usage"]["completion_tokens"] < 1:
            raise InvalidReceipt("Empty first completion")
        digest = reply["tensorfold"]["token_sha256"]
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise InvalidReceipt("First response has no full token hash")
        return digest
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        if isinstance(error, InvalidReceipt):
            raise
        raise InvalidReceipt("Malformed first completion") from None


def paired(ready, done, digest):
    if done["startup"]["event"] != "first_completion" or done["token_sha256"] != digest:
        raise InvalidReceipt("Receipt and response token hashes differ")
    stable = lambda rows: [r for r in rows if r["phase"] not in ("first_completion", "metal_source_build", "metal_pipeline_build")]
    if stable(ready["startup"]["phases"]) != stable(done["startup"]["phases"]):
        raise InvalidReceipt("Startup phases changed after ready")
    if ready["resources"] is not None and done["resources"] is not None:
        for key in ("lifetime_peak_physical_footprint_bytes", "disk_io_bytes_read"):
            if done["resources"][key] < ready["resources"][key]:
                raise InvalidReceipt("Lifetime resource counter moved backwards")
    return {
        "ready": ready, "first_completion": done, "reply_token_sha256": digest,
        "counter_scope": "process lifetime; disk_io_bytes_read is kernel physical I/O, not logical file bytes",
    }


def fixed_request():
    return {"model": "startup-receipt", "prompt": "Write the numbers from 1 to 20, separated by spaces.",
            "max_tokens": 64, "temperature": 0, "seed": 0, "stream": False, "draft": False}
