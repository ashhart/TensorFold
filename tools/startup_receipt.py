#!/usr/bin/env python3
"""Measure one local Metal startup without changing global caches or touching another process."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from startup_receipt_format import InvalidReceipt, completion_hash, fixed_request, paired, parse_line


class RunFailed(RuntimeError):
    pass


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def warm_checkpoint(model):
    config = json.loads((model / "config.json").read_bytes())
    text = config.get("text_config", config)
    quant = config.get("quantization", config.get("quantization_config", {}))
    if (text.get("hidden_size"), text.get("num_hidden_layers"), quant.get("bits"), quant.get("group_size")) != (
        5120, 64, 4, 64
    ) or text.get("num_experts", 0):
        raise RunFailed("This receipt requires the dense 27B MLX affine 4-bit g64 checkpoint")
    index = model / "model.safetensors.index.json"
    names = ["model.safetensors"]
    if index.exists():
        names = list(json.loads(index.read_bytes())["weight_map"].values())
    if not names or any(not isinstance(n, str) or Path(n).is_absolute() or ".." in Path(n).parts for n in names):
        raise RunFailed("Invalid checkpoint shard names")
    names += [n for n in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
                         "generation_config.json") if (model / n).is_file()]
    files, total = [], 0
    start = time.monotonic_ns()
    for name in sorted(set(names)):
        path = model / name
        size = path.stat().st_size
        files.append({"name": name, "bytes": size, "sha256": file_hash(path)})
        total += size
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"files": files, "sha256": identity, "sequential_bytes_read": total,
            "elapsed_ns": time.monotonic_ns() - start, "page_cache": "prewarmed_best_effort"}


class Child:
    def __init__(self, argv, environ, log_path):
        self.events = queue.Queue(maxsize=64)
        self.log = log_path.open("wb")
        self.log_lock = threading.Lock()
        self.log_bytes = 0
        self.process = None
        self.threads = []
        try:
            self.process = subprocess.Popen(argv, env=environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for stream in (self.process.stdout, self.process.stderr):
                thread = threading.Thread(target=self.read, args=(stream,), daemon=True)
                thread.start()
                self.threads.append(thread)
        except BaseException:
            if self.process is not None:
                self.stop()
            else:
                self.log.close()
            raise

    def publish(self, record):
        try:
            self.events.put_nowait(record)
        except queue.Full:
            pass

    def read(self, stream):
        try:
            while line := stream.readline(16385):
                with self.log_lock:
                    if self.log_bytes < 1 << 20:
                        self.log.write(line[:(1 << 20) - self.log_bytes])
                        self.log_bytes += len(line)
                if len(line) > 16384:
                    self.publish(RunFailed("Server log line exceeded the receipt bound"))
                    return
                try:
                    record = parse_line(line)
                    if record is not None:
                        self.publish(record)
                except InvalidReceipt as error:
                    self.publish(RunFailed(str(error)))
        except OSError:
            self.publish(RunFailed("Could not read child output"))

    def wait_record(self, event, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                record = self.events.get(timeout=min(0.05, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                if self.process.poll() is not None and not any(t.is_alive() for t in self.threads):
                    raise RunFailed("Server exited before " + event)
                continue
            if isinstance(record, Exception):
                raise record
            if record["startup"]["event"] != event:
                raise RunFailed("Unexpected or repeated startup event")
            return record
        raise RunFailed("Timed out waiting for " + event)

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        else:
            self.process.wait()
        for thread in self.threads:
            thread.join(timeout=2)
        if any(t.is_alive() for t in self.threads):
            raise RunFailed("Child output did not close after child exit")
        self.process.stdout.close()
        self.process.stderr.close()
        self.log.close()


def request(port, timeout, method, route, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        data = None if body is None else json.dumps(body).encode()
        conn.request(method, route, body=data, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        payload = response.read(4 * 1024 * 1024 + 1)
        if response.status != 200 or len(payload) > 4 * 1024 * 1024:
            raise RunFailed("First request returned an error or oversized response")
        return payload
    finally:
        conn.close()


def attempt(argv, environ, directory, ready_timeout, request_timeout):
    directory.mkdir()
    child = None
    result = {"status": "failed", "ready": None, "first_completion": None}
    launch = time.monotonic_ns()
    try:
        child = Child(argv, environ, directory / "server-private.log")
        ready = child.wait_record("ready", ready_timeout)
        result["ready"] = ready
        result["launch_to_ready_observed_ns"] = time.monotonic_ns() - launch
        port = ready["port"]
        request(port, request_timeout, "GET", "/health")
        started = time.monotonic_ns()
        body = request(port, request_timeout, "POST", "/v1/completions", fixed_request())
        result["request_delivery_ns"] = time.monotonic_ns() - started
        digest = completion_hash(body)
        done = child.wait_record("first_completion", request_timeout)
        result.update(paired(ready, done, digest))
        if child.process.poll() is not None:
            raise RunFailed("Server exited before the receipt was collected")
        result["status"] = "ok"
    except KeyboardInterrupt:
        result["status"], result["error"] = "interrupted", "Interrupted"
    except (RunFailed, InvalidReceipt, OSError, http.client.HTTPException, ValueError) as error:
        result["error"] = str(error) if isinstance(error, (RunFailed, InvalidReceipt)) else type(error).__name__
    finally:
        if child is not None:
            try:
                child.stop()
                result["child_reaped"] = child.process.poll() is not None
            except (RunFailed, subprocess.TimeoutExpired) as error:
                result["status"], result["error"] = "failed", type(error).__name__
        (directory / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--ready-timeout", type=float, default=600)
    p.add_argument("--request-timeout", type=float, default=180)
    p.add_argument("--case", choices=("warm", "cold"), default="warm")
    args = p.parse_args(argv)
    if args.runs < 1 or not 0 < args.ready_timeout <= 1800 or not 0 < args.request_timeout <= 1800:
        p.error("positive bounded timeouts and at least one run are required")
    if args.output.exists():
        p.error("output must be new; previous attempts are never overwritten")
    args.output.mkdir(parents=True)
    summary = {"schema": 1, "case": args.case, "status": "unchecked", "request": fixed_request(), "runs": [],
               "deployment": "Qwen3.8-27B MLX affine 4-bit g64, Metal, plain, 1 stream, context 4096, ANE off",
               "prepared_artifacts": "No engine-managed target pack; driver shader cache is uncontrolled",
               "cold_page_cache": "Unavailable: no privileged purge or global memory pressure is attempted"}
    try:
        if args.case == "cold":
            summary["status"] = "unsupported"
            return 2
        if sys.platform != "darwin":
            raise RunFailed("The selected device receipt requires macOS")
        binary, model = args.binary.resolve(strict=True), args.model.resolve(strict=True)
        summary["binary_sha256"] = file_hash(binary)
        env = dict(os.environ, TF_STARTUP_RECEIPT="1", TF_ANE_COLUMNS="0", TENSORFOLD_NO_LIVE="1",
                   TENSORFOLD_API_KEY="")
        command = [str(binary), "serve", str(model), "--host", "127.0.0.1", "--port", "0", "--name", "startup-receipt",
                   "--parallel", "1", "--context", "4096", "--prompt-cache-gib", "0", "--no-thinking", "--no-drafts",
                   "--no-update-check"]
        for index in range(args.runs):
            if file_hash(binary) != summary["binary_sha256"]:
                raise RunFailed("Server binary changed between attempts")
            warm = warm_checkpoint(model)
            result = attempt(command, env, args.output / f"run-{index + 1}", args.ready_timeout, args.request_timeout)
            result["prewarm"] = warm
            if file_hash(binary) != summary["binary_sha256"]:
                result["status"], result["error"] = "failed", "Server binary changed during attempt"
            summary["runs"].append(result)
            if result["status"] != "ok":
                summary["status"] = "failed"
                return 1
            if result["first_completion"]["resources"] is None:
                raise RunFailed("Kernel memory and physical-read counters were unavailable")
        hashes = {r["reply_token_sha256"] for r in summary["runs"]}
        identities = {r["prewarm"]["sha256"] for r in summary["runs"]}
        summary["status"] = "ok" if len(hashes) == len(identities) == 1 else "failed"
        return 0 if summary["status"] == "ok" else 1
    except KeyboardInterrupt:
        summary["status"], summary["error"] = "interrupted", "Interrupted"
        return 130
    except (OSError, ValueError, KeyError, RunFailed) as error:
        summary["status"] = "failed"
        summary["error"] = str(error) if isinstance(error, RunFailed) else type(error).__name__
        return 1
    finally:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
