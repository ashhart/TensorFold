"""Exercise a running sleep-enabled server with public prompts, streamed drain and stored conversation history."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request


def qualify(base, model, token, output, *, tokens=64, require_cache=False):
    def call(route, body=None, *, method=None, authenticated=False, origin=False):
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = "Bearer " + token
        if origin:
            headers["Origin"] = "https://example.invalid"
        request = urllib.request.Request(base + route, method=method or ("POST" if body is not None else "GET"),
                                         data=json.dumps(body).encode() if body is not None else None, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=1800)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if "application/json" in response.headers.get("Content-Type", "") else raw

    def control(route, *, method="POST"):
        start = time.perf_counter()
        status, body = call(route, method=method, authenticated=True)
        assert status == 200, (status, body)
        return body, time.perf_counter() - start

    def chat(draft=True):
        body = {"model": model, "messages": [{"role": "user", "content":
                 "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."}],
                "max_tokens": tokens, "temperature": 1.0, "top_k": 20, "top_p": .95, "seed": 1234,
                "ignore_eos": True, "draft": draft, "chat_template_kwargs": {"enable_thinking": False}}
        status, result = call("/v1/chat/completions", body)
        assert status == 200, (status, result)
        assert result["usage"]["completion_tokens"] == tokens
        return result["tensorfold"]["token_sha"]

    def stream(entered):
        body = {"model": model, "prompt": "Write a short Python function that computes the Fibonacci sequence and explain it.",
                "max_tokens": tokens, "temperature": 0, "ignore_eos": True, "stream": True,
                "stream_options": {"include_usage": True}}
        if require_cache:
            body["draft"] = False  # The serial twin drains without evicting the retained conversation prefix.
        request = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        result = {}
        with urllib.request.urlopen(request, timeout=1800) as response:
            for line in response:
                if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                    continue
                part = json.loads(line[5:])
                if any(c.get("text") for c in part.get("choices", [])):
                    entered.set()
                if part.get("usage"):
                    result["usage"] = part["usage"]
                if part.get("tensorfold"):
                    result["tensorfold"] = part["tensorfold"]
        assert result["usage"]["completion_tokens"] == tokens
        return result

    assert call("/is_sleeping")[0] == 401
    assert call("/sleep", method="POST", authenticated=True, origin=True)[0] == 403
    assert call("/sleep?level=1", method="POST", authenticated=True)[0] == 400
    initial, _ = control("/is_sleeping", method="GET")
    assert initial["ready"]
    if require_cache:
        assert initial.get("cache", {}).get("mode") == "disk", "server cache preservation is disabled"
    serial, drafted = chat(False), chat()
    assert serial == drafted, "drafted HTTP reply differs from serial"

    common = {"model": model, "max_output_tokens": 32, "temperature": 0,
              "chat_template_kwargs": {"enable_thinking": False}}
    status, first = call("/v1/responses", {**common, "input": "Remember the marker ORCHID-42. Reply with OK."})
    assert status == 200, (status, first)
    continuation = {**common, "previous_response_id": first["id"],
                    "input": "What marker did I ask you to remember? Reply with the marker only."}
    status, before = call("/v1/responses", continuation)
    assert status == 200, (status, before)
    if require_cache:
        status, warm = call("/v1/responses", continuation)
        assert status == 200 and warm["tensorfold"]["token_sha"] == before["tensorfold"]["token_sha"]
        before = warm
        assert before["usage"]["input_tokens_details"]["cached_tokens"] > 0, "awake prefix was not reused"

    entered = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        generating = pool.submit(stream, entered)
        assert entered.wait(120), "stream did not begin"
        sleeping = pool.submit(control, "/sleep?level=2")
        deadline = time.monotonic() + 10
        while True:
            state, _ = control("/is_sleeping", method="GET")
            if state["state"] == "draining":
                break
            assert time.monotonic() < deadline, "draining state was not observed"
            time.sleep(.02)
        assert call("/v1/chat/completions", {})[0] == 503
        assert call("/wake_up", method="POST", authenticated=True)[0] == 409
        streamed = generating.result(timeout=1800)
        asleep, sleep_s = sleeping.result(timeout=1800)

    assert asleep["is_sleeping"] and asleep["memory"]["allocated_bytes"] == 0, asleep
    if require_cache:
        assert asleep["memory"]["reserved_bytes"] == 0 and asleep["cache"]["saved_prefixes"] > 0
    assert call("/v1/completions", {})[0] == 503
    assert call("/v1/models")[0] == 200
    assert call("/metrics")[0] == 200
    status, health = call("/health")
    assert status == 200 and not health["ready"] and health["lifecycle"]["state"] == "sleeping"
    status, stored = call("/v1/responses/" + first["id"])
    assert status == 200 and stored == first, "stored response changed during sleep"
    awake, wake_s = control("/wake_up")
    assert awake["ready"]
    if require_cache:
        assert awake["cache"]["loaded_prefixes"] == 0, "wake loaded prefixes eagerly"
    assert chat() == drafted and chat(False) == serial, "HTTP token hash changed after wake"
    status, after = call("/v1/responses", continuation)
    assert status == 200 and after["previous_response_id"] == first["id"], (status, after)
    assert after["tensorfold"]["token_sha"] == before["tensorfold"]["token_sha"], "conversation continuation changed"
    assert after["usage"]["input_tokens"] == before["usage"]["input_tokens"], "conversation history length changed"
    report = dict(passed=True, reply_tokens=tokens, initial=initial, asleep=asleep, awake=awake,
                  sleep_s=sleep_s, wake_s=wake_s, serial_hash=serial, drafted_hash=drafted,
                  stream_hash=streamed["tensorfold"]["token_sha"],
                  conversation_hash=after["tensorfold"]["token_sha"],
                  conversation_input_tokens=after["usage"]["input_tokens"],
                  checks=["auth", "Origin", "level refusal", "stream drain", "503 admission", "409 conflict",
                          "discovery while asleep", "stored response", "continued conversation", "token equality"])
    if require_cache:
        cached = after["usage"]["input_tokens_details"]["cached_tokens"]
        assert cached == before["usage"]["input_tokens_details"]["cached_tokens"], "conversation prefix reuse lost"
        state, _ = control("/is_sleeping", method="GET")
        assert state["cache"]["loaded_prefixes"] > 0 and state["cache"]["load_failures"] == 0
        report.update(conversation_cached_tokens=cached, restored_cache=state["cache"])
        report["checks"].extend(["zero reserved memory", "lazy cache restore", "conversation prefix reuse"])
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "sleep_s": sleep_s, "wake_s": wake_s,
                      "asleep_allocated_bytes": asleep["memory"]["allocated_bytes"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("model")
    parser.add_argument("--token-env", default="TENSORFOLD_SLEEP_TOKEN")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--require-cache", action="store_true", help="verify disk preservation and cached-token reuse")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    token = os.environ.get(args.token_env, "")
    if not token or args.tokens < 1:
        parser.error("a configured bearer secret and positive token count are required")
    qualify(args.base.rstrip("/"), args.model, token, args.output, tokens=args.tokens, require_cache=args.require_cache)


if __name__ == "__main__":
    main()
