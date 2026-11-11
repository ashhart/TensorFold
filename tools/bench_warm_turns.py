"""Measure a chat turn and its prefetched continuation without retaining prompt or reply text."""
import argparse
import hashlib
import json
import sys
import time
import urllib.request


def request_json(base, path, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(base + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def load(base):
    return request_json(base, "/v1/load")


def public_load(value):
    return {key: item for key, item in value.items() if key not in ("instance", "timestamp", "model")}


def tokenize(base, messages):
    body = {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}}
    return request_json(base, "/v1/tokenize", body)["tokens"]


def stream_chat(base, model, messages, max_tokens, seed):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "seed": seed, "stream": True, "stream_options": {"include_usage": True},
            "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    sent = time.perf_counter()
    first_output = None
    contents, reasoning = [], []
    usage, runtime = {}, {}
    finish_reason = None
    done = False
    with urllib.request.urlopen(req, timeout=900) as response:
        headers = time.perf_counter()
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            runtime = event.get("tensorfold") or runtime
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                thought = delta.get("reasoning_content") or ""
                if content or thought:
                    first_output = first_output or time.perf_counter()
                contents.append(content)
                reasoning.append(thought)
                finish_reason = choice.get("finish_reason") or finish_reason
    ended = time.perf_counter()
    content, thought = "".join(contents), "".join(reasoning)
    details = usage.get("prompt_tokens_details") or {}
    return {"prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": details.get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "client_ttft_seconds":
            round(first_output - sent, 6) if first_output else None,
            "client_headers_seconds": round(headers - sent, 6), "client_end_to_end_seconds": round(ended - sent, 6),
            "engine_prefill_seconds": runtime.get("prefill_seconds"),
            "engine_time_to_first_token_seconds": runtime.get("time_to_first_token"),
            "prefill_widths": runtime.get("prefill_widths"), "token_sha256": runtime.get("token_sha256"),
            "token_sha": runtime.get("token_sha"), "output_sha256": hashlib.sha256(
                json.dumps([content, thought], ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "finish_reason": finish_reason, "complete": done and finish_reason is not None,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(), "_content": content}


def int_at(value, *path):
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if type(value) is int and value >= 0 else None


def run(args):
    base = args.base.rstrip("/")
    fixture = (f"Warm-prefill qualification {args.case_id}.\n" + "alpha beta gamma delta\n" * args.repeat_lines
               + "Answer with only the word READY.")
    user = {"role": "user", "content": fixture}
    first_messages = [user]
    initial_ids = tokenize(base, first_messages)
    before = load(base)
    first = stream_chat(base, args.model, first_messages, args.max_tokens, args.seed)
    first_finished_at = time.perf_counter()
    answer = first.pop("_content", None)
    if answer is None:
        raise RuntimeError("first turn returned no answer")
    follow_user = {"role": "user", "content": "What should the next line say? Answer with one word."}
    history = [user, {"role": "assistant", "content": answer}, follow_user]
    continuation_ids = tokenize(base, history)
    a_ids = tokenize(base, [user, {"role": "assistant", "content": answer}, {"role": "user", "content": "A"}])
    b_ids = tokenize(base, [user, {"role": "assistant", "content": answer}, {"role": "user", "content": "B"}])
    common = next((i for i, pair in enumerate(zip(a_ids, b_ids)) if pair[0] != pair[1]), min(len(a_ids), len(b_ids)))
    after_first = load(base)
    ready = False
    waited = 0.0
    started_before = int_at(before, "requests_started") or 0
    warm_started = False
    if args.phase == "ready":
        start_wait = time.perf_counter()
        deadline = start_wait + args.warm_timeout
        while time.perf_counter() < deadline:
            state = load(base)
            warm_started = (int_at(state, "requests_started") or 0) >= started_before + 2
            running = int_at(state, "running")
            if warm_started and running == 0:
                ready = True
                break
            time.sleep(0.05)
        waited = round(time.perf_counter() - start_wait, 6)
    elif args.phase == "yield":
        time.sleep(args.warm_delay_ms / 1000)
        waited = args.warm_delay_ms / 1000
    else:
        time.sleep(args.warm_delay_ms / 1000)
        waited = args.warm_delay_ms / 1000
    before_continuation = load(base)
    warm_started = warm_started or (int_at(before_continuation, "requests_started") or 0) >= started_before + 2
    warm_running_at_continuation = (int_at(before_continuation, "running") or 0) > 0
    continuation_sent_at = time.perf_counter()
    continuation = stream_chat(base, args.model, history, args.max_tokens, args.seed)
    continuation.pop("_content", None)
    after_continuation = load(base)
    warm_yields_before = int_at(after_first, "background_yields") or 0
    warm_yields_after = int_at(after_continuation, "background_yields") or 0
    candidate_cached = continuation.get("cached_tokens")
    if args.phase == "cold":
        cache_ok = first.get("cached_tokens") == 0
    else:
        cache_ok = type(candidate_cached) is int and candidate_cached >= min(4096, common)
    yield_ok = args.phase != "yield" or (warm_running_at_continuation and warm_yields_after > warm_yields_before)
    crossed_grid = len(initial_ids) < 4096 <= common
    compare = None
    if args.compare_file:
        with open(args.compare_file, encoding="utf-8") as source:
            reference = json.load(source)
        ref_first = reference.get("first_turn", {})
        ref_cont = reference.get("continuation", {})
        compare = {"first_token_ids_equal": bool(first.get("token_sha256") and
                                                   first["token_sha256"] == ref_first.get("token_sha256")),
                   "continuation_token_ids_equal": bool(continuation.get("token_sha256") and
                                                         continuation["token_sha256"] == ref_cont.get("token_sha256")),
                   "first_output_equal": first.get("output_sha256") == ref_first.get("output_sha256"),
                   "continuation_output_equal": continuation.get("output_sha256") == ref_cont.get("output_sha256")}
    report = {"schema_version": 1, "phase": args.phase, "case_id": args.case_id, "model": args.model,
              "prompt_fixture_sha256": hashlib.sha256(fixture.encode("utf-8")).hexdigest(),
              "repeat_lines": args.repeat_lines, "temperature": 0, "seed": args.seed,
              "context_window": request_json(base, "/v1/models")["data"][0].get("context_length"),
              "capabilities": request_json(base, "/v1/models")["data"][0].get("capabilities", {}),
              "first_prompt_tokens_tokenizer": len(initial_ids), "shared_prefix_tokens_tokenizer": common,
              "continuation_prompt_tokens_tokenizer": len(continuation_ids), "wait_seconds": waited,
              "first_turn_to_continuation_seconds": round(continuation_sent_at - first_finished_at, 6),
              "warm_ready_observed": ready, "warm_request_started": warm_started,
              "warm_running_at_continuation": warm_running_at_continuation, "first_turn": first,
              "continuation": continuation, "background_yields_delta": warm_yields_after - warm_yields_before,
              "load_before": public_load(before), "load_after_first_turn": public_load(after_first),
              "load_before_continuation": public_load(before_continuation),
              "load_after_continuation": public_load(after_continuation),
              "comparison": compare,
              "assertions": {"first_prompt_tokenizer_count_matches_usage": first.get("prompt_tokens") == len(initial_ids),
                             "continuation_tokenizer_count_matches_usage": continuation.get("prompt_tokens") == len(continuation_ids),
                             "warm_prefix_crosses_4096_grid": args.phase == "cold" or crossed_grid,
                             "first_turn_starts_cold": first.get("cached_tokens") == 0,
                             "warm_pass_started": args.phase == "cold" or warm_started,
                             "continuation_cache_requirement": cache_ok, "yield_observed_when_requested": yield_ok,
                             "first_turn_complete": first.get("complete") is True,
                             "continuation_complete": continuation.get("complete") is True,
                             "comparison_exact_when_requested": compare is None or all(compare.values())}}
    report["qualified"] = all(report["assertions"].values())
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as out:
            out.write(encoded)
    print(encoded, end="")
    return 0 if report["qualified"] else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", help="native server base URL")
    p.add_argument("model")
    p.add_argument("--phase", choices=("cold", "ready", "yield"), required=True,
                   help="cold: do not wait for a candidate; ready: wait for the candidate pass; yield: arrive during it")
    p.add_argument("--case-id", default="4k")
    p.add_argument("--repeat-lines", type=int, default=820)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--warm-delay-ms", type=float, default=200)
    p.add_argument("--warm-timeout", type=float, default=120)
    p.add_argument("--compare-file")
    p.add_argument("--output")
    args = p.parse_args(argv)
    if args.repeat_lines < 1 or args.max_tokens < 1 or args.warm_delay_ms < 0 or args.warm_timeout < 0:
        p.error("repeat-lines and max-tokens must be positive; delays cannot be negative")
    try:
        return run(args)
    except Exception as exc:  # Keep failures concise and avoid copying server error bodies into receipts.
        print(f"qualification error: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
