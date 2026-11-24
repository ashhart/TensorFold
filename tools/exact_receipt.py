"""Collection and comparison helpers for tools/exact_restarts.py's public exact receipt."""
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PUBLIC = re.compile(r"[A-Za-z0-9_.:+@-]{1,128}\Z")


def public_id(value, field):
    """A caller-supplied public identifier, never a path, address, account or free-form text."""
    if not isinstance(value, str) or not PUBLIC.fullmatch(value) or "@" in value or value.startswith("~"):
        raise ValueError(f"--{field}-id must be a public identifier without paths, accounts or whitespace")
    return value


def token_sha256(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def comparison(kind, prompt, left_id, left, right_id, right):
    pos = first_diff(left["token_ids"], right["token_ids"])
    return {"kind": kind, "prompt": prompt, "reference_request": left_id, "candidate_request": right_id,
            "reference_token_ids_json_sha256": left["token_ids_json_sha256"],
            "candidate_token_ids_json_sha256": right["token_ids_json_sha256"],
            "first_diff": pos, "equal": pos is None}


def compare_results(reference, candidate, kind):
    """Compare candidate-request keys; a restart candidate may be a focused subset of the first run."""
    left = reference.get("results", reference)
    right = candidate.get("results", candidate)
    return [comparison(kind, v["prompt"], k, left[k], k, v) for k, v in right.items() if k in left]


def result(request, item, sampling, stop, streams, draft, phase, peer, ids, cached, finish, wall, round_wall,
           drafted, accepted):
    return {"request": request, "prompt": item["name"], "sampling": sampling, "stop": stop, "streams": streams,
            "draft": draft, "phase": phase, "peer": peer, "token_ids": ids,
            "token_ids_json_sha256": token_sha256(ids), "token_count": len(ids), "cached_tokens": cached,
            "finish_reason": finish, "wall_seconds": wall, "round_wall_seconds": round_wall,
            "drafted_tokens": drafted, "accepted_tokens": accepted}


def request_body(item, tokens, sampling, stop, seed, draft, followup=False, reply=""):
    sampled = sampling == "seeded"
    body = {"model": "served", "max_tokens": tokens, "temperature": 0.8 if sampled else 0,
            "top_p": 0.95 if sampled else 1, "top_k": 20 if sampled else 0, "min_p": 0,
            "ignore_eos": stop == "budget", "return_token_ids": True, "draft": draft, "stream": False}
    if sampled:
        body["seed"] = seed
    if item["kind"] == "chat":
        messages = [{"role": "user", "content": item["prompt"]}]
        if followup:
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": item.get("followup", "Continue.")}]
        body["messages"] = messages
    elif isinstance(item["prompt"], list):
        suffix = (reply + item["followup_token_ids"]) if followup else []
        body["prompt"] = item["prompt"] + suffix
    else:
        body["prompt"] = item["prompt"] + (("\n" + reply + "\n" + item.get("followup", "Continue."))
                                           if followup else "")
    return body


def request_name(item, sampling, stop, streams, draft, phase, peer):
    return "/".join((item["name"], sampling, stop, f"s{streams}", "draft" if draft else "plain", phase,
                     f"peer{peer}"))


def fetch(base, spec, api_key, timeout):
    body, request = spec["body"], spec["request"]
    path = "/v1/chat/completions" if "messages" in body else "/v1/completions"
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            reply = json.load(response)
        wall = time.perf_counter() - started
        runtime = reply.get("tensorfold") or {}
        if runtime.get("drafts") is not body["draft"]:
            raise ValueError("reply did not acknowledge the requested draft mode")
        ids = reply.get("tensorfold", {}).get("token_ids")
        usage = reply.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        prompt = usage.get("prompt_tokens")
        if not isinstance(ids, list) or any(type(x) is not int or x < 0 or x >= 2**32 for x in ids):
            raise ValueError("reply carried no valid tensorfold.token_ids")
        if type(cached) is not int or type(prompt) is not int or not 0 <= cached <= prompt:
            raise ValueError("reply carried no valid cached/prompt token counts")
        if usage.get("completion_tokens") != len(ids):
            raise ValueError("reply completion token count did not match tensorfold.token_ids")
        choice = (reply.get("choices") or [{}])[0]
        finish = choice.get("finish_reason")
        if finish not in ("stop", "length"):
            raise ValueError("reply carried no valid finish reason")
        speculative = reply.get("speculative") or {}
        content = ((choice.get("message") or {}).get("content") if "messages" in body else choice.get("text")) or ""
        return {"ok": True, "ids": ids, "cached": cached, "prompt": prompt, "finish": finish, "wall": wall,
                "drafted": speculative.get("drafted") or 0, "accepted": speculative.get("accepted") or 0,
                "content": content}
    except urllib.error.HTTPError as e:
        return {"ok": False, "request": request, "reason": f"HTTP {e.code}"}
    except (urllib.error.URLError, TimeoutError, OSError):
        return {"ok": False, "request": request, "reason": "request transport failure"}
    except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as e:
        reason = str(e) if isinstance(e, ValueError) and str(e).startswith("reply ") else "invalid response"
        return {"ok": False, "request": request, "reason": reason}


def spec_for(item, args, sampling, stop, streams, draft, phase, peer, followup=False, reply=""):
    tokens = len(item["expected_token_ids"]) if stop == "budget" else args.eos_tokens
    return {"request": request_name(item, sampling, stop, streams, draft, phase, peer), "item": item,
            "sampling": sampling, "stop": stop, "streams": streams, "draft": draft, "phase": phase, "peer": peer,
            "budget": tokens, "body": request_body(item, tokens, sampling, stop, args.seed, draft, followup, reply)}


def check_result(out, value):
    request, ids, item = value["request"], value["token_ids"], value["_item"]
    if not ids:
        out["failures"].append({"request": request, "reason": "empty reply makes equality vacuous"})
    expected = "length" if value["stop"] == "budget" else "stop"
    if value["finish_reason"] != expected:
        out["failures"].append({"request": request,
                                "reason": f"finish {value['finish_reason']}; expected {expected} for {value['stop']}"})
    if value["stop"] == "budget" and value["token_count"] != value["_budget"]:
        out["failures"].append({"request": request, "reason": "token-budget stop returned the wrong token count"})
    if value["phase"] in ("cold", "fresh") and value["cached_tokens"] != 0:
        out["failures"].append({"request": request, "reason": "cold/fresh request reused cached tokens"})
    if value["phase"] in ("warm", "resumed") and value["cached_tokens"] == 0:
        out["failures"].append({"request": request, "reason": "cache miss on warm/resumed request"})
    if value["draft"] and value["drafted_tokens"] <= 0:
        out["failures"].append({"request": request, "reason": "drafting was not observed in this request"})
    if not value["draft"] and value["drafted_tokens"] != 0:
        out["failures"].append({"request": request, "reason": "plain request performed draft work"})
    source = value["sampling"] == "greedy" and value["stop"] == "budget" and value["streams"] == 1 \
        and not value["draft"] and value["phase"] == "cold"
    if source and ids != item["expected_token_ids"]:
        out["failures"].append({"request": request, "reason": "recorded greedy window not reproduced"})


def perform_round(out, base, specs, args):
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        replies = list(pool.map(lambda s: fetch(base, s, args.api_key, args.timeout), specs))
    round_wall = time.perf_counter() - started
    out["rounds"].append({"request_ids": [s["request"] for s in specs], "streams": len(specs),
                          "round_wall_seconds": round_wall})
    made = {}
    for spec, reply in zip(specs, replies):
        if not reply["ok"]:
            out["failures"].append({"request": spec["request"], "reason": reply["reason"]})
            continue
        value = result(spec["request"], spec["item"], spec["sampling"], spec["stop"], spec["streams"],
                       spec["draft"], spec["phase"], spec["peer"], reply["ids"], reply["cached"],
                       reply["finish"], reply["wall"], round_wall, reply["drafted"], reply["accepted"])
        value["_budget"], value["_item"], value["content"] = spec["budget"], spec["item"], reply["content"]
        value["prompt_tokens"] = reply["prompt"]
        check_result(out, value)
        out["results"][spec["request"]] = value
        made[spec["peer"]] = value
    return made


def pair_phase(out, kind, left, right, peers):
    for peer in peers:
        if peer in left and peer in right:
            out["comparisons"].append(comparison(kind, left[peer]["prompt"], left[peer]["request"], left[peer],
                                                 right[peer]["request"], right[peer]))
            if kind == "fresh_vs_resumed" and left[peer]["prompt_tokens"] != right[peer]["prompt_tokens"]:
                out["failures"].append({"request": right[peer]["request"],
                                        "reason": "fresh and resumed prompt token counts differ"})


def run_matrix(base, prompts, args, fresh_base=None):
    """All exact cells on a cache-enabled base and a separate cache-disabled fresh reference."""
    out = {"results": {}, "comparisons": [], "failures": [], "rounds": []}
    streams = sorted(set(args.streams))
    drafts = [False] if "draft" in args.skip else [False, True]
    for sampling in args.sampling:
        for stop in args.stops:
            for draft in drafts:
                for count in streams:
                    for item in prompts:
                        qualified = sampling == "greedy" and stop == "budget" and not draft and count == 1
                        initial, repeated = ("cold", "warm") if qualified else ("first", "repeat")
                        initial_specs = [spec_for(item, args, sampling, stop, count, draft, initial, peer)
                                         for peer in range(count)]
                        first = perform_round(out, base, initial_specs, args)
                        repeat_specs = [spec_for(item, args, sampling, stop, count, draft, repeated, peer)
                                        for peer in range(count)]
                        repeat = perform_round(out, base, repeat_specs, args)
                        pair_phase(out, "cold_vs_warm" if qualified else "repeatability", first, repeat, range(count))
                        if count != 1 or 0 not in first:
                            continue
                        reply = first[0]["token_ids"] if isinstance(item["prompt"], list) else first[0]["content"]
                        fresh_spec = spec_for(item, args, sampling, stop, 1, draft, "fresh", 0, True, reply)
                        resumed_spec = spec_for(item, args, sampling, stop, 1, draft, "resumed", 0, True, reply)
                        if fresh_base is None:
                            out["failures"].append({"request": fresh_spec["request"],
                                                    "reason": "cache-disabled fresh endpoint was not supplied"})
                            continue
                        fresh = perform_round(out, fresh_base, [fresh_spec], args)
                        resumed = perform_round(out, base, [resumed_spec], args)
                        pair_phase(out, "fresh_vs_resumed", fresh, resumed, [0])
                        if sampling == "seeded" and not draft and 0 in first:
                            other = spec_for(item, args, sampling, stop, 1, False, "seed-control", 0)
                            other["body"]["seed"] = args.seed + 1
                            changed = perform_round(out, base, [other], args)
                            if 0 in changed and first[0]["token_ids"] == changed[0]["token_ids"]:
                                out["failures"].append({"request": other["request"],
                                                        "reason": "changed seed gave identical tokens; seed unproven"})
    add_cross_cell_comparisons(out)
    cases = [{"prompt": p["name"], "recording_sha256": p["recording_sha256"],
              "near_tie_offset": p["near_tie_offset"], "source_position": p.get("source_position")} for p in prompts]
    out["coverage"] = {"streams": streams, "sampling": args.sampling, "stops": args.stops, "draft": drafts,
                       "phases": ["cold", "warm", "first", "repeat", "fresh", "resumed", "seed-control"],
                       "near_tie_cases": cases, "fresh_reference": "cache-disabled" if fresh_base else "missing"}
    return out


def add_cross_cell_comparisons(out):
    results = out["results"]
    for key, value in list(results.items()):
        if value["draft"]:
            plain = key.replace("/draft/", "/plain/")
            if plain in results:
                out["comparisons"].append(comparison("drafted_vs_plain", value["prompt"], plain, results[plain],
                                                     key, value))
        if value["streams"] > 1:
            solo = key.replace(f"/s{value['streams']}/", "/s1/").replace(f"/peer{value['peer']}", "/peer0")
            if solo in results:
                out["comparisons"].append(comparison("solo_vs_concurrent", value["prompt"], solo, results[solo],
                                                     key, value))


def public_matrix(matrix):
    """Drop raw token ids and reply text; retain only the public hashes and counted/timed metadata."""
    requests = {k: {key: v for key, v in value.items() if not key.startswith("_") and key != "content"
                    and key != "token_ids"} for k, value in matrix["results"].items()}
    return {**{k: v for k, v in matrix.items() if k != "results"}, "requests": requests}


def binary_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
