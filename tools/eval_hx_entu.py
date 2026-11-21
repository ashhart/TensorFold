"""A five-turn agent eval of heat-exchanger physics and tool use against an OpenAI-compatible server."""
import argparse
import http.client
import json
import math
import sys
import time
import urllib.error
import urllib.request

from bench_auth import authorize, read_api_key, redact
import eval_sandbox

if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11+ is required (this is {sys.version.split()[0]}); TensorFold's tools use PEP 604 unions.")

TH_IN, TC_IN = 200.0, 20.0  # inlet temperatures, degrees C
TOLERANCE = 0.005  # relative error a value may have and pass
FAR = 0.05  # a duty this far off its reference, with both stream balances exact, is a balance by construction
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
SYSTEM = ("You are an engineering assistant. You have a run_python tool that runs a Python 3 program and returns "
          "what it prints. Use it for every calculation.")
TOOL = {"type": "function", "function": {
    "name": "run_python",
    "description": "Run a complete Python 3 program in a fresh directory and return its standard output and errors. "
                   "Print every value you need.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The program"}},
                   "required": ["code"]}}}
TURNS = (
    {"name": "counterflow, equal capacity rates", "flow": "counterflow", "C_h": 500.0, "C_c": 500.0, "UA": 1000.0,
     "ask": "rate"},
    {"name": "parallel flow, same exchanger", "flow": "parallel-flow", "C_h": 500.0, "C_c": 500.0, "UA": 1000.0,
     "ask": "rate"},
    {"name": "counterflow, cold flow doubled", "flow": "counterflow", "C_h": 500.0, "C_c": 1000.0, "UA": 1000.0,
     "ask": "rate"},
    {"name": "counterflow sizing for Th_out", "flow": "counterflow", "C_h": 500.0, "C_c": 500.0, "Th_out": 100.0,
     "ask": "size"},
    {"name": "turn 1 in 200 cells", "flow": "counterflow", "C_h": 500.0, "C_c": 500.0, "UA": 1000.0, "cells": 200,
     "ask": "cells"},
)
ANSWER = {"rate": '{"q_W": <number>, "Th_out_C": <number>, "Tc_out_C": <number>}',
          "size": '{"UA_W_per_K": <number>}', "cells": '{"Th_out_C": <number>}'}


class EvalError(RuntimeError):
    """A request that got no usable reply; the run stops at the turn it happened in."""


def effectiveness(flow: str, ntu: float, cr: float) -> float:
    """The effectiveness-NTU closed form for one exchanger; counterflow at Cr = 1 takes its limit NTU / (1 + NTU)."""
    if flow == "parallel-flow":
        return (1 - math.exp(-ntu * (1 + cr))) / (1 + cr)
    if math.isclose(cr, 1.0, rel_tol=1e-12):
        return ntu / (1 + ntu)
    e = math.exp(-ntu * (1 - cr))
    return (1 - e) / (1 - cr * e)


def counterflow_ntu(eps: float, cr: float) -> float:
    """The counterflow NTU that gives effectiveness eps, the inverse of effectiveness()."""
    if math.isclose(cr, 1.0, rel_tol=1e-12):
        return eps / (1 - eps)
    return math.log((1 - eps * cr) / (1 - eps)) / (1 - cr)


def rating(flow: str, c_h: float, c_c: float, ua: float) -> dict:
    """Duty and outlet temperatures of an exchanger with the module's inlets."""
    c_min, c_max = min(c_h, c_c), max(c_h, c_c)
    q = effectiveness(flow, ua / c_min, c_min / c_max) * c_min * (TH_IN - TC_IN)
    return {"q_W": q, "Th_out_C": TH_IN - q / c_h, "Tc_out_C": TC_IN + q / c_c}


def reference(turn: dict) -> dict:
    """The values a turn asks for, from the closed forms."""
    if turn["ask"] == "size":
        c_min, c_max = min(turn["C_h"], turn["C_c"]), max(turn["C_h"], turn["C_c"])
        eps = turn["C_h"] * (TH_IN - turn["Th_out"]) / (c_min * (TH_IN - TC_IN))
        return {"UA_W_per_K": counterflow_ntu(eps, c_min / c_max) * c_min}
    rated = rating(turn["flow"], turn["C_h"], turn["C_c"], turn["UA"])
    return {"Th_out_C": rated["Th_out_C"]} if turn["ask"] == "cells" else rated


def prompt(turn: dict) -> str:
    """The user message for a turn: its numbers, the tool and the JSON line, with no hint at its formula's limits."""
    streams = (f"Hot gas: capacity rate C_h = {turn['C_h']:g} W/K, inlet {TH_IN:g} °C; "
               f"cold stream: C_c = {turn['C_c']:g} W/K, inlet {TC_IN:g} °C")
    if turn["ask"] == "rate":
        task = (f"A {turn['flow']} heat exchanger. {streams}. Overall conductance UA = {turn['UA']:g} W/K. "
                "Using the effectiveness-NTU method, find the heat duty q (W) and both outlet temperatures (°C).")
    elif turn["ask"] == "size":
        task = (f"Sizing: a {turn['flow']} heat exchanger. {streams}. The hot gas must leave at "
                f"{turn['Th_out']:g} °C. Using the effectiveness-NTU method, find the UA (W/K) it needs.")
    else:
        task = (f"Check your first answer numerically. Take that {turn['flow']} exchanger again ({streams}; "
                f"UA = {turn['UA']:g} W/K) and split it into N = {turn['cells']} cells in series along its length, "
                "each with UA/N. Solve the cells' energy balances numerically, by iteration or shooting, without "
                "the closed-form effectiveness of the whole exchanger, and report the hot outlet temperature from "
                "that solution.")
    return (f"{task} Use the run_python tool to compute it. End your final answer with one line of JSON: "
            f"{ANSWER[turn['ask']]}")


def answer_json(text: str, keys=()) -> dict | None:
    """The last JSON object in a reply that holds one of keys (any, without keys), among prose or code fences."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    decoder, found, i = json.JSONDecoder(), [], text.find("{")
    while i >= 0:
        try:
            value, end = decoder.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(value, dict):
            found.append(value)
        i = text.find("{", end)
    keyed = [v for v in found if any(k in v for k in keys)]
    return (keyed or found or [None])[-1]


def number(value) -> float | None:
    """A finite number from a JSON value or numeric string, else None."""
    if isinstance(value, bool):
        return None
    try:
        out = float(value.replace(",", "")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def balance_by_construction(turn: dict, got: dict, ref: dict) -> bool:
    """Both stream balances equal q exactly while q is zero or far from its reference: the outlets came from q."""
    if turn["ask"] != "rate" or any(got.get(k) is None for k in ref):
        return False
    q = got["q_W"]
    hot, cold = turn["C_h"] * (TH_IN - got["Th_out_C"]), turn["C_c"] * (got["Tc_out_C"] - TC_IN)
    slack = max(1.0, 1e-3 * abs(q))
    balanced = abs(hot - q) <= slack and abs(cold - q) <= slack
    return balanced and (abs(q) < 1.0 or abs(q - ref["q_W"]) > FAR * abs(ref["q_W"]))


def score(turn: dict, text: str) -> dict:
    """PASS when every requested value is within TOLERANCE of the reference, with each value's relative error."""
    ref = reference(turn)
    parsed = answer_json(text, ref)
    got = {k: number(parsed.get(k)) if parsed else None for k in ref}
    errors = {k: None if got[k] is None else abs(got[k] - r) / abs(r) for k, r in ref.items()}
    missing = [k for k, v in got.items() if v is None]
    if parsed is None:
        reason = "no JSON line in the final answer"
    elif missing:
        reason = "missing or not a number: " + ", ".join(missing)
    else:
        worst = max(errors, key=errors.get)
        reason = "" if errors[worst] <= TOLERANCE else f"{worst} off by {errors[worst]:.2%}"
    return {"status": "FAIL" if reason else "PASS", "reason": reason, "answer": got, "reference": ref,
            "relative_error": errors, "balance_by_construction": balance_by_construction(turn, got, ref)}


class Client:
    """Posts chat requests with the run_python tool to an OpenAI-compatible server."""

    def __init__(self, base_url: str, model: str, fields: dict, api_key: str | None = None, timeout: float = 3600):
        self.url, self.model, self.fields = base_url.rstrip("/"), model, fields
        self.api_key, self.timeout = api_key, timeout

    def request(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        authorize(req, self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", errors="replace")
            raise EvalError(redact(f"HTTP {exc.code}: {detail}", self.api_key)) from None
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise EvalError(f"request failed: {type(exc).__name__}: {exc}") from None

    def chat(self, messages: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "tools": [TOOL], "tool_choice": "auto", **self.fields}
        reply = self.request("/chat/completions", body)
        if not isinstance(reply, dict) or not reply.get("choices"):
            raise EvalError("reply has no choices")
        return reply

    def first_model(self) -> str:
        models = self.request("/models").get("data") or []
        if not models:
            raise EvalError("the server lists no models; pass --model")
        return models[0]["id"]


def call_code(call: dict) -> str | None:
    """The code string of a well-formed run_python call: valid JSON arguments holding a code string."""
    function = call.get("function") or {}
    args = function.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if function.get("name") != "run_python" or not isinstance(args, dict) or not isinstance(args.get("code"), str):
        return None
    return args["code"]


def run_turn(client: Client, sandbox: eval_sandbox.Sandbox, messages: list[dict], stats: dict, max_calls: int,
             log: list[dict]) -> str:
    """Send, run tool calls and send again until the model answers without one or the call limit ends the turn."""
    while True:
        reply = client.chat(messages)
        stats["requests"] += 1
        usage = reply.get("usage") or {}
        stats["completion_tokens"] += usage.get("completion_tokens") or 0
        thought = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if isinstance(thought, int):
            stats["thinking_tokens"] = (stats["thinking_tokens"] or 0) + thought
        choice = reply["choices"][0]
        message, stats["finish_reason"] = choice.get("message") or {}, choice.get("finish_reason")
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        stats["reasoning_chars"] += len(reasoning)
        calls = [c for c in message.get("tool_calls") or [] if isinstance(c, dict)]
        sent = {"role": "assistant", "content": message.get("content") or ""}
        if reasoning:
            sent["reasoning_content"] = reasoning
        if calls:
            for n, call in enumerate(calls):
                call.setdefault("id", f"call_{stats['requests']}_{n}")
                call.setdefault("type", "function")
            sent["tool_calls"] = calls
        messages.append(sent)
        if not calls:
            stats["ended"] = "answer"
            return sent["content"]
        for call in calls:
            stats["tool_calls"] += 1
            code = call_code(call)
            stats["well_formed"] += code is not None
            if stats["tool_calls"] > max_calls:
                output = "[not run: the tool-call limit for this question was reached]"
            elif code is None:
                output = "[not run: call run_python with JSON arguments holding a code string]"
            else:
                result = sandbox.run(code)
                stats["exec_errors"] += result["error"]
                output = result["text"]
                log.append({"code": code, "ended": result["ended"], "exit": result["exit"], "output": output})
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": "run_python", "content": output})
        if stats["tool_calls"] >= max_calls:
            stats["ended"] = "tool-call limit"
            return ""


def run_conversation(client: Client, sandbox: eval_sandbox.Sandbox, max_calls: int) -> dict:
    """One whole conversation; a transport error ends it at that turn."""
    messages, turns = [{"role": "system", "content": SYSTEM}], []
    for index, turn in enumerate(TURNS, 1):
        messages.append({"role": "user", "content": prompt(turn)})
        stats = {"tool_calls": 0, "well_formed": 0, "exec_errors": 0, "requests": 0, "completion_tokens": 0,
                 "thinking_tokens": None, "reasoning_chars": 0, "finish_reason": None, "ended": None}
        log, start = [], time.perf_counter()
        try:
            text = run_turn(client, sandbox, messages, stats, max_calls, log)
            result = score(turn, text)
            if stats["ended"] == "tool-call limit":
                result.update(status="FAIL", reason=f"no answer within {max_calls} tool calls")
            elif result["status"] == "FAIL" and stats["finish_reason"] == "length":
                result["reason"] += "; the reply hit its token limit"
        except EvalError as exc:
            text, result = "", {"status": "ERROR", "reason": str(exc), "reference": reference(turn)}
        turns.append({"turn": index, "name": turn["name"], **result, **stats,
                      "wall_s": round(time.perf_counter() - start, 3), "final_text": text[-2000:], "executions": log})
        if result["status"] == "ERROR":
            break
    return {"turns": turns, "turns_reached": sum(t["status"] != "ERROR" for t in turns),
            "passed": sum(t["status"] == "PASS" for t in turns), "messages": messages}


def fmt(value: float | None) -> str:
    return "none" if value is None else f"{value:.6g}"


def describe_turn(t: dict) -> list[str]:
    """Two summary lines for a turn: verdict with values against references, then its tool and token counts."""
    if t["status"] == "ERROR":
        values = t["reason"][:160]
    else:
        values = ", ".join(f"{k} {fmt(t['answer'][k])} (ref {fmt(r)})" for k, r in t["reference"].items())
        values += f"; {t['reason']}" if t["reason"] else ""
        values += "; energy balance by construction" if t.get("balance_by_construction") else ""
    thinking = "" if t["thinking_tokens"] is None else f", thinking tokens {t['thinking_tokens']}"
    counts = (f"    tool calls {t['tool_calls']} ({t['well_formed']} well-formed, {t['exec_errors']} errored), "
              f"completion tokens {t['completion_tokens']}{thinking}, {t['wall_s']:.1f} s")
    return [f"  {t['turn']} {t['name']:<34} {t['status']:<5} {values}", counts]


def summary(runs: list[dict]) -> str:
    """The human summary: each run's turns, then totals over every run."""
    lines = []
    for n, run in enumerate(runs, 1):
        lines.append(f"run {n}: {run['passed']}/{len(TURNS)} turns pass, {run['turns_reached']} reached")
        for t in run["turns"]:
            lines += describe_turn(t)
    turns = [t for run in runs for t in run["turns"]]
    thinking = [t["thinking_tokens"] for t in turns if t["thinking_tokens"] is not None]
    total = {k: sum(t[k] for t in turns) for k in ("tool_calls", "well_formed", "exec_errors", "completion_tokens")}
    thought = f", thinking tokens {sum(thinking)}" if thinking else ""
    lines.append(f"totals: tool calls {total['tool_calls']} ({total['well_formed']} well-formed, "
                 f"{total['exec_errors']} errored), completion tokens {total['completion_tokens']}{thought}, "
                 f"{sum(t['wall_s'] for t in turns):.1f} s")
    passes = [sum(t["turn"] == i and t["status"] == "PASS" for t in turns) for i in range(1, len(TURNS) + 1)]
    per_turn = ", ".join(f"t{i} {n}/{len(runs)}" for i, n in enumerate(passes, 1))
    lines.append(f"{sum(r['passed'] == len(TURNS) for r in runs)}/{len(runs)} runs pass every turn; turn passes: "
                 f"{per_turn}; turns reached: {', '.join(str(r['turns_reached']) for r in runs)}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, epilog="example: python3 tools/eval_hx_entu.py --model "
                                "local-model --reasoning-effort low --runs 3 --out hx.json")
    p.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="the server's /v1 URL")
    p.add_argument("--model", help="model name to request (default: the first the server lists)")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tool-calls", type=int, default=8, help="tool calls a turn may make before it ends")
    p.add_argument("--reasoning-effort", choices=EFFORTS, help="sent as reasoning_effort; none turns thinking off")
    p.add_argument("--thinking-budget", type=int, metavar="TOKENS",
                   help="sent as thinking_budget: the server closes the reasoning after this many tokens")
    p.add_argument("--max-tokens", type=int, help="reply token limit per request (default: the server's)")
    p.add_argument("--seed", type=int, help="seed for run 1; run N sends seed + N - 1")
    p.add_argument("--runs", type=int, default=1, help="repeat the whole conversation this many times")
    p.add_argument("--exec-timeout", type=float, default=30.0, help="wall seconds a tool program may run")
    p.add_argument("--exec-memory-mb", type=int, default=2048, help="memory a tool program may use")
    p.add_argument("--tool-output-chars", type=int, default=4000, help="tool output kept for the model")
    p.add_argument("--request-timeout", type=float, default=3600.0, help="seconds to wait for one reply")
    p.add_argument("--api-key-env", metavar="NAME", help="read the bearer API key from this environment variable")
    p.add_argument("--unsafe-no-sandbox", action="store_true",
                   help="run model code with rlimits only when this host has no sandbox")
    p.add_argument("--out", help="write the results JSON here")
    args = p.parse_args(argv)
    for name in ("max_tool_calls", "runs", "exec_memory_mb", "tool_output_chars", "thinking_budget", "max_tokens"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be greater than 0")
    if not math.isfinite(args.temperature) or args.exec_timeout <= 0 or args.request_timeout <= 0:
        p.error("--temperature must be finite and the timeouts greater than 0")
    args.api_key = read_api_key(p, args.api_key_env)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sandbox = eval_sandbox.Sandbox.detect(args.unsafe_no_sandbox, warn=lambda m: print(m, file=sys.stderr),
                                              timeout_s=args.exec_timeout, memory_mb=args.exec_memory_mb,
                                              output_chars=args.tool_output_chars)
    except eval_sandbox.SandboxUnavailable as exc:
        print(f"refusing to run model code: {exc}", file=sys.stderr)
        return 2
    fields = {"temperature": args.temperature}
    for key in ("reasoning_effort", "thinking_budget", "max_tokens"):
        if getattr(args, key) is not None:
            fields[key] = getattr(args, key)
    client = Client(args.base_url, args.model or "", fields, args.api_key, args.request_timeout)
    try:
        client.model = args.model or client.first_model()
    except (EvalError, KeyError, TypeError, AttributeError) as exc:
        print(f"cannot find a model: {exc}", file=sys.stderr)
        return 2
    print(f"eval_hx_entu: {client.model} at {args.base_url}, sandbox {sandbox.kind}, {args.runs} run(s)", flush=True)
    runs = []
    for n in range(args.runs):
        if args.seed is not None:
            client.fields["seed"] = args.seed + n
        runs.append(run_conversation(client, sandbox, args.max_tool_calls))
    print(redact(summary(runs), args.api_key))
    settings = {k: v for k, v in vars(args).items() if k not in ("api_key", "api_key_env")}
    result = {"eval": "hx_entu", "model": client.model, "sandbox": sandbox.kind, "settings": settings,
              "request_fields": fields, "tolerance": TOLERANCE, "runs": runs,
              "runs_passing_every_turn": sum(r["passed"] == len(TURNS) for r in runs)}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(redact(result, args.api_key), f, indent=1)
    return 0 if result["runs_passing_every_turn"] == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
