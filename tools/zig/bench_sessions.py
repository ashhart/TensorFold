#!/usr/bin/env python3
"""Long agent sessions on one shared harness, sent in turn patterns; one JSON line per request (#2346)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11+ is required (this is {sys.version.split()[0]}); try python3.12 tools/zig/bench_sessions.py")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_timing  # noqa: E402
from bench_auth import authorize, read_api_key, redact  # noqa: E402
from bench_evidence import Evidence, count, token_equal  # noqa: E402
from metrics_text import change, series  # noqa: E402
from reuse_served import TOOLS  # noqa: E402

USAGE = """Every session is the shared harness (a system message of about --system-tokens, then reuse_served's six
tools) followed by its own history, packed from the --text files in --chunk-tokens pieces (a user task, assistant tool
calls and tool results, now and then a user message) to --history-tokens rendered prompt tokens, counted by
/v1/tokenize.
Each session first sends one cold request. Each turn then resends the last reply's content (and its tool calls) plus a
new tool result, or user message when the reply called no tool, of --turn-tokens, streamed at temperature 0.

patterns (sessions A, B and C; any further --sessions only add their cold request):
  alt      A B A B ...                        --turns rounds
  burst    --burst A turns, then as many B turns, until each session has --turns
  three    A B C A B C ...
  overlap  A, then B --overlap-gap s later, then C --overlap-gap s after B while both run;
           the cold requests and every turn round are sent this way

examples:
  python3 tools/zig/bench_sessions.py http://127.0.0.1:8080 MODEL --text a.txt b.txt --pattern alt --out alt.jsonl
  python3 tools/zig/bench_sessions.py http://127.0.0.1:8080 MODEL --text a.txt --pattern three --turns 6 \\
      --replay-url http://127.0.0.1:8081 --replay-session B --out three.jsonl

--replay-url (for example a server started with --prompt-cache-gib 0) receives the replay session's exact request
bodies after the run, in order, and each reply's token_sha256 is compared with the first server's.
Output, to --out and stdout: a plan line, one line per request (with /stats and the prompt-cache series of /metrics
before and after, when the server serves them), the replay lines and a summary. The exit status is 1 when a request
fails or a replayed reply differs. --plan-only builds the histories, writes the plan line and sends no chat request."""
NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NEEDS = {"alt": 2, "burst": 2, "three": 3, "overlap": 3}
COUNTER = re.compile(r"cache.*(hit|miss|evict)|(hit|miss|evict).*cache", re.IGNORECASE)
CUT_TRIES = 8
FIX_TRIES = 8
FIRST_OVERHEAD = 16  # tokens a template adds around one message, until the first measurement replaces it


class Client:
    """One server's /v1/tokenize, /stats, /metrics and streamed /v1/chat/completions."""

    def __init__(self, base: str, model: str, key: str | None, timeout: float, thinking: bool):
        self.base, self.model, self.key, self.timeout, self.thinking = base.rstrip("/"), model, key, timeout, thinking

    def open(self, path: str, body: dict | bytes | None = None):
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"} if data is not None else {})
        authorize(req, self.key)
        return urllib.request.urlopen(req, timeout=self.timeout)

    def tokenize(self, body: dict) -> dict:
        with self.open("/v1/tokenize", {"model": self.model, **body}) as r:
            got = json.loads(r.read())
        if not isinstance(got, dict) or count(got.get("count")) is None:
            raise ValueError("/v1/tokenize returned no count")
        return got

    def text_tokens(self, text: str) -> int:
        """A plain text's token count, without special tokens."""
        return self.tokenize({"prompt": text, "add_special_tokens": False})["count"]

    def rendered(self, messages: list) -> tuple[int, int | None, list | None]:
        """The chat prompt's length as the chat route renders it (harness tools, thinking flag), context and ids."""
        got = self.tokenize({"messages": messages, "tools": TOOLS, "add_generation_prompt": True,
                             "chat_template_kwargs": {"enable_thinking": self.thinking}})
        ids = got.get("tokens")
        return got["count"], count(got.get("max_model_len")), ids if isinstance(ids, list) else None

    def text(self, path: str) -> str | None:
        """A GET route's body when the server answers it; None otherwise."""
        try:
            with self.open(path) as r:
                return r.read().decode("utf-8", "replace")
        except OSError as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            return None

    def stats(self) -> dict | None:
        """GET /stats when the server serves it (native --dashboard); None otherwise."""
        try:
            got = json.loads(self.text("/stats") or "null")
        except ValueError:
            return None
        return got if isinstance(got, dict) else None

    def chat(self, data: bytes) -> dict:
        """One streamed reply: client times, usage, the tensorfold block, the answer text and its tool calls."""
        run = {"sent": time.perf_counter(), "pieces": [], "complete": False, "error": None, "finish": None}
        evidence, usage, runtime, spec, text, calls = Evidence(), {}, {}, None, [], {}
        try:
            with self.open("/v1/chat/completions", data) as resp:
                for raw in resp:
                    line = raw.decode().strip()
                    if line == "data: [DONE]":
                        run["complete"] = True
                        break
                    if not line.startswith("data:"):
                        continue
                    chunk = json.loads(line[5:])
                    if "error" in chunk:
                        raise RuntimeError("server_stream_error")
                    usage = chunk.get("usage") or usage
                    runtime = chunk.get("tensorfold") or runtime
                    spec = chunk.get("speculative") or spec
                    for choice in chunk.get("choices") or []:
                        evidence.add(choice)
                        delta = choice.get("delta") or {}
                        run["finish"] = choice.get("finish_reason") or run["finish"]
                        for call in delta.get("tool_calls") or []:
                            merge(calls, call)
                        text.append(delta.get("content") or "")
                        if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                            run["pieces"].append((time.perf_counter(), len(text[-1])))
        except Exception as exc:  # noqa: BLE001 - a failed request is recorded, not fatal
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            run["error"], run["complete"] = type(exc).__name__, False
        run["completed"] = time.perf_counter()
        run.update(evidence.finish(usage, runtime, run["complete"]))
        details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
        run["reasoning_tokens"] = count(details.get("reasoning_tokens")) if isinstance(details, dict) else None
        run["tensorfold"] = {k: v for k, v in runtime.items() if k != "token_ids"} if isinstance(runtime, dict) else {}
        run["speculative"] = spec if isinstance(spec, dict) else None
        run["content"] = "".join(text)
        run["calls"] = [calls[k] for k in sorted(calls)] if run["finish"] != "length" else []  # a cut call is dropped
        return run


def merge(calls: dict, call: dict) -> None:
    """Join one streamed tool-call delta into the call at its index."""
    at = call.get("index") if type(call.get("index")) is int else 0
    slot = calls.setdefault(at, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
    slot["id"] = call.get("id") or slot["id"]
    fn = call.get("function") or {}
    slot["function"]["name"] += fn.get("name") or ""
    args = fn.get("arguments")
    slot["function"]["arguments"] += args if isinstance(args, str) else json.dumps(args) if args else ""


class Corpus:
    """The given texts in order, read as one ring, cut into pieces whose lengths the server's tokenizer measures."""

    def __init__(self, texts: list[str], client: Client):
        self.text, self.client = "\n\n".join(texts) + "\n\n", client
        self.pos, self.wrapped, self.ratio = 0, False, 4.0

    def peek(self, chars: int) -> str:
        out, at = [], self.pos
        while chars > 0:
            piece = self.text[at:at + chars]
            out.append(piece)
            chars -= len(piece)
            at = (at + len(piece)) % len(self.text)
        return "".join(out)

    def cut(self, want: int, lo: int = 1, hi: float = math.inf) -> tuple[str, int]:
        """About `want` tokens of text, in lo..hi when a try lands there (else the closest try); the ring moves on."""
        tol = max(4, want // 100)
        best, chars = None, max(1, round(want * self.ratio))
        for _ in range(CUT_TRIES):
            text = self.peek(chars)
            n = self.client.text_tokens(text)
            score = (not lo <= n <= hi, abs(n - want))
            if n and (best is None or score < best[0]):
                best = (score, text, n)
            if n and max(lo, want - tol) <= n <= min(hi, want + tol):
                break
            self.ratio = len(text) / n if n else self.ratio * 2
            chars = max(1, round(want * self.ratio))
        if best is None:
            raise ValueError("the --text files produced no tokens")
        _, text, n = best
        end = self.pos + len(text)
        self.wrapped |= end >= len(self.text)
        self.pos = end % len(self.text)
        return text, n


def next_role(prev: str | None, last: bool, rng: random.Random) -> str:
    """A user task opens; a tool result answers each assistant call; after a result, sometimes the user speaks."""
    if prev is None:
        return "user"
    if prev == "assistant":
        return "tool"
    return "user" if last or (prev == "tool" and rng.random() < 1 / 6) else "assistant"


def message(role: str, text: str, call: str | None) -> dict:
    if role == "assistant":
        args = json.dumps({"path": f"notes/{call}.txt"})
        return {"role": role, "content": text,
                "tool_calls": [{"id": call, "type": "function", "function": {"name": "read_file", "arguments": args}}]}
    return {"role": role, "tool_call_id": call, "content": text} if role == "tool" else {"role": role, "content": text}


def tolerance(target: int) -> int:
    return max(32, target // 1000)


def history(corpus: Corpus, client: Client, name: str, harness: dict, base: int, target: int,
            chunk: tuple[int, int], rng: random.Random) -> tuple[list, list, int, list | None]:
    """A session's messages after the harness, in chunk-sized pieces, to `target` rendered prompt tokens, and ids."""
    lo, hi = chunk
    head = f"Session {name}.\n\n"
    msgs, sizes, over, est, call = [], [], FIRST_OVERHEAD, base, None
    while True:
        if msgs and target - est - over <= 2 * hi:  # the last pieces are planned from exact counts
            est = client.rendered([harness, *msgs])[0]
        room = target - est - over
        last = room <= hi
        want = max(lo, room) if last else rng.randint(lo, max(lo, min(hi, room - lo - over)))
        role = next_role(msgs[-1]["role"] if msgs else None, last, rng)
        call = f"call_{name}_{len(msgs)}" if role == "assistant" else call
        text, n = corpus.cut(want, lo, hi)
        msgs.append(message(role, (head if not msgs else "") + text, call))
        sizes.append(n)
        est += n + over
        if len(msgs) == 1:
            first = est = client.rendered([harness, *msgs])[0]
        elif len(msgs) == 3:  # the template's cost per message, from the two messages after the first
            est = client.rendered([harness, *msgs])[0]
            over = max(0, round((est - first - sizes[1] - sizes[2]) / 2))
        if last:
            break
    (total, _, ids), k = client.rendered([harness, *msgs]), len(msgs) - 1
    for _ in range(FIX_TRIES):  # recut the last piece, or the one before when a piece is at its bound
        want = min(hi, max(lo, sizes[k] + target - total))
        while k > 0 and want == sizes[k]:
            k -= 1
            want = min(hi, max(lo, sizes[k] + target - total))
        if abs(target - total) <= tolerance(target) or want == sizes[k]:
            break
        text, sizes[k] = corpus.cut(want, lo, hi)
        msgs[k]["content"] = (head if k == 0 else "") + text
        total, _, ids = client.rendered([harness, *msgs])
    return msgs, sizes, total, ids


def shared(a: list | None, b: list | None) -> list | None:
    """The token ids two prompts share from the start; None when either is unknown."""
    if a is None or b is None:
        return None
    return a[:next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))]


def schedule(pattern: str, turns: int, burst: int) -> list[list[int]]:
    """The turn requests as rounds of session indices; a round of several is sent overlapped."""
    if pattern == "overlap":
        return [[0, 1, 2] for _ in range(turns)]
    if pattern == "burst":
        out = []
        for done in range(0, turns, burst):
            k = min(burst, turns - done)
            out += [[0] for _ in range(k)] + [[1] for _ in range(k)]
        return out
    return [[i] for _ in range(turns) for i in range(3 if pattern == "three" else 2)]


def leaves(value, path: str = "") -> dict:
    """Integer leaves of a /stats snapshot whose dotted key path names a cache hit, miss or eviction."""
    out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{path}.{key}" if path else str(key)
            if isinstance(item, dict):
                out.update(leaves(item, name))
            elif count(item) is not None and COUNTER.search(name):
                out[name] = item
    return out


def counters(before: dict | None, after: dict | None) -> dict | None:
    """The snapshots' cache hit, miss and eviction counters and their change; None when the server reports none."""
    return change(leaves(before), leaves(after))


def seconds(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


class Bench:
    """One run: the harness, the sessions, the shared clock and the output lines."""

    def __init__(self, args, client: Client, corpus: Corpus, out, key: str | None):
        self.args, self.client, self.corpus, self.out, self.key = args, client, corpus, out, key
        self.rng = random.Random(args.seed)
        self.lock = threading.Lock()
        self.flying: set[str] = set()
        self.records: list[dict] = []
        self.sessions: list[dict] = []
        self.seq, self.failed, self.t0, self.prefix = 0, 0, time.perf_counter(), None
        self.stats = client.stats() is not None
        self.metrics, self.found = None, {}
        for path in ("/metrics", "/v1/metrics"):
            if (text := client.text(path)) is not None:
                self.metrics, self.found = path, series(text)
                break

    def snapshot(self) -> tuple[dict | None, dict]:
        """/stats and the prompt-cache series of /metrics, from the routes that answered at startup."""
        return (self.client.stats() if self.stats else None,
                series(self.client.text(self.metrics)) if self.metrics else {})

    def emit(self, record: dict) -> None:
        line = json.dumps(redact(record, self.key))
        with self.lock:
            if record["kind"] == "request":
                self.records.append(record)
            print(line, flush=True)
            self.out.write(line + "\n")
            self.out.flush()

    def setup(self, digests: list[dict]) -> None:
        a = self.args
        text, system = self.corpus.cut(a.system_tokens)
        harness = {"role": "system", "content": "You are a coding agent working in a repository.\n\n" + text}
        base, window, _ = self.client.rendered([harness, {"role": "user", "content": "Begin."}])
        if window is not None and a.history_tokens + a.max_tokens > window:
            raise ValueError(f"--history-tokens plus --max-tokens pass the server's context of {window} tokens")
        if a.history_tokens - base < a.chunk_tokens[0]:
            raise ValueError(f"--history-tokens leaves no history piece after the {base}-token harness")
        plan, common = {}, None
        for i in range(a.sessions):
            name = NAMES[i]
            msgs, sizes, total, ids = history(self.corpus, self.client, name, harness, base, a.history_tokens,
                                              a.chunk_tokens, self.rng)
            common = ids if i == 0 else shared(common, ids)
            keep = a.replay_url is not None and name == a.replay_session
            self.sessions.append({"name": name, "messages": [harness, *msgs], "turn": 0, "reply": None,
                                  "alive": True, "bodies": [] if keep else None})
            plan[name] = {"prompt_tokens": total, "off_by": total - a.history_tokens, "messages": len(msgs),
                          "roles": "".join(m["role"][0] for m in msgs), "pieces": sizes}
        self.prefix = None if common is None else len(common)
        note = None if self.stats or self.metrics else "neither /stats nor /metrics answered; cache fields are empty"
        self.emit({"kind": "plan", "pattern": a.pattern, "sessions": a.sessions, "turns": a.turns, "burst": a.burst,
                   "overlap_gap_s": a.overlap_gap, "thinking": a.thinking == "on", "max_tokens": a.max_tokens,
                   "seed": a.seed, "system_tokens": system, "history_tokens": a.history_tokens,
                   "chunk_tokens": a.chunk_tokens, "turn_tokens": a.turn_tokens, "max_model_len": window,
                   "tools": [t["function"]["name"] for t in TOOLS], "stats": self.stats, "metrics": self.metrics,
                   "metrics_cache_series": sorted(self.found), "cache_note": note, "plan_only": a.plan_only,
                   "shared_prefix_tokens": self.prefix, "texts": digests, "corpus_wrapped": self.corpus.wrapped,
                   "history": plan})

    def prepare(self, s: dict, phase: str) -> dict:
        """The session's next body: the cold history, or the last reply resent with a new tool or user message."""
        a, new = self.args, None
        if phase == "turn":
            text, new = self.corpus.cut(self.rng.randint(*a.turn_tokens), *a.turn_tokens)
            reply = s["reply"]
            calls = [{**c, "id": c["id"] or f"call_{s['name']}_t{s['turn']}_{i}"} for i, c in enumerate(reply["calls"])]
            said = {"role": "assistant", "content": reply["content"]}
            s["messages"].append({**said, "tool_calls": calls} if calls else said)
            s["messages"] += [{"role": "tool", "tool_call_id": c["id"], "content": text if i == 0 else "(no output)"}
                              for i, c in enumerate(calls)] or [{"role": "user", "content": text}]
            s["turn"] += 1
        body = {"model": a.model, "messages": s["messages"], "tools": TOOLS,
                "chat_template_kwargs": {"enable_thinking": a.thinking == "on"}, "temperature": 0, "seed": a.seed,
                "max_tokens": a.max_tokens, "stream": True, "stream_options": {"include_usage": True}}
        return {"session": s, "phase": phase, "turn": s["turn"], "new_tokens": new, "data": json.dumps(body).encode()}

    def send(self, p: dict) -> None:
        s = p["session"]
        before = self.snapshot()
        with self.lock:
            others, seq = sorted(self.flying), self.seq
            self.flying.add(s["name"])
            self.seq += 1
        sent = time.perf_counter() - self.t0
        run = self.client.chat(p["data"])
        with self.lock:
            self.flying.discard(s["name"])
        after = self.snapshot()
        first, last = (run["pieces"][0][0], run["pieces"][-1][0]) if run["pieces"] else (None, None)
        record = {"kind": "request", "pattern": self.args.pattern, "session": s["name"], "turn": p["turn"],
                  "phase": p["phase"], "seq": seq, "sent_s": seconds(sent), "in_flight_at_send": others,
                  "new_tokens": p["new_tokens"], "prompt_tokens": run["prompt_tokens"],
                  "cached_tokens": run["cached_tokens"], "cache_state": run["cache_state"],
                  "completion_tokens": run["tokens"], "reasoning_tokens": run["reasoning_tokens"],
                  "finish_reason": run["finish"], "tool_calls": len(run["calls"]),
                  "ttft_s": seconds(first and first - run["sent"]), "decode_s": seconds(first and last - first),
                  "total_s": seconds(run["completed"] - run["sent"]),
                  "chunk_gaps_ms": bench_timing.request(run)["chunk_gaps_ms"], "tensorfold": run["tensorfold"],
                  "speculative": run["speculative"], "token_sha256": run["token_sha256"], "token_sha": run["token_sha"],
                  "output_sha256": run["output_sha256"], "complete": run["complete"], "error": run["error"],
                  "cache_counters": counters(before[0], after[0]), "cache_metrics": change(before[1], after[1])}
        if self.stats:
            record.update(stats_before=before[0], stats_after=after[0])
        self.emit(record)
        s["reply"] = run
        if run["error"] or not run["complete"]:
            s["alive"] = False
            with self.lock:
                self.failed += 1
        if s["bodies"] is not None:
            s["bodies"].append((p, run))

    def round(self, sessions: list[dict], phase: str, gap: float | None) -> None:
        """One request per live session: one after another, or each `gap` seconds after the last, together."""
        prepared = [self.prepare(s, phase) for s in sessions if s["alive"]]
        if gap is None:
            for p in prepared:
                self.send(p)
            return
        start = time.perf_counter()

        def go(k: int, p: dict) -> None:
            time.sleep(max(0.0, start + k * gap - time.perf_counter()))
            self.send(p)

        threads = [threading.Thread(target=go, args=(k, p)) for k, p in enumerate(prepared)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def run(self) -> None:
        a = self.args
        gap = a.overlap_gap if a.pattern == "overlap" else None
        firsts = self.sessions[:3] if gap is not None else []
        if firsts:
            self.round(firsts, "cold", gap)
        for s in self.sessions[len(firsts):]:
            self.round([s], "cold", None)
        for group in schedule(a.pattern, a.turns, a.burst):
            self.round([self.sessions[i] for i in group], "turn", gap)

    def replay(self, ref: Client) -> dict:
        """The replay session's bodies, in order, on the reference server; each reply's token hash compared."""
        s = next(x for x in self.sessions if x["name"] == self.args.replay_session)
        tally = {"session": s["name"], "equal": 0, "unequal": 0, "unverified": 0}
        for p, primary in s["bodies"]:
            run = ref.chat(p["data"])
            equal = token_equal(primary, run)
            tally["equal" if equal else "unverified" if equal is None else "unequal"] += 1
            cached = run["cached_tokens"]
            self.emit({"kind": "replay", "session": s["name"], "turn": p["turn"], "phase": p["phase"],
                       "prompt_tokens": run["prompt_tokens"], "cached_tokens": cached,
                       "fresh": None if cached is None else cached == 0,
                       "total_s": seconds(run["completed"] - run["sent"]), "token_sha256": run["token_sha256"],
                       "primary_token_sha256": primary["token_sha256"], "equal": equal, "error": run["error"]})
        return tally

    def summary(self, tally: dict | None) -> dict:
        per = {}
        for r in self.records:
            got = per.setdefault(r["session"], {k: [] for k in ("phase", "prompt_tokens", "cached_tokens", "ttft_s")})
            for k in got:
                got[k].append(r[k])
        for got in per.values():  # a turn without a reported cached count is in neither tally
            cached = [c for p, c in zip(got["phase"], got["cached_tokens"]) if p == "turn" and count(c) is not None]
            past = None if self.prefix is None else sum(c > self.prefix for c in cached)
            within = None if past is None else len(cached) - past
            got.update(turns_past_shared_prefix=past, turns_within_shared_prefix=within)
        return {"kind": "summary", "pattern": self.args.pattern, "requests": len(self.records), "failed": self.failed,
                "corpus_wrapped": self.corpus.wrapped, "shared_prefix_tokens": self.prefix, "sessions": per,
                "replay": tally}


def span(value: str) -> tuple[int, int]:
    """LO,HI token counts with 0 < LO <= HI."""
    try:
        lo, hi = (int(x) for x in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected LO,HI integers") from None
    if not 0 < lo <= hi:
        raise argparse.ArgumentTypeError("expected LO,HI with 0 < LO <= HI")
    return lo, hi


def check_url(p: argparse.ArgumentParser, value: str, name: str) -> str:
    split = urllib.parse.urlsplit(value)
    if split.scheme not in ("http", "https") or not split.netloc or split.query or split.fragment:
        p.error(f"{name} must be an http or https URL without a query or fragment")
    return value.rstrip("/")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--text", nargs="+", required=True, metavar="FILE", help="UTF-8 text files the histories come from")
    p.add_argument("--pattern", choices=sorted(NEEDS), default="alt")
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--system-tokens", type=int, default=6144, help="the shared system message's length")
    p.add_argument("--history-tokens", type=int, default=215_000, help="each cold request's rendered prompt tokens")
    p.add_argument("--chunk-tokens", type=span, default=(2048, 8192), metavar="LO,HI", help="history piece lengths")
    p.add_argument("--turn-tokens", type=span, default=(512, 4096), metavar="LO,HI", help="a turn's new message")
    p.add_argument("--turns", type=int, default=4, help="turns per session after its cold request")
    p.add_argument("--burst", type=int, default=4, help="burst: turns in a row per session")
    p.add_argument("--overlap-gap", type=float, default=5.0, help="overlap: seconds between the three sends")
    p.add_argument("--thinking", choices=("on", "off"), default="on", help="chat_template_kwargs.enable_thinking")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1234, help="the requests' seed and the packing's random choices")
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds per HTTP request")
    p.add_argument("--replay-url", help="a second server for the replay session's bodies (resumed == fresh)")
    p.add_argument("--replay-session", default="A")
    p.add_argument("--plan-only", action="store_true", help="build the histories, write the plan line and stop")
    p.add_argument("--out", required=True, help="JSON lines output")
    p.add_argument("--api-key-env", metavar="NAME", help="read the bearer API key (both servers) from this variable")
    return p


def check(p: argparse.ArgumentParser, a) -> None:
    """Refuse a flag combination before any request."""
    a.base = check_url(p, a.base, "base")
    if a.replay_url is not None and check_url(p, a.replay_url, "--replay-url") == a.base:
        p.error("--replay-url must name a second server")
    if not NEEDS[a.pattern] <= a.sessions <= len(NAMES):
        p.error(f"--pattern {a.pattern} needs {NEEDS[a.pattern]} to {len(NAMES)} sessions")
    lo, hi = a.chunk_tokens
    if hi < 2 * lo:
        p.error("--chunk-tokens needs HI >= 2 * LO so every history packs")
    if min(a.system_tokens, a.max_tokens, a.burst) < 1 or a.turns < 0:
        p.error("--system-tokens, --max-tokens and --burst must be positive; --turns nonnegative")
    if a.history_tokens < a.system_tokens + 2 * lo:
        p.error("--history-tokens must leave at least two --chunk-tokens LO pieces after --system-tokens")
    if not bench_timing.timestamp(a.overlap_gap):
        p.error("--overlap-gap must be finite and nonnegative")
    if not (math.isfinite(a.timeout) and a.timeout > 0):
        p.error("--timeout must be finite and positive")
    if a.replay_session not in NAMES[:a.sessions]:
        p.error(f"--replay-session must be one of {NAMES[:a.sessions]}")


def read_texts(p: argparse.ArgumentParser, paths: list[str]) -> tuple[list[str], list[dict]]:
    texts, digests = [], []
    for path in paths:
        try:
            raw = Path(path).read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            p.error(f"--text {path}: not a readable UTF-8 file")
        if not text.strip():
            p.error(f"--text {path}: empty")
        texts.append(text)
        digests.append({"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    return texts, digests


def main(argv: list[str] | None = None) -> int:
    p = parser()
    a = p.parse_args(argv)
    check(p, a)
    key = read_api_key(p, a.api_key_env)
    texts, digests = read_texts(p, a.text)
    try:
        out = open(a.out, "w", encoding="utf-8")
    except OSError as exc:
        p.error(f"--out: {exc.strerror}")
    client = Client(a.base, a.model, key, a.timeout, a.thinking == "on")
    with out:
        bench = Bench(a, client, Corpus(texts, client), out, key)
        try:
            bench.setup(digests)
        except (OSError, ValueError) as exc:
            print(redact(f"bench_sessions: setup failed: {type(exc).__name__}: {exc}", key), file=sys.stderr)
            return 1
        if a.plan_only:
            return 0
        bench.run()
        ref = Client(a.replay_url, a.model, key, a.timeout, a.thinking == "on") if a.replay_url else None
        tally = bench.replay(ref) if ref else None
        bench.emit(bench.summary(tally))
    return 1 if bench.failed or (tally and (tally["unequal"] or tally["unverified"])) else 0


if __name__ == "__main__":
    sys.exit(main())
