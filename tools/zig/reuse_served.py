"""Served prompt reuse: agent transcripts on a cache server, replayed on a fresh one, replies compared by token SHA."""

import argparse
import http.client
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request

COLD = (1024, 2048, 8192, 16384, 32768, 65536)  # cold prompt lengths: a warm-up, then the PP rule's
PROBE = chr(0x2063) + "probe"  # the server's system-prefix probe (prompt.zig systemPrefixLen)
USAGE = """modes:
  record BASE MODEL ROWS.jsonl     every scenario on a server with the prompt cache on; cached tokens checked per reply
  replay BASE MODEL ROWS.jsonl OUT the same bodies, in order, on a server started with --prompt-cache-gib 0
  compare ROWS.jsonl OUT           token_sha equal, drafted == "draft": false, resumed first tokens sooner than fresh
  evict BASE MODEL ROWS.jsonl GIB  a budget that holds one conversation: alternating ones evict, a big one is refused
  cold BASE MODEL PROMPTS.json OUT cold prompts 2k-64k (built once through /v1/tokenize): time to first token
  pp OUT...                        cold medians by length, each run against the first
  ids BASE MODEL OUT.json TOKENS   a chat prompt's ids as the server renders it, for the engine gates
  warm BASE MODEL OUT.json         an agent's short turns (--turns, --new-tokens, --max-tokens): cached tokens and TTFT a turn
  ranks R0.log R1.log              speed-up mode: each reply's token SHA on rank 0 (done lines) equals rank 1's (its reply lines)
  split BASE MODEL ROWS.jsonl      speed-up mode: a system cut one row before, at and after a pair call's split, each resumed by another conversation"""
TOOLS = [
    {"type": "function", "function": {"name": name, "description": about, "parameters": {
        "type": "object", "properties": {p: {"type": "string", "description": d} for p, d in params},
        "required": [p for p, _ in params]}}}
    for name, about, params in [
        ("read_file", "Read a file from the repository and return its text.", [("path", "Path from the repository root.")]),
        ("list_dir", "List a directory's entries.", [("path", "Directory path from the repository root.")]),
        ("grep", "Search files for a regular expression.", [("pattern", "The regular expression."), ("path", "Where to search.")]),
        ("run", "Run a shell command in the repository and return its output.", [("command", "The command line.")]),
        ("edit_file", "Replace one exact span of a file.", [("path", "The file."), ("old", "Text to replace."), ("new", "Replacement.")]),
        ("write_file", "Write a whole file.", [("path", "The file."), ("content", "The new text.")]),
    ]
]


class Client:
    def __init__(self, base: str, model: str) -> None:
        self.base, self.model = base.rstrip("/"), model

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=7200) as r:
            return json.loads(r.read())

    def chat(self, body: dict) -> dict:
        t0 = time.perf_counter()
        try:
            out = self.post("/v1/chat/completions", body)
        except urllib.error.HTTPError as e:
            return {"error": f"{e.code} {e.read()[:300]!r}", "wall": time.perf_counter() - t0}
        wall = time.perf_counter() - t0
        tf, usage, choice = out.get("tensorfold") or {}, out.get("usage") or {}, out["choices"][0]
        msg = choice["message"]
        return {"sha": tf.get("token_sha"), "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "prompt": usage.get("prompt_tokens"), "tokens": usage.get("completion_tokens"),
                "ttft": tf.get("time_to_first_token"), "prefill": tf.get("prefill_seconds"), "wall": wall,
                "finish": choice.get("finish_reason"), "content": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or "", "calls": msg.get("tool_calls") or []}

    def cancel(self, body: dict, after: float) -> dict:
        """Sends the request and closes the socket after `after` seconds, before any reply (a cancel in prefill)."""
        u = urllib.parse.urlparse(self.base)
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=7200)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        time.sleep(after)
        c.sock.shutdown(2)
        c.close()
        return {"cancelled_after": after}

    def tokens(self, messages: list, tools: list | None, thinking: bool, generation: bool) -> list[int]:
        body = {"model": self.model, "messages": messages, "add_generation_prompt": generation,
                "chat_template_kwargs": {"enable_thinking": thinking}}
        if tools:
            body["tools"] = tools
        return self.post("/v1/tokenize", body)["tokens"]


class Texts:
    """Deterministic text from a directory's larger .py files (default: this Python's standard library)."""

    def __init__(self, root: str) -> None:
        names = sorted(n for n in os.listdir(root) if n.endswith(".py") and os.path.getsize(os.path.join(root, n)) > 40_000)
        self.root, self.names = root, names

    def get(self, k: int, chars: int) -> tuple[str, str]:
        name = self.names[k % len(self.names)]
        with open(os.path.join(self.root, name), encoding="utf-8", errors="replace") as f:
            return name, f.read()[:chars]


class Conv:
    """One conversation as a client resends it every turn: the system prompt, the user's task, then tool turns."""

    def __init__(self, name: str, system: str, user: str, tools: list | None, thinking: bool, echo_reasoning: bool):
        self.name, self.tools, self.thinking, self.echo = name, tools, thinking, echo_reasoning
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        self.turn = 0
        self.history = 0  # the rendered history's length after the last turn: where the next turn resumes
        self.kept = False  # whether the last turn's prompt was long enough to keep its history

    def body(self, model: str, max_tokens: int, plain: bool) -> dict:
        b = {"model": model, "messages": json.loads(json.dumps(self.messages)), "max_tokens": max_tokens,
             "temperature": 0, "chat_template_kwargs": {"enable_thinking": self.thinking}}
        if self.tools:
            b["tools"] = self.tools
        if plain:
            b["draft"] = False
        return b

    def extend(self, reply: dict, tool_text: tuple[str, str]) -> None:
        """The reply as the assistant's turn (its own calls, else a read of the next file) and the tool's result."""
        k = self.turn
        calls = reply.get("calls") or [{"id": f"call_{self.name}_{k}", "type": "function",
                                       "function": {"name": "read_file", "arguments": json.dumps({"path": tool_text[0]})}}]
        msg = {"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls}
        if self.echo and reply.get("reasoning"):
            msg["reasoning_content"] = reply["reasoning"]
        self.messages.append(msg)
        for c in calls:
            self.messages.append({"role": "tool", "tool_call_id": c.get("id", f"call_{k}"), "content": tool_text[1]})


def fits(cached, allowed) -> bool:
    """A cached count against a list of positions, or a {"min", "max"} range (planned families floor marks to chunk starts)."""
    if allowed is None:
        return True
    if isinstance(allowed, dict):
        return cached is not None and allowed["min"] <= cached <= allowed["max"]
    return cached in allowed


class Recorder:
    def __init__(self, client: Client, out: str, max_tokens: int, planned: bool = False, min_prompt: int = 4096) -> None:
        self.c, self.out, self.max_tokens, self.planned, self.min_prompt = client, open(out, "w"), max_tokens, planned, min_prompt
        self.i, self.failures = 0, 0

    def prev(self, conv: "Conv") -> list[int] | dict:
        """A follow-up's resume: the last turn's history if that prompt kept it, else nothing."""
        return self.hist(conv.history) if conv.kept else [0]

    def hist(self, history: int) -> list[int] | dict:
        """A resume at a rendered history: exactly there, or for planned families at the chunk start it floors to."""
        return {"min": max(1, history - 8192), "max": history} if self.planned else [history]

    def cuts(self, cuts: list[int], miss: bool) -> list[int] | dict:
        """A system cut's expectation: the cuts themselves, or for planned families any start at or below the last."""
        if not self.planned:
            return ([0] if miss else []) + cuts
        return {"min": 0 if miss else 1, "max": max(cuts) if cuts else 0}

    def system_cuts(self, conv: Conv, prompt: list[int]) -> list[int]:
        """The server's shared-prefix marks: the system block's length, and 512 and 2048 before it (each 512+)."""
        first_user = next(i for i, m in enumerate(conv.messages) if m["role"] == "user")
        probe = json.loads(json.dumps(conv.messages[:first_user])) + [{"role": "user", "content": PROBE}]
        other = self.c.tokens(probe, conv.tools, conv.thinking, True)
        n = 0
        while n < min(len(prompt), len(other)) and prompt[n] == other[n]:
            n += 1
        return [x for x in (n - 2048, n - 512, n) if x >= 512] if n >= 512 else []

    def row(self, conv: Conv, arm: str, body: dict, got: dict, allowed: list[int] | None, history: int, group: str = "") -> None:
        cached = got.get("cached")
        ok = "error" not in got and fits(cached, allowed)
        self.failures += 0 if ok else 1
        r = {"i": self.i, "conv": conv.name, "turn": conv.turn + 1, "arm": arm, "group": group, "body": body,
             "history": history, "allowed": allowed, "got": {k: v for k, v in got.items() if k not in ("content", "reasoning", "calls")},
             "ok": ok}
        self.out.write(json.dumps(r) + "\n")
        self.out.flush()
        print(f"{self.i:3d} {conv.name:>4} t{conv.turn + 1} {arm:<7} prompt {got.get('prompt')} history {history} "
              f"cached {cached} (allowed {allowed if not isinstance(allowed, list) or len(allowed) < 6 else str(allowed[:5]) + '...'}) "
              f"ttft {got.get('ttft') or 0:.2f}s sha {got.get('sha')} {got.get('finish')} {'ok' if ok else 'FAIL ' + str(got.get('error', ''))}",
              flush=True)
        self.i += 1

    def turn(self, conv: Conv, allowed: list[int] | None, tool_text: tuple[str, str], plain: bool = True, plain_allowed: list[int] | None = None) -> dict:
        """One turn: drafted, then "draft": false on the same messages (it resumes this turn's own history)."""
        prompt = self.c.tokens(conv.messages, conv.tools, conv.thinking, True)
        history = len(self.c.tokens(conv.messages, conv.tools, conv.thinking, False))
        body = conv.body(self.c.model, self.max_tokens, False)
        got = self.c.chat(body)
        self.row(conv, "drafted", body, got, allowed, history)
        if plain:
            pb = conv.body(self.c.model, self.max_tokens, True)
            kept = len(prompt) >= self.min_prompt
            self.row(conv, "plain", pb, self.c.chat(pb), plain_allowed or (self.hist(history) if kept else [0]), history)
        conv.history = history
        conv.kept = len(prompt) >= self.min_prompt
        conv.extend(got, tool_text)
        conv.turn += 1
        return {"prompt": prompt, "history": history}


def scenarios(rec: Recorder, texts: Texts, turns: int, cancel_after: float) -> None:
    sys_a = "You are a careful coding agent working in a Python repository. Read before you edit, keep changes small, and explain each step.\n\nProject guide:\n" + texts.get(0, 22_000)[1]
    file_k = iter(range(3, 10_000))
    tool = lambda: texts.get(next(file_k), 11_000)
    # 1: one agent session, the reply's reasoning echoed back, growing turn by turn (cached: the last turn's history)
    s1 = Conv("S1", sys_a, "Task: find why the parser below rejects nested groups, then propose a fix.\n\n" + texts.get(1, 4_000)[1], TOOLS, True, True)
    first = rec.turn(s1, [0], tool())
    cuts_a = rec.system_cuts(s1, first["prompt"])
    print(f"system cuts {cuts_a}", flush=True)
    for _ in range(turns - 1):
        rec.turn(s1, rec.prev(s1), tool())
    # 2: an earlier request with another system prompt, then two conversations on S1's system prompt, alternating
    x = Conv("X", "You translate technical prose into plain English.\n\n" + texts.get(2, 9_000)[1], "Rewrite the module summary above for a beginner.", None, True, False)
    rec.turn(x, [0], tool(), plain=False)
    a = Conv("A", sys_a, "Review this module for error handling gaps.\n\n" + texts.get(11, 3_000)[1], TOOLS, True, False)
    b = Conv("B", sys_a, "List every public function below with a one-line summary.\n\n" + texts.get(12, 3_000)[1], TOOLS, True, False)
    for k in range(3):
        for conv in (a, b):
            rec.turn(conv, rec.cuts(cuts_a, False) if k == 0 else rec.prev(conv), tool())
    # 3: S1's transcript with its first user message edited mid-way: nothing past the edit resumes
    ed = Conv("E", sys_a, "", TOOLS, True, True)
    ed.messages = json.loads(json.dumps(s1.messages))
    u = ed.messages[1]["content"]
    ed.messages[1]["content"] = u[: len(u) // 2] + " (edited) " + u[len(u) // 2:]
    rec.turn(ed, rec.cuts(cuts_a, True), tool())
    # 4: thinking off
    t = Conv("T", "You answer briefly.\n\n" + texts.get(13, 6_000)[1], "Summarize the code above in three sentences.", None, False, False)
    for k in range(2):
        rec.turn(t, [0] if k == 0 else rec.prev(t), tool())
    # 5: a request cancelled in prefill, sent again (resumes only a mark kept whole before the cancel), then its next turn
    sys_c = "You audit code for security problems.\n\n" + texts.get(14, 40_000)[1]
    c = Conv("C", sys_c, "Audit the file above and list concrete issues with line references.", TOOLS, True, True)
    body = c.body(rec.c.model, rec.max_tokens, False)
    rec.row(c, "cancel", body, rec.c.cancel(body, cancel_after), None, 0)
    prompt = rec.c.tokens(c.messages, c.tools, c.thinking, True)
    rec.turn(c, rec.cuts(rec.system_cuts(c, prompt), True), tool())
    rec.turn(c, rec.prev(c), tool())
    # 6: two requests at once: each waits its turn and resumes its own conversation
    pair = [(conv, conv.body(rec.c.model, rec.max_tokens, False)) for conv in (a, b)]
    hist = [len(rec.c.tokens(conv.messages, conv.tools, conv.thinking, False)) for conv, _ in pair]
    got = [None, None]
    threads = [threading.Thread(target=lambda j=j: got.__setitem__(j, rec.c.chat(pair[j][1]))) for j in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for j, (conv, body) in enumerate(pair):
        rec.row(conv, "drafted", body, got[j], rec.prev(conv), hist[j], group="concurrent")


def split(client: Client, out: str, texts: Texts, max_tokens: int) -> int:
    """System cuts one row before, at and after a fresh pair call's split (rank 0 holds rows up to ceil(L/2)), each resumed by a second conversation."""
    rec = Recorder(client, out, max_tokens)
    for k, delta in enumerate((-1, 0, 1)):
        user = f"Case {k}. Read the file below and say what it does.\n\n" + texts.get(50 + k, 14_000)[1]
        probe = Conv("probe", "You are agent Z.", user, None, False, False)
        prompt = client.tokens(probe.messages, None, False, True)
        tail = len(prompt) - rec.system_cuts(probe, prompt)[-1] if rec.system_cuts(probe, prompt) else None
        if tail is None:  # the probe's system block is under 512 tokens: measure it from a longer one
            probe.messages[0]["content"] = "You are agent Z.\n\n" + texts.get(60, 4_000)[1]
            prompt = client.tokens(probe.messages, None, False, True)
            tail = len(prompt) - rec.system_cuts(probe, prompt)[-1]
        want = next(sz for sz in range(tail - 4, tail + 8) if sz - delta == (sz + tail + 1) // 2)  # S - ceil((S + tail) / 2) == delta
        text = texts.get(70 + k, 200_000)[1]
        lo, hi = 0, len(text)
        while lo < hi:  # the most characters whose system block stays within `want` tokens
            mid = (lo + hi + 1) // 2
            c = Conv("probe", "You are agent P.\n\n" + text[:mid], user, None, False, False)
            cut = rec.system_cuts(c, client.tokens(c.messages, None, False, True))
            if cut and cut[-1] <= want:
                lo = mid
            else:
                hi = mid - 1
        sysp = "You are agent P.\n\n" + text[:lo]
        p = Conv(f"P{k}", sysp, user, None, False, False)
        prompt = client.tokens(p.messages, None, False, True)
        cut = rec.system_cuts(p, prompt)
        print(f"split case {delta:+d}: prompt {len(prompt)}, split at {(len(prompt) + 1) // 2}, cuts {cut} (want {want})", flush=True)
        rec.turn(p, [0], texts.get(80 + k, 4_000), plain=False)
        q = Conv(f"Q{k}", sysp, user.replace(f"Case {k}.", f"Case {k}, again."), None, False, False)
        rec.turn(q, [cut[-1]] if cut else [0], texts.get(90 + k, 4_000))
        rec.turn(p, rec.prev(p), texts.get(100 + k, 4_000), plain=False)
    print("split: the server log's pass lines show where each pair call crossed its marks", flush=True)
    return rec.failures


def evict(client: Client, out: str, budget_gib: float, texts: Texts, max_tokens: int) -> int:
    """At 1 GiB one 15k-31k-token state fits and two do not: conversations take turns evicting, a 40k-token state is refused."""
    rec = Recorder(client, out, max_tokens)
    tool = lambda k: texts.get(20 + k, 11_000)
    body = lambda k, chars: "".join(texts.get(k + j, 40_000)[1] for j in range(chars // 40_000 + 1))[:chars]
    p, q = (Conv(n, f"You are agent {n}.", "Work through these files.\n\n" + body(30 + 3 * i, 70_000), None, True, True) for i, n in enumerate("PQ"))
    rec.turn(p, [0], tool(0), plain=False)  # keeps P1's state
    rec.turn(q, [0], tool(1), plain=False)  # Q1's state evicts P1's
    rec.turn(p, [0], tool(2), plain=False)  # P1's state is gone: a miss, never the evicted position; P2's evicts Q1's
    rec.turn(q, [0], tool(3), plain=False)  # likewise Q1's: Q2's state evicts P2's
    rec.turn(q, rec.prev(q), tool(4), plain=False)  # the survivor: Q3 resumes Q2's state
    r = Conv("R", "You are agent R.", "Summarize these files.\n\n" + body(40, 160_000), None, True, False)
    rec.turn(r, [0], tool(5), plain_allowed=[0])  # its state passes the budget: refused, so its resend misses too
    rec.turn(q, rec.prev(q), tool(6), plain=False)  # the refusal evicted nothing: Q4 resumes Q3's state
    print(f"budget {budget_gib} GiB: the server log's prompt cache lines show kept, evicted and refused", flush=True)
    return rec.failures


def replay(client: Client, rows_path: str, out_path: str) -> int:
    rows = [json.loads(line) for line in open(rows_path)]
    out = open(out_path, "w")
    k = 0
    while k < len(rows):
        r = rows[k]
        if r["arm"] == "cancel":
            k += 1
            continue
        batch = [r]
        while r["group"] and k + len(batch) < len(rows) and rows[k + len(batch)]["group"] == r["group"]:
            batch.append(rows[k + len(batch)])
        got = [None] * len(batch)
        threads = [threading.Thread(target=lambda j=j: got.__setitem__(j, client.chat(batch[j]["body"]))) for j in range(len(batch))]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        for row, g in zip(batch, got):
            g = {key: v for key, v in g.items() if key not in ("content", "reasoning", "calls")}
            out.write(json.dumps({"i": row["i"], "got": g}) + "\n")
            out.flush()
            print(f"{row['i']:3d} {row['conv']:>4} t{row['turn']} {row['arm']:<7} prompt {g.get('prompt')} cached {g.get('cached')} "
                  f"ttft {g.get('ttft') or 0:.2f}s sha {g.get('sha')}", flush=True)
        k += len(batch)
    return 0


def compare(rows_path: str, fresh_path: str) -> int:
    rows = {r["i"]: r for r in map(json.loads, open(rows_path))}
    fresh = {r["i"]: r["got"] for r in map(json.loads, open(fresh_path))}
    bad, faster, total = 0, [], 0
    by_messages: dict[tuple, set] = {}
    for i, r in sorted(rows.items()):
        if r["arm"] == "cancel":
            continue
        f, g = fresh.get(i), r["got"]
        total += 1
        problems = []
        if f is None:
            problems.append("no fresh reply")
        else:
            if g.get("sha") != f.get("sha"):
                problems.append(f"sha {g.get('sha')} != fresh {f.get('sha')}")
            if f.get("cached"):
                problems.append(f"fresh cached {f.get('cached')}")
            if g.get("cached") and f.get("ttft") and g.get("ttft") is not None:
                faster.append((r["conv"], r["turn"], r["arm"], g["prompt"], g["cached"], g["ttft"], f["ttft"]))
                if g["ttft"] >= f["ttft"]:
                    problems.append(f"resumed ttft {g['ttft']:.2f} >= fresh {f['ttft']:.2f}")
        if not r["ok"]:
            problems.append(f"cached {g.get('cached')} not in {r['allowed']}")
        key = json.dumps({k: v for k, v in r["body"].items() if k != "draft"}, sort_keys=True)
        by_messages.setdefault(key, set()).add(g.get("sha"))
        bad += 1 if problems else 0
        print(f"{i:3d} {r['conv']:>4} t{r['turn']} {r['arm']:<7} {'SAME' if not problems else 'FAIL ' + '; '.join(problems)}")
    split = [k for k, shas in by_messages.items() if len(shas) > 1]
    print(f"drafted == plain on the same messages: {len(by_messages) - len(split)}/{len(by_messages)}")
    for conv, turn, arm, prompt, cached, t, ft in faster:
        print(f"  {conv:>4} t{turn} {arm:<7} prompt {prompt:6d} cached {cached:6d}: ttft {t:6.2f} s vs fresh {ft:6.2f} s ({ft / max(t, 1e-6):5.1f}x)")
    print(f"{'PASS' if bad == 0 and not split else 'FAIL'}: {total - bad}/{total} replies equal and as cached as planned")
    return 1 if bad or split else 0


def ranks(r0_path: str, r1_path: str) -> int:
    """Rank 0's done and ended lines against rank 1's reply lines, in request order (rank 1 also logs rank 0's engine
    warm-up first); a reply rank 0 ended early (a client that left) is checked by its prompt only."""
    line0 = re.compile(r"\] (?:done \S+ prompt=(\d+) .*? tokens=(\d+) sha=([0-9a-f]+)|ended \S+ reason=.*? prompt=(\d+) tokens=(\d+))")
    line1 = re.compile(r"speed-up rank 1: resumed at \d+ of (\d+) .*?reply (\d+) tokens, sha ([0-9a-f]+)")
    a = []
    for m in map(line0.search, open(r0_path, errors="replace")):
        if m:
            a.append((m[1], m[2], m[3]) if m[1] else (m[4], None, None))
    b = [m.groups() for m in map(line1.search, open(r1_path, errors="replace")) if m]
    while b and len(b) > len(a) and (not a or b[0][0] != a[0][0]):
        b = b[1:]  # the engine warm-up: no server request on rank 0
    bad = 0
    for k, (x, y) in enumerate(zip(a, b)):
        same = x[0] == y[0] and (x[2] is None or (x[1], x[2]) == (y[1], y[2]))
        if not same:
            bad += 1
            print(f"request {k}: rank 0 prompt {x[0]} tokens {x[1]} sha {x[2]}; rank 1 prompt {y[0]} tokens {y[1]} sha {y[2]}")
    if len(a) != len(b):
        bad += 1
        print(f"{len(a)} replies on rank 0, {len(b)} on rank 1")
    ended = sum(1 for x in a if x[2] is None)
    print(f"{'PASS' if bad == 0 else 'FAIL'}: {min(len(a), len(b)) - bad}/{len(a)} replies equal on both ranks ({ended} ended early, prompt only)")
    return 1 if bad else 0


def cold_prompts(client: Client, path: str, root: str) -> list[dict]:
    """Single-message prompts of fixed token lengths, a unique first line each so nothing resumes (prefill_cold.py's)."""
    if os.path.exists(path):
        return json.load(open(path))
    text = "".join(open(os.path.join(root, n), encoding="utf-8", errors="replace").read() for n in sorted(os.listdir(root)) if n.endswith(".py"))
    items = []
    for length in COLD:
        for rep in range(1 if length == 1024 else 3):
            start = (rep * 1_000_003 + length * 7) % (len(text) - 6 * length)
            msgs = lambda chars: [{"role": "user", "content": f"Request {length}-{rep}.\n" + text[start:start + chars] + "\nSay in one sentence what the code above does."}]
            lo, hi = 0, 6 * length
            while lo < hi:  # the most characters that stay within the length
                mid = (lo + hi + 1) // 2
                if len(client.tokens(msgs(mid), None, False, True)) <= length:
                    lo = mid
                else:
                    hi = mid - 1
            items.append({"length": length, "rep": rep, "messages": msgs(lo)})
    json.dump(items, open(path, "w"))
    return items


def ids(client: Client, out_path: str, length: int, root: str) -> int:
    """A chat prompt of at most `length` tokens as the server renders it, saved as {"prompt": ids} for the engine gates."""
    text = "".join(open(os.path.join(root, n), encoding="utf-8", errors="replace").read() for n in sorted(os.listdir(root)) if n.endswith(".py"))
    msgs = lambda chars: [{"role": "user", "content": "Explain what this code does, section by section.\n\n" + text[:chars]}]
    lo, hi = 0, 6 * length
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(client.tokens(msgs(mid), None, True, True)) <= length:
            lo = mid
        else:
            hi = mid - 1
    toks = client.tokens(msgs(lo), None, True, True)
    json.dump({"prompt": toks}, open(out_path, "w"))
    print(f"{out_path}: {len(toks)} tokens", flush=True)
    return 0


def warm(client: Client, out_path: str, root: str, turns: int, new_tokens: int, max_tokens: int) -> int:
    """An agent's short turns as an agent harness sends them (thinking off, ~new_tokens of tool output a turn): each turn's cached tokens and first token."""
    texts = Texts(root)
    system = "You are a careful coding agent working in a Python repository. Read before you edit, keep changes small.\n\n" + texts.get(0, 22_000)[1]
    conv = Conv("W", system, "Task: find why the parser below rejects nested groups.\n\n" + texts.get(1, 4_000)[1], TOOLS, False, True)
    rows = []
    for k in range(turns):
        body = conv.body(client.model, max_tokens, False)
        got = client.chat(body)
        rows.append({"turn": k + 1, "body": body, "got": {key: v for key, v in got.items() if key not in ("content", "reasoning", "calls")}})
        print(f"turn {k + 1:2d}: prompt {got.get('prompt')} cached {got.get('cached')} new {(got.get('prompt') or 0) - (got.get('cached') or 0)} "
              f"ttft {got.get('ttft') or 0:.3f} s wall {got.get('wall') or 0:.2f} s sha {got.get('sha')}", flush=True)
        name, text = texts.get(5 + k, 40_000)
        conv.extend(got, (name, text[: 4 * new_tokens]))
        conv.turn += 1
    json.dump(rows, open(out_path, "w"))
    return 0


def cold(client: Client, prompts_path: str, out_path: str, root: str) -> int:
    """Time to first streamed token for each cold prompt (2 reply tokens, thinking off), the first a warm-up."""
    rows = []
    for it in cold_prompts(client, prompts_path, root):
        body = {"model": client.model, "messages": it["messages"], "max_tokens": 2, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(client.base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        sent, first, usage = time.perf_counter(), None, {}
        with urllib.request.urlopen(req, timeout=7200) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                for c in chunk.get("choices") or []:
                    d = c.get("delta") or {}
                    if first is None and (d.get("content") or d.get("reasoning_content")):
                        first = time.perf_counter()
                usage = chunk.get("usage") or usage
        rows.append({"length": it["length"], "rep": it["rep"], "ttft": (first or time.perf_counter()) - sent,
                     "prompt": usage.get("prompt_tokens"), "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")})
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(out_path, "w"))
    return 0


def pp(paths: list[str]) -> int:
    """Median cold TTFT and prompt tok/s by length for each run, and each run against the first."""
    runs = [json.load(open(p)) for p in paths]
    for length in COLD[1:]:
        med = []
        for rows in runs:
            ts = sorted(r["ttft"] for r in rows if r["length"] == length)
            med.append((ts[len(ts) // 2], [r["prompt"] for r in rows if r["length"] == length][0]))
        cells = "  ".join(f"{t:7.3f} s {n / t:7.1f} tok/s" + (f" ({100 * (t / med[0][0] - 1):+5.1f}%)" if k else "") for k, (t, n) in enumerate(med))
        print(f"{length:6d}: {cells}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["record", "replay", "compare", "evict", "cold", "pp", "ids", "warm", "ranks", "split"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--turns", type=int, default=9)
    ap.add_argument("--text", default=os.path.dirname(os.__file__))
    ap.add_argument("--cancel-after", type=float, default=4.0)
    ap.add_argument("--planned", action="store_true", help="the family keeps states only at chunk starts (Nemotron)")
    ap.add_argument("--min-prompt", type=int, default=4096, help="the server keeps nothing for shorter prompts (Rules.min_prompt)")
    ap.add_argument("--new-tokens", type=int, default=200, help="warm: tool output tokens a turn adds")
    o = ap.parse_args()
    if o.mode == "compare":
        return compare(*o.args)
    if o.mode == "pp":
        return pp(o.args)
    if o.mode == "ranks":
        return ranks(*o.args)
    client = Client(o.args[0], o.args[1])
    if o.mode == "replay":
        return replay(client, o.args[2], o.args[3])
    if o.mode == "warm":
        return warm(client, o.args[2], o.text, o.turns, o.new_tokens, o.max_tokens)
    if o.mode == "ids":
        return ids(client, o.args[2], int(o.args[3]), o.text)
    if o.mode == "cold":
        return cold(client, o.args[2], o.args[3], o.text)
    if o.mode == "split":
        return 1 if split(client, o.args[2], Texts(o.text), o.max_tokens) else 0
    if o.mode == "evict":
        return 1 if evict(client, o.args[2], float(o.args[3]), Texts(o.text), o.max_tokens) else 0
    rec = Recorder(client, o.args[2], o.max_tokens, o.planned, o.min_prompt)
    scenarios(rec, Texts(o.text), o.turns, o.cancel_after)
    print(f"record: {rec.i} requests, {rec.failures} not cached as planned", flush=True)
    return 1 if rec.failures else 0


if __name__ == "__main__":
    sys.exit(main())
