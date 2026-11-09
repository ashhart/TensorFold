#!/usr/bin/env python3
"""Served exactness gates for one family profile: drafted, repeat, concurrent, resumed, refusals, cancel and evict.

Each gate prints one PASS or FAIL line with its counts; any FAIL exits 1. Equality is the reply's full token-id hash
(``tensorfold.token_sha256``), so a reply without one is unverified and fails its gate.

- drafted: every prompt, greedy and seeded-sampled, equals its ``"draft": false`` reply; at least one draft must be
  accepted, or the gate fails as vacuous.
- repeat: the drafted replies again, one at a time, equal the first ones.
- concurrent: at each level, that many requests at once (drafted and plain) each equal their solo reply; a long
  prompt sent among three decoding streams leaves them unchanged and equals its solo reply.
- resumed: a multi-turn chat on the cached server equals the same chat on a server with the prompt cache off; the
  fresh side must report ``cached_tokens == 0`` and every later resumed turn ``cached_tokens > 0``. A profile's
  ``conversation_context_items`` puts a long document in the first turn, for families that keep only long prompts.
  A profile's ``fresh_env`` is set on every server started with ``--prompt-cache-gib 0``, for families whose streams
  resume their own last chat without the prompt cache (``TF_QWEN27_SLOT_RESUME=0`` on Metal); with ``--fresh-url``,
  start that server with it yourself.
- refusals: each named request the profile lists is refused with its status and error text.
- cancel, evict: a reply cut mid-stream and a prompt pushed out of the cache leave the server as a fresh one
  (``conformance_lifecycle.py``). A profile's ``skip`` names either one its family lacks, with the reason; it prints
  SKIP, and ``--plant`` refuses it.

``--binary`` starts the servers itself (the cache-off one first, then the cached one), one at a time, from the
profile's flags; ``--url`` and ``--fresh-url`` run against servers you started. ``--plant GATE`` plants a fault that
must turn that gate red (the arm under test gets another seed, the cached server keeps no prompts, or a refusal is
sent without its refused field; cancel and evict: see ``conformance_lifecycle.py``). ``--stream`` asks for every
exactness reply as a stream and reads it through the benchmark clients' reader (``bench_stream``), with the same
hash, counts and text; a stream that errs or ends before ``[DONE]`` is unverified. Refusals stay plain JSON requests:
the server answers a streamed context-window refusal with an error event after a 200 head, not with the status the
profile lists. Standard library only, like its sibling ``bench_stream``.

  python3 tools/conformance.py tools/conformance_profiles/qwen35-cuda.json --binary BIN --model-dir DIR \\
    --drafter DRAFTER --output receipt.json
"""

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import bench_stream

GATES = ("drafted", "repeat", "concurrent", "resumed", "refusals", "cancel", "evict")
SKIPPABLE = ("cancel", "evict")  # a family may lack these; the exactness gates hold for every family
NEEDS = {"cancel": "prompt", "evict": "fill_prompts"}  # the key each lifecycle block cannot run without


def lifecycle_gaps(profile: dict) -> list[str]:
    """Names each lifecycle gate the profile neither runs (its block with its key set) nor skips with a reason."""
    skip = profile.get("skip") or {}
    return [g for g in SKIPPABLE if not (isinstance(skip, dict) and skip.get(g))
            and not (isinstance(profile.get(g), dict) and profile[g].get(NEEDS[g]))]
SHA256 = re.compile(r"[0-9a-f]{64}")


def reply_row(status: int, doc) -> dict:
    """One reply's evidence: status, token hash, prompt and cached counts, draft counters, text or error."""
    doc = doc if isinstance(doc, dict) else {}
    row = {"status": status}
    if status != 200 or "error" in doc:
        err = doc.get("error")
        row["error"] = (err.get("message") if isinstance(err, dict) else err) or f"HTTP {status}"
        return row
    runtime = doc.get("tensorfold") if isinstance(doc.get("tensorfold"), dict) else {}
    sha = runtime.get("token_sha256")
    row["sha"] = sha if isinstance(sha, str) and SHA256.fullmatch(sha) else None
    usage = doc.get("usage") if isinstance(doc.get("usage"), dict) else {}
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    row["prompt"] = count(usage.get("prompt_tokens"))
    row["cached"] = count(details.get("cached_tokens"))
    row["tokens"] = count(usage.get("completion_tokens"))
    spec = doc.get("speculative") if isinstance(doc.get("speculative"), dict) else {}
    row["accepted"], row["drafted"] = count(spec.get("accepted")) or 0, count(spec.get("drafted")) or 0
    choices = doc.get("choices") if isinstance(doc.get("choices"), list) else []
    message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
    row["text"] = message.get("content") if isinstance(message.get("content"), str) else ""
    return row


def stream_row(reply) -> dict:
    """A streamed reply read through ``bench_stream``, reduced to the row its JSON reply would give."""
    events = bench_stream.Collector(time.perf_counter())
    events.read(reply)
    if events.end != "done":
        return reply_row(reply.status, {"error": events.error or "incomplete_stream"})
    doc = {"tensorfold": events.runtime, "usage": events.usage, "speculative": events.speculative,
           "choices": [{"message": {"content": "".join(events.text)}}]}
    return {**reply_row(reply.status, doc), "completion": events.completion()}


def count(value):
    """A nonnegative integer the server reported, else None (booleans and floats never count)."""
    return value if type(value) is int and value >= 0 else None


def same(a: dict, b: dict):
    """True or False when both replies carry a full token hash; None (unverified) otherwise."""
    if a.get("error") or b.get("error") or not a.get("sha") or not b.get("sha"):
        return None
    return a["sha"] == b["sha"]


def tally(pairs) -> dict:
    """Equal, unequal and unverified counts over (left, right) reply pairs."""
    verdicts = [same(a, b) for a, b in pairs]
    return {"equal": verdicts.count(True), "unequal": verdicts.count(False), "unverified": verdicts.count(None)}


def equal_line(counts: dict) -> str:
    total = counts["equal"] + counts["unequal"] + counts["unverified"]
    return f"{counts['equal']}/{total} equal, {counts['unequal']} unequal, {counts['unverified']} unverified"


def equality_gate(name: str, pairs, extra_ok=True, extra="") -> dict:
    """A gate that passes when every pair is verified equal (and the extra condition holds) and there is a pair."""
    counts = tally(pairs)
    passed = bool(pairs) and counts["unequal"] == 0 and counts["unverified"] == 0 and extra_ok
    return {"gate": name, "passed": passed, "counts": counts, "line": equal_line(counts) + extra}


def drafted_gate(pairs) -> dict:
    """drafted == plain over (plain, drafted) pairs; fails as vacuous when no drafted reply accepted a draft."""
    accepted = sum(d.get("accepted", 0) for _, d in pairs)
    drafted = sum(d.get("drafted", 0) for _, d in pairs)
    gate = equality_gate("drafted == plain", pairs, accepted > 0, f"; {accepted} of {drafted} drafts accepted")
    gate["counts"].update(accepted=accepted, drafted=drafted)
    if accepted == 0:
        gate["line"] += " (vacuous: no draft accepted)"
    return gate


def resumed_gate(turns) -> dict:
    """resumed == fresh over per-turn (fresh, resumed) pairs, with the cache counts both sides must show."""
    counts = tally(turns)
    fresh_cached = [f.get("cached") for f, _ in turns]
    later = [r.get("cached") for _, r in turns if r.get("turn", 0) > 0]
    problems = []
    if any(c != 0 for c in fresh_cached):
        problems.append(f"fresh side cached {[c for c in fresh_cached if c != 0]} (needs --prompt-cache-gib 0)")
    if not later or any(not c for c in later):
        problems.append(f"later resumed turns cached {later} (each must be > 0)")
    passed = bool(turns) and counts["unequal"] == 0 and counts["unverified"] == 0 and not problems
    line = equal_line(counts) + f"; resumed cached_tokens {later}"
    if problems:
        line += "; " + "; ".join(problems)
    return {"gate": "resumed == fresh", "passed": passed, "counts": counts, "line": line}


def refusal_verdict(row: dict, expect: dict) -> bool:
    """A refusal passes when the status matches and the error text contains the expected words."""
    return row.get("status") == expect.get("status", 400) and expect["error"] in (row.get("error") or "")


def refusals_gate(results) -> dict:
    """Every named refusal (name, row, expect) refused as listed; an empty list fails as vacuous."""
    bad = [name for name, row, expect in results if not refusal_verdict(row, expect)]
    line = f"{len(results) - len(bad)}/{len(results)} refused as listed"
    if bad:
        line += "; not refused as listed: " + ", ".join(bad)
    if not results:
        line += " (vacuous: the profile lists none)"
    return {"gate": "refusals", "passed": bool(results) and not bad, "counts": {"refused": len(results) - len(bad),
            "wrong": len(bad)}, "line": line}


def print_gate(gate: dict) -> None:
    word = "SKIP " if gate.get("skipped") else "PASS " if gate["passed"] else "FAIL "
    print(word + gate["gate"] + ": " + gate["line"], flush=True)


class Client:
    """One served endpoint: JSON or streamed chat requests, every outcome reduced to a reply row."""

    def __init__(self, base: str, model: str, timeout: float = 900, stream: bool = False):
        self.base, self.model, self.timeout, self.stream = base.rstrip("/"), model, timeout, stream

    def post(self, body: dict, stream: bool | None = None) -> dict:
        stream = self.stream if stream is None else stream
        body = {"model": self.model, **body}
        if stream:
            body.update(stream=True, stream_options={"include_usage": True})
        request = urllib.request.Request(self.base + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                return stream_row(reply) if stream else reply_row(reply.status, json.load(reply))
        except urllib.error.HTTPError as exc:
            raw = exc.read(1 << 16)
            try:
                doc = json.loads(raw)
            except ValueError:
                doc = {"error": raw.decode(errors="replace")}
            return reply_row(exc.code, doc)
        except Exception as exc:  # noqa: BLE001 - a failed request is evidence, not a crash
            return {"status": 0, "error": f"{type(exc).__name__}: {exc}"}

    def get(self, path: str):
        """A GET's status and raw body; status 0 when the server did not answer."""
        try:
            with urllib.request.urlopen(self.base + path, timeout=30) as reply:
                return reply.status, reply.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(1 << 16)
        except Exception:  # noqa: BLE001 - an unanswered probe is evidence, not a crash
            return 0, b""


class Plan:
    """A profile's requests: prompts, sampling modes, seeds and the planted fault."""

    def __init__(self, profile: dict, plant: str | None):
        self.profile, self.plant = profile, plant
        self.tokens = profile.get("tokens", 64)
        self.prompts = dict(profile["prompts"])
        self.prompts["long"] = long_prompt(profile.get("long_items", 500))
        self.modes = {"greedy": {"temperature": 0}, "sampled": dict(profile["sampled"])}
        self.seed = profile.get("seed", 7)

    def body(self, messages, mode: str, draft: bool, planted: bool = False, ignore_eos=None) -> dict:
        eos = self.profile.get("ignore_eos", True) if ignore_eos is None else ignore_eos
        return {"messages": messages, "max_tokens": self.tokens, "seed": self.seed + (1 if planted else 0),
                "draft": draft, "ignore_eos": eos, "chat_template_kwargs": {"enable_thinking": False},
                **self.modes[mode]}

    def ask(self, name: str, mode: str, draft: bool, planted: bool = False) -> dict:
        return self.body([{"role": "user", "content": self.prompts[name]}], mode, draft, planted)


def long_prompt(items: int) -> str:
    """A prompt of a few thousand tokens, so it is prefilled in several chunks."""
    return "Summarise this list in one sentence: " + " ".join(f"item{i % 97}x{(i * 7919) % 1013}" for i in range(items))


def inventory(items: int) -> str:
    """A document for the chat's first turn, long enough that a family keeping only long prompts resumes it."""
    return "An inventory: " + " ".join(f"crate{i % 89}-{(i * 6151) % 997}" for i in range(items))


def together(client: Client, bodies, delays=None) -> list:
    """Send every body at once (or after its delay) and return the reply rows in order."""
    out = [None] * len(bodies)
    gate = threading.Barrier(len(bodies))

    def run(i):
        try:
            gate.wait(timeout=60)
        except threading.BrokenBarrierError:
            pass
        if delays:
            time.sleep(delays[i])
        out[i] = client.post(bodies[i])

    threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(len(bodies))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=client.timeout + 120)
    return [row if row is not None else {"status": 0, "error": "request thread did not finish"} for row in out]


class Runner:
    """Runs the gates on one cached server, with solo replies computed once and shared between gates."""

    def __init__(self, client: Client, plan: Plan):
        self.client, self.plan, self.solos, self.rows = client, plan, {}, []

    def solo(self, name: str, mode: str, draft: bool) -> dict:
        key = (name, mode, draft)
        if key not in self.solos:
            planted = draft and self.plan.plant == "drafted"
            self.solos[key] = self.record("solo", key, self.client.post(self.plan.ask(name, mode, draft, planted)))
        return self.solos[key]

    def record(self, kind, key, row):
        self.rows.append({"kind": kind, "key": list(key), **{k: v for k, v in row.items() if k != "text"}})
        return row

    def drafted(self) -> dict:
        pairs = [(self.solo(n, m, False), self.solo(n, m, True)) for n in self.plan.profile["prompts"]
                 for m in self.plan.modes]
        return drafted_gate(pairs)

    def repeat(self) -> dict:
        pairs = []
        for n in self.plan.profile["prompts"]:
            for m in self.plan.modes:
                first = self.solo(n, m, True)
                again = self.client.post(self.plan.ask(n, m, True, self.plan.plant == "repeat"))
                pairs.append((first, self.record("repeat", (n, m, True), again)))
        return equality_gate("repeat == first", pairs)

    def concurrent(self) -> dict:
        names, pairs, planted = list(self.plan.profile["prompts"]), [], self.plan.plant == "concurrent"
        levels = self.plan.profile.get("levels", [1, 2, 4, 8])
        for level in levels:
            for draft in (True, False):
                for mode in self.plan.modes:
                    picks = [names[i % len(names)] for i in range(level)]
                    rows = together(self.client, [self.plan.ask(n, mode, draft, planted) for n in picks])
                    for n, row in zip(picks, rows):
                        row = self.record(f"concurrent{level}", (n, mode, draft), row)
                        pairs.append((self.solo(n, mode, draft), row))
        mixed = 0
        for draft in (True, False):
            for mode in self.plan.modes:
                picks = names[:3] + ["long"]
                bodies = [self.plan.ask(n, mode, draft, planted) for n in picks]
                rows = together(self.client, bodies, delays=[0, 0, 0, 0.5])
                mixed += len(rows)
                for n, row in zip(picks, rows):
                    pairs.append((self.solo(n, mode, draft), self.record("mixed", (n, mode, draft), row)))
        return equality_gate("concurrent == solo", pairs, extra=f" (levels {levels}, drafted and plain, "
                                                                  f"{mixed} with a long prompt among decoders)")

    def refusals(self) -> dict:
        results = []
        for item in self.plan.profile.get("refusals", []):
            body = self.plan.ask(next(iter(self.plan.profile["prompts"])), "greedy", True)
            body["max_tokens"] = 4
            if self.plan.plant != "refusals":
                body.update(item["body"])
            row = self.client.post(body, stream=False)
            self.record("refusal", (item["name"],), row)
            results.append((item["name"], row, item))
        return refusals_gate(results)


def turn_text(plan: Plan, turn: int) -> str:
    """The chat's user text at ``turn``; the first carries the profile's long document, if it names one."""
    text = plan.profile["conversation"][turn]
    if turn == 0 and plan.profile.get("conversation_context_items"):
        text = inventory(plan.profile["conversation_context_items"]) + "\n\n" + text
    return text


def last_turn(plan: Plan, rows) -> list:
    """The chat's last request, its history carrying the replies in ``rows``."""
    history = []
    for turn in range(len(plan.profile["conversation"])):
        history.append({"role": "user", "content": turn_text(plan, turn)})
        if turn < len(plan.profile["conversation"]) - 1:
            history.append({"role": "assistant", "content": rows[turn].get("text", "")})
    return history


def conversation(client: Client, plan: Plan, mode: str, draft: bool) -> list:
    """One multi-turn chat; each turn's history carries this server's own earlier replies."""
    history, rows = [], []
    for turn in range(len(plan.profile["conversation"])):
        history.append({"role": "user", "content": turn_text(plan, turn)})
        row = client.post(plan.body(list(history), mode, draft, ignore_eos=False))
        row["turn"] = turn
        rows.append(row)
        history.append({"role": "assistant", "content": row.get("text", "")})
    return rows


def resumed(fresh: dict, runner: Runner) -> dict:
    """resumed == fresh: the cached server's drafted turns against the cache-off server's plain turns."""
    turns = []
    for mode in runner.plan.modes:
        for f, r in zip(fresh[mode], conversation(runner.client, runner.plan, mode, True)):
            turns.append((runner.record("fresh", (mode, f["turn"]), f), runner.record("resumed", (mode, r["turn"]), r)))
    return resumed_gate(turns)


class Server:
    """A loopback server this tool owns: started from the profile's flags, stopped on exit."""

    def __init__(self, args, profile: dict, cache_gib: float, logs: Path):
        self.args, self.profile, self.cache_gib, self.logs = args, profile, cache_gib, logs
        self.process = None

    def command(self, port: int) -> list:
        flags = [f.replace("{drafter}", str(self.args.drafter)) for f in self.profile.get("serve", [])]
        if any("{drafter}" in f for f in self.profile.get("serve", [])) and not self.args.drafter:
            raise SystemExit("the profile's flags name {drafter}: pass --drafter")
        return [str(self.args.binary), "serve", str(self.args.model_dir), "--host", "127.0.0.1", "--port", str(port),
                "--name", self.profile["name"], "--no-update-check", "--prompt-cache-gib", str(self.cache_gib), *flags]

    def env(self) -> dict:
        """The inherited environment, plus the profile's ``fresh_env`` on a server that keeps no prompts."""
        extra = self.profile.get("fresh_env", {}) if self.cache_gib == 0 else {}
        return {**os.environ, **{k: str(v) for k, v in extra.items()}}

    def __enter__(self):
        if self.args.before_start:
            subprocess.run(self.args.before_start, shell=True, check=True, timeout=600)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        self.cmd = self.command(port)
        self.log = tempfile.NamedTemporaryFile("w+b", prefix=f"server-cache{self.cache_gib}-", suffix=".log",
                                               dir=self.logs, delete=False)
        print("START " + " ".join(self.cmd), flush=True)
        self.process = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
                                        env=self.env())
        deadline = time.monotonic() + self.args.start_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.__exit__()
                raise SystemExit(f"server exited {self.process.returncode}; see {self.log.name}")
            try:
                with urllib.request.urlopen(self.base + "/v1/models", timeout=2) as reply:
                    if reply.status == 200:
                        return self
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(1)
        self.__exit__()
        raise SystemExit(f"server not ready in {self.args.start_timeout}s; see {self.log.name}")

    def __exit__(self, *_):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30)
        self.log.close()


def run_gates(runner: Runner, gates, fresh) -> list:
    """The selected gates in order; resumed follows the gates that want no conversation in the cache."""
    import conformance_lifecycle as lifecycle
    out, skip = [], runner.plan.profile.get("skip") or {}
    for name in gates:
        if name in skip:
            gate = {"gate": name, "passed": True, "skipped": True, "counts": {}, "line": str(skip[name])}
        elif name in ("cancel", "evict"):
            gate = getattr(lifecycle, name)(runner, fresh)
        elif name == "resumed":
            if fresh is None:
                gate = {"gate": "resumed == fresh", "passed": False, "counts": {},
                        "line": "no cache-off server (pass --fresh-url or --binary)"}
            else:
                gate = resumed(fresh, runner)
        else:
            gate = getattr(runner, name)()
        print_gate(gate)
        out.append(gate)
    return out


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("profile", type=Path)
    p.add_argument("--url", help="a running server with its prompt cache on")
    p.add_argument("--fresh-url", help="the same server started with --prompt-cache-gib 0")
    p.add_argument("--binary", type=Path, help="start the servers from the profile instead")
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--drafter", type=Path)
    p.add_argument("--before-start", help="a shell command run before each server start (drop the page cache)")
    p.add_argument("--start-timeout", type=float, default=900)
    p.add_argument("--gates", default=",".join(GATES))
    p.add_argument("--plant", choices=GATES)
    p.add_argument("--stream", action="store_true", help="read the exactness replies as streams (refusals stay JSON)")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    args.gates = [g for g in args.gates.split(",") if g]
    if bad := [g for g in args.gates if g not in GATES]:
        p.error(f"unknown gates {bad}: choose from {', '.join(GATES)}")
    if bool(args.url) == bool(args.binary):
        p.error("pass exactly one of --url or --binary")
    if args.binary and not args.model_dir:
        p.error("--binary needs --model-dir")
    if args.plant and args.plant not in args.gates:
        p.error(f"--plant {args.plant} needs that gate in --gates")
    skip = json.loads(args.profile.read_text()).get("skip") or {}
    if not isinstance(skip, dict) or not all(isinstance(why, str) and why.strip() for why in skip.values()):
        p.error(f"the profile's skip must map each skipped gate to its reason, a non-empty string: {skip!r}")
    if bad := sorted(set(skip) - set(SKIPPABLE)):
        p.error(f"the profile skips {bad}: only {' and '.join(SKIPPABLE)} can be skipped")
    if args.plant in skip:
        p.error(f"--plant {args.plant}: the profile skips that gate ({skip[args.plant]}), so nothing would turn red")
    return args


def main(argv=None) -> int:
    args = parse(argv)
    profile = json.loads(args.profile.read_text())
    plan = Plan(profile, args.plant)
    receipt = {"profile": profile, "plant": args.plant, "stream": args.stream, "gates": [], "servers": []}
    fresh = None
    if args.binary:
        receipt["binary_sha256"] = hashlib.sha256(args.binary.read_bytes()).hexdigest()
        logs = Path(tempfile.mkdtemp(prefix="conformance-"))
        if {"resumed", "evict"} & set(args.gates):
            with Server(args, profile, 0, logs) as server:
                receipt["servers"].append({"cmd": server.cmd, "log": server.log.name})
                client = Client(server.base, profile["name"], stream=args.stream)
                fresh = {mode: conversation(client, plan, mode, False) for mode in plan.modes}
        cache = 0 if args.plant == "resumed" else profile.get("prompt_cache_gib", 2)
        with Server(args, profile, cache, logs) as server:
            receipt["servers"].append({"cmd": server.cmd, "log": server.log.name})
            runner = Runner(Client(server.base, profile["name"], stream=args.stream), plan)
            receipt["gates"] = run_gates(runner, args.gates, fresh)
    else:
        if args.fresh_url and {"resumed", "evict"} & set(args.gates):
            client = Client(args.fresh_url, profile["name"], stream=args.stream)
            fresh = {mode: conversation(client, plan, mode, False) for mode in plan.modes}
        url = args.fresh_url if args.plant == "resumed" and args.fresh_url else args.url
        runner = Runner(Client(url, profile["name"], stream=args.stream), plan)
        receipt["gates"] = run_gates(runner, args.gates, fresh)
    receipt["rows"] = runner.rows
    failed = sum(not g["passed"] for g in receipt["gates"])
    skipped = sum(bool(g.get("skipped")) for g in receipt["gates"])
    print(f"{profile['name']}: {len(receipt['gates']) - failed - skipped} passed, {failed} failed, {skipped} skipped"
          + (f" (planted: {args.plant})" if args.plant else ""), flush=True)
    if args.output:
        args.output.write_text(json.dumps(receipt, indent=1) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
