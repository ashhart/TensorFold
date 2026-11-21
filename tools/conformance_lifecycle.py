"""The conformance kit's lifecycle gates: a cancelled reply and an evicted prompt leave the server as a fresh one.

- cancel: a streamed reply is closed mid-way. The server must count the disconnect, stay healthy, settle back to the
  census it showed before the cut (``GET /v1/load``: running, waiting, active chat requests, free lanes, and each
  pool's promised and unheld tokens), and answer the same request again exactly as its solo reply. A reply that
  ended before the cut, or a cut the server did not count, fails the gate as vacuous.
- evict: the chat's last turn (the cache-off server's history) is asked twice, so the second resumes from the kept
  prompt; the profile's fill prompts then push it out of the prompt cache. Asked once more it must equal the cache-off
  server's reply, resume fewer tokens than before, and leave the census as it was. Nothing cached before the fill
  fails as vacuous; the same cached count after it fails as not evicted.

Each gate first waits for the server to go idle (no request running, queued or open), up to the block's ``settle_s``
or else the profile's (default 60 s): a server that prefills a chat's next turn in the background after each reply
needs longer. A server that does not go idle in time fails the gate; it is never compared against a busy census. The
census holds only the fields the server reports (a server without ``capacity.lanes_free`` is compared without it).

Each gate reads its block from the profile (``cancel``, ``evict``); a family without the capability names the gate
under the profile's ``skip`` with a reason. ``--plant cancel`` asks the repeat after each cut with another seed;
``--plant evict`` sends no fill prompts, so nothing is evicted.
"""
import http.client
import json
import re
import socket
import time
import urllib.parse

from conformance import equal_line, last_turn, tally

DISCONNECTS = re.compile(r"^tensorfold:client_disconnections_total (\d+)$", re.M)


def census(client):
    """The counts an idle server must return to, from ``GET /v1/load``; None when it serves none."""
    status, raw = client.get("/v1/load")
    try:
        doc = json.loads(raw) if status == 200 else None
    except ValueError:
        doc = None
    if not isinstance(doc, dict) or "running" not in doc:
        return None
    active = doc.get("http_active") if isinstance(doc.get("http_active"), dict) else {}
    out = {"running": doc.get("running"), "waiting": doc.get("waiting"), "active": active.get("chat_completions")}
    capacity = doc.get("capacity") if isinstance(doc.get("capacity"), dict) else {}
    if "lanes_free" in capacity:
        out["lanes_free"] = capacity["lanes_free"]
    pools = [p for p in capacity.get("pools") or [] if isinstance(p, dict)]
    if pools:
        out["pools"] = [{"promised": p.get("promised_tokens"), "unheld": unheld(p)} for p in pools]
    return out


def unheld(pool: dict):
    """A pool's tokens no request holds: available plus reclaimable."""
    return (pool.get("available_tokens") or 0) + (pool.get("reclaimable_tokens") or 0)


def quiet(seen) -> bool:
    """No request running, queued or open."""
    return bool(seen) and not seen.get("running") and not seen.get("waiting") and not seen.get("active")


def idle(client, seconds: float):
    """The census once the server is idle, or None when it is not idle within ``seconds``."""
    seen, held = settle(client, quiet, seconds)
    return seen if held else None


def wait_s(plan, spec: dict) -> float:
    """Seconds a gate waits for the server to settle: its block's, else the profile's, else 60."""
    return spec.get("settle_s", plan.profile.get("settle_s", 60))


def settle(client, until, seconds: float):
    """Polls the census until ``until(census)`` holds or the deadline passes; returns (census, held)."""
    deadline = time.monotonic() + seconds
    while True:
        seen = census(client)
        if seen is not None and until(seen):
            return seen, True
        if time.monotonic() >= deadline:
            return seen, False
        time.sleep(0.25)


def disconnects(client):
    """The server's count of replies a client left, from ``GET /metrics``; None when it reports none."""
    status, raw = client.get("/metrics")
    found = DISCONNECTS.search(raw.decode(errors="replace")) if status == 200 else None
    return int(found.group(1)) if found else None


def healthy(client) -> bool:
    status, raw = client.get("/health")
    try:
        return status == 200 and json.loads(raw).get("status") == "ok"
    except (ValueError, AttributeError):
        return False


def stream_cut(client, body: dict, after: int) -> dict:
    """Streams the request, reads ``after`` chunks, then closes the connection; reports what it saw before the cut."""
    url = urllib.parse.urlsplit(client.base)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=client.timeout)
    out = {"chunks": 0, "ended": False}
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps({"model": client.model, **body, "stream": True}),
                     {"Content-Type": "application/json"})
        sock, reply = conn.sock, conn.getresponse()
        if reply.status != 200:
            out["error"] = f"HTTP {reply.status}"
            return out
        while out["chunks"] < after:
            line = reply.readline()
            if not line:
                out["ended"] = True
                break
            data = line.decode(errors="replace").strip()
            if not data.startswith("data:"):
                continue
            data = data[5:].strip()
            if data == "[DONE]" or any(c.get("finish_reason") for c in json.loads(data).get("choices", [])):
                out["ended"] = True
                break
            out["chunks"] += 1
        sock.shutdown(socket.SHUT_RDWR)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()
    return out


def cancel_gate(pairs, cuts, settled, health, counted, base) -> dict:
    """cancel == solo: every repeat after a cut equals its solo reply, and every cut was clean and counted."""
    counts = tally(pairs)
    clean = sum(c["chunks"] > 0 and not c["ended"] and not c.get("error") for c in cuts)
    problems = []
    if base is None:
        problems.append("no idle census from GET /v1/load (busy past settle_s)")
    if clean < len(cuts):
        problems.append(f"{len(cuts) - clean} cuts came after the reply ended or failed (vacuous)")
    if counted != len(cuts):
        problems.append(f"the server counted {counted} disconnects for {len(cuts)} cuts")
    if sum(settled) < len(cuts):
        problems.append(f"{len(cuts) - sum(settled)} censuses did not return to {base} (a lane or page held)")
    if sum(health) < len(cuts):
        problems.append(f"{len(cuts) - sum(health)} health checks not ok")
    passed = bool(pairs) and counts["unequal"] == 0 and counts["unverified"] == 0 and not problems
    line = (equal_line(counts) + f" after a cut; {clean}/{len(cuts)} cut mid-reply, {counted} disconnects counted, "
            f"{sum(settled)}/{len(cuts)} censuses back to {base}, {sum(health)}/{len(cuts)} healthy")
    if problems:
        line += "; " + "; ".join(problems)
    return {"gate": "cancel", "passed": passed, "counts": {**counts, "cuts": len(cuts), "clean": clean,
            "counted": counted, "settled": sum(settled)}, "line": line}


def cancel(runner, _fresh=None) -> dict:
    """Cuts the profile's prompt mid-reply, ``rounds`` times in each sampling mode, drafted."""
    spec, plan, client = runner.plan.profile.get("cancel") or {}, runner.plan, runner.client
    if "prompt" not in spec:
        return missing("cancel")
    wait = wait_s(plan, spec)
    bodies = {m: {**plan.ask(spec["prompt"], m, True), "max_tokens": spec.get("tokens", 256)} for m in plan.modes}
    solos = {m: runner.record("cancel-solo", (m,), client.post(b)) for m, b in bodies.items()}
    base = idle(client, wait)
    before = disconnects(client)
    pairs, cuts, settled, health = [], [], [], []
    for mode, body in bodies.items():
        for r in range(spec.get("rounds", 2)):
            cuts.append(stream_cut(client, body, spec.get("after_chunks", 4)))
            runner.rows.append({"kind": "cancel-cut", "key": [mode, r], **cuts[-1]})
            seen, held = settle(client, lambda s: base is not None and s == base, wait)
            settled.append(held)
            health.append(healthy(client))
            again = {**body, "seed": body["seed"] + 1} if plan.plant == "cancel" else body
            pairs.append((solos[mode], runner.record("cancel-again", (mode, r), client.post(again))))
    after = disconnects(client)
    counted = after - before if after is not None and before is not None else None
    return cancel_gate(pairs, cuts, settled, health, counted, base)


def filler(k: int, items: int) -> str:
    """A long prompt sharing no text with the chat, so keeping it pushes the chat out of the prompt cache."""
    parts = " ".join(f"part{(i + k) % 83}/{(i * 4093 + k) % 991}" for i in range(items))
    return f"Document {k}, a parts list: {parts}"


def evict_gate(fresh, warm, again, fills, held, base) -> dict:
    """evict == fresh: kept and evicted replies equal the cache-off reply, the eviction shows, the census holds."""
    pairs = [(fresh, w) for w in warm] + [(fresh, again)]
    counts = tally(pairs)
    kept, now = warm[-1].get("cached"), again.get("cached")
    failed = [i for i, f in enumerate(fills) if f.get("error")]
    problems = []
    if not kept:
        problems.append(f"the kept turn resumed {kept} tokens (vacuous: nothing cached to evict)")
    elif now is None or now >= kept:
        problems.append(f"still resumed {now} of {kept} after {len(fills)} fill prompts (nothing evicted)")
    if failed:
        problems.append(f"fill prompts {failed} failed")
    if base is None:
        problems.append("no idle census from GET /v1/load (busy past settle_s)")
    elif not held:
        problems.append(f"census did not return to {base} (a lane or page held)")
    passed = counts["unequal"] == 0 and counts["unverified"] == 0 and not problems
    line = (equal_line(counts) + f" against cache-off (kept, resumed, evicted); cached {kept} -> {now} after "
            f"{len(fills)} fill prompts; census {'back to ' + str(base) if held else 'changed'}")
    if problems:
        line += "; " + "; ".join(problems)
    return {"gate": "evict", "passed": passed, "counts": {**counts, "kept": kept, "evicted_to": now,
            "fills": len(fills)}, "line": line}


def evict(runner, fresh) -> dict:
    """Keeps the chat's last turn, fills the prompt cache past its budget, then asks the turn again."""
    spec, plan, client = runner.plan.profile.get("evict") or {}, runner.plan, runner.client
    if "fill_prompts" not in spec:
        return missing("evict")
    if fresh is None:
        return {"gate": "evict", "passed": False, "counts": {}, "line": "no cache-off server (pass --fresh-url or "
                                                                           "--binary)"}
    wait = wait_s(plan, spec)
    base = idle(client, wait)
    body = plan.body(last_turn(plan, fresh["greedy"]), "greedy", True, ignore_eos=False)
    warm = [runner.record("evict-kept", (i,), client.post(body)) for i in range(2)]
    fills = []
    for k in range(0 if plan.plant == "evict" else spec["fill_prompts"]):
        fill = plan.body([{"role": "user", "content": filler(k, spec.get("fill_items", 900))}], "greedy", True)
        fills.append(runner.record("evict-fill", (k,), client.post({**fill, "max_tokens": spec.get("fill_tokens", 1)})))
    again = runner.record("evict-again", (), client.post(body))
    _, held = settle(client, lambda s: base is not None and s == base, wait)
    return evict_gate(fresh["greedy"][-1], warm, again, fills, held, base)


def missing(name: str) -> dict:
    return {"gate": name, "passed": False, "counts": {},
            "line": f"the profile has no {name} block (add one, or name the gate under skip with a reason)"}
