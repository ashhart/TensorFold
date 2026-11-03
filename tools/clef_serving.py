#!/usr/bin/env python3
"""Gate 6 for Clef-Flash: two 16K readouts and a short one at once, judged from the server's TF_LANE_TRACE lines."""
# usage: clef_serving.py <url> <records> <trace file> [--long rec_39,rec_38] [--short rec_00] [--lanes 3] [--report FILE]
# The server runs with TF_LANE_TRACE=<trace file> and --parallel = --lanes. The records are clef_e2e.py's. Each phase's
# requests are the trace's new submit lines (exactly as many as sent), told apart by id; equal prompt lengths are
# matched in the order they were sent. Rules: every prompt call of a readout is ok and one piece of at most --piece
# rows, contiguous, the last holding its ok completion; no engine step overlaps another; a submit comes before its first
# call; at most 8 + lanes other steps run between a readout's turns (a step running at its submit counts). Then a
# client leaving inside a 16K readout must be cancelled mid-prompt and free its lane: of --lanes requests sent after,
# each makes its first call before any 16K one ends. The short request gives its alone bytes beside and after them.
import argparse, json, socket, sys, threading, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

PREFERENCE_ROUNDS = 8  # lane_host_fill.zig: a long prompt's turn after eight skipped choices
STAGGER = 0.05  # seconds between a phase's sends, so equal-length requests submit in send order


def post(url, body):
    """The reply's bytes, the latency and the HTTP status (an error status returns too, with its body)."""
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data, status = resp.read(), resp.status
    except urllib.error.HTTPError as e:
        data, status = e.read(), e.code
    return data, time.monotonic() - t0, status


def leave(url, body, after):
    """Sends `body` and closes the connection after `after` seconds without reading the reply: true if the client
    closed first (the server had sent nothing and not closed its side)."""
    u = urllib.parse.urlparse(url)
    data = json.dumps(body).encode()
    s = socket.create_connection((u.hostname, u.port))
    s.sendall(f"POST /v1/systemone HTTP/1.1\r\nHost: {u.hostname}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n\r\n".encode() + data)
    time.sleep(after)
    s.setblocking(False)
    try:
        s.recv(1, socket.MSG_PEEK)  # b"": the server closed; a byte: it replied before the client left
        first = False
    except BlockingIOError:
        first = True
    except OSError:  # a reset: the server's side went first
        first = False
    s.close()
    return first


def lines(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def concurrent(url, bodies):
    """Every body sent from its own thread, STAGGER apart; the replies' bytes and latencies in send order."""
    out, threads = [None] * len(bodies), []
    for k, b in enumerate(bodies):
        th = threading.Thread(target=lambda k=k, b=b: out.__setitem__(k, post(url, b)))
        threads.append(th)
        th.start()
        time.sleep(STAGGER)
    for th in threads:
        th.join()
    return out


def phase_ids(trace, start, lengths):
    """The ids of the submits after line `start`, in submit order, matched to the sent lengths: exactly one each."""
    subs = sorted((l for l in trace[start:] if l["kind"] == "submit"), key=lambda l: l["t0_ns"])
    if len(subs) != len(lengths):
        sys.exit(f"the trace holds {len(subs)} new submits for {len(lengths)} requests")
    ids, free = [None] * len(lengths), list(subs)
    for k, n in enumerate(lengths):
        hit = next((l for l in free if l["rows_to"] == n), None)
        if hit is None:
            sys.exit(f"no submit of {n} rows in the trace for request {k}")
        free.remove(hit)
        ids[k] = hit["id"]
    return ids


def settle(path, ids, ended, timeout=120, missing="the trace never showed every request's last line"):
    """The trace once every id has its ending line (`ended(lines of id)`), or exits with `missing` after `timeout` s."""
    deadline = time.monotonic() + timeout
    while True:
        trace = lines(path)
        if all(ended([l for l in trace if l["id"] == i]) for i in ids):
            return trace
        if time.monotonic() > deadline:
            sys.exit(missing)
        time.sleep(0.1)


def completed(mine):
    """The completion and the prompt call holding it are both written (the call's line comes last)."""
    done = [l for l in mine if l["kind"] == "complete"]
    return bool(done) and any(l["kind"] == "fill" and l["t0_ns"] <= done[0]["t0_ns"] and l["t1_ns"] >= done[0]["t1_ns"] for l in mine)


def judge(trace, ids, lengths, lanes, piece):
    """The rules over one phase: `ids[k]` is request k, of `lengths[k]` rows."""
    if not trace or not ids:
        return [], False
    steps = sorted((l for l in trace if l["kind"] in ("fill", "round")), key=lambda l: l["t0_ns"])
    serial = all(a["t1_ns"] <= b["t0_ns"] for a, b in zip(steps, steps[1:]))
    out, ok = [], serial
    for i, n in zip(ids, lengths):
        subs = [l for l in trace if l["kind"] == "submit" and l["id"] == i]
        mine = [l for l in steps if l["kind"] == "fill" and l["id"] == i]
        done = [l for l in trace if l["kind"] == "complete" and l["id"] == i]
        if len(subs) != 1 or not mine or len(done) != 1:
            out.append({"id": i, "rows": n, "rules": {"lines": False}})
            ok = False
            continue
        sub, last = subs[0], mine[-1]
        pieces = [l["rows_to"] - l["rows_from"] for l in mine]
        one_piece = all(0 < p <= piece for p in pieces) and all(l["ok"] for l in mine)
        contiguous = mine[0]["rows_from"] == 0 and mine[-1]["rows_to"] == n and all(a["rows_to"] == b["rows_from"] for a, b in zip(mine, mine[1:]))
        nested = last["t0_ns"] <= done[0]["t0_ns"] and done[0]["t1_ns"] <= last["t1_ns"] and done[0]["ok"]
        ordered = sub["t1_ns"] <= mine[0]["t0_ns"]
        # other steps between this readout's turns, from its submit on; one running at the submit counts in the first
        ends, starts = [sub["t1_ns"]] + [l["t1_ns"] for l in mine], [l["t0_ns"] for l in mine]
        between = [sum(1 for l in steps if l["id"] != i and l["t1_ns"] > e and l["t0_ns"] < s) for e, s in zip(ends, starts)]
        bound = PREFERENCE_ROUNDS + lanes
        rule = {"lines": True, "pieces": len(mine), "ok_one_piece_each": one_piece, "contiguous": contiguous, "completion_in_last_step": nested,
                "submit_before_first_call": ordered, "most_steps_between_turns": max(between), "bound": bound, "within_bound": max(between) <= bound}
        ok &= one_piece and contiguous and nested and ordered and rule["within_bound"]
        ms = lambda ls: [(l["t1_ns"] - l["t0_ns"]) / 1e6 for l in ls]
        out.append({"id": i, "rows": n, "rules": rule,
                    "t_piece_ms": {"mean": sum(ms(mine[:-1])) / max(len(mine) - 1, 1), "max": max(ms(mine[:-1]), default=0.0)},
                    "t_last_piece_and_finish_ms": ms([last])[0], "completion_ms": ms(done)[0],
                    "longest_wait_ms": max((s - e) / 1e6 for e, s in zip(ends, starts)),
                    "submit_to_done_ms": (done[0]["t1_ns"] - sub["t0_ns"]) / 1e6})
    return out, ok and serial


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("records")
    p.add_argument("trace")
    p.add_argument("--long", default="rec_39,rec_38")
    p.add_argument("--short", default="rec_00")
    p.add_argument("--lanes", type=int, default=3)
    p.add_argument("--piece", type=int, default=1024)
    p.add_argument("--report")
    a = p.parse_args()
    index = {r["name"]: r for r in json.loads((Path(a.records) / "index.json").read_text())["records"]}
    names = a.long.split(",") + [a.short]
    bodies = [json.loads((Path(a.records) / f"{n}.json").read_text()) for n in names]
    lengths = [index[n]["tokens"] for n in names]
    run0 = len(lines(a.trace))
    alone, _, alone_status = post(a.url, bodies[-1])  # the short request alone: the bytes it must give beside the readouts and after
    # phase 1: the readouts and the short request at once
    start = len(lines(a.trace))
    replies = concurrent(a.url, bodies)
    ids = phase_ids(lines(a.trace), start, lengths)
    trace = settle(a.trace, ids, completed)[start:]
    out, ok = judge(trace, ids, lengths, a.lanes, a.piece)
    beside = replies[-1][0] == alone
    ok &= beside and alone_status == 200 and all(r[2] == 200 for r in replies)
    # phase 2: a client leaving inside a 16K readout is cancelled while its prompt still runs
    start = len(lines(a.trace))
    client_closed = leave(a.url, bodies[0], 0.5)
    (gone,) = phase_ids(lines(a.trace), start, [lengths[0]])
    ended = lambda mine: any(l["kind"] == "fill" and not l["ok"] for l in mine) or completed(mine)
    # a cancel taken between two calls writes no line: alone, the readout's own calls fill almost all of its time
    no_end = "FAIL gate 6: the left readout shows no failed call and no completion (a cancel between calls writes no line)"
    left = [l for l in settle(a.trace, [gone], ended, missing=no_end)[start:] if l["id"] == gone]
    own_fail = any(l["kind"] == "fill" and not l["ok"] and l["rows_to"] < lengths[0] for l in left)
    cancelled = client_closed and not completed(left) and own_fail
    # phase 3: its lane is back: --lanes requests are all in flight at once, the short one with its alone bytes
    start = len(lines(a.trace))
    longs = [k % (len(bodies) - 1) for k in range(a.lanes - 1)]  # the long requests fill all lanes but one, the short one last
    after_bodies = [bodies[k] for k in longs] + [bodies[-1]]
    after_lengths = [lengths[k] for k in longs] + [lengths[-1]]
    after = concurrent(a.url, after_bodies)
    after_ids = phase_ids(lines(a.trace), start, after_lengths)
    third = settle(a.trace, after_ids, completed)[start:]
    first_call = [min(l["t0_ns"] for l in third if l["kind"] == "fill" and l["id"] == i) for i in after_ids]
    last_done = [max(l["t1_ns"] for l in third if l["kind"] == "complete" and l["id"] == i) for i in after_ids]
    # every request holds a lane before any 16K readout ends: a lane the left client kept would hold one back until then
    together = max(first_call) < min(last_done[:-1])
    served = all(r[2] == 200 for r in after) and all(l["ok"] for l in third if l["kind"] == "complete" and l["id"] in after_ids)
    # the left request's own failed call is the only ok=false line of the whole run
    stray = [l for l in lines(a.trace)[run0:] if not l["ok"] and l["id"] != gone]
    freed = cancelled and together and served and not stray and after[-1][0] == alone
    ok &= freed
    print("| request | id | rows | pieces | T_piece mean / max ms | T_last_piece + T_finish ms | completion ms | longest wait ms | "
          "most steps between turns (bound) | rules |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n, o in zip(names, out):
        r = o["rules"]
        if not r["lines"]:
            print(f"| {n} | {o['id']} | {o['rows']} | lines missing | | | | | | FAIL |")
            continue
        passed = r["ok_one_piece_each"] and r["contiguous"] and r["completion_in_last_step"] and r["submit_before_first_call"] and r["within_bound"]
        print(f"| {n} | {o['id']} | {o['rows']} | {r['pieces']} | {o['t_piece_ms']['mean']:.1f} / {o['t_piece_ms']['max']:.1f} | "
              f"{o['t_last_piece_and_finish_ms']:.1f} | {o['completion_ms']:.1f} | {o['longest_wait_ms']:.1f} | "
              f"{r['most_steps_between_turns']} ({r['bound']}) | {'pass' if passed else 'FAIL'} |")
    steps = sorted((l for l in trace if l["kind"] in ("fill", "round")), key=lambda l: l["t0_ns"])
    print(f"engine steps never overlap: {all(x['t1_ns'] <= y['t0_ns'] for x, y in zip(steps, steps[1:]))}; client latency: "
          + ", ".join(f"{n} {r[1]:.2f} s" for n, r in zip(names, replies)))
    print(f"the short request beside the readouts gives its alone bytes: {beside}")
    print(f"a leaving client is cancelled mid-prompt (an ok=false call before row {lengths[0]}, no completion): {cancelled}; "
          f"then {a.lanes} requests are in flight at once: {together}; the short one's bytes after equal alone: {after[-1][0] == alone}")
    print(f"the left client closed first: {client_closed}; after it every request gives 200 and an ok completion: {served}; "
          f"ok=false lines of other requests in the run: {len(stray)}")
    if a.report:
        Path(a.report).write_text(json.dumps({"requests": [dict(o, name=n) for n, o in zip(names, out)], "beside_equals_alone": beside,
                                              "leave_cancelled_mid_prompt": cancelled, "client_closed": client_closed,
                                              "after_served": served, "stray_failed_lines": stray, "lanes_in_flight_after": together, "pass": ok}, indent=1))
    print("PASS gate 6" if ok else "FAIL gate 6")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
