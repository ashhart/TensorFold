"""Matches each run's draft counts file against its receipt, its run log and, when kept, the server's own log lines."""
import copy
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = ("out/27b-auto", "out/35b", "out/35b-serial", "out/35b-attached", "out/35b-hipvis0")
DONE = re.compile(r"\] done (\S+) .* tokens=(\d+) sha=([0-9a-f]+) finish=(\w+) rounds=(\d+) accepted=(\d+)/(\d+)$")
POST = re.compile(r'"POST (\S+) HTTP/1\.1" (\d+)')
SAID = re.compile(r"Draft counts: (\d+) of (\d+) drafted tokens accepted over (\d+) rounds \((\d+) of (\d+) attempts")
ROUTE = {"code": "/v1/completions", "chat": "/v1/chat/completions"}
KEYS = ("rounds", "drafted", "accepted")


def served(log):
    """Every finished request in a server log: its line, the POST line before it and the counters it printed."""
    out, post = [], None
    for n, line in enumerate(log.read_text().splitlines(), 1):
        if m := POST.search(line):
            post = (n, m[1], m[2])
        elif m := DONE.search(line):
            out.append({"line": n, "post_line": post[0], "route": post[1], "http": post[2], "id": m[1], "tokens": int(m[2]),
                        "sha": m[3], "finish": m[4], "rounds": int(m[5]), "accepted": int(m[6]), "drafted": int(m[7])})
    return out


def cells(samples):
    """The receipt's samples grouped by cell, in run order: [(fixture, temperature, [index, ...]), ...]."""
    out = []
    for i, s in enumerate(samples):
        if not out or out[-1][:2] != (s["fixture_id"], s["temperature"]):
            out.append((s["fixture_id"], s["temperature"], []))
        out[-1][2].append(i)
    return out


def check(run, drafts, rec, log_text, serve):
    """Problems with one run, plus the lines to print; an empty problem list means every count matches."""
    bad, lines = [], []
    rows, kept = drafts["samples"], rec["samples"]
    if drafts["run_id"] != rec["run_id"] or drafts["receipt"] != Path(run).name + ".json" or len(rows) != len(kept):
        bad.append(f"{run}: the draft counts file is not this receipt's")
        return bad, lines
    for d, r in zip(rows, kept):
        if any(d[k] != r[k] for k in ("fixture_id", "temperature", "repeat", "status", "completion_tokens")):
            bad.append(f"{run}: sample {d['fixture_id']} T={d['temperature']} r{d['repeat']} differs from the receipt's")
    sums = {k: sum(d[k] for d in rows) for k in KEYS}
    t = drafts["totals"]
    if any(t[k] != sums[k] for k in KEYS) or t["attempts"] != len(rows) or t["reported"] != sum(d["rounds"] is not None for d in rows):
        bad.append(f"{run}: totals {t} are not the sum of its samples {sums}")
    said = SAID.findall(log_text)
    if [tuple(map(int, x)) for x in said] != [(t["accepted"], t["drafted"], t["rounds"], t["reported"], t["attempts"])]:
        bad.append(f"{run}: the run's printed line {said} is not its totals")
    lines.append(f"{run}: drafter_passed={drafts['drafter_passed']} attempts={t['attempts']} reported={t['reported']} "
                 f"rounds={t['rounds']} drafted={t['drafted']} accepted={t['accepted']} (sum of samples, and the run's line)")
    if serve is None:
        lines.append("  no server log: a managed run unlinks its server's log as it starts")
        return bad, lines
    groups, done, at, log = cells(kept), serve, 0, f"{Path(run).parent}/35b-serve.log"
    if len(done) != len(kept) + len(groups):
        bad.append(f"{run}: {len(done)} finished requests in the server log for {len(kept)} samples and {len(groups)} warm-ups")
        return bad, lines
    from_log = dict.fromkeys(KEYS, 0)
    for fixture, temp, members in groups:
        w = done[at]
        lines.append(f"  warm-up {fixture} T={temp}: {log}:{w['line']} rounds={w['rounds']} accepted={w['accepted']}/{w['drafted']} (not a sample)")
        at += 1
        for i in members:
            d, r, s = rows[i], kept[i], done[at]
            at += 1
            same = all(d[k] == s[k] for k in KEYS) and d["completion_tokens"] == s["tokens"] and r["token_sha"] == s["sha"]
            if not same or s["route"] != ROUTE[fixture] or s["http"] != "200":
                bad.append(f"{run}: sample {fixture} T={temp} r{d['repeat']} does not match {log}:{s['line']}")
            for k in KEYS:
                from_log[k] += s[k]
            lines.append(f"  {fixture} T={temp} r{d['repeat']}: rounds={d['rounds']} drafted={d['drafted']} accepted={d['accepted']} "
                         f"sha={r['token_sha']} = {log}:{s['line']} ({s['route']} at line {s['post_line']}, finish={s['finish']}, "
                         f"rounds={s['rounds']} accepted={s['accepted']}/{s['drafted']} sha={s['sha']})")
    if from_log != sums:
        bad.append(f"{run}: the server log's samples sum to {from_log}, the file to {sums}")
    lines.append(f"  server log, warm-ups left out: rounds={from_log['rounds']} drafted={from_log['drafted']} accepted={from_log['accepted']}")
    return bad, lines


def load(run):
    """One run's draft counts, receipt, run log and, for an attached run, its server's log."""
    folder = (HERE / run).parent
    serve = folder / "35b-serve.log" if run.endswith("attached") else None
    return (json.loads((HERE / f"{run}.drafts.json").read_text()), json.loads((HERE / f"{run}.json").read_text()),
            (HERE / f"{run}.log").read_text(), served(serve) if serve else None)


def main():
    problems = []
    for run in RUNS:
        bad, lines = check(run, *load(run))
        problems += bad
        print("\n".join(lines))
    # controls: the same check refuses a count off by one and a server log read one request out of step
    drafts, rec, text, serve = load("out/35b-attached")
    off = copy.deepcopy(drafts)
    off["samples"][7]["drafted"] += 1
    off["totals"]["drafted"] += 1
    shifted = serve[1:] + serve[:1]
    for name, args in (("one drafted count +1", (off, rec, text, serve)), ("log one request out of step", (drafts, rec, text, shifted))):
        bad, _ = check("out/35b-attached", *args)
        print(f"control, {name}: {'refused: ' + '; '.join(bad) if bad else 'NOT REFUSED'}")
        if not bad:
            problems.append(f"control {name} was not refused")
    print("\n".join(problems) if problems else "every count matches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
