"""Parse kept/refused lesson receipts without inventing diagnostics absent from older binaries."""
import hashlib
import json
import re


def events(text):
    result, data, kind = [], [], "message"

    def flush():
        if not data:
            return
        raw = "\n".join(data)
        if raw == "[DONE]":
            result.append({"event": "done", "data": None})
        else:
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("lesson event must contain an object")
            result.append({"event": kind, "data": value})

    for line in text.replace("\r\n", "\n").split("\n"):
        if not line:
            flush()
            data, kind = [], "message"
        elif line.startswith("event:"):
            kind = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    flush()
    return result


def outcome(stream, graph):
    learned = [e["data"] for e in stream if e["event"] == "learned"]
    failures = [e["data"] for e in stream if e["event"] in ("failed", "error")]
    saved = any(e["event"] == "saved" and e["data"].get("saved") is True for e in stream)
    done = bool(stream) and stream[-1]["event"] == "done"
    facts = graph.get("facts", [])
    kept = [f for f in facts if f.get("state") == "learned"]
    recalled = bool(learned) and all(e.get("recalled") is True for e in learned)
    accepted = done and saved and recalled and bool(kept) and not failures
    status = "kept" if accepted else "refused" if done and learned and not recalled and not failures else "error"
    if not accepted and kept:
        status = "error"
    return {"status": status, "accepted": accepted, "learned_events": learned, "failures": failures,
            "steps": [e.get("steps") for e in learned], "saved": saved, "done": done,
            "kept_facts": kept}


ROUND = re.compile(r"slide: round (\d+): (\d+) steps .*?the held-out answers "
                   r"(come back|do not come back) token for token, and (come back|do not come back) when asked")
COUNTS = re.compile(r"fact (\d+): held plain (\d+)/(\d+)(?: FAIL)?, "
                    r"held thinking (\d+)/(\d+)(?: FAIL)?, probes (\d+)/(\d+)")
REDRAW = re.compile(r"wordings redrawn, attempt (\d+)/(\d+)")
SEED = re.compile(r"\bdone (\S+).*\bseed=(\d+)\b")


def diagnostics(text):
    rounds, seeds, redraws, pending = [], [], [], []
    attempt = 0
    for number, line in enumerate(text.splitlines(), 1):
        seed = SEED.search(line)
        if seed:
            row = {"line": number, "request_id": seed[1], "seed": int(seed[2]), "raw": line}
            seeds.append(row)
            pending.append(row)
        redraw = REDRAW.search(line)
        if redraw:
            attempt = int(redraw[1])
            redraws.append({"line": number, "attempt": attempt, "limit": int(redraw[2]), "raw": line})
        match = ROUND.search(line)
        if not match:
            continue
        counts = []
        for c in COUNTS.finditer(line):
            pairs = [(int(c[i]), int(c[i + 1])) for i in (2, 4, 6)]
            if any(passed > asked for passed, asked in pairs):
                raise ValueError("invalid per-round recall counts")
            counts.append({"fact": int(c[1]), **{name: {"passed": pair[0], "asked": pair[1]}
                           for name, pair in zip(("plain", "thinking", "probes"), pairs)}})
        rounds.append({"round": int(match[1]), "cumulative_steps": int(match[2]), "wording_attempt": attempt,
                       "held_token_equal": match[3] == "come back", "recalled": match[4] == "come back",
                       "counts": counts or None, "request_seeds_since_previous_round": pending,
                       "line": number, "raw": line})
        pending = []
    return {"rounds": rounds, "request_seeds": seeds, "redraws": redraws,
            "trailing_request_seeds": pending,
            "rounds_missing_counts": sum(r["counts"] is None for r in rounds),
            "note": "Seeds belong to logged requests, including writer and check calls; none are inferred."}


def tokens(reply):
    runtime = reply.get("tensorfold", {})
    ids = runtime.get("token_ids")
    if not isinstance(ids, list) or any(type(x) is not int or not 0 <= x < 2 ** 32 for x in ids):
        raise ValueError("reply lacks valid u32 tensorfold.token_ids")
    digest = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    if runtime.get("token_sha256", digest) != digest:
        raise ValueError("server token digest disagrees with its IDs")
    return {"ids": ids, "sha256": digest}


def changed(before, after):
    if len(before) != len(after):
        raise ValueError("unrelated panel size changed")
    return [i for i, (a, b) in enumerate(zip(before, after)) if a["tokens"]["ids"] != b["tokens"]["ids"]]


def summary(rows, runs):
    cells = []
    for arm in ("base", "candidate"):
        for thinking in (False, True):
            for seeded in (False, True):
                group = [r for r in rows if (r["arm"], r["thinking"], r["prompt_seed"]) == (arm, thinking, seeded)]
                kept = sum(r.get("lesson_status", r.get("status")) == "kept" for r in group)
                refused = sum(r.get("lesson_status", r.get("status")) == "refused" for r in group)
                errors = sum(r.get("status") == "error" for r in group)
                cells.append({"arm": arm, "thinking": thinking, "prompt_seed": seeded, "scheduled": runs,
                              "recorded": len(group), "kept": kept, "refused": refused, "errors": errors,
                              "not_recorded": runs - len(group), "keep_rate_of_scheduled": kept / runs,
                              "unresolved_lessons": runs - kept - refused,
                              "complete": len(group) == runs and errors == 0,
                              "steps": [r.get("steps") for r in group],
                              "rounds_missing_counts": sum(r.get("diagnostics", {}).get("rounds_missing_counts", 0)
                                                          for r in group)})
    arms = {}
    for arm in ("base", "candidate"):
        group = [c for c in cells if c["arm"] == arm]
        arms[arm] = {name: sum(c[name] for c in group) for name in ("scheduled", "recorded", "kept", "refused", "errors")}
        arms[arm]["keep_rate_of_scheduled"] = arms[arm]["kept"] / arms[arm]["scheduled"]
    return {"cells": cells, "arms": arms, "complete": all(c["complete"] for c in cells),
            "note": "Errors in post-lesson measurements may overlap a known kept/refused verdict; all scheduled trials remain."}
