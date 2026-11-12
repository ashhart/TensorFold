"""Living cards."""

from __future__ import annotations

import json
import time

import httpx

from . import living


def _read_events(r: httpx.Response, it: living.Item) -> bool:

    kind = None
    for line in r.iter_lines():
        if line.startswith("event:"):
            kind = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if kind == "learning" and payload.get("id") is not None and \
                all(f["id"] != payload.get("id") for f in it.found):
            it.found.append({"id": payload.get("id"), "text": None, "outcome": "found"})   # never announced
        if kind == "learning" and not living.matches_approved(it.fact, it.found):
            read = "; ".join(f"\"{f['text']}\"" for f in it.found) or "nothing"
            it.state = "refused"
            it.message = f"the engine read {read} instead of the approved fact; stopped before it learned"
            return True                          # leaving closes the request: the engine stops there
        if kind == "fact":
            it.found.append({"id": payload.get("id"), "text": payload.get("text"), "outcome": "found"})
        elif kind == "learned" or "recalled" in payload:
            f = next((x for x in it.found if x["id"] == payload.get("id")), None)
            if f is None:
                f = {"id": payload.get("id"), "text": it.fact, "outcome": "found"}
                it.found.append(f)
            f["outcome"] = "kept" if payload.get("recalled") else "missed"
            f["message"] = payload.get("message")
            it.steps = (it.steps or 0) + int(payload.get("steps") or 0)
        elif kind == "failed":
            it.message = payload.get("message")
    return False


def why_missed(lines: list[str], fact: str | None = None) -> str | None:

    def mine(line: str) -> bool:
        m = living._KEPT_FOR.search(line)
        return bool(m) and (fact is None or living.canonical(m.group(2)) == living.canonical(fact))
    since = max((i for i, line in enumerate(lines) if mine(line)), default=None)
    if since is None:
        return None
    tail = lines[since:]
    kept = int(living._KEPT_FOR.search(lines[since]).group(1))
    if kept == 0:
        return "the model wrote no clean question to learn this fact from; try another wording"
    rounds = [m for m in (living._ROUND.search(x) for x in tail) if m]
    if rounds and all(m.group(3) == "do not come back" for m in rounds):
        return f"it never came back when asked, in {len(rounds)} rounds; nothing of it was kept"
    return None


def record(copy: str, job: living.Job, it: living.Item) -> None:
    """The copy's record of what was taught: one entry per fact sent."""

    with living._record_lock:
        doc = living.lessons(copy)
        doc.append({"job": job.id, "at": time.time(), "source": job.source, "fact": it.fact, "question": it.question,
                    "outcome": it.state, "message": it.message, "steps": it.steps, "seconds": it.seconds,
                    "before": it.before, "after": it.after, "found": it.found})
        living._write(living._lessons_path(copy), doc)


def forget_plan(name: str, fact: str, manifests: list[dict]) -> dict:

    record = living.lessons(name)
    kept = [x for x in record if x.get("outcome") in living.KEPT]
    # Prefer taught wording, then the engine graph's rewrite, then the recorded lesson wording.
    taught = next((x for x in kept if x.get("fact") == fact), None) or next(
        (x for x in kept if living.canonical(fact) in {living.canonical(str(t)) for t in
                                                [x.get("fact") or "", *(f.get("text") or "" for f in x.get("found") or [])]}),
        None)
    if taught is None:
        raise ValueError("TensorFold Web did not teach this fact, so there is no version from before it; restore an "
                         "earlier version instead")
    version = next((m for m in manifests if m.get("label") == f"before lesson {taught['job']}"), None)
    if version is None:
        raise ValueError("no version was kept before the lesson that taught this fact")
    fact = taught["fact"]
    later, seen = [], set()
    for x in record:                                 # everything kept since that version, but the fact itself
        if x.get("outcome") in living.KEPT and x.get("at", 0) > version["at"] and x["fact"] != fact and x["fact"] not in seen:
            seen.add(x["fact"])
            later.append({"fact": x["fact"], "question": x.get("question") or ""})
    return {"version": version["id"], "again": later, "fact": fact}


def graph_summary(body: dict) -> list[dict]:
    """The engine's fact graph as rows: each fact with its state, neurons and questions."""

    out = []
    for f in body.get("facts") or []:
        if isinstance(f, dict):
            out.append({"id": f.get("id"), "text": f.get("text"), "state": f.get("state"),
                        "neurons": len(f.get("neurons") or []), "questions": (f.get("questions") or [])[:6],
                        "source": f.get("source")})
    return out


def _kl_cell(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def card(name: str, *, graph: dict, versions: list[dict], runs: list[dict], engine: str | None) -> str:

    meta = living._read(living._meta_path(name), {})
    record = living.lessons(name)
    facts = graph_summary(graph)
    when = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "—"
    lines = [f"# {name}: a living copy of {meta.get('base') or 'an unknown model'}", "",
             f"Made {when(meta.get('created'))}" + (f", fetched from {meta['fetched_from']}" if meta.get("fetched_from")
                                                    else "") + f". Served by {engine or 'TensorFold'} with `--slide`.", "",
             ("Facts taught into its weights (TensorFold Living Weights): they are part of the model files, answer "
              "every client, and cannot be removed one by one except by restoring a version kept before them. Do not "
              "share this folder if a fact is personal or confidential."), "",
             f"## Facts ({sum(1 for f in facts if f['state'] == 'learned')} kept of {len(facts)} tried)", "",
             "| Fact | State | Source | Neurons |", "|---|---|---|---|"]
    lines += [f"| {f['text']} | {f['state']} | {f.get('source') or '—'} | {f['neurons']} |" for f in facts]
    lines += ["", f"## Lessons sent by TensorFold Web ({len(record)})", "", "| When | Fact | Outcome | Answer before | After |",
              "|---|---|---|---|---|"]
    clip = lambda s: " ".join(str(s or "—").split())[:80].replace("|", "/")
    lines += [f"| {when(x.get('at'))} | {clip(x['fact'])} | {x.get('outcome')} | {clip(x.get('before'))} | "
              f"{clip(x.get('after'))} |" for x in record]
    lines += ["", f"## Versions ({len(versions)})", "", "| When | Version | Facts kept |", "|---|---|---|"]
    lines += [f"| {when(v['at'])} | {v['label']} | {v.get('kept', '—')} |" for v in versions]
    if runs:
        lines += ["", "## Measured (Bench)", "", "| When | Run | Facts kept | Speed | Same tokens drafts off | KL |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {when(r['created'])} | {r['mode']} | {r.get('kept', '—')} | "
                  f"{round(r['tps']) if r.get('tps') else '—'} tok/s | "
                  f"{(str(r['exact']) + ' of ' + str(r['checked'])) if r.get('checked') else '—'} | "
                  f"{_kl_cell(r.get('mean_kl'))} |" for r in runs]
    return "\n".join(lines) + "\n"
