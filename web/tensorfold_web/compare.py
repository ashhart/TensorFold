"""Two Bench runs side by side: prompts paired one by one, and a test that says whether a difference is noise.

Pairs are made on what was sent (a suite's prompt, or a prefill size), so two runs compare only when they asked the
same things: runs whose suites have different digests are refused. Pass and fail (exactness now, the quality suite
later) go through an exact McNemar test; speed goes through a sign test on the per-prompt direction. Both are exact
binomial tests, small-sample safe, and p-values are given as they are, never rounded to a threshold.
"""

from __future__ import annotations

import html
import json
import math
import statistics
import time

from . import report

NOISE_P = 0.05            # below this, a difference is reported as one
SAME_SPEED = 0.02         # a median change smaller than this is "the same speed", whatever the test says


def binomial_two_sided(k: int, n: int) -> float:
    """P of a split at least as uneven as k out of n when both sides are equally likely: the exact sign test."""

    if n == 0:
        return 1.0
    lo = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / 2**n
    return min(1.0, 2 * tail)


def mcnemar(only_a: int, only_b: int) -> float:
    """The exact McNemar test: only the discordant pairs count, and they should split evenly if A and B are alike."""

    return binomial_two_sided(only_a, only_a + only_b)


class Refused(ValueError):
    """Two runs that cannot be compared, and why."""


def _digest(run: dict) -> dict | None:
    return run.get("suites_version")


def check(a: dict, b: dict) -> list[str]:
    """Raises Refused when the runs asked different things; returns the caveats when they may still be compared."""

    notes: list[str] = []
    if (a.get("mode") == "prefill") != (b.get("mode") == "prefill"):
        raise Refused("a prefill sweep compares only with another prefill sweep")
    if (a.get("mode") == "load") != (b.get("mode") == "load"):
        raise Refused("a load sweep compares only with another load sweep")
    if (a.get("mode") == "quality") != (b.get("mode") == "quality"):
        raise Refused("a quality run compares only with another quality run")
    if "kl" in (a.get("mode"), b.get("mode")):
        raise Refused("a KL run is a comparison already: open each one")
    if (a.get("mode") == "tools") != (b.get("mode") == "tools"):
        raise Refused("a tool-call run compares only with another tool-call run")
    if "load" in (a.get("mode"), b.get("mode")):
        # a load sweep's replies are known by their place in a wave: they pair only when both sent the same prompts
        if a.get("mode") != b.get("mode"):
            raise Refused("a load sweep compares only with another load sweep")
        if sorted(a.get("suites") or []) != sorted(b.get("suites") or []):
            raise Refused("the two load sweeps did not send the same suites: their replies are not the same prompts")
    da, db = _digest(a), _digest(b)
    if da and db and a.get("mode") in ("tools", "quality") and da.get("digest") != db.get("digest"):
        raise Refused(f"the two runs did not use the same {'tool-eval-bench version and scenarios' if a.get('mode') == 'tools' else 'quality items'} "
                      f"({da.get('digest')} against {db.get('digest')})")
    if da and db:
        if da.get("digest") != db.get("digest") and set(a.get("suites") or []) & {
                s for s in b.get("suites") or [] if not s.startswith("custom-")}:
            raise Refused(f"the built-in suites differ (digest {da.get('digest')} against {db.get('digest')}): "
                          "the prompts are not the same")
        for sid in set(da.get("custom") or {}) & set(db.get("custom") or {}):
            if da["custom"][sid] != db["custom"][sid]:
                raise Refused(f"the suite {sid} changed between the two runs")
    elif a.get("mode") != "prefill":
        notes.append("one of the runs predates suite versions (10 Oct 2026); prompts are paired by name, and the "
                     "built-in prompts have not changed since")
    if a.get("model") != b.get("model"):
        notes.append("the two runs served different models")
    ma, mb = a.get("machine") or {}, b.get("machine") or {}
    if (ma.get("chip"), ma.get("ram_bytes")) != (mb.get("chip"), mb.get("ram_bytes")):
        notes.append("the two runs were made on different machines")
    return notes


def _pairs(a: dict, b: dict) -> list[dict]:
    """One row per prompt both runs answered with drafts as they were (the measured reply)."""

    def index(run: dict) -> dict:
        out: dict = {}
        for r in run.get("results") or []:
            key = (r["suite"], str(r.get("size") or r["prompt"]))
            if r.get("draft", True):
                out.setdefault(key, {})["on"] = r
            else:
                out.setdefault(key, {})["off"] = r
        return out

    ia, ib = index(a), index(b)
    names = {**report._names(), **(a.get("suite_names") or {}), **(b.get("suite_names") or {})}
    rows = []
    for key in [k for k in ia if k in ib]:
        ra, rb = ia[key].get("on"), ib[key].get("on")
        if not ra or not rb or ra.get("error") or rb.get("error"):
            continue                                    # a failed or cut reply says nothing of speed or text
        ma, mb = ra.get("m") or {}, rb.get("m") or {}

        def passed(side: dict) -> bool | None:
            if "passed" in (side.get("on") or {}):          # a quality run: the item's own score
                p = side["on"]["passed"]
                return None if p is None else bool(p)          # a request that failed was not graded
            off = side.get("off")
            return None if off is None or off.get("same") is None else bool(off["same"])

        row = {"suite": key[0], "suite_name": names.get(key[0], key[0]), "prompt": key[1],
               "a": {"tps": ma.get("tps"), "ttft": ma.get("ttft"), "tokens": ma.get("tokens"),
                     "prefill_tps": ma.get("prefill_tps"), "error": ra.get("error"), "pass": passed(ia[key])},
               "b": {"tps": mb.get("tps"), "ttft": mb.get("ttft"), "tokens": mb.get("tokens"),
                     "prefill_tps": mb.get("prefill_tps"), "error": rb.get("error"), "pass": passed(ib[key])},
               # the stored text stops at 4,000 characters: the digest of the whole reply decides when both have it
               "same_text": (ra["text_sha"] == rb["text_sha"]) if ra.get("text_sha") and rb.get("text_sha") else
                            bool(ra.get("text")) and ra.get("text") == rb.get("text") and len(ra["text"]) < 4000,
               "same_tokens": bool(ma.get("sha")) and ma.get("sha") == mb.get("sha")}
        speed = "prefill_tps" if a.get("mode") == "prefill" else "tps"
        va, vb = row["a"][speed], row["b"][speed]
        row["change"] = (vb / va - 1) if va and vb else None
        rows.append(row)
    return rows


def _overall(run: dict) -> dict:
    o = (run.get("summary") or {}).get("overall") or {}
    hw = run.get("hardware") or {}
    return {"tps": o.get("tps"), "tps_no_drafts": o.get("tps_no_drafts"), "gain": o.get("gain"), "ttft": o.get("ttft"),
            "exact": o.get("exact"), "checked": o.get("checked"), "j_per_token": hw.get("j_per_token"),
            "avg_watts": hw.get("avg_watts"), "engine_peak_bytes": (run.get("memory") or {}).get("engine_peak_bytes")}


def _who(run: dict) -> dict:
    m = run.get("machine") or {}
    return {"id": run.get("id"), "created": run.get("created"), "mode": run.get("mode"), "model": run.get("model"),
            "engine": run.get("engine"), "chip": m.get("chip"), "ram_bytes": m.get("ram_bytes"),
            "studio": run.get("studio"), "suites_version": run.get("suites_version"), "status": run.get("status")}


def compare(a: dict, b: dict) -> dict:
    """A against B: the pairs, the tests, the overall figures and plain sentences; raises Refused when they differ in
    what they asked."""

    notes = check(a, b)
    rows = _pairs(a, b)
    if not rows:
        raise Refused("the two runs have no prompt in common")
    changes = [r["change"] for r in rows if r["change"] is not None]
    faster_b = sum(1 for c in changes if c > 0)
    slower_b = sum(1 for c in changes if c < 0)
    speed_p = binomial_two_sided(faster_b, faster_b + slower_b)
    median_change = statistics.median(changes) if changes else None
    checked = [r for r in rows if r["a"]["pass"] is not None and r["b"]["pass"] is not None]
    only_a = sum(1 for r in checked if r["a"]["pass"] and not r["b"]["pass"])
    only_b = sum(1 for r in checked if r["b"]["pass"] and not r["a"]["pass"])
    both = sum(1 for r in checked if r["a"]["pass"] and r["b"]["pass"])
    neither = len(checked) - both - only_a - only_b
    out = {
        "a": _who(a), "b": _who(b), "notes": notes, "pairs": rows,
        "speed": {"median_change": median_change, "b_faster": faster_b, "b_slower": slower_b, "p": speed_p,
                  "too_few": binomial_two_sided(0, faster_b + slower_b) >= NOISE_P,
                  "within_noise": median_change is None or abs(median_change) < SAME_SPEED or speed_p >= NOISE_P,
                  "measure": "prefill tokens per second" if a.get("mode") == "prefill" else "tokens per second"},
        "exactness": {"checked": len(checked), "both": both, "only_a": only_a, "only_b": only_b, "neither": neither,
                      "p": mcnemar(only_a, only_b), "within_noise": mcnemar(only_a, only_b) >= NOISE_P,
                      "too_few": mcnemar(0, only_a + only_b) >= NOISE_P},
        "same_text": sum(1 for r in rows if r["same_text"]),
        "by_suite": _by_suite(checked),
        "overall": {"a": _overall(a), "b": _overall(b)},
    }
    out["findings"] = findings(out)
    return out


def _by_suite(checked: list[dict]) -> dict:
    """McNemar suite by suite (category by category in a quality run): a loss confined to one kind of task can hide in
    the overall count."""

    out: dict = {}
    for r in checked:
        d = out.setdefault(r["suite"], {"name": r["suite_name"], "checked": 0, "only_a": 0, "only_b": 0})
        d["checked"] += 1
        d["only_a"] += int(bool(r["a"]["pass"]) and not r["b"]["pass"])
        d["only_b"] += int(bool(r["b"]["pass"]) and not r["a"]["pass"])
    for d in out.values():
        d["p"] = mcnemar(d["only_a"], d["only_b"])
    return out


def _p(p: float) -> str:
    return f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"


def findings(c: dict) -> list[str]:
    out: list[str] = []
    sp, ex = c["speed"], c["exactness"]
    n = sp["b_faster"] + sp["b_slower"]
    if sp["median_change"] is not None:
        ch = sp["median_change"]
        size = f"{abs(ch) * 100:.0f}% {'faster' if ch > 0 else 'slower'}" if abs(ch) >= 0.005 else "as fast"
        if sp.get("too_few") and abs(ch) >= SAME_SPEED:
            out.append(f"Speed: B is {size} at the median, but {n} prompts are too few to tell it from chance "
                       f"(B faster on {sp['b_faster']} of {n}; even all of them would not be enough).")
        elif sp["within_noise"]:
            out.append(f"Speed: B is {size} at the median, no clear difference (B faster on {sp['b_faster']} of {n} "
                       f"prompts, {_p(sp['p'])}): not shown to differ, which is not the same as shown equal.")
        else:
            out.append(f"Speed: B is {size} at the median, a real difference (B faster on {sp['b_faster']} of {n} "
                       f"prompts, {_p(sp['p'])}).")
    graded = c["a"].get("mode") in ("quality", "tools")
    what, good = ("Quality", "right") if graded else ("Exactness", "identical")
    if ex["checked"]:
        if ex["only_a"] == ex["only_b"] == 0:
            out.append(f"{what}: the same outcome on all {ex['checked']} items ({ex['both']} {good} on both sides).")
        else:
            verdict = ("too few differing items to tell" if ex.get("too_few") else
                       "no clear difference" if ex["within_noise"] else "a real difference")
            out.append(f"{what}: {ex['only_a']} {good} only in A, {ex['only_b']} only in B, {ex['both']} on both, "
                       f"{verdict} (McNemar, {_p(ex['p'])}).")
    split = [d for d in (c.get("by_suite") or {}).values() if d["p"] < NOISE_P]
    if split and len(c.get("by_suite") or {}) > 1:
        many = len(c["by_suite"])
        for d in split:
            better = "A" if d["only_a"] > d["only_b"] else "B"
            # several groups tested at once: below 0.05 / groups it holds, above it may be chance (Bonferroni)
            caveat = (f"it holds even counting the {many} groups tested" if d["p"] < NOISE_P / many else
                      f"one of {many} groups tested, so it can be chance: worth a second run")
            out.append(f"{d['name']}: {d['only_a']} {good} only in A, {d['only_b']} only in B, {better} better "
                       f"(McNemar, {_p(d['p'])}); {caveat}.")
    total = len(c["pairs"])
    out.append(f"Replies: {c['same_text']} of {total} word for word the same in both runs." if total else "")
    return [x for x in out if x]


# -- the HTML report of a comparison -------------------------------------------------------------------------------------

def _e(x: object) -> str:
    return html.escape(str(x), quote=True)


def html_report(a: dict, b: dict, c: dict) -> str:
    """One self-contained file: the verdicts, the overall deltas, every pair, and both receipts embedded."""

    fmt = {"tps": report._tps, "tps_no_drafts": report._tps, "gain": report._pct, "ttft": report._secs,
           "j_per_token": report._joules, "avg_watts": lambda x: f"{report._n(x)} W",
           "engine_peak_bytes": report._gb}
    labels = {"tps": "Writing, tok/s", "tps_no_drafts": "Without drafts", "gain": "Gain from drafts",
              "ttft": "First token", "j_per_token": "Energy per token", "avg_watts": "Average power",
              "engine_peak_bytes": "Engine memory, peak"}
    oa, ob = c["overall"]["a"], c["overall"]["b"]
    over = "".join(f"<tr><td>{_e(labels[k])}</td><td>{_e(f(oa[k]))}</td><td>{_e(f(ob[k]))}</td><td>"
                   f"{_e(report._pct(ob[k] / oa[k] - 1)) if oa[k] and ob[k] and k != 'gain' else ''}</td></tr>"
                   for k, f in fmt.items() if oa.get(k) is not None or ob.get(k) is not None)
    speed_key = "prefill_tps" if c["a"]["mode"] == "prefill" else "tps"

    def mark(v: bool | None) -> str:
        return "—" if v is None else "yes" if v else "no"

    rows = "".join(
        f"<tr><td>{_e(r['suite_name'])}: {_e(r['prompt'])}</td><td>{_e(report._tps(r['a'][speed_key]))}</td>"
        f"<td>{_e(report._tps(r['b'][speed_key]))}</td><td>{_e(report._pct(r['change']))}</td>"
        f"<td>{mark(r['a']['pass'])} / {mark(r['b']['pass'])}</td><td>{'same' if r['same_text'] else 'differ'}</td></tr>"
        for r in c["pairs"])

    def who(w: dict, tag: str) -> str:
        when = time.strftime("%d %b %Y, %H:%M", time.localtime(w.get("created") or 0))
        sv = w.get("suites_version") or {}
        return (f"<dt>{tag}</dt><dd>{_e(w.get('model'))} · {_e(w.get('engine') or 'engine not identified')} · "
                f"{_e(w.get('chip') or '')} · {_e(when)} · run {_e(w.get('id'))}"
                f"{' · suites ' + _e(sv.get('digest')) if sv else ''}</dd>")

    data = json.dumps({"kind": "tensorfold-studio-bench-comparison", "a": a, "b": b},
                      ensure_ascii=False).replace("</", "<\\/")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Comparison: {_e(c['a']['model'])} and {_e(c['b']['model'])}</title><style>{report.CSS}</style></head>"
        "<body><main><header><div class=\"brand\">TensorFold Web<small>Bench comparison</small></div></header>"
        f"<h1>A against B</h1><section><dl class=\"kv\">{who(c['a'], 'A')}{who(c['b'], 'B')}</dl>"
        + ("<ul>" + "".join(f"<li>{_e(n)}</li>" for n in c["notes"]) + "</ul>" if c["notes"] else "")
        + "</section><section><h2>What the comparison shows</h2><ul class=\"find\">"
        + "".join(f"<li>{_e(f)}</li>" for f in c["findings"]) + "</ul></section>"
        "<section><h2>Overall</h2><div class=\"tw\"><table><thead><tr><th></th><th>A</th><th>B</th><th>B against A</th>"
        f"</tr></thead><tbody>{over}</tbody></table></div></section>"
        "<section><h2>Prompt by prompt</h2><div class=\"tw\"><table><thead><tr><th>Prompt</th><th>A</th><th>B</th>"
        "<th>change</th><th>identical without drafts, A / B</th><th>replies</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div></section>"
        "<footer>Speed: a sign test on which run was faster prompt by prompt. Exactness: an exact McNemar test on the "
        "prompts where only one run was identical with drafts off. Both receipts are embedded at the end of this file "
        "(<code>#runs</code>).</footer>"
        f'</main><script type="application/json" id="runs">{data}</script></body></html>')
