"""A Bench run told for people: findings in plain sentences, a short text to share, and a self-contained HTML report.

Everything here reads the run alone (the receipt), so a report can be rebuilt from a receipt on any machine. Each
finding is written only when the data supports it; a figure the run does not have is left out, never guessed.
"""

from __future__ import annotations

import base64
import html
import itertools
import json
import math
import time
from pathlib import Path

from . import bench


def kl_line(run: dict) -> str:
    """A KL run in one line: against what, the divergence, the agreement."""

    k, ref = run["kl"], run.get("kl_reference") or {}
    against = ("the same engine with drafts off" if ref.get("drafts_off") is not False else
               "the same engine asked for drafts off, which it did not confirm") if ref.get("kind") == "drafts-off" \
        else f"{ref.get('model')} at {ref.get('url')}"
    done = run.get("kl_load_done") or {}
    full = done.get("full_share")
    if not run.get("kl_load"):
        load = ""
    elif run.get("kl_load_verified") is False:
        load = f", {run['kl_load']} background requests asked but the load not established (see the notes)"
    else:
        load = (f", the candidate under {run['kl_load']} requests at once ({done.get('ok', 0)} background replies, "
                f"{done.get('failed', 0)} refused"
                + (f"; all {run['kl_load']} in flight {full * 100:.0f}% of the time" if full is not None else "") + ")")
    if run.get("kl_text") is not None:             # one fixed text, read by both: every position, no replies
        return (f"Against {against}{load}: mean KL {k['mean_kl']:.4f} nats (p99 {k['p99_kl']:.4f}, max "
                f"{k['max_kl']:.4f}) over {k['positions']:,} positions of a fixed text; same top token at "
                f"{k['same_top'] * 100:.1f}%, the same distribution at {k.get('same_distribution', 0) * 100:.1f}%.")
    return (f"Against {against}{load}: mean KL {k['mean_kl']:.4f} nats (p99 {k['p99_kl']:.4f}, max {k['max_kl']:.4f}) "
            f"over {k['positions']:,} positions of {k['prompts']} prompts; same top token at {k['same_top'] * 100:.1f}%, "
            f"replies identical for {k['identical'] * 100:.0f}% of prompts.")


KIND = {"quick": "Quick score", "profile": "Usage profile", "prefill": "Prefill sweep", "load": "Load sweep",
        "quality": "Quality", "tools": "Tool calls", "kl": "KL divergence"}
STATUS = {"done": "complete", "cancelled": "stopped before the end", "failed": "failed",
          "interrupted": "interrupted by a TensorFold Web restart", "running": "still running"}
LOGO = Path(__file__).parent / "ui_dist" / "logo.png"


# -- numbers ------------------------------------------------------------------------------------------------------------

def _n(x: float | None, digits: int = 0) -> str:
    return "—" if x is None else f"{x:,.{digits}f}"


def _tps(x: float | None) -> str:
    return "—" if x is None else f"{x:,.0f}" if x >= 20 else f"{x:,.1f}"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{'+' if x >= 0 else '−'}{abs(x) * 100:.0f}%"


def _secs(x: float | None) -> str:
    if x is None:
        return "—"
    if x < 1:
        return f"{x * 1000:.0f} ms"
    if x < 60:
        return f"{x:.1f} s"
    m, sec = divmod(round(x), 60)
    return f"{m} min {sec:02d} s"


def _k(n: int) -> str:
    return f"{n // 1024}k" if n >= 1024 else str(n)


def _joules(x: float | None) -> str:
    return "—" if x is None else f"{x * 1000:.0f} mJ" if x < 1 else f"{x:.2f} J"


def _money(x: float | None, cur: str) -> str:
    if x is None:
        return "—"
    sym = {"EUR": "€", "USD": "$", "GBP": "£"}.get(cur)
    v = f"{x:.2g}" if x < 1 else f"{x:,.2f}"
    return f"{v} {sym}" if sym == "€" else f"{sym}{v}" if sym else f"{v} {cur}"


def _gb(b: float | None) -> str:
    return "—" if not b else f"{b / 1024 ** 3:.1f} GB"


def _apart(ratio: float) -> str:
    return f"{ratio:.1f}× apart" if ratio >= 2 else f"{(ratio - 1) * 100:.0f}% apart"


def _names() -> dict[str, str]:
    try:
        return {s["id"]: s["name"] for s in bench.load_suites()["suites"]}
    except (OSError, ValueError, KeyError):
        return {}


def _sweep_points(run: dict) -> list[dict]:
    return [p for p in (run.get("summary") or {}).get("sweep") or [] if not p.get("error") and p.get("prefill_tps")]



# -- findings ---------------------------------------------------------------------------------------------------------

def findings(run: dict) -> list[str]:
    """Plain sentences on what the run shows, each one only when its figures are there."""

    out: list[str] = []
    o = (run.get("summary") or {}).get("overall") or {}
    hw = run.get("hardware") or {}
    names = {**_names(), **(run.get("suite_names") or {})}
    if run.get("status") in ("cancelled", "failed", "interrupted"):
        out.append(f"This run was {STATUS[run['status']]}: the figures cover the replies it finished.")
    th = run.get("thinking") or {}
    if th and th.get("off") is False:
        out.append("Thinking stayed on whichever way it was turned off, so the speeds include reasoning tokens.")
    k = run.get("kl")
    if run.get("mode") == "kl" and k and not k.get("positions") and k.get("unscored"):
        out.append(f"No position could be compared: the engine gave no log-probabilities at the {k['unscored']} "
                   "positions it wrote, so this run says nothing of the distributions.")
    if run.get("mode") == "kl" and k and k.get("positions"):
        out.append(kl_line(run))
        if (run.get("kl_reference") or {}).get("kind") == "drafts-off" and \
                (run.get("kl_reference") or {}).get("drafts_off") is False:
            out.append("The engine did not confirm the reference replies were written with drafts off: if it ignored "
                       "the request, this run compares the engine with itself and its agreement says nothing.")
        if k.get("unscored"):
            out.append(f"{k['unscored']} positions came without log-probabilities on one side and were left out of "
                       "the figures.")
        width = k.get("width", k.get("top", 20))
        if k.get("same_distribution") == 1:
            out.append(f"Every compared position gave the same {width} most likely tokens with the same probabilities, "
                       "checked token by token: the candidate's distributions match the reference's exactly "
                       + ("at every position of the text." if run.get("kl_text") is not None
                          else "where both wrote the same text."))
        elif k.get("same_distribution") is not None:
            out.append(f"{k['same_distribution'] * 100:.1f}% of the compared positions gave the same {width} tokens "
                       "with the same probabilities.")
        if width < k.get("top", 20):
            out.append(f"One side gave {width} alternatives per position where {k.get('top', 20)} were asked: the "
                       "comparison reads that narrower distribution, so a match says less and a mismatch may be "
                       "the width.")
        if k.get("worst") and k["worst"][0]["kl"] > 0:
            w = k["worst"][0]
            where = f"at {w['token']!r}, token {w['at']} of the text" if run.get("kl_text") is not None and \
                w.get("token") is not None else f"at token {w['at']} of {w['prompt']}"
            out.append(f"The largest divergence, {w['kl']:.4f} nats, was {where} "
                       f"(reference {w['ref']!r}, candidate {w['cand']!r}).")
        out.append("Every position of the fixed text is compared: both engines read the same tokens, so none is left "
                   "out after a difference. The divergence is estimated on the top 20 log-probabilities."
                   if run.get("kl_text") is not None else
                   "Positions are compared up to the first token where the two replies differ: after it they no longer "
                   "write on the same text. The divergence is estimated on the top 20 log-probabilities.")
    tl = run.get("tools")
    if run.get("mode") == "tools" and tl:
        res = [r for r in run.get("results") or [] if "status" in r]
        n = {k: sum(1 for r in res if r["status"] == k) for k in ("pass", "partial", "fail")}
        out.append(f"tool-eval-bench {tl.get('version')}: {tl.get('final_score')} of 100 ({tl.get('points')} of "
                   f"{tl.get('max_points')} points), {n['pass']} scenarios passed, {n['partial']} partly, {n['fail']} failed.")
        weak = [c for c in tl.get("categories") or [] if (c.get("percent") or 0) < 100]
        if weak:
            out.append("Short of full marks in " + ", ".join(f"{c['label']} ({c['percent']}%)" for c in weak) + ".")
        if tl.get("safety_warnings"):
            out.append(f"{tl['safety_warnings']} safety warnings: the receipt has them.")
    q = (run.get("summary") or {}).get("quality")
    if run.get("mode") == "quality" and q:
        ov = q["overall"]
        if ov["items"]:
            out.append(f"{ov['passed']} of {ov['items']} items right ({ov['passed'] / ov['items'] * 100:.0f}%), scored "
                       "strictly: the last value counts, an answer that offers two fails.")
        if ov.get("unscored"):
            out.append(f"{ov['unscored']} items got no answer (the request failed): not graded, and not in the share "
                       "right.")
        parts = [f"{names.get(k, k)} {v['passed']}/{v['items']}" for k, v in q["categories"].items()]
        out.append("By category: " + ", ".join(parts) + ".")
        cut = sum(1 for r in run.get("results") or [] if "passed" in r and r["passed"] is False
                  and str(r.get("read", "")).startswith("cut at"))
        if cut:
            out.append(f"{cut} of the items missed were cut at the reply limit before their answer, not answered wrong.")
        lc = q["categories"].get("long_context")
        if lc and lc["passed"] < lc["items"]:
            out.append(f"In long texts the model kept the stale value or no value on {lc['items'] - lc['passed']} of "
                       f"{lc['items']} corrected facts.")
    load = (run.get("summary") or {}).get("load") or []
    if run.get("mode") == "load" and load:
        # "alone" is the level of one request; a comparison only between levels that were measured
        one = next((x for x in load if x["concurrency"] == 1), load[0])
        measured = [x for x in load if x.get("total_tps")]
        unmeasured = [x["concurrency"] for x in load if not x.get("total_tps")]
        best = max(measured or load, key=lambda x: x.get("total_tps") or 0)
        if best["concurrency"] > 1 and one.get("total_tps") and best.get("total_tps"):
            out.append(f"Serving several at once raises the total from {_tps(one['total_tps'])} tokens per second (one "
                       f"request) to {_tps(best['total_tps'])} at {best['concurrency']} at once, "
                       f"{best['total_tps'] / one['total_tps']:.1f}× as much.")
            out.append(f"Each of those {best['concurrency']} requests then writes at {_tps(best['per_request_tps'])} "
                       f"tokens per second, against {_tps(one['per_request_tps'])} alone, and the slowest waits "
                       f"{_secs(best['ttft_p95'])} for its first token.")
        elif not any(x.get("total_tps") for x in load):
            out.append("The engine gave no token counts, so no total speed can be given: a stream event is not a "
                       "token.")
        elif not one.get("total_tps"):
            out.append("The speed of one request alone was not measured (no token counts), so nothing is said of what "
                       "serving several at once adds.")
        elif unmeasured:
            out.append(f"{_tps(one.get('total_tps'))} tokens per second alone; the levels of "
                       f"{', '.join(str(c) for c in unmeasured)} at once were not measured (no token counts), so "
                       "nothing is said of what serving several at once gives.")
        else:
            out.append(f"Serving several at once does not raise the total: {_tps(one.get('total_tps'))} tokens per "
                       "second alone is the most this engine gives here.")
        if unmeasured and best["concurrency"] > 1:
            out.append(f"Not measured at {', '.join(str(c) for c in unmeasured)} at once (no token counts): the best "
                       "total above is among the levels measured only.")
        # a plateau only between two levels both measured
        flat = next((x for x, y in itertools.pairwise(load) if x.get("total_tps") and y.get("total_tps")
                     and y["total_tps"] < 1.1 * x["total_tps"]), None)
        if flat and best["concurrency"] > 1:
            out.append(f"Past {flat['concurrency']} at once the total gains less than 10% per doubling: more requests "
                       "mostly wait longer.")
    if run.get("mode") in ("quick", "profile") and o:
        if o.get("checked"):
            by = "same token digest" if any(r.get("compared_by") == "token_sha" for r in run.get("results", [])) \
                else "same text"
            if o["exact"] == o["checked"]:
                out.append(f"All {o['checked']} replies came out identical with drafts off ({by}): drafts change "
                           "the speed, not the answers.")
            else:
                out.append(f"{o['checked'] - o['exact']} of {o['checked']} replies differed with drafts off; "
                           "the replies are in the receipt.")
        gain = o.get("gain")
        if gain is not None:
            if gain > 0.05:
                faster = f"{1 + gain:.1f}× as fast" if gain >= 1 else f"{gain * 100:.0f}% faster"
                out.append(f"Drafts made writing {faster}: {_tps(o['tps'])} tokens per second against "
                           f"{_tps(o['tps_no_drafts'])} without.")
            elif gain < -0.05:
                out.append(f"Drafts made writing {-gain * 100:.0f}% slower on this machine.")
            else:
                out.append("Drafts made no clear difference in speed (within 5%).")
        suites = [(k, b["tps"]) for k, b in ((run.get("summary") or {}).get("suites") or {}).items() if b.get("tps")]
        if len(suites) >= 2:
            hi, lo = max(suites, key=lambda x: x[1]), min(suites, key=lambda x: x[1])
            if hi[1] / lo[1] > 1.15:
                out.append(f"Fastest on {names.get(hi[0], hi[0])} ({_tps(hi[1])} tok/s), slowest on "
                           f"{names.get(lo[0], lo[0])} ({_tps(lo[1])} tok/s), {_apart(hi[1] / lo[1])}.")
        if o.get("ttft") is not None:
            start = "in under half a second" if o["ttft"] < 0.5 else "after " + _secs(o["ttft"])
            answer = f"; the answer itself after {_secs(o['ttft_answer'])}" if o.get("ttft_answer") else ""
            out.append(f"Replies start {start} (median {_secs(o['ttft'])}{answer}).")
    pts = _sweep_points(run)
    if run.get("mode") == "prefill" and pts:
        first, last = pts[0], pts[-1]
        if len(pts) >= 2:
            out.append(f"Reading prompts: {_n(first['prefill_tps'])} tokens per second at {_k(first['size'])}, "
                       f"{_n(last['prefill_tps'])} at {_k(last['size'])}.")
        if last.get("ttft") is not None:
            out.append(f"A {_k(last['size'])}-token prompt waits {_secs(last['ttft'])} before its first token.")
        if any(p.get("cached") for p in pts):
            out.append("Some prompt tokens came from the cache; those sizes read faster than they would cold.")
    if hw.get("energy_j"):
        whole = "for the whole machine" if hw.get("whole_machine") else f"for the {hw.get('scope')} only (an estimate)"
        if run.get("mode") == "prefill" and hw.get("j_per_1k_prompt") is not None:
            out.append(f"{_joules(hw['j_per_1k_prompt'])} per thousand prompt tokens read, "
                       f"{whole}, at {_n(hw.get('avg_watts'))} W on average.")
        elif hw.get("j_per_token") is not None:
            out.append(f"{_joules(hw['j_per_token'])} per token written, {whole}, at {_n(hw.get('avg_watts'))} W on "
                       "average.")
        cost = hw.get("cost") or {}
        if cost.get("per_million_tokens") is not None and run.get("mode") != "prefill":
            out.append(f"At {_money(cost['price_per_kwh'], cost['currency'])} per kWh, a million tokens cost "
                       f"{_money(cost['per_million_tokens'], cost['currency'])} in electricity.")
        if hw.get("coverage") is not None and hw["coverage"] < 0.98:
            out.append(f"Power readings cover {hw['coverage'] * 100:.0f}% of the run, so the energy is a lower bound.")
    t = hw.get("throttle") or {}
    if t.get("throttled") is True:
        why = f" ({', '.join(t['reasons'])})" if t.get("reasons") else ""
        out.append(f"The machine slowed itself down for about {_n(t.get('seconds'))} s{why}: the speeds understate it.")
    elif t.get("throttled") is False:
        out.append("No slowdown: the machine kept its pace through the run.")
    if o.get("errors"):
        out.append(f"{o['errors']} {'reply' if o['errors'] == 1 else 'replies'} failed; the notes say why.")
    return out


# -- the text to share --------------------------------------------------------------------------------------------------

def _machine(run: dict) -> str:
    if run.get("elsewhere"):
        return "an engine on another machine (" + (run.get("target") or {}).get("url", "") + ")"
    m = run.get("machine") or {}
    ram = f", {round(m['ram_bytes'] / 1024 ** 3)} GB" if m.get("ram_bytes") else ""
    return f"{m.get('chip') or 'unknown machine'}{ram}"


def share_text(run: dict) -> str:
    """A few lines to paste in a chat or a post: the headline figures, the machine and the protocol."""

    o = (run.get("summary") or {}).get("overall") or {}
    hw = run.get("hardware") or {}
    lines = [f"TensorFold Web Bench, {KIND.get(run.get('mode'), run.get('mode'))}: {run.get('model')}",
             f"{_machine(run)} · {run.get('engine') or 'engine not identified'}"]
    if run.get("mode") == "prefill":
        pts = _sweep_points(run)
        if pts:
            lines.append(f"Prompt reading {_n(pts[0]['prefill_tps'])} tok/s at {_k(pts[0]['size'])}, "
                         f"{_n(pts[-1]['prefill_tps'])} at {_k(pts[-1]['size'])} · first token at "
                         f"{_k(pts[-1]['size'])}: {_secs(pts[-1].get('ttft'))}")
    elif run.get("mode") == "kl" and (run.get("kl") or {}).get("positions"):
        lines.append(kl_line(run))
    elif run.get("mode") == "tools" and run.get("tools"):
        tl = run["tools"]
        lines.append(f"tool-eval-bench {tl.get('version')}: {tl.get('final_score')}/100, {tl.get('points')}/"
                     f"{tl.get('max_points')} points over {tl.get('scenarios')} scenarios")
    elif run.get("mode") == "quality" and (run.get("summary") or {}).get("quality"):
        q = run["summary"]["quality"]
        lines.append(f"{q['overall']['passed']}/{q['overall']['items']} right · " + " · ".join(
            f"{(run.get('suite_names') or {}).get(k, k)} {v['passed']}/{v['items']}" for k, v in q["categories"].items()))
    elif run.get("mode") == "load" and (run.get("summary") or {}).get("load"):
        lv = run["summary"]["load"]
        best = max(lv, key=lambda x: x.get("total_tps") or 0)
        lines.append(f"{_tps(lv[0].get('total_tps'))} tok/s alone, {_tps(best.get('total_tps'))} total at "
                     f"{best['concurrency']} at once ({_tps(best.get('per_request_tps'))} each, first token p95 "
                     f"{_secs(best.get('ttft_p95'))})")
    elif o:
        parts = [f"{_tps(o.get('tps'))} tok/s"]
        if o.get("tps_no_drafts"):
            parts.append(f"{_tps(o['tps_no_drafts'])} without drafts ({_pct(o.get('gain'))})")
        if o.get("checked"):
            parts.append(f"{o['exact']}/{o['checked']} identical")
        parts.append(f"first token {_secs(o.get('ttft'))}")
        lines.append(" · ".join(parts))
    if hw.get("j_per_token") is not None and run.get("mode") != "prefill":
        scope = "whole machine" if hw.get("whole_machine") else f"{hw.get('scope')}, estimate"
        slow = {True: " · slowed down", False: " · no slowdown"}.get((hw.get("throttle") or {}).get("throttled"), "")
        lines.append(f"{_joules(hw['j_per_token'])}/token, {_n(hw.get('avg_watts'))} W avg ({scope}){slow}")
    sv = run.get("suites_version") or {}
    which = ((f"quality v{sv['quality']}" if "quality" in sv else f"tool-eval-bench {sv['tool_eval_bench']}"
              if "tool_eval_bench" in sv else f"suites v{sv.get('version')}") if sv else "")
    th = run.get("thinking") or {}
    # a run before the warm-up check recorded nothing: thinking was asked off, not checked
    thinking = "thinking off" if th.get("off") else "thinking on" if th else "thinking asked off"
    at_once = f"{run['max_concurrency']} at once" if run.get("mode") == "load" else "warm-up"
    proto = f"temperature 0, {at_once}, {thinking}" + (f", {which} {sv.get('digest')}" if sv else "")
    lines.append(f"{proto} · {time.strftime('%d %b %Y', time.localtime(run.get('created') or 0))}")
    return "\n".join(lines)


# -- the HTML report ------------------------------------------------------------------------------------------------------

CSS = """
:root{--bg:#F4F5F7;--card:#FFF;--ink:#1A1C21;--ink2:#5B6270;--ink3:#8B919C;--line:#E2E5EA;--accent:#5157E4;
--soft:#ECEDFC;--good:#1F9D63;--bad:#C8364C;--warn:#D96A3C}
@media (prefers-color-scheme:dark){:root{--bg:#0F1114;--card:#171A1F;--ink:#E9EBEE;--ink2:#A4AAB5;--ink3:#6F7682;
--line:#272B33;--accent:#8A8FFF;--soft:#23264A;--good:#3FCB8E;--bad:#F27085;--warn:#F0916B}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
main{max-width:960px;margin:0 auto;padding:32px 16px 48px}
header{display:flex;gap:14px;align-items:center;margin-bottom:6px}header img{width:40px;height:40px}
.brand{font-weight:700;letter-spacing:-.01em}.brand small{display:block;font-weight:500;color:var(--ink2)}
h1{font-size:26px;letter-spacing:-.02em;margin:18px 0 4px}h2{font-size:17px;margin:0 0 10px}
.meta{color:var(--ink2);margin:0 0 18px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:14px 0}
.figs{display:flex;flex-wrap:wrap;gap:12px 32px;margin:0}.figs div{min-width:110px}
.figs dt{font-size:12.5px;color:var(--ink2)}.figs dd{margin:0;font-size:24px;font-weight:650;
font-variant-numeric:tabular-nums;letter-spacing:-.02em}.good{color:var(--good)}.bad{color:var(--bad)}
ul.find{margin:0;padding-left:20px}ul.find li{margin:4px 0}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:600}
svg{width:100%;height:auto;display:block;margin:6px 0 10px}svg text{font-size:11px;fill:var(--ink3)}
svg text.v{fill:var(--ink)}.legend{font-size:12.5px;color:var(--ink2)}
dl.kv{display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;margin:0;font-size:13.5px}
dl.kv dt{color:var(--ink2)}dl.kv dd{margin:0}
details{margin:6px 0}summary{cursor:pointer;color:var(--ink2)}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--bg);padding:10px 12px;border-radius:8px;font-size:12.5px}
code{font:12.5px ui-monospace,"SF Mono",Menlo,Consolas,monospace}
footer{color:var(--ink3);font-size:12.5px;margin-top:22px}
"""


def _e(x: object) -> str:
    return html.escape(str(x), quote=True)


def _logo() -> str:
    try:
        return "data:image/png;base64," + base64.b64encode(LOGO.read_bytes()).decode()
    except OSError:
        return ""


def _identical(b: dict) -> str:
    return f"{b['exact']}/{b['checked']}" if b.get("checked") else "—"


def _bars(run: dict, names: dict[str, str]) -> str:
    """Speed per suite, with drafts and without, as horizontal bars."""

    rows = [(k, b) for k, b in ((run.get("summary") or {}).get("suites") or {}).items() if b.get("tps")]
    if not rows:
        return ""
    top = max(max(b["tps"], b.get("tps_no_drafts") or 0) for _, b in rows) or 1
    w, lab, rh = 640, 150, 30
    out = [f'<svg viewBox="0 0 {w} {len(rows) * rh + 8}" role="img" aria-label="Speed per suite">']
    for i, (k, b) in enumerate(rows):
        y = i * rh + 4
        span = w - lab - 70
        out.append(f'<text x="0" y="{y + 15}">{_e(names.get(k, k))}</text>')
        out.append(f'<rect x="{lab}" y="{y + 2}" width="{span * b["tps"] / top:.1f}" height="10" rx="3" '
                   'fill="var(--accent)"/>')
        if b.get("tps_no_drafts"):
            out.append(f'<rect x="{lab}" y="{y + 14}" width="{span * b["tps_no_drafts"] / top:.1f}" height="7" rx="3" '
                       'fill="var(--ink3)" opacity=".55"/>')
        out.append(f'<text class="v" x="{lab + span * b["tps"] / top + 6:.1f}" y="{y + 12}">{_tps(b["tps"])}</text>')
    out.append("</svg>")
    return "".join(out) + ('<div class="legend">Tokens per second: with drafts (colour), without (grey).</div>')


def _sweep_chart(pts: list[dict]) -> str:
    if len(pts) < 2:
        return ""
    w, h, left, bottom = 640, 200, 44, 24
    xs = [math.log2(p["size"]) for p in pts]
    top = max(p["prefill_tps"] for p in pts) * 1.1

    def x(v: float) -> float:
        return left + (v - xs[0]) / ((xs[-1] - xs[0]) or 1) * (w - left - 16)

    def y(v: float) -> float:
        return 10 + (h - bottom - 10) * (1 - v / top)

    path = " ".join(f"{'M' if i == 0 else 'L'}{x(xs[i]):.1f},{y(p['prefill_tps']):.1f}" for i, p in enumerate(pts))
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Prefill speed by prompt size">',
           f'<path d="{path}" fill="none" stroke="var(--accent)" stroke-width="2"/>']
    for i, p in enumerate(pts):
        out.append(f'<circle cx="{x(xs[i]):.1f}" cy="{y(p["prefill_tps"]):.1f}" r="3" fill="var(--accent)"/>')
        out.append(f'<text x="{x(xs[i]):.1f}" y="{h - 6}" text-anchor="middle">{_k(p["size"])}</text>')
        out.append(f'<text class="v" x="{x(xs[i]):.1f}" y="{y(p["prefill_tps"]) - 8:.1f}" text-anchor="middle">'
                   f'{_n(p["prefill_tps"])}</text>')
    out.append("</svg>")
    return "".join(out) + '<div class="legend">Prompt tokens read per second, by prompt size.</div>'


def _load_chart(lv: list[dict]) -> str:
    """Total and per-request speed against requests at once, on a doubling axis."""

    pts = [x for x in lv if x.get("total_tps")]
    if len(pts) < 2:
        return ""
    w, h, left, bottom = 640, 200, 44, 24
    top = max(x["total_tps"] for x in pts) * 1.1

    def x_(i: int) -> float:
        return left + i / (len(pts) - 1) * (w - left - 16)

    def y_(v: float) -> float:
        return 10 + (h - bottom - 10) * (1 - (v or 0) / top)

    total = " ".join(f"{'M' if i == 0 else 'L'}{x_(i):.1f},{y_(p['total_tps']):.1f}" for i, p in enumerate(pts))
    each = " ".join(f"{'M' if i == 0 else 'L'}{x_(i):.1f},{y_(p.get('per_request_tps') or 0):.1f}" for i, p in enumerate(pts))
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Speed by requests at once">',
           f'<path d="{total}" fill="none" stroke="var(--accent)" stroke-width="2"/>',
           f'<path d="{each}" fill="none" stroke="var(--ink3)" stroke-width="2" stroke-dasharray="4 4"/>']
    for i, p in enumerate(pts):
        out.append(f'<text x="{x_(i):.1f}" y="{h - 6}" text-anchor="middle">{p["concurrency"]}</text>')
        out.append(f'<text class="v" x="{x_(i):.1f}" y="{y_(p["total_tps"]) - 8:.1f}" text-anchor="middle">'
                   f'{_tps(p["total_tps"])}</text>')
    out.append("</svg>")
    return "".join(out) + '<div class="legend">Total tokens per second (line) and each request\'s (dashes), by requests at once.</div>'


def _power_chart(hw: dict) -> str:
    rows = [r for r in hw.get("series") or [] if r[1] is not None]
    if len(rows) < 2:
        return ""
    w, h = 640, 120
    t0, t1 = rows[0][0], rows[-1][0] or 1
    top = max(r[1] for r in rows) or 1

    def x(t: float) -> float:
        return (t - t0) / ((t1 - t0) or 1) * w

    pts = " ".join(f"L{x(r[0]):.1f},{10 + (h - 20) * (1 - r[1] / top):.1f}" for r in rows)
    area = f"M{x(rows[0][0]):.1f},{h - 10} {pts} L{x(rows[-1][0]):.1f},{h - 10} Z"
    slow = "".join(f'<rect x="{x(r[0]) - 2:.1f}" y="{h - 8}" width="4" height="6" fill="var(--bad)"/>'
                   for r in rows if len(r) > 5 and r[5])
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Power during the run"><path d="{area}" '
            f'fill="var(--soft)" stroke="var(--accent)" stroke-width="1.5"/>{slow}</svg>'
            f'<div class="legend">Power, up to {_n(top)} W, over {_n(t1 - t0)} s{"; red marks: slowed down" if slow else ""}.</div>')


def html_report(run: dict) -> str:
    """One HTML file with no outside resource: the report, then the receipt itself as JSON at the end."""

    names = {**_names(), **(run.get("suite_names") or {})}
    o = (run.get("summary") or {}).get("overall") or {}
    hw = run.get("hardware") or {}
    m = run.get("machine") or {}
    kind = KIND.get(run.get("mode"), run.get("mode"))
    when = time.strftime("%d %B %Y, %H:%M", time.localtime(run.get("created") or 0))
    body: list[str] = []
    figs: list[tuple[str, str, str]] = []
    if run.get("mode") == "prefill":
        pts = _sweep_points(run)
        if pts:
            figs += [(f"Prefill at {_k(pts[0]['size'])}", f"{_n(pts[0]['prefill_tps'])} tok/s", ""),
                     (f"Prefill at {_k(pts[-1]['size'])}", f"{_n(pts[-1]['prefill_tps'])} tok/s", ""),
                     (f"First token at {_k(pts[-1]['size'])}", _secs(pts[-1].get("ttft")), "")]
    elif run.get("mode") == "kl" and (run.get("kl") or {}).get("positions"):
        k = run["kl"]
        figs += [("Mean KL", f"{k['mean_kl']:.4f} nats", ""), ("KL p99", f"{k['p99_kl']:.4f}", ""),
                 ("Same top token", f"{k['same_top'] * 100:.1f}%", ""),
                 ("Same distributions", f"{k.get('same_distribution', 0) * 100:.1f}%", "") if run.get("kl_text") is not None
                 else ("Identical replies", f"{k['identical'] * 100:.0f}%", ""), ("Positions", f"{k['positions']:,}", "")]
    elif run.get("mode") == "tools" and run.get("tools"):
        tl = run["tools"]
        figs += [("Score", f"{tl.get('final_score')} of 100", ""), ("Points", f"{tl.get('points')}/{tl.get('max_points')}", ""),
                 ("Scenarios", str(tl.get("scenarios")), "")]
    elif run.get("mode") == "quality" and (run.get("summary") or {}).get("quality"):
        q = run["summary"]["quality"]
        figs.append(("Right", f"{q['overall']['passed']} of {q['overall']['items']}", ""))
        figs += [(names.get(k, k), f"{v['passed']}/{v['items']}", "") for k, v in q["categories"].items()]
    elif run.get("mode") == "load" and (run.get("summary") or {}).get("load"):
        lv = run["summary"]["load"]
        best = max(lv, key=lambda x: x.get("total_tps") or 0)
        figs += [("Alone", f"{_tps(lv[0].get('total_tps'))} tok/s", ""),
                 (f"Total at {best['concurrency']} at once", f"{_tps(best.get('total_tps'))} tok/s", ""),
                 ("Each, then", f"{_tps(best.get('per_request_tps'))} tok/s", ""),
                 ("First token p95, then", _secs(best.get("ttft_p95")), "")]
    elif o:
        figs.append(("Writing", f"{_tps(o.get('tps'))} tok/s", ""))
        if o.get("tps_no_drafts"):
            figs += [("Without drafts", f"{_tps(o['tps_no_drafts'])} tok/s", ""), ("Gain", _pct(o.get("gain")), "")]
        if o.get("checked"):
            figs.append(("Identical", f"{o['exact']} of {o['checked']}", "good" if o["exact"] == o["checked"] else "bad"))
        figs.append(("First token", _secs(o.get("ttft")), ""))
    if (run.get("memory") or {}).get("engine_peak_bytes"):
        figs.append(("Engine memory, peak", _gb(run["memory"]["engine_peak_bytes"]), ""))
    if figs:
        body.append('<section><dl class="figs">' + "".join(
            f'<div><dt>{_e(a)}</dt><dd class="{c}">{_e(b)}</dd></div>' for a, b, c in figs) + "</dl></section>")
    found = findings(run)
    if found:
        body.append("<section><h2>What this run shows</h2><ul class=\"find\">" +
                    "".join(f"<li>{_e(f)}</li>" for f in found) + "</ul></section>")
    suites = ((run.get("summary") or {}).get("suites") or {})
    if run.get("mode") in ("quality", "tools"):
        missed = [r for r in run.get("results") or [] if r.get("passed") is False]   # an ungraded request is not missed
        if missed:
            rows = "".join(f"<tr><td>{_e(names.get(r['suite'], r['suite']))}</td><td>{_e(r['prompt'])}</td>"
                           f"<td>{_e(r.get('expected') or '—')}</td><td>{_e(r.get('read') or '')}</td></tr>" for r in missed)
            body.append("<section><h2>Items missed</h2><div class=\"tw\"><table><thead><tr><th>Category</th><th>Item</th>"
                        f"<th>expected</th><th>read</th></tr></thead><tbody>{rows}</tbody></table></div></section>")
    if run.get("mode") == "load" and (run.get("summary") or {}).get("load"):
        lv = run["summary"]["load"]
        rows = "".join(
            f"<tr><td>{x['concurrency']}</td><td>{_tps(x.get('total_tps'))}</td><td>{_tps(x.get('per_request_tps'))}</td>"
            f"<td>{_secs(x.get('ttft_p50'))}</td><td>{_secs(x.get('ttft_p95'))}</td><td>{x['requests']}</td>"
            f"<td>{x['errors'] or ''}</td></tr>" for x in lv)
        body.append("<section><h2>Requests at once</h2>" + _load_chart(lv) +
                    '<div class="tw"><table><thead><tr><th>At once</th><th>total tok/s</th><th>each, tok/s</th>'
                    "<th>first token p50</th><th>p95</th><th>requests</th><th>failed</th></tr></thead><tbody>" + rows +
                    "</tbody></table></div></section>")
    if run.get("mode") in ("quick", "profile") and suites:
        rows = "".join(
            f"<tr><td>{_e(names.get(k, k))}</td><td>{_tps(b.get('tps'))}</td><td>{_tps(b.get('tps_no_drafts'))}</td>"
            f"<td>{_pct(b.get('gain'))}</td><td>{_secs(b.get('ttft'))}</td>"
            f"<td>{_secs(b.get('gap_p50'))} / {_secs(b.get('gap_p95'))}</td>"
            f"<td>{_identical(b)}</td></tr>"
            for k, b in suites.items())
        body.append("<section><h2>By use</h2>" + _bars(run, names) +
                    '<div class="tw"><table><thead><tr><th>Suite</th><th>tok/s</th><th>no drafts</th><th>gain</th>'
                    "<th>first token</th><th>gaps p50 / p95</th><th>identical</th></tr></thead><tbody>" + rows +
                    "</tbody></table></div></section>")
    if run.get("mode") == "prefill":
        allpts = (run.get("summary") or {}).get("sweep") or []
        rows = "".join(
            f"<tr><td>{_k(p['size'])}</td><td>{_n(p.get('prompt_tokens'))}</td><td>{_n(p.get('cached'))}</td>"
            f"<td>{_secs(p.get('prefill_s'))}</td><td>{_n(p.get('prefill_tps'))}</td>"
            f"<td>{_e(p['error']) if p.get('error') else _secs(p.get('ttft'))}</td></tr>" for p in allpts)
        body.append("<section><h2>Prompt reading by size</h2>" + _sweep_chart(_sweep_points(run)) +
                    '<div class="tw"><table><thead><tr><th>Size</th><th>prompt tokens</th><th>from cache</th>'
                    "<th>prefill</th><th>prefill tok/s</th><th>first token</th></tr></thead><tbody>" + rows +
                    "</tbody></table></div></section>")
    if hw:
        t = hw.get("throttle") or {}
        slow = {True: f"yes, {_n(t.get('seconds'))} s", False: "no"}.get(t.get("throttled"), "not reported")
        pf = [("Average power", f"{_n(hw.get('avg_watts'))} W"), ("Peak", f"{_n(hw.get('peak_watts'))} W"),
              ("Idle", f"{_n(hw.get('idle_watts'), 1)} W"), ("Hottest GPU", f"{_n(hw.get('peak_gpu_c'))} °C"),
              ("Slowed down", slow)]
        if run.get("mode") != "prefill":
            pf.insert(0, ("Energy per token", _joules(hw.get("j_per_token"))))
        cost = hw.get("cost") or {}
        if cost:
            pf.append(("Energy cost", _money(cost.get("run"), cost.get("currency", ""))))
        scope = ("Measured for the whole machine" if hw.get("whole_machine")
                 else f"An estimate: {_e(hw.get('source'))} measures the {_e(hw.get('scope'))} only")
        body.append("<section><h2>Power and heat</h2><dl class=\"figs\">" + "".join(
            f"<div><dt>{_e(a)}</dt><dd>{_e(b)}</dd></div>" for a, b in pf) + "</dl>" + _power_chart(hw) +
            f'<div class="legend">{scope} ({_e(hw.get("source"))}), over the measured part of the run.</div></section>')
    p = run.get("protocol") or {}
    th = run.get("thinking") or {}
    sv = run.get("suites_version") or {}
    at_once = (f"up to {run['max_concurrency']} requests at once" if run.get("mode") == "load" and run.get("max_concurrency")
               else "one request at a time")
    kv = [("Sampling", (f"temperature {p.get('temperature')}, top_p {p.get('top_p')}, {at_once}, "
                        f"{p.get('warmup_tokens')}-token warm-up")),
          ("Thinking", (("off" if th.get("off") else "still on") + (f", sent as {th['sent']}" if th.get("sent") else ""))
           if th else p.get("thinking")),
          ("Output format", p.get("output_format"))]
    if sv:
        kv.append(("Suites", f"version {sv.get('version')}, digest {sv.get('digest')}" + "".join(
            f"; {names.get(k, k)} (custom), digest {d}" for k, d in (sv.get("custom") or {}).items())))
    kv += [("Machine", f"{_machine(run)}, {m.get('os') or ''} {m.get('os_version') or ''}".strip(", ")),
           ("Engine", run.get("engine") or "not identified"), ("TensorFold Web", run.get("studio")),
           ("Run", f"{run.get('id')}, {STATUS.get(run.get('status'), run.get('status'))}")]
    body.append("<section><h2>Protocol</h2><dl class=\"kv\">" + "".join(
        f"<dt>{_e(a)}</dt><dd>{_e(b)}</dd>" for a, b in kv) + "</dl>" +
        ("<ul>" + "".join(f"<li>{_e(n)}</li>" for n in run.get("notes") or []) + "</ul>" if run.get("notes") else "") +
        "</section>")
    replies = [r for r in run.get("results") or [] if r.get("draft") and run.get("mode") in ("quick", "profile")]
    if replies:
        body.append("<section><h2>Replies</h2>" + "".join(
            f"<details><summary>{_e(names.get(r['suite'], r['suite']))}: {_e(r['prompt'])} · "
            f"{_n((r.get('m') or {}).get('tokens'))} tokens, {_tps((r.get('m') or {}).get('tps'))} tok/s</summary>"
            f"<pre>{_e(r.get('error') or r.get('text') or '')}</pre></details>" for r in replies) + "</section>")
    data = json.dumps(bench.receipt(run), ensure_ascii=False).replace("</", "<\\/")
    logo = _logo()
    img = f'<img src="{logo}" alt="">' if logo else ""
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_e(kind)}: {_e(run.get('model'))}</title><style>{CSS}</style></head><body><main>"
        f"<header>{img}"
        '<div class="brand">TensorFold Web<small>Bench report</small></div></header>'
        f"<h1>{_e(kind)}: {_e(run.get('model'))}</h1>"
        f'<p class="meta">{_e(when)} · {_e(_machine(run))} · {_e(run.get("engine") or "engine not identified")}</p>'
        + "".join(body) +
        "<footer>Made by TensorFold Web from the run's receipt, which is embedded at the end of this file as JSON "
        "(<code>#receipt</code>). Nothing here reaches the network.</footer>"
        f'</main><script type="application/json" id="receipt">{data}</script></body></html>')
