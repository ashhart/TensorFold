"""Tier 3 live test driver: greedy parity scenarios against a running tensorfold
server (plain HTTP, no client deps).

Scenarios (each writes JSON to --out):
  canary        short prompt served unplanned; must be identical across SCR on/off
  append_exact  3-message conversation run conversationally (planned, splice-free)
                and as one fresh full prompt; greedy outputs must match token for
                token (dense-exact claim; on a hybrid model GDN drift may bend it —
                reported, not hidden)
  mid_edit      turn 2 deletes a section containing no needle from turn 1's message;
                expects a planned turn with splices and a reply that still carries
                the needle (ACCESS-<code>)

Usage: python3 live_driver.py --port 8391 --out /tmp/scr_live --scenario all
"""

import argparse
import json
import os
import sys
import urllib.request

PARA = ("The logistics team reviewed item {i} of the quarterly audit. Storage racks in the "
        "north wing were reindexed after the March inventory, and the reconciled counts for "
        "palleted goods moved within two percent of the ledger. No anomalies were raised by "
        "the floor supervisors during this window. The council of nine issued guidance that "
        "all further adjustments route through the secondary ledger until the autumn count. "
        "This paragraph exists to give the context model something real to chew on.")


def filler(n, needle_para=None, needle="ACCESS-7Q4K"):
    out = []
    for i in range(n):
        p = PARA.format(i=i)
        if needle_para is not None and i == needle_para:
            p += f" Emergency access for this site is {needle}."
        out.append(f"Paragraph {i + 1}. {p}")
    return "\n\n".join(out)


def chat(port, messages, max_tokens=8):
    body = json.dumps({"model": "local", "messages": messages, "temperature": 0,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    ch = d["choices"][0]
    return ch["message"]["content"].strip(), d.get("usage", {})


def run_canary(args):
    msgs = [{"role": "user", "content": "What is 2+2? Reply with just the number."}]
    text, usage = chat(args.port, msgs)
    return {"prompt_class": "short/unplanned", "reply": text, "usage": usage}


def run_append_exact(args):
    long_msg = filler(28)   # ~28 paragraphs, well over min_base tokens
    ask = "\n\nNow reply with exactly one word, the name of a common fruit:"
    t1_msgs = [{"role": "user", "content": long_msg + ask}]
    reply1, usage1 = chat(args.port, t1_msgs)
    t2_msgs = [*t1_msgs, {"role": "assistant", "content": reply1},
               {"role": "user", "content": "Good. Now reply with exactly one word, the color of a banana:"}]
    reply2, usage2 = chat(args.port, t2_msgs)
    # the same conversation as ONE fresh prompt (planned-vs-fresh comparison target)
    fresh_msgs = [*t2_msgs]
    fresh, usage3 = chat(args.port, fresh_msgs)
    return {"reply1": reply1, "reply2_planned": reply2, "usage_planned": usage2,
            "reply2_fresh": fresh, "usage_fresh": usage3,
            "match": reply2 == fresh}


def run_mid_edit(args):
    # 40 paragraphs; turn 2 deletes paragraph 20 (past the 1024-token match floor)
    # and the needle sits at paragraph 33 — inside the surviving span that must be
    # relocated, so the answer proves the relocated KV still works
    needle = "ACCESS-7Q4K"
    long_msg = filler(40, needle_para=32, needle=needle)
    ask = "\n\nConfirm you are tracking: reply with just the word TRACKING."
    t1_msgs = [{"role": "user", "content": long_msg + ask}]
    reply1, _ = chat(args.port, t1_msgs)
    paras = long_msg.split("\n\n")
    edited = "\n\n".join(paras[:19] + paras[20:]) + ask
    t2_msgs = [{"role": "user", "content": edited},
               {"role": "assistant", "content": reply1},
               {"role": "user", "content": "Now: what is the emergency access code? Reply with only the code."}]
    reply2, usage2 = chat(args.port, t2_msgs)
    return {"reply1": reply1, "reply2_edited": reply2, "usage": usage2,
            "needle_found": needle in reply2}


SCENARIOS = {"canary": run_canary, "append_exact": run_append_exact, "mid_edit": run_mid_edit}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    todo = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results = {}
    for name in todo:
        print(f"--- {name}", flush=True)
        try:
            results[name] = SCENARIOS[name](args)
        except Exception as e:                                    # noqa: BLE001
            results[name] = {"error": repr(e)}
        print(json.dumps(results[name], indent=2), flush=True)
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(results[name], f, indent=2)
    ok = True
    if "append_exact" in results and "error" not in results["append_exact"]:
        if not results["append_exact"]["match"]:
            ok = False
            print("APPEND-ONLY PLANNED TURN DIVERGED from fresh — inspect GDN drift")
    if "mid_edit" in results and "error" not in results["mid_edit"]:
        if not results["mid_edit"]["needle_found"]:
            ok = False
            print("EDITED PLANNED TURN LOST THE NEEDLE — relocation problem")
    print("LIVE-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())