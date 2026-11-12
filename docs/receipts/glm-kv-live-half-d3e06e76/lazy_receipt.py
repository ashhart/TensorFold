"""The lazy KV codec's served workload, for the receipt (README.md has the run command and its arguments).

    lazy_receipt.py run --base URL --config NAME --out DIR [--log PATH] [--lazy-ms 20000] [--dry-run] ...
    lazy_receipt.py collect DIR --log R0_LOG [--log R1_LOG]     # after the server stopped (its KVCODEC totals)
    lazy_receipt.py compare DIR_OFF DIR_EAGER DIR_LAZY [--json OUT]
    lazy_receipt.py selftest                                    # against an in-process fake server

`run` is one greedy chat (temperature 0) whose rendered history passes 2048 tokens by turn 2 and grows by a
document piece of `--piece-tokens` (more than one 2048-token prompt chunk) every turn. The prompt cache keeps the
history before the last user message at its last 2048-token mark, so turn 2 keeps the first state and every turn
from 3 resumes the last turn's kept state and keeps a new one. Each reply is the next turn's assistant message.
The keep falls before the last user message's prefill, so a state's lazy delay runs during that tail and the
decode. Turns:

  1-6   main      turns 2-6 sent `--pause` s after the last reply (default lazy-ms + 10 s: a compacted restore)
  7     raw-keep  after the pause: keeps the state turn 8 resumes
  8     raw       sent at once after turn 7: resumes turn 7's state within the lazy delay (a raw restore)
  9     partial   sent so that its restore falls inside the compaction of turn 8's state (a part-compacted
                  restore): at the keep's estimated time + lazy-ms + `--partial-frac` of the compaction's
                  estimated length (from this run's earlier `compacted` lines; `--compact-ms` before any)

The three configurations run the same turns at the same times, so the replies compare byte for byte.
The lazy configuration needs TF_GLM_KV_CODEC_LAZY_MS equal to `--lazy-ms` (default 20000: turn 7's prompt tail and
decode must end within it for turn 8 to resume raw). If `collect` shows partial_restores=0, the restore fell
before or after the compaction: run again with `--partial-frac` moved, or widen the compaction with
TF_GLM_KV_CODEC_THREADS=1 (noted in the receipt, as it changes lazy_encode_ms).

Per turn: time to first token (the first streamed chunk carrying text), total seconds, completion tokens, the
reply's token sha256 (`return_token_ids`), cached_tokens, the server's prefill seconds, and with `--log` the
server log's new `prompt cache:` and `KVCODEC compacted|trimmed` lines and the restore form they imply.
"""
import argparse, glob, json, os, random, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import served

GRID = 2048
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ASKS = ["Summarize the part above in three sentences.", "Name the two most important points so far.",
        "Quote one sentence that supports the first point.", "What changed between the last two parts?",
        "Which part was the most technical, and why?", "List three keywords for the whole text.",
        "Now restate the first answer in one sentence.", "What would you add to the last part?",
        "Give one question the text leaves open."]
WORDS = ("the engine keeps a state for each prompt chunk and resumes it when the same history comes back while "
         "pages are given back after compaction so the cache holds more conversations at once and every restore "
         "must reproduce the bytes exactly or the reply would differ from the first send").split()


def plan(turns_main=6):
    out = [{"turn": i + 1, "kind": "main"} for i in range(turns_main)]
    n = turns_main
    out += [{"turn": n + 1, "kind": "raw-keep"}, {"turn": n + 2, "kind": "raw"}, {"turn": n + 3, "kind": "partial"}]
    return out


def source_text(a):
    if a.text:
        return open(a.text, encoding="utf-8", errors="replace").read()
    if a.source == "docs":
        paths = sorted(glob.glob(os.path.join(ROOT, "docs", "*.md")))
        if paths:
            return "\n\n".join(open(p, encoding="utf-8", errors="replace").read() for p in paths)
    rng = random.Random(1101)  # synthetic prose: fixed, so every configuration sends the same bytes
    return " ".join(rng.choice(WORDS) for _ in range(400_000))


def pieces(text, srv, sizes):
    """Consecutive pieces of `text` of `sizes` tokens each (the server's tokenizer); without one, ~4 chars a token."""
    if srv is None:
        out, at = [], 0
        for n in sizes:
            out.append(text[at:at + 4 * n])
            at += 4 * n
        return out
    ids = srv.ids(text[: 6 * sum(sizes) + 4096])
    if len(ids) < sum(sizes):
        raise SystemExit(f"lazy_receipt: refused: the source text has {len(ids)} tokens, {sum(sizes)} needed")
    out, at = [], 0
    for n in sizes:
        out.append(srv.text(ids[at:at + n]))
        at += n
    return out


def user_message(i, piece):
    return {"role": "user", "content": f"Part {i + 1} of the text:\n\n{piece}\n\n{ASKS[i % len(ASKS)]}"}


class Codec:
    """What the server log says about keeps and compactions so far (rank 0's lines)."""

    def __init__(self, path):
        self.tail = served.LogTail(path)
        self.keeps = 0  # keep ids run 1, 2, ... in keep order (cuda_snapshots.keep)
        self.mark = {}  # keep id -> the prompt tokens before it (its grid mark, estimated)
        self.compacted = {}  # keep id -> its compacted line
        self.before = set()

    def read(self, mark=None):
        """This turn's new lines; `before` holds the keeps compacted before its restore (earlier reads, or earlier in
        this read than its `prompt cache:` line)."""
        lines = self.tail.new()
        got = served.parse_log(lines)
        first = min([c["at"] for c in got["cache"]] or [len(lines)])
        self.before = set(self.compacted) | {c["keep"] for c in got["compacted"] if c["rank"] == 0 and c["at"] < first}
        for c in got["cache"]:
            for _ in range(c["kept"]):
                self.keeps += 1
                self.mark[self.keeps] = mark
        for c in got["compacted"]:
            if c["rank"] == 0:
                self.compacted[c["keep"]] = c
        return {"cache": got["cache"], "compacted": got["compacted"], "trimmed": got["trimmed"], "done": got["done"],
                "refused": got["refused"]}

    def compaction_ms(self, tokens, lazy_ms, fallback):
        """The compaction's length for a state of `tokens`, at the least ms a token of this run's compacted lines: a
        state restored before its compaction restarts its delay, so its after_ms overstates it (the floor does not)."""
        rates = [max(1.0, c["after_ms"] - lazy_ms) / self.mark[k] for k, c in self.compacted.items() if self.mark.get(k)]
        return min(rates) * tokens if rates else fallback


def run(a):
    srv = None if a.dry_run else served.Server(a.base)
    steps = plan(a.main_turns)
    sizes = [a.first_tokens] + [a.piece_tokens] * (len(steps) - 1)
    parts = pieces(source_text(a), srv, sizes)
    os.makedirs(a.out, exist_ok=True)
    pause = a.pause if a.pause is not None else a.lazy_ms / 1000 + 10
    msgs, rows = [], []
    codec = Codec(None if a.dry_run else a.log)
    last = None  # the previous turn's row
    for i, st in enumerate(steps):
        msgs.append(user_message(i, parts[i]))
        body = served.chat_body(list(msgs), a.max_tokens)
        if a.dry_run:
            json.dump(body, open(os.path.join(a.out, f"body-{st['turn']}.json"), "w"), indent=1)
            print(f"turn {st['turn']} {st['kind']}: {len(msgs)} messages, about {sum(len(m['content']) for m in msgs) // 4} tokens")
            msgs.append({"role": "assistant", "content": f"(reply {st['turn']})"})
            continue
        if i > 0:
            wait_for(st, last, pause, a, codec)
        mark = ((len(srv.render(msgs[:-1])) if len(msgs) > 1 else 0) // GRID) * GRID if a.marks else None  # no history: no /tokenize (an empty messages list is refused)
        t_send = time.time()
        r = srv.chat(body, stream=True)
        time.sleep(a.log_settle)
        log = codec.read(mark)
        row = {**st, "sent_at": t_send, "messages": len(msgs), "mark": mark, **{k: r.get(k) for k in (
            "ttft_s", "total_s", "server_ttft_s", "prefill_s", "decode_s", "decode_tps", "prompt_tokens", "cached_tokens",
            "completion_tokens", "token_sha256", "text_sha256", "rounds", "drafted", "accepted")}, "log": log}
        row["restored_keep"] = codec.keeps - sum(c["kept"] for c in log["cache"]) if i > 0 else None
        row["form"] = form(a.config, row, codec, last)
        if last is not None:
            row["since_keep_ms"] = (t_send - keep_time(last)) * 1000 if keep_time(last) else None
        rows.append(row)
        print(f"turn {st['turn']:2d} {st['kind']:8s} ttft {fmt(row['ttft_s'])} total {fmt(row['total_s'])} cached {row['cached_tokens']} "
              f"tokens {row['completion_tokens']} sha {(row['token_sha256'] or '')[:12]} form {row['form']}", flush=True)
        msgs.append({"role": "assistant", "content": r["content"]})
        last = row
    if a.dry_run:
        return
    meta = {"config": a.config, "base": a.base, "lazy_ms": a.lazy_ms, "pause_s": pause, "max_tokens": a.max_tokens,
            "first_tokens": a.first_tokens, "piece_tokens": a.piece_tokens, "partial_frac": a.partial_frac,
            "source": a.text and os.path.basename(a.text) or a.source, "model": srv.name(), "log": bool(a.log)}
    json.dump({"meta": meta, "turns": rows}, open(os.path.join(a.out, "turns.json"), "w"), indent=1)
    print(f"wrote {os.path.join(a.out, 'turns.json')}; stop the server (SIGINT to rank 0), then: collect {a.out} --log ...")


def fmt(x):
    return "-" if x is None else f"{x:.2f}"


def keep_time(row):
    """When `row`'s prompt pass kept its state: its first token less the prompt's tail past the mark, at the pass's rate."""
    if row.get("server_ttft_s") is None or not row.get("prefill_s") or not row.get("prompt_tokens") or row.get("mark") is None:
        return row["sent_at"] + (row.get("ttft_s") or 0) if row.get("ttft_s") else None
    rate = max(1, row["prompt_tokens"] - (row.get("cached_tokens") or 0)) / row["prefill_s"]
    return row["sent_at"] + row["server_ttft_s"] - max(0, row["prompt_tokens"] - row["mark"]) / rate


def wait_for(st, last, pause, a, codec):
    if st["kind"] == "raw":
        return  # at once: within the lazy delay of turn 7's keep
    if st["kind"] != "partial":
        time.sleep(pause)
        return
    kt = keep_time(last) or time.time()
    est = codec.compaction_ms(last.get("mark") or last.get("prompt_tokens") or 0, a.lazy_ms, a.compact_ms)
    target = kt + (a.lazy_ms + a.partial_frac * est) / 1000
    st["partial_plan"] = {"keep_at": kt, "compaction_ms_est": est, "send_at": target}
    time.sleep(max(0.0, target - time.time()))


def form(config, row, codec, last):
    """The restore's form as the log shows it before the turn: compacted, raw-or-partial (lazy), or none."""
    if row["kind"] == "main" and row["turn"] == 1:
        return "none (first turn)"
    if not row.get("cached_tokens"):
        return "none (nothing resumed)"
    if config == "off" or not codec.tail.path:
        return "uncompacted" if config == "off" else "unknown (no --log)"
    if config == "eager" or row.get("restored_keep") in codec.before:
        return "compacted"
    return "raw-or-partial"


def collect(a):
    lines, logs = [], []
    for i, p in enumerate(a.log):
        dst = os.path.join(a.dir, f"server-r{i}.log")
        if os.path.abspath(p) != os.path.abspath(dst):
            shutil.copyfile(p, dst)
        logs.append(served.parse_log(served.read_lines(p)))
    meta = json.load(open(os.path.join(a.dir, "turns.json")))["meta"]
    out = {"config": meta["config"], "ranks": logs}
    json.dump(out, open(os.path.join(a.dir, "log-summary.json"), "w"), indent=1)
    for r, lg in enumerate(logs):
        for t in lg["totals"]:
            print(f"log {r}: totals " + " ".join(f"{k}={v}" for k, v in t.items() if k != "at"))
        if not lg["totals"] and meta["config"] != "off":
            print(f"log {r}: no KVCODEC totals line (stop rank 0 with SIGINT so the ranks tear down and print them)")
    return out


# ---- compare -------------------------------------------------------------------------------------------------------
TIMERS = ("encode_ms", "lazy_encode_ms", "decode_ms", "copy_ms", "lazy_copy_ms", "keep_io_ms", "wait_ms")


def load(d):
    t = json.load(open(os.path.join(d, "turns.json")))
    p = os.path.join(d, "log-summary.json")
    return {"dir": d, "meta": t["meta"], "turns": t["turns"], "log": json.load(open(p)) if os.path.exists(p) else None}


def compare(dirs):
    runs = [load(d) for d in dirs]
    names = [r["meta"]["config"] for r in runs]
    n = min(len(r["turns"]) for r in runs)
    res = {"configs": names, "turns": [], "first_difference": None, "codec": {}, "pages": {}, "forms": {}}
    for i in range(n):
        ts = [r["turns"][i] for r in runs]
        shas = [t["token_sha256"] for t in ts]
        same = len(set(shas)) == 1 and shas[0] is not None
        if not same and res["first_difference"] is None:
            res["first_difference"] = ts[0]["turn"]
        res["turns"].append({"turn": ts[0]["turn"], "kind": ts[0]["kind"], "same": same, "sha": shas,
                             "ttft_s": [t["ttft_s"] for t in ts], "total_s": [t["total_s"] for t in ts],
                             "cached": [t["cached_tokens"] for t in ts], "prompt": ts[0]["prompt_tokens"],
                             "tokens": [t["completion_tokens"] for t in ts], "form": [t["form"] for t in ts]})
    for name, r in zip(names, runs):
        ranks = (r["log"] or {}).get("ranks") or []
        res["codec"][name] = [lg["totals"][-1] if lg["totals"] else None for lg in ranks]
        res["pages"][name] = [{"trimmed": len(lg["trimmed"]), "freed": sum(x["pages"][0] - x["pages"][1] for x in lg["trimmed"]),
                               "free_last": lg["trimmed"][-1]["free"] if lg["trimmed"] else None,
                               "compacted": [(x["keep"], x["pages"], x["after_ms"]) for x in lg["compacted"]],
                               "spare_mib_last": lg["cache"][-1]["spare_mib"] if lg["cache"] else None,
                               "held_mib_last": lg["cache"][-1]["held_mib"] if lg["cache"] else None} for lg in ranks]
        res["forms"][name] = {t["kind"]: t["form"] for t in r["turns"] if t["kind"] in ("raw", "partial")}
    return res


def table(res):
    names = res["configs"]
    out = ["| turn | kind | prompt | cached " + "/".join(names) + " | TTFT s " + " / ".join(names) + " | reply sha256 | same |",
           "|---|---|---|---|---|---|---|"]
    for t in res["turns"]:
        out.append(f"| {t['turn']} | {t['kind']} | {t['prompt']} | {' / '.join(str(c) for c in t['cached'])} | "
                   f"{' / '.join(fmt(x) for x in t['ttft_s'])} | `{(t['sha'][0] or '')[:12]}` | {'yes' if t['same'] else 'NO'} |")
    out.append("")
    out.append("replies identical in every turn" if res["first_difference"] is None else f"replies differ from turn {res['first_difference']}")
    out += ["", "| config | rank | keeps | restores (raw / partial) | ratio | pages | " + " | ".join(TIMERS) + " |",
            "|---|---|---|---|---|---|" + "---|" * len(TIMERS)]
    for name in names:
        for rk, t in enumerate(res["codec"].get(name) or []):
            if t is None:
                out.append(f"| {name} | {rk} | (no totals) | | | |" + " |" * len(TIMERS))
                continue
            out.append(f"| {name} | {rk} | {t.get('keeps')} | {t.get('restores')} ({t.get('raw_restores')} / {t.get('partial_restores')}) | "
                       f"{t.get('ratio')} | {t.get('raw_pages')} -> {t.get('kept_pages')} | " + " | ".join(str(t.get(k, "-")) for k in TIMERS) + " |")
    out += ["", "| config | rank | trimmed | pages freed | free= last | held / spare MiB last |", "|---|---|---|---|---|---|"]
    for name in names:
        for rk, p in enumerate(res["pages"].get(name) or []):
            out.append(f"| {name} | {rk} | {p['trimmed']} | {p['freed']} | {p['free_last']} | {p['held_mib_last']} / {p['spare_mib_last']} |")
    out += ["", "restore forms (log-inferred, extra cases): " + "; ".join(f"{n}: {res['forms'][n]}" for n in names)]
    return "\n".join(out)


def cmd_compare(a):
    res = compare([a.off, a.eager, a.lazy])
    print(table(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="send the conversation to one configuration's server")
    r.add_argument("--base", default="http://127.0.0.1:18080")
    r.add_argument("--config", required=True, help="off, eager or lazy (labels the run; the server's env decides)")
    r.add_argument("--out", required=True)
    r.add_argument("--log", help="rank 0's server log (read as it grows: prompt cache and KVCODEC lines)")
    r.add_argument("--lazy-ms", type=int, default=20000, help="the lazy run's TF_GLM_KV_CODEC_LAZY_MS (same value in every configuration)")
    r.add_argument("--pause", type=float, help="seconds before turns 2-7 (default lazy-ms/1000 + 10)")
    r.add_argument("--max-tokens", type=int, default=48)
    r.add_argument("--first-tokens", type=int, default=3072, help="turn 1's piece of the text")
    r.add_argument("--piece-tokens", type=int, default=2200, help="every later turn's piece (> 2048: a new keep each turn)")
    r.add_argument("--main-turns", type=int, default=6)
    r.add_argument("--partial-frac", type=float, default=0.4)
    r.add_argument("--compact-ms", type=float, default=800, help="the compaction's estimated length before any is logged")
    r.add_argument("--source", choices=("docs", "synthetic"), default="docs", help="the repository's docs/*.md, or fixed synthetic prose")
    r.add_argument("--text", help="a text file instead of --source")
    r.add_argument("--no-marks", dest="marks", action="store_false", help="skip /tokenize of each history (the keep's mark)")
    r.add_argument("--log-settle", type=float, default=0.5, help="seconds after a reply before the log is read")
    r.add_argument("--dry-run", action="store_true", help="write the bodies (approximate pieces), contact no server")
    c = sub.add_parser("collect", help="after the server stopped: copy and parse its logs (KVCODEC totals)")
    c.add_argument("dir")
    c.add_argument("--log", action="append", required=True, help="rank 0's log, then rank 1's")
    m = sub.add_parser("compare", help="the receipt's tables from the three runs")
    m.add_argument("off")
    m.add_argument("eager")
    m.add_argument("lazy")
    m.add_argument("--json")
    sub.add_parser("selftest", help="every subcommand against an in-process fake server")
    a = p.parse_args(argv)
    if a.cmd == "run":
        if a.piece_tokens <= GRID:
            raise SystemExit(f"lazy_receipt: refused: --piece-tokens {a.piece_tokens} must pass {GRID} (a new keep each turn)")
        return run(a)
    if a.cmd == "collect":
        return collect(a)
    if a.cmd == "compare":
        return cmd_compare(a)
    import fake_server
    return fake_server.selftest_lazy(main)


if __name__ == "__main__":
    main()
