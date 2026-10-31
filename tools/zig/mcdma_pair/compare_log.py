"""tf-mcdma-pair's rank logs as compare.py reads them, and the verdict it prints and writes as a receipt."""
import os
import re
import shlex
import sys


def parse_kv(rest):
    out = {}
    try:
        parts = shlex.split(rest)
    except ValueError:
        parts = rest.split()
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k] = v
    return out


class Log:
    def __init__(self, path, tree=None):
        self.path = path
        self.tree = tree or os.path.dirname(path)
        self.dir = os.path.dirname(path)
        self.header = {}
        self.config = {}
        self.link = {}
        self.settings = None
        self.settings_kv = {}
        self.passes = {}  # (test, round) -> dict
        self.bufs = {}  # (test, round, n) -> (what, len, sha)
        self.fails = []
        self.mismatch = []
        self.arm = None
        self.result = None
        self.hash_all = None
        self.rates = []
        self.rounds_done = 0
        self.exit = None
        self.meta = {}
        self.rank = None
        self.run_id = None
        self.expect_log = []
        self.refuse_log = []
        self.text = []  # every other line, for the EXPECT-LOG and REFUSE-LOG search
        with open(path, errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                word, _, rest = line.partition(" ")
                if word == "EXPECT-LOG":
                    self.expect_log.append(rest)
                    continue
                if word == "REFUSE-LOG":
                    self.refuse_log.append(rest)
                    continue
                self.text.append(line)
                if word == "TF-MCDMA-PAIR":
                    self.header = parse_kv(rest)
                    self.rank = int(self.header.get("rank", "-1"))
                elif word == "CONFIG":
                    self.config = parse_kv(rest)
                elif word == "LINK":
                    self.link = parse_kv(rest)
                elif word == "SETTINGS":
                    self.settings = rest
                    self.settings_kv = parse_kv(rest)
                elif word == "RUN":
                    self.run_id = parse_kv(rest).get("id")
                elif word == "PASS":
                    kv = parse_kv(rest)
                    self.passes[(kv["test"], int(kv["round"]))] = kv
                elif word == "BUF":
                    kv = parse_kv(rest)
                    self.bufs[(kv["test"], int(kv["round"]), int(kv["n"]))] = (kv.get("what"), kv.get("len"), kv.get("sha"))
                elif word == "FAIL":
                    self.fails.append(parse_kv(rest))
                elif word == "MISMATCH":
                    self.mismatch.append(rest)
                elif word == "ARM":
                    self.arm = parse_kv(rest)
                elif word == "RESULT":
                    self.result = parse_kv(rest)  # the last one wins (the run script's outer guard appends)
                elif word == "HASH":
                    self.hash_all = parse_kv(rest)
                elif word == "RATE":
                    self.rates.append(parse_kv(rest))
                elif word == "ROUND":
                    self.rounds_done += 1
        if self.rank is None or self.rank < 0:
            m = re.search(r"rank(\d)\.log$", path)
            self.rank = int(m.group(1)) if m else -1
        for name in ("rank%d.exit" % self.rank,):
            p = os.path.join(self.dir, name)
            if os.path.exists(p):
                try:
                    self.exit = int(open(p).read().strip() or "-1")
                except ValueError:
                    self.exit = None
        meta = os.path.join(self.dir, "meta.txt")
        if os.path.exists(meta):
            for line in open(meta, errors="replace"):
                k, _, v = line.rstrip("\n").partition(" ")
                if k == "fact":
                    self.meta.setdefault("facts", []).append(v)
                else:
                    self.meta[k] = v
        if self.exit is None and self.result is not None and "code" in self.result:
            try:
                self.exit = int(self.result["code"])
            except ValueError:
                pass

    def host(self):
        return self.header.get("host", "?")

    def label(self):
        return self.header.get("label", "?")

    def library(self):
        return self.header.get("library") or self.meta.get("library", "?")

    def fact(self, key):
        """The first meta.txt fact line starting with `key `, without the key; None when there is none."""
        for f in self.meta.get("facts", []):
            if f.startswith(key + " "):
                return f[len(key) + 1:]
        return None

    def name(self):
        return "rank %d (%s, %s)" % (self.rank, self.host(), os.path.relpath(self.path))


def find_logs(paths, label=None):
    """The logs under `paths` (with `label`, each rank's newest), and the candidates --label left out."""
    logs = []
    for p in paths:
        if os.path.isfile(p):
            logs.append((p, p))
            continue
        if not os.path.isdir(p):
            sys.exit("compare.py: no such file or directory: %s" % p)
        for root, _dirs, files in os.walk(p):
            for f in sorted(files):
                if re.fullmatch(r"rank\d\.log", f):
                    logs.append((os.path.join(root, f), p))
    out = [Log(p, tree) for p, tree in logs]
    if label is None:
        return out, []
    newest = {}
    for lg in out:
        if lg.label() != label:
            continue
        key = lg.rank
        # result directories start with a UTC stamp: the newest run of each rank wins
        if key not in newest or os.path.basename(lg.dir) > os.path.basename(newest[key].dir):
            newest[key] = lg
    chosen = set(id(lg) for lg in newest.values())
    dropped = [lg for lg in out if lg.label() == label and id(lg) not in chosen]
    return list(newest.values()), dropped


def as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class Verdict:
    def __init__(self):
        self.lines = []
        self.bad = 0
        self.partial = False  # --partial let a run of fewer tests through: it passes, but it is not the gate
        self.na = None  # why this run cannot show what it is for on this host (G5a at an active MTU it cannot exceed)
        self.relaxed = []  # the checks --same-host or --stub let through: a pass is then an offline proof, not the gate

    def ok(self, what):
        self.lines.append("ok    " + what)

    def fail(self, what):
        self.bad += 1
        self.lines.append("FAIL  " + what)

    def note(self, what):
        self.lines.append("      " + what)

    def check(self, cond, what):
        (self.ok if cond else self.fail)(what)
        return cond

    def relax(self, what):
        """A check the gate would fail that --same-host or --stub let through."""
        self.relaxed.append(what)
        self.lines.append("off   " + what)

    def word(self):
        return "FAIL" if self.bad else "N/A" if self.na else "PARTIAL" if self.partial else \
            "OFFLINE PASS" if self.relaxed else "PASS"


RECEIPT_FACTS = ("kext", "macos ", "os ", "kernel ", "gid ", "active_mtu", "port_state", "port ", "ioreg ", "native_version",
                 "relaxed", "pcie", "neighbour", "clock ", "meeting", "WARN")


def receipt(logs, v):
    out = ["## tf-mcdma-pair receipt", ""]
    for lg in sorted(logs, key=lambda x: x.rank):
        m = lg.meta
        out.append("- rank %d on %s (%s %s): device %s, via %s, gid %s, path MTU %s, kind %s, connect %s s" % (
            lg.rank, lg.host(), lg.header.get("os"), lg.header.get("arch"), lg.config.get("device"), lg.config.get("via"),
            lg.config.get("gid"), lg.config.get("mtu"), lg.link.get("kind"), lg.link.get("connect_s")))
        for k in ("commit", "binary_sha256", "library", "library_sha256", "date_utc"):
            if k in m:
                out.append("  - %s: %s" % (k, m[k]))
        for f in m.get("facts", []):
            if f.startswith(RECEIPT_FACTS):
                out.append("  - %s" % f)
        for r in lg.rates:
            out.append("  - bulk %s MiB each way in %s s (round %s)" % (r.get("mib_each_way"), r.get("s"), r.get("round")))
        if lg.hash_all:
            out.append("  - run hash %s (%s tests)" % (lg.hash_all.get("all"), lg.hash_all.get("tests")))
        if lg.result:
            out.append("  - RESULT %s" % " ".join("%s=%s" % kv for kv in lg.result.items()))
    out.append("")
    out.append("verdict: %s" % ("FAIL (%d check(s))" % v.bad if v.bad else "N/A (%s)" % v.na if v.na else
                                "PARTIAL (not the whole suite)" if v.partial else
                                "OFFLINE PASS (not the gate: %s)" % "; ".join(v.relaxed) if v.relaxed else "GATE PASS"))
    return "\n".join(out) + "\n"
