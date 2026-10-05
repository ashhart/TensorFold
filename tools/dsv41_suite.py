"""Suites of the serial engine's checks (``tools/dsv41_serial_run.py --suite``): the tests, the construction each runs
under, the tiers, and the PASS / FAIL report. No torch: ``tools/dsv41_suite2.sh`` and the tests read it on the host.

A test is one of dsv41_serial_run.py's modes with fixed arguments. Tests that share a construction (``--cap``,
``--slots``, ``--graph``, ``--dspark`` and the import-time environment, e.g. ``TF_DSV41_KV``) form a group: one process,
one weight load, the engine's request state reset between tests (``fresh_state``). A test inside a suite parses the
same argument list a fresh process gets (``Test.argv``), so the fresh command printed by ``fresh`` reproduces it:

    python3 tools/dsv41_suite.py groups quick       # the groups of a tier (one process each)
    python3 tools/dsv41_suite.py env fp8            # the environment a group's process needs (KEY=VALUE lines)
    python3 tools/dsv41_suite.py fresh quick        # each test as a fresh run (dsv41_run2.sh arguments)
    python3 tools/dsv41_suite.py merge LOG...       # every [result] line of the logs, a summary; exit 1 on a FAIL
    python3 tools/dsv41_suite.py prove SUITE_LOG FRESH_LOG...   # the same fingerprints in a suite and fresh

Tiers (notes/dsv41/DEV.md): ``quick`` is engine smoke at short context, ``full`` adds long context and quality,
``long`` is the long-context group alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Group:
    """A construction: one process and one weight load for every test in it."""

    name: str
    cap: int
    slots: int
    graph: bool
    dspark: int
    env: tuple[tuple[str, str], ...] = ()          # import-time knobs the process must be started with

    def flags(self) -> list[str]:
        out = ["--cap", str(self.cap), "--slots", str(self.slots)]
        if self.graph:
            out.append("--graph")
        if self.dspark:
            out += ["--dspark", str(self.dspark)]
        return out


@dataclass(frozen=True)
class Test:
    name: str
    group: str
    mode: str                                       # dsv41_serial_run.py's mode function
    args: tuple[tuple[str, object], ...]            # the mode's flags (argparse dests) and values
    env: tuple[tuple[str, str], ...] = ()           # mode selectors read from the environment (TF_CHUNK_PREFILL)
    what: str = ""

    def cli(self) -> list[str]:
        out = []
        for dest, value in self.args:
            out.append("--" + dest.replace("_", "-"))
            if value is not True:
                out.append(str(value))
        return out

    def argv(self) -> list[str]:
        """The flags after MODEL ENGRAM --rank R --master M: the group's construction, then the test's mode."""

        return GROUPS[self.group].flags() + self.cli()


GROUPS = {g.name: g for g in (
    # short context, two slots (multi / resume need them), the decode graphs and DSpark (step-test drafts)
    Group("q", cap=16384, slots=2, graph=True, dspark=3),
    # the same under V4's fp8 caches (TF_DSV41_KV is read at import: its own process)
    Group("fp8", cap=16384, slots=2, graph=True, dspark=3, env=(("TF_DSV41_KV", "fp8"),)),
    # long context: one slot whose limit holds a 131072-token needle and its reply
    Group("long", cap=140000, slots=1, graph=True, dspark=0),
)}

# env keys that select a mode (never inherited by a suite test from the caller's environment)
MODE_ENV = ("TF_CHUNK_PREFILL", "TF_TAIL_TEST")

TESTS = [
    Test("multi-test", "q", "multi_test", (("multi_test", 24),),
         what="two streams batched == each alone, and a verify window beside a stream"),
    Test("step-test", "q", "step_test", (("step_test", 1), ("step_len", 8000)),
         what="whole prefill == 16K-chunk prefill == drafted rounds (DSpark), 8000 tokens"),
    Test("resume-test", "q", "resume_test", (("resume_test", "3000,600"),),
         what="a kept 3000-token state resumed (COPY, NVMe bytes, TAKEOVER) == fresh prefill"),
    Test("views-test", "q", "views_test", (("views_test", 3000),),
         what="a kept state's bytes equal on both ranks"),
    Test("chunk-prefill", "q", "chunk_test", (("chunk_test", "6000,1000,3000,4500"),), env=(("TF_CHUNK_PREFILL", "1"),),
         what="one 6000-token prefill == prefills split at 1000 / 3000 / 4500 (last-row logits bit-equal)"),
    Test("fp8-multi-test", "fp8", "multi_test", (("multi_test", 24),),
         what="fp8 caches: batched == alone (identity spot check)"),
    Test("fp8-resume-test", "fp8", "resume_test", (("resume_test", "3000,600"),),
         what="fp8 caches: kept-state resume == fresh (identity spot check)"),
    Test("decode-bench", "long", "decode_bench", (("decode_bench", "32768,131072"),),
         what="decode speed after a 32K / 128K prefill (report; slower than a baseline: WARN)"),
    Test("needle", "long", "quality", (("needle", "32768,131072"),),
         what="a number at 10/50/90% of 32K / 128K tokens of code, recalled"),
    Test("tf-compare", "long", "quality", (("tf_compare", "32768,98304"),),
         what="teacher-forced NLL / top-1 of 32K and 96K documents (gate: vs a baseline)"),
    Test("prefill-bench", "long", "prefill_bench", (("prefill_bench", "8192,65536"),),
         what="whole-prompt prefill speed (report)"),
]
BY_NAME = {t.name: t for t in TESTS}
TIERS = {
    "quick": [t.name for t in TESTS if t.group in ("q", "fp8")],
    "long": [t.name for t in TESTS if t.group == "long"],
}
TIERS["full"] = TIERS["quick"] + TIERS["long"]

# tf-compare against a baseline: per document (same tokens only), NLL may rise this much, top-1 fall this many points
NLL_TOL = 0.01
TOP1_TOL = 1.0
# decode-bench against a baseline: ms a step this much slower is a WARN (timing is noisy; never a FAIL)
SLOW_TOL = 0.10


def resolve(spec: str) -> list[Test]:
    """``quick`` / ``full`` / ``long`` or test names (comma separated), in suite order, each once."""

    names: list[str] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if part in TIERS:
            names += TIERS[part]
        elif part in BY_NAME:
            names.append(part)
        else:
            raise ValueError(f"unknown suite test or tier {part!r} (tiers {', '.join(TIERS)}; tests "
                             f"{', '.join(BY_NAME)})")
    if not names:
        raise ValueError("an empty suite")
    return [t for t in TESTS if t.name in names]        # suite order (groups together, as TESTS lists them), once


def groups_of(tests: list[Test]) -> list[str]:
    out: list[str] = []
    for t in tests:
        if t.group not in out:
            out.append(t.group)
    return out


def check_env(group: str, env: dict) -> str | None:
    """Why this process cannot run ``group`` (its import-time knobs differ), or None."""

    for k, v in GROUPS[group].env:
        if env.get(k, "") != v:
            return f"group {group} needs {k}={v} at start (got {env.get(k, '')!r}): tools/dsv41_suite2.sh sets it"
    return None


def mode_env(test: Test, env: dict) -> dict:
    """The environment a test's mode reads: the caller's without mode selectors, plus the test's."""

    out = {k: v for k, v in env.items() if k not in MODE_ENV}
    out.update(dict(test.env))
    return out


# -- results -------------------------------------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    status: str                                     # PASS FAIL INFO WARN ERROR SKIP
    facts: object = None                            # deterministic outputs (tokens, diffs, hashes): the fingerprint
    summary: str = ""
    metrics: dict = field(default_factory=dict)     # timings, quality numbers (not fingerprinted)
    seconds: float = 0.0

    @property
    def fp(self) -> str:
        return fingerprint(self.facts)


def _canon(x):
    if isinstance(x, float):
        return "nan" if math.isnan(x) else repr(x)
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    return x


def fingerprint(facts) -> str:
    return hashlib.sha256(json.dumps(_canon(facts), sort_keys=True, default=repr).encode()).hexdigest()[:12]


LINE = re.compile(r"\[result\] (?P<name>\S+) (?P<status>[A-Z]+) fp=(?P<fp>[0-9a-f]{12}|-) (?P<secs>\d+)s ?(?P<summary>.*)")


def line(r: Result) -> str:
    fp = r.fp if r.facts is not None else "-"
    return f"[result] {r.name} {r.status} fp={fp} {r.seconds:.0f}s {r.summary}".rstrip()


def parse_lines(text: str) -> list[dict]:
    """Rank 0's [result] lines, one a test (the last: a [baseline] line rejudges an earlier one); rank 1's lines
    ("[rank 1] [result] ...") are left out."""

    out: dict[str, dict] = {}
    for ln in text.splitlines():
        m = LINE.match(ln.removeprefix("[baseline] "))
        if m:
            out[m["name"]] = m.groupdict()
    return list(out.values())


def summary(results: list[Result] | list[dict], label: str = "suite") -> tuple[str, bool]:
    """``[suite] ...`` with the counts, and whether it passed (no FAIL / ERROR)."""

    get = (lambda r: r["status"]) if results and isinstance(results[0], dict) else (lambda r: r.status)
    counts: dict[str, int] = {}
    for r in results:
        counts[get(r)] = counts.get(get(r), 0) + 1
    ok = not (counts.get("FAIL") or counts.get("ERROR")) and bool(results)
    names = (lambda r: r["name"]) if results and isinstance(results[0], dict) else (lambda r: r.name)
    bad = [names(r) for r in results if get(r) in ("FAIL", "ERROR")]
    text = (f"[{label}] {'PASS' if ok else 'FAIL'}: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
            + (f"; failed: {', '.join(bad)}" if bad else ""))
    return text, ok


def compare(results: list[Result], baseline: dict) -> None:
    """Judge tf-compare (FAIL past NLL_TOL / TOP1_TOL on a document with the same tokens) and decode-bench (WARN past
    SLOW_TOL) against a saved suite (``--suite-out`` of an earlier run), in place."""

    base = {r["name"]: r for r in baseline.get("results", [])}
    for r in results:
        b = base.get(r.name)
        if b is None or r.status in ("ERROR", "SKIP"):
            continue
        if r.name == "tf-compare":
            worse, same = [], 0
            for doc, m in r.metrics.get("docs", {}).items():
                bm = b.get("metrics", {}).get("docs", {}).get(doc)
                if bm is None or bm.get("sha") != m.get("sha"):
                    continue                        # another tree's documents: not comparable
                same += 1
                if m["nll"] > bm["nll"] + NLL_TOL or m["top1"] < bm["top1"] - TOP1_TOL:
                    worse.append(f"{doc} NLL {bm['nll']:.4f}->{m['nll']:.4f} top-1 {bm['top1']:.1f}->{m['top1']:.1f}")
            if worse:
                r.status, r.summary = "FAIL", f"{r.summary}; worse than the baseline: {'; '.join(worse)}"
            elif same and r.status == "INFO":
                r.status, r.summary = "PASS", f"{r.summary}; within the baseline on {same} documents"
        elif r.name == "decode-bench":
            slow = []
            for k, ms in r.metrics.get("ms", {}).items():
                bms = b.get("metrics", {}).get("ms", {}).get(k)
                if bms and ms > bms * (1 + SLOW_TOL):
                    slow.append(f"{k} {bms:.2f}->{ms:.2f} ms")
            if slow and r.status in ("INFO", "PASS"):
                r.status, r.summary = "WARN", f"{r.summary}; slower than the baseline: {'; '.join(slow)}"


def to_json(results: list[Result], **extra) -> dict:
    return {**extra, "results": [{"name": r.name, "status": r.status, "fp": r.fp if r.facts is not None else None,
                                  "summary": r.summary, "metrics": r.metrics, "seconds": round(r.seconds, 1)}
                                 for r in results]}


def prove(suite_text: str, fresh_texts: list[str]) -> tuple[list[str], bool]:
    """Every fresh run's [result] fingerprint against the suite's line of the same test."""

    in_suite = {d["name"]: d for d in parse_lines(suite_text)}
    out, ok = [], True
    for text in fresh_texts:
        for d in parse_lines(text):
            s = in_suite.get(d["name"])
            if s is None:
                out.append(f"{d['name']}: not in the suite log")
                ok = False
            elif s["fp"] != d["fp"]:
                out.append(f"{d['name']}: DIFFERENT suite fp={s['fp']} fresh fp={d['fp']}")
                ok = False
            else:
                out.append(f"{d['name']}: same fp={d['fp']} ({d['status']})")
    return out, ok


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "groups":
        print(" ".join(groups_of(resolve(rest[0]))))
    elif cmd == "env":
        for k, v in GROUPS[rest[0]].env:
            print(f"{k}={v}")
    elif cmd == "flags":
        print(" ".join(GROUPS[rest[0]].flags()))
    elif cmd == "fresh":
        for t in resolve(rest[0]):
            env = " ".join(f"{k}={v}" for k, v in (*GROUPS[t.group].env, *t.env))
            print(f"{t.name}\t{(env + ' ') if env else ''}tools/dsv41_run2.sh {' '.join(t.argv())}")
    elif cmd == "argv":
        print(" ".join(BY_NAME[rest[0]].argv()))
    elif cmd == "testenv":
        for k, v in (*GROUPS[BY_NAME[rest[0]].group].env, *BY_NAME[rest[0]].env):
            print(f"{k}={v}")
    elif cmd == "merge":
        rows = []
        for path in rest:
            with open(path, errors="replace") as f:
                rows += parse_lines(f.read())
        for d in rows:
            print(f"[result] {d['name']} {d['status']} fp={d['fp']} {d['secs']}s {d['summary']}".rstrip())
        text, ok = summary(rows, "suite")
        print(text)
        return 0 if ok else 1
    elif cmd == "prove":
        with open(rest[0], errors="replace") as f:
            suite_text = f.read()
        fresh = []
        for path in rest[1:]:
            with open(path, errors="replace") as f:
                fresh.append(f.read())
        lines, ok = prove(suite_text, fresh)
        print("\n".join(lines))
        print(f"[prove] {'PASS: every fresh run printed the suite fingerprint' if ok else 'FAIL'}")
        return 0 if ok else 1
    else:
        print(f"unknown command {cmd!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
