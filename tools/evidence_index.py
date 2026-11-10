#!/usr/bin/env python3
"""Index TensorFold's qualification evidence.

For every family x backend x capability this writes one JSON index that says what
was measured, in which file, at which commit (when the source records one), on what
date, at which grade, and whether the evidence is stale or failed.

The sources walked on a checkout are:

- ``docs/receipts/**``            the device and host receipts (free-form README + JSON)
- ``docs/optimization-ledger.md`` the family x optimization status table
- ``tools/conformance_profiles/`` the served-exactness gate profiles (the kit's inputs)
- ``docs/recipes/lfm3-native.md`` the LFM3 evidence ledger
- ``docs/known-issues.md``        the open-issue list
- a capabilities file, if given with ``--capabilities`` (the output of
  ``tensorfold-native capabilities --json``); it is a command, not a file in the
  checkout, so it is marked unavailable when not supplied.

Standard library only. Run from anywhere:

    python3 tools/evidence_index.py --output out/evidence-index.json

Grades. Each piece of evidence carries one of four grades, or ``unknown`` when the
source does not state which kind of measurement it is:

- ``host_test``        a no-device host step (``zig build test-*-host``)
- ``kernel_contract``  the kernel contract runner (gates and planted controls)
- ``served_exactness`` served replies compared by full token hash
- ``served_speed``     served timing (TTFT, tokens/s, prefill/decode)

A grade is never inferred: a source that says only "it ran" leaves the grade
``unknown``. Staleness is computed with ``git log``: a record is stale when the
commit it was measured at is an ancestor of HEAD and a file it covers changed after
that commit. A record whose commit is not in this history, or that covers no files,
is ``unknown`` (null).
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from evidence_capabilities import CapabilitiesError, read_facts

GRADES = ("host_test", "kernel_contract", "served_exactness", "served_speed")

# Canonical family names, from the optimization ledger's family table. The ledger's
# short codes map to the source directory names under zig/src/families/.
FAMILY_ALIASES = {
    "q3.5": "qwen3_5", "fnext": "flashnext", "glm": "glm", "gem4": "gemma4",
    "nemo": "nemotron", "kol1": "kolibri1", "spk25": "spark25", "dory": "dory",
    "mm3": "minimax_m3", "dsv41": "deepseek_v41", "lfm2m": "lfm2_moe",
    "lfm3": "lfm3", "qwen4": "qwen4", "ling": "ling", "q27": "qwen27",
    "lfm2": "lfm2", "llama": "llama", "step5": "step5", "gdsa": "glm_dsa",
    "kimi": "kimi_k3", "qwen3": "qwen3",
}

# Receipt directory -> the family and backend(s) it covers. The backend is the one
# the receipt's own README names (GB10/DGX Spark = cuda, M-series = metal,
# gfx1151/Radeon = hip). A directory not listed here is indexed with family and
# backend left unknown rather than guessed.
RECEIPT_COVERAGE = {
    "prompt-tail-gemma-nemo": [("gemma4", "cuda"), ("nemotron", "cuda")],
    "lfm3": [("lfm3", "metal")],
    "rocm-host-identity": [("qwen27", "hip")],
    "qwen-cuda-first-token": [("qwen27", "cuda")],
    "pr1381-review": [("lfm2_moe", "metal")],
    "cuda-cancelled-prefill": [("qwen27", "cuda")],
    "drafter-counts": [("qwen27", "hip")],
    "flashnext-cuda-sampled-heads": [("flashnext", "cuda")],
    "flashnext-lanes": [("flashnext", "cuda")],
    "flashnext-mtp-chain-2026-10-10": [("flashnext", "metal")],
    "flashnext-readout-rows": [("flashnext", "cuda")],
    "gfx1151-cancel-76aafcf2": [("qwen27", "hip")],
    "gfx1151-qwen-833bbfcb": [("qwen27", "hip")],
    "glm-fp4local-80517fc3": [("glm", "cuda")],
    "glm-kv-pages-0b094b74": [("glm", "cuda")],
    "glm-kv-pages-8e705835": [("glm", "cuda")],
    "glm-sampling-95818b3": [("glm", "metal")],
    "hip-refuses-drafter": [("qwen27", "hip")],
}

# Backend letter -> canonical name, as used in the optimization ledger cells.
BACKEND_LETTERS = {"C": "cuda", "M": "metal", "H": "hip", "X": "xpu"}

COMMIT_RE = re.compile(r"\b[0-9a-f]{40}\b")
ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
MON_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{4})\b"
)
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}

# A commit is treated as the measured (primary) one when it follows one of these
# words, which is how the receipts name the source they built and ran.
# "base" is deliberately absent: it names the reference commit, not the one that was
# built and run.
_SOURCE_WORDS = ("source", "tested", "candidate", "head", "validation", "final",
                 "measured", "built from")


def _run_git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def commit_in_history(repo, commit):
    return bool(commit) and _run_git(repo, "cat-file", "-e", commit).returncode == 0


def is_ancestor(repo, commit):
    return _run_git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode == 0


def is_stale(repo, commit, covers):
    """True when ``commit`` is an ancestor of HEAD and a covered file changed after it.

    Returns None (unknown) when the commit is absent, not in this history, or there
    is nothing to cover.
    """
    if not commit or not covers:
        return None
    if not commit_in_history(repo, commit) or not is_ancestor(repo, commit):
        return None
    r = _run_git(repo, "log", "--format=%H", f"{commit}..HEAD", "--", *covers)
    if r.returncode != 0:
        return None
    return len(r.stdout.strip().splitlines()) > 0


def covers_for(family, backend):
    """The source files a measurement of this family and backend exercises.

    The family's own source directory, plus the CUDA kernel tree for CUDA backends.
    Shared-core changes are not captured here; that is a known limitation.
    """
    if not family:
        return []
    covers = [f"zig/src/families/{family}/"]
    if backend == "cuda":
        covers.append("zig/kernels/cuda/")
    return covers


def extract_commits(text):
    """Return (primary, all) 40-hex commits found in ``text``.

    The primary is the first commit that follows a source-naming word, else the first
    commit, else None.
    """
    all_commits = list(dict.fromkeys(COMMIT_RE.findall(text)))
    if not all_commits:
        return None, []
    for m in re.finditer(r"([^\n:]*?)\b([0-9a-f]{40})\b", text):
        if any(w in m.group(1).lower() for w in _SOURCE_WORDS):
            return m.group(2), all_commits
    return all_commits[0], all_commits


def extract_date(text):
    """The first recorded date in ``text`` as an ISO string, else None."""
    m = ISO_DATE_RE.search(text)
    if m:
        return m.group(1)
    m = MON_DATE_RE.search(text)
    if m:
        day, mon_name, year = int(m.group(1)), m.group(2), m.group(3)
        return f"{year}-{_MONTHS[mon_name[:3]]:02d}-{day:02d}"
    return None


def extract_title(text):
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def grade_from_text(text):
    """The grades the source text states, in GRADES order; empty when none is stated."""
    t = text.lower()
    found = []
    if any(k in t for k in ("kernel contract", "tf-kernel-contract", "contract run",
                            "gates 1-4", "controls red", "planted controls")):
        found.append("kernel_contract")
    if any(k in t for k in ("token hash", "token sha", "tokens equal", "replies equal",
                            "replies were equal", "exactness", "conformance", "http equality",
                            "served exactness", "replies matched", "token for token",
                            "replies match", "replies differ", "served checks",
                            "cancellation checks", "focused-state checks")):
        found.append("served_exactness")
    if any(k in t for k in ("ttft", "tok/s", "tokens/s", "prefill", "decode rate",
                            "throughput", "faster", "quicker", "sooner", "timing",
                            "time to first", "cold prefill")):
        found.append("served_speed")
    if any(k in t for k in ("zig build test", "host test", "host checks", "host controls",
                            "host fixture", "host-side", "host regression")):
        found.append("host_test")
    return [g for g in GRADES if g in found]


def failed_from_text(text):
    """True when the source records a real (non-control) failure, else None.

    Planted controls that are caught are not failures, so this is conservative: it
    only fires on explicit result failures or explicit all-pass statements.
    """
    t = text.lower()
    # Only unambiguous result failures. "gate out" and "a candidate failed the
    # comparison" are left out: a gate can be out on the good side, and an excluded
    # candidate is not the measurement's result.
    if any(k in t for k in ("the gate failed", "regression confirmed",
                            "no performance-gate pass", "a real regression",
                            "the result failed")):
        return True
    if any(k in t for k in ("all passed", "zero failed", "zero failures", "passed all",
                            "all checks passed", "gate in", "passed 303/303",
                            "zero failed or skipped")):
        return False
    return None


def _norm_capability(name):
    name = re.sub(r"^\d+\.\s*", "", name.strip())
    name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return name or "optimization"


def _clean(text):
    return re.sub(r"\s+", " ", text).strip() if text else text


def _item(source, file, family, backend, capability, grade, commit, date, measured,
          failed, covers, extra=None):
    d = {
        "source": source, "file": file, "family": family, "backend": backend,
        "capability": capability, "grade": grade, "commit": commit, "date": date,
        "measured": _clean(measured), "failed": failed, "stale": None, "covers": covers,
    }
    if extra:
        d.update(extra)
    return d


# --------------------------------------------------------------------------- sources


def read_conformance_profiles(root):
    """The served-exactness gate profiles: what the conformance kit would run."""
    records = []
    prof_dir = root / "tools" / "conformance_profiles"
    if not prof_dir.is_dir():
        return records
    for p in sorted(prof_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        m = re.match(r"^\s*(\S+)\s+on\s+(\S+)", data.get("family", ""))
        fam, backend = (m.group(1), m.group(2).lower()) if m else (data.get("family"), None)
        gates = data.get("gates") or ["drafted", "repeat", "concurrent", "resumed", "refusals"]
        records.append(_item(
            "conformance_profiles", p.relative_to(root).as_posix(), fam, backend,
            "served_exactness", "served_exactness", None, None,
            f"conformance profile {data.get('name', p.stem)}: gates {', '.join(gates)}",
            None, covers_for(fam, backend)))
    return records


def read_optimization_ledger(root):
    """The family x optimization status table. Grade is unknown: the ledger states a
    status (done/yes/no/failed/open/?), not a kind of measurement."""
    records = []
    path = root / "docs" / "optimization-ledger.md"
    if not path.is_file():
        return records
    text = path.read_text(encoding="utf-8")

    fam_backend = {}
    for m in re.finditer(r"(\S+)\s+`([a-z0-9_]+)`\s*\(([^)]*)\)", text):
        code, canonical, btxt = m.group(1), m.group(2), m.group(3)
        backends = [BACKEND_LETTERS[c] for c in re.findall(r"[CMHX]", btxt) if c in BACKEND_LETTERS]
        fam_backend[code] = (canonical, backends)

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if (line.startswith("|") and "Optimization" in line and i + 1 < len(lines)
                and set(lines[i + 1].replace("|", "").strip()) <= set("- :")):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            codes = header[1:]
            # Only the family tables: every column is a known family code. The
            # "Rough gain" table (Prefill/Decode/Memory/...) is not a family table.
            if not codes or not all(c in fam_backend for c in codes):
                i += 1
                continue
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if len(cells) >= 2 and cells[0]:
                    opt = _norm_capability(cells[0])
                    for code, cell in zip(codes, cells[1:]):
                        canonical, backends = fam_backend.get(code, (FAMILY_ALIASES.get(code, code), []))
                        status_m = re.match(r"^(done|failed|open|yes|no|\?)(\u00b9)?", cell)
                        status = status_m.group(1) if status_m else None
                        prs = re.findall(r"\[(\d+)\]", cell)
                        cell_backends = [BACKEND_LETTERS[c] for c in
                                         re.findall(r"(?<![A-Za-z])([CMHX])(?=[\[\s]|$)", cell)]
                        for backend in (cell_backends or backends or ["unspecified"]):
                            measured = f"optimization {cells[0]}: {status or cell}"
                            if prs:
                                measured += f" (PR {', '.join(prs)})"
                            records.append(_item(
                                "optimization_ledger", path.relative_to(root).as_posix(),
                                canonical, backend, opt, "unknown", None, None, measured,
                                (status == "failed"), covers_for(canonical, backend)))
                i += 1
        i += 1
    return records


def read_lfm3_ledger(root):
    """The LFM3 evidence ledger table in the family recipe."""
    records = []
    path = root / "docs" / "recipes" / "lfm3-native.md"
    if not path.is_file():
        return records
    text = path.read_text(encoding="utf-8")
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Cell / evidence"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or set(cells[0]) <= set("- "):
            continue
        cell, ran, showed = cells[0], cells[1], cells[2]
        primary, _ = extract_commits(ran)
        grades = grade_from_text(cell + " " + ran + " " + showed)
        capability = _norm_capability(re.sub(r"\[.*?\]\(.*?\)", "", cell).split(",")[0])
        records.append(_item(
            "lfm3_ledger", path.relative_to(root).as_posix(), "lfm3", "metal", capability,
            grades[0] if grades else "unknown", primary, extract_date(ran),
            re.sub(r"\[.*?\]\(.*?\)", "", showed)[:200],
            failed_from_text(showed), covers_for("lfm3", "metal")))
    return records


def read_known_issues(root):
    """The open-issue list: each bullet is an open limitation for a family."""
    records = []
    path = root / "docs" / "known-issues.md"
    if not path.is_file():
        return records
    text = path.read_text(encoding="utf-8")
    frag = [
        ("GLM-5.3", "glm"), ("Kolibri", "kolibri1"), ("Qwen3.5-2B", "qwen3_5"),
        ("Qwen3.8-27B", "qwen27"), ("Qwen3.8-Flash-Next", "flashnext"),
        ("Flash Next", "flashnext"), ("Nemotron", "nemotron"),
        ("DeepSeek-V4.1", "deepseek_v41"), ("DeepSeek", "deepseek_v41"),
        ("Qwen3.8", "qwen27"), ("Qwen", "qwen27"),
    ]
    for block in re.split(r"\n## ", text)[1:]:
        title = block.splitlines()[0].strip()
        for m in re.finditer(r"^- (.*?)(?=\n- |\n\n|\Z)", block, re.M | re.S):
            body = m.group(1).strip()
            family = next((f for name, f in frag if name.lower() in body.lower()), None)
            records.append(_item(
                "known_issues", path.relative_to(root).as_posix(), family, None, "known_issue",
                "unknown", None, None, f"[{title}] {body[:160]}", True,
                covers_for(family, None)))
    return records


def read_receipts(root):
    """The device and host receipts under docs/receipts/."""
    records = []
    base = root / "docs" / "receipts"
    if not base.is_dir():
        return records
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        readme = d / "README.md"
        if not readme.is_file():
            continue
        text = readme.read_text(encoding="utf-8")
        title = extract_title(text) or d.name
        primary, all_commits = extract_commits(text)
        grades = grade_from_text(text)
        failed = failed_from_text(text)
        targets = RECEIPT_COVERAGE.get(d.name, [(None, None)])
        for family, backend in targets:
            covers = covers_for(family, backend)
            # One record per grade the receipt states, so each (family, backend,
            # capability) entry carries its own grade. A receipt that states none
            # becomes a single unknown-grade record.
            for grade in (grades or ["unknown"]):
                records.append(_item(
                    "receipts", readme.relative_to(root).as_posix(), family, backend,
                    grade, grade, primary, extract_date(text), title, failed, covers,
                    extra={"all_commits": all_commits}))
    return records


def read_capabilities(root, path):
    """Index advertised facts at unknown grade; invalid supplied input raises a bounded error."""
    if path is None:
        return []
    facts = read_facts(path)
    try:
        source = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        source = "external capabilities input"
    return [_item("capabilities", source, family, backend, capability, "unknown", None, None,
                  "advertised capability " + capability, None, covers_for(family, backend))
            for family, backend, capability in facts]


# --------------------------------------------------------------------------- index


def build_index(root, capabilities=None):
    root = Path(root).resolve()
    raw = []
    raw += read_conformance_profiles(root)
    raw += read_optimization_ledger(root)
    raw += read_lfm3_ledger(root)
    raw += read_known_issues(root)
    raw += read_receipts(root)
    raw += read_capabilities(root, capabilities)

    for item in raw:
        item["stale"] = is_stale(root, item.get("commit"), item.get("covers"))

    groups = {}
    for item in raw:
        key = (item["family"], item["backend"], item["capability"])
        g = groups.setdefault(key, {
            "family": item["family"], "backend": item["backend"],
            "capability": item["capability"], "grade": "unknown", "evidence": [],
        })
        ev = {
            "source": item["source"], "file": item["file"], "commit": item["commit"],
            "date": item["date"], "measured": item["measured"],
            "failed": item["failed"], "stale": item["stale"],
        }
        if item.get("all_commits"):
            ev["all_commits"] = item["all_commits"]
        g["evidence"].append(ev)
        if item["grade"] != "unknown":
            g["grade"] = item["grade"]

    entries = []
    for g in groups.values():
        stales = [e["stale"] for e in g["evidence"]]
        fails = [e["failed"] for e in g["evidence"] if e["failed"] is not None]
        g["stale"] = (any(s is True for s in stales)
                      if any(s is not None for s in stales) else None)
        if not fails:
            g["failed"] = None
        elif any(f is True for f in fails):
            g["failed"] = True
        else:
            g["failed"] = False
        entries.append(g)
    entries.sort(key=lambda g: (str(g["family"]), str(g["backend"]), g["capability"]))

    src_counts = {}
    for item in raw:
        src_counts[item["source"]] = src_counts.get(item["source"], 0) + 1
    if capabilities is not None:
        src_counts.setdefault("capabilities", 0)

    head = _run_git(root, "log", "-1", "--format=%H").stdout.strip()
    sources = [{"name": n, "records": c} for n, c in sorted(src_counts.items())]
    if capabilities is None:
        sources.append({"name": "capabilities", "records": 0,
                        "note": "not supplied; run tensorfold-native capabilities --json "
                                "and pass --capabilities"})
    return {
        "tool": "evidence_index",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkout_commit": head or None,
        "grades": list(GRADES),
        "sources": sources,
        "entries": entries,
        "summary": summarize(entries),
    }


def summarize(entries):
    by_grade, by_family, by_backend = {}, {}, {}
    stale = failed = unknown_grade = 0
    for g in entries:
        by_grade[g["grade"]] = by_grade.get(g["grade"], 0) + 1
        by_family[str(g["family"])] = by_family.get(str(g["family"]), 0) + 1
        by_backend[str(g["backend"])] = by_backend.get(str(g["backend"]), 0) + 1
        if g["stale"]:
            stale += 1
        if g["failed"]:
            failed += 1
        if g["grade"] == "unknown":
            unknown_grade += 1
    return {
        "entries": len(entries),
        "by_grade": {k: by_grade.get(k, 0) for k in list(GRADES) + ["unknown"]},
        "by_family": dict(sorted(by_family.items())),
        "by_backend": dict(sorted(by_backend.items())),
        "stale": stale, "failed": failed, "unknown_grade": unknown_grade,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=".", help="the checkout root (default: cwd)")
    ap.add_argument("--output", default=None, help="write the JSON index here (default: stdout)")
    ap.add_argument("--capabilities", default=None,
                    help="a capabilities --json file to include (optional)")
    args = ap.parse_args(argv)

    try:
        index = build_index(args.root, args.capabilities)
    except CapabilitiesError as error:
        ap.error(str(error))
    text = json.dumps(index, indent=2, ensure_ascii=False)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out} ({index['summary']['entries']} entries)")
    else:
        print(text)
    s = index["summary"]
    print(f"\nsummary: {s['entries']} entries; by grade {s['by_grade']}; "
          f"stale {s['stale']}, failed {s['failed']}, unknown-grade {s['unknown_grade']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
