"""Every environment variable the engine's Zig source reads, by lookup call and scope, checked against the doc."""
import re
import sys
from pathlib import Path

import env_lookups as lk
import env_reads as er
import env_scope as es
from lean_check import ROOT, test_only

DOC = "docs/environment-names.md"
# Test data that only environment_test.zig imports.
TEST_DATA = ("zig/src/platform/environment_valid_cases.zig",)
# A whole literal shaped like a variable name, for the cross-check that every such spelling is a read or explained.
SHAPED = re.compile(r'"([A-Z][A-Z0-9]*_[A-Z0-9_]*[A-Z0-9])"')
# A build option a read can sit behind: the read is in a function only that option's builds call.
GATES = {("zig/src/server/main.zig", "forgetFail"): "slide_test_hooks"}
ROW = re.compile(r"^\| `([A-Za-z_][A-Za-z0-9_]*)` \| ([a-z, ]+) \| ([a-z, -]+) \|$")
# A wrapper's callers: "file" (calls in its file), "public" (also `<stem or alias>.fn(`), or (file, regex, arg)s.
CALLERS = ("file", "public")
TAGS = ("writes-content", "writes", "logs-content", "logs", "loads", "learns", "other")


def sources(root: Path) -> list[er.Source]:
    """The engine's non-test Zig sources under zig/src."""
    paths = sorted((root / "zig/src").rglob("*.zig"))
    rels = [p.relative_to(root).as_posix() for p in paths]
    return [er.Source(r, p.read_text(encoding="utf-8", errors="replace"))
            for r, p in zip(rels, paths) if not test_only(r) and r not in TEST_DATA]


def wrapper_patterns(src: er.Source) -> list[tuple[re.Pattern, int, tuple[str, str]]]:
    """The wrapper and value-function calls that count as lookups in one file, with each one's wrapper."""
    out = []
    for file, fn, index, callers in lk.WRAPPERS:
        key = (file, fn)
        if callers in CALLERS and src.rel == file:
            spaces = "|".join(re.findall(r"\bconst\s+(\w+)\s*=\s*struct\s*\{", src.code))
            rx = rf"(?<![\w.])(?:(?:{spaces})\.)?{fn}\(" if spaces else rf"(?<![\w.]){fn}\("
            out.append((re.compile(rx), index, key))
        if callers == "public":
            stem = Path(file).stem
            receivers = [stem, rf'@import\("(?:[^"]*/)?{stem}(?:\.zig)?"\)'] + sorted(src.aliases.get(stem, set()))
            out.append((re.compile(rf"(?<![\w])(?:[\w.]*\.)?(?:{'|'.join(receivers)})\.{fn}\("), index, key))
            out += [(rx, index, key) for rx in alias_calls(src, rf"(?:{'|'.join(receivers)})\.{fn}")]
        if isinstance(callers, tuple):
            out += [(re.compile(rx), i, key) for where, rx, i in callers if where in ("*", src.rel)]
    receivers = {"environment"} | src.aliases.get("environment", set())
    for fn, index in lk.VALUES:
        rx = rf"(?<![\w])(?:[\w.]*\.)?(?:{'|'.join(sorted(receivers))})\.{fn}\("
        out.append((re.compile(rx), index, ("zig/src/platform/environment.zig", fn)))
    return out


def alias_calls(src: er.Source, wrapper: str) -> list[re.Pattern]:
    """Calls through constants aliasing the wrapper, hop by hop: bare, or a member via its struct or a comptime type."""
    owners = re.findall(r"\bconst\s+(\w+)\s*=\s*struct\s*\{", src.code)
    owners += re.findall(r"\bcomptime\s+(\w+)\s*:\s*type\b", src.code)
    owners = "|".join(sorted(set(owners))) or "(?!)"
    out, done, todo = [], set(), [wrapper]
    while todo:
        target = todo.pop()
        for m in re.finditer(rf"^([ \t]*)(?:pub\s+)?const\s+(\w+)\s*=\s*{target}\s*;", src.code, re.M):
            name = m.group(2)
            if name in done:
                continue
            done.add(name)
            member = bool(m.group(1))
            out.append(re.compile(rf"(?<![\w.])(?:{owners})\.{name}\(" if member else rf"(?<![\w.]){name}\("))
            todo.append(rf"(?:(?:{owners})\.)?{name}" if member else name)
    return out


def reads(srcs: list[er.Source]) -> tuple[list[er.Read], list[str]]:
    """Every lookup with its names, the computed ones off the allowlist, and warnings for entries nothing matches."""
    fields, consts = er.map_fields(srcs), er.file_consts(srcs)
    defs = {(f, fn) for f, fn, _, _ in lk.WRAPPERS}
    by_rel = {s.rel: s for s in srcs}
    params = {(f, fn): forwarded(by_rel[f], fn, i) for f, fn, i, _ in lk.WRAPPERS if f in by_rel}
    found, problems, used = [], [], set()
    for src in srcs:
        got = er.base(src, fields, consts)
        for pattern, index, _ in wrapper_patterns(src):
            got += er.calls(src, pattern, index, consts)
        seen = set()
        for r in got:
            if (r.line, r.call) in seen:
                continue
            seen.add((r.line, r.call))
            if r.names:
                found.append(r)
            elif (r.file, r.function) in defs and forwards(r.arg, params[(r.file, r.function)]):
                used.add((r.file, r.function))
            elif (r.file, r.call) in lk.NAMED:
                used.add((r.file, r.call))
                given = lk.NAMED[(r.file, r.call)]
                if (names := named(src, given, srcs)) is None:
                    problems.append(f"{r.file}:{r.line}: {r.call} reads {given}, not an array of strings there")
                    continue
                sweep = given in ("profile", "registry")
                found.append(er.Read(r.file, r.line, r.call, names, r.function, r.test, sweep))
            else:
                problems.append(f"{r.file}:{r.line}: computed lookup {r.call} is not in tools/zig/env_lookups.py")
    problems += [f"warning: tools/zig/env_lookups.py: wrapper {f} {fn} reads nothing computed" for f, fn in sorted(defs - used)
                 if f in {s.rel for s in srcs}]
    problems += [f"warning: tools/zig/env_lookups.py: {f} {c} matches no computed lookup" for f, c in lk.NAMED
                 if (f, c) not in used]
    return found, problems


def forwarded(src: er.Source, fn: str, index: int) -> set[str]:
    """A wrapper's name parameter at `index`, and a loop's capture over it: the only computed names it may read."""
    m = re.search(rf"\bfn\s+{fn}\s*\(", src.code)
    params = er.arguments(src.code, src.text, m.end() - 1)[0] if m else []
    words = re.findall(r"\w+", params[index].split(":", 1)[0]) if index < len(params) else []
    if not words:
        return set()
    return {words[-1]} | set(re.findall(rf"\bfor\s*\(\s*{words[-1]}\s*\)\s*\|\s*(\w+)\s*\|", src.code))


def forwards(arg: str, params: set[str]) -> bool:
    """Whether a wrapper's computed read is its forwarding read: the name, its `.ptr`, or `buf[0..name.len :0]`."""
    return any(re.fullmatch(rf"{p}(?:\.ptr)?|\w+\[\s*0\s*\.\.\s*{p}\.len\s*:\s*0\s*\]", arg.strip())
               for p in params)


def named(src: er.Source, given: tuple[str, ...] | str, srcs: list[er.Source]) -> tuple[str, ...] | None:
    """A NAMED entry's names: given, its file's array constant's, the profile registry's or the validation table's."""
    if not isinstance(given, str):
        return given
    if given.startswith("list:"):
        return src.array(given[5:])
    path = {"profile": "zig/src/core/profile.zig", "registry": "zig/src/platform/environment.zig"}[given]
    text = next(s.text for s in srcs if s.rel == path)
    if given == "registry":
        text += next(s.text for s in srcs if s.rel == "zig/src/platform/environment_safe_rules.zig")
        return tuple(re.findall(r'\.name = "([A-Za-z0-9_]+)"', text))
    switches = re.search(r"pub const Switch = enum \{\n(.*?)\n\};", text, re.S).group(1)
    return tuple(re.findall(r"^\s*([A-Z][A-Z0-9_]*),", switches, re.M)) + tuple(re.findall(r'\.name = "([A-Z0-9_]+)"', text))


def gate_problems(srcs: list[er.Source]) -> list[str]:
    """A gated function called anywhere its option does not guard is no longer test-only."""
    out = []
    for (file, fn), option in GATES.items():
        for src in srcs:
            for i, line in enumerate(src.code.split("\n"), 1):
                if re.search(rf"(?<![\w.]){fn}\(", line) and not re.search(rf"\bfn\s+{fn}\b", line) and option not in line:
                    out.append(f"{src.rel}:{i}: {fn} is called outside {option}, so its reads are not test-only")
    return out


def scopes(found: list[er.Read], reached: dict[str, set[str]], srcs: list[er.Source]) -> dict[str, set[str]]:
    """Each name's scopes from its lookups' files, gates and the test runner; a table sweep counts for no other name."""
    mains = {s.rel for s in srcs if re.search(r"^pub fn main\(", s.code, re.M)}
    out: dict[str, set[str]] = {}
    direct = {n for r in found if not r.sweep for n in r.names}
    for r in found:
        if r.test or (r.file, r.function) in GATES:
            where = {"test"}
        else:
            where = reached.get(r.file) or {"tool" if r.file in mains else "test"}
        for n in r.names:
            if not r.sweep or n not in direct:
                out.setdefault(n, set()).update(where)
    return out


def spelled(srcs: list[er.Source]) -> dict[str, str]:
    """Each whole literal shaped like a variable name, outside comments and test blocks, with where it first is."""
    out = {}
    for src in srcs:
        for i, (line, code) in enumerate(zip(src.text.split("\n"), src.code.split("\n")), 1):
            if i not in src.tests and not line.lstrip().startswith("//"):
                kept = line if er.LINE_STRING.match(line) or "//" not in line else line[:len(code.rstrip())]
                for n in SHAPED.findall(kept):
                    out.setdefault(n, f"{src.rel}:{i}")
    return out


def listed(text: str) -> dict[str, tuple[str, str]]:
    """The doc's rows: name to (scopes, tags)."""
    return {m.group(1): (m.group(2), m.group(3)) for line in text.splitlines() if (m := ROW.match(line))}


def joined(where: set[str]) -> str:
    return ", ".join(s for s in es.SCOPES if s in where)


def problems(names: dict[str, set[str]], rows: dict[str, tuple[str, str]], shaped: dict[str, str]) -> list[str]:
    """Reads the doc lacks, rows nothing reads, scopes that disagree, tags missing or unknown, and unexplained names."""
    found = [f"{DOC}: {n} is read and has no row (run with --write, then tag it)" for n in sorted(names) if n not in rows]
    found += [f"{DOC}: {n} is listed but nothing reads it" for n in sorted(rows) if n not in names]
    for n in sorted(set(names) & set(rows)):
        have = set(rows[n][0].split(", "))
        if ("release" in have) != ("release" in names[n]) or not names[n] <= have or not have <= set(es.SCOPES):
            found.append(f"{DOC}: {n} scope {rows[n][0]!r}, reads say {joined(names[n])!r}")
        tags = rows[n][1].split(", ")
        found += [f"{DOC}: {n} has tag {t!r}, not one of {', '.join(TAGS)}" for t in tags if t not in TAGS]
    found += [f"{at}: {n} looks like a variable, but no lookup reads it and env_lookups.py's NOT_READ does not explain it"
              for n, at in sorted(shaped.items()) if n not in names and n not in lk.NOT_READ]
    found += [f"tools/zig/env_lookups.py: NOT_READ {n} is read, or spelled nowhere" for n in sorted(lk.NOT_READ)
              if n in names or n not in shaped]
    return found


def written(text: str, names: dict[str, set[str]]) -> str:
    """The doc rebuilt: rows keep their tags and tool and test scopes, new reads come untagged, gone reads leave."""
    rows = listed(text)
    head = text.split("\n## Variables\n", 1)[0].rstrip("\n")
    out = [head, "", "## Variables", "", "| Name | Scope | Tags |", "| --- | --- | --- |"]
    for n, where in sorted(names.items()):
        other_hosts = set(rows[n][0].split(", ")) - {"release"} if n in rows else set()
        out.append(f"| `{n}` | {joined(where | other_hosts)} | {rows.get(n, ('', 'untagged'))[1]} |")
    return "\n".join(out) + "\n"


def collect(root: Path, graph: str) -> tuple[dict[str, set[str]], dict[str, str], list[str]]:
    """Each read name's scopes, the shaped spellings, and the lookup and gate problems."""
    srcs = sources(root)
    found, issues = reads(srcs)
    return scopes(found, es.reach(root, graph), srcs), spelled(srcs), issues + gate_problems(srcs)


def main(argv: list[str], root: Path = ROOT, graph: str | None = None) -> int:
    doc = root / DOC
    names, shaped, issues = collect(root, es.graph_text(root) if graph is None else graph)
    # a scan with errors has missed reads, so writing it would drop rows that are still read
    if argv == ["--write"] and not any(not f.startswith("warning: ") for f in issues):
        doc.write_text(written(doc.read_text(encoding="utf-8"), names), encoding="utf-8")
    found = issues + problems(names, listed(doc.read_text(encoding="utf-8")), shaped)
    for f in found:
        print(f)
    # an entry nothing matches only warns, so a branch and main a change apart both pass until the branch is rebased
    failed = [f for f in found if not f.startswith("warning: ")]
    print(f"{len(names)} variables read, {len(failed)} problems")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
