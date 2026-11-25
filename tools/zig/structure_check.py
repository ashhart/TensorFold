#!/usr/bin/env python3
"""Import cycles between Zig modules, from file text alone: no build, no index, no checkout of the base."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import posixpath
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = "zig/src/"
# Whitespace, newlines and a trailing comma inside the parens are legal Zig (zig fmt writes the multiline form).
IMPORT = re.compile(r'@import\(\s*"([^"]+)"\s*,?\s*\)')
# Names the compiler provides; they can never close a cycle in zig/src.
COMPILER = frozenset({"std", "builtin"})
# Core sits under every module; a module it newly imports is an ERROR even inside an inherited cycle.
CORE = "core"
ADD_IMPORT = re.compile(r'addImport\(\s*"(\w+)"\s*,\s*([^;]*?)\)\s*;', re.S)
MODULE_VAR = re.compile(r'(?:const|var)\s+(\w+)\s*=\s*([^;]*?createModule[^;]*?);', re.S)
ROOT_PATH = re.compile(r'"(zig/src/[^"]+\.zig)"')


def module_of(path):
    """zig/src/<dir>, zig/src/families/<name>, or a file directly under zig/src as its own module."""
    parts = path[len(SRC):].split("/")
    if parts[0] == "families" and len(parts) >= 3:
        return "families/" + parts[1]
    return parts[0]


def code(text):
    """The text with // comments and \\\\ string lines blanked, so a quoted or commented import is not read."""
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("\\\\"):
            out.append("")
            continue
        quote, i = None, 0
        while i < len(line):
            ch = line[i]
            if quote:
                if ch == "\\":
                    i += 1
                elif ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif line.startswith("//", i):
                line = line[:i]
                break
            i += 1
        out.append(line)
    return "\n".join(out)


def build_files(files):
    return {n: t for n, t in files.items() if n == "build.zig" or (n.startswith("zig/build/") and n.endswith(".zig"))}


def named_modules(files):
    """Each named import's module, only from literal evidence; a name with roots in two modules maps to None."""
    found = defaultdict(set)
    for text in build_files(files).values():
        text = code(text)
        bound = {m[1]: ROOT_PATH.findall(m[2]) for m in MODULE_VAR.finditer(text)}
        for m in ADD_IMPORT.finditer(text):
            roots = ROOT_PATH.findall(m[2]) or bound.get(m[2].strip(), [])
            found[m[1]].update(module_of(r) for r in roots)
    dirs = {n[len(SRC):].split("/")[0] for n in files if n.startswith(SRC) and n.count("/") >= 3}
    for name in dirs - {"families"}:
        found[name].add(name)
    return {name: next(iter(mods)) if len(mods) == 1 else None for name, mods in found.items() if mods}


def graph(files):
    """Module edges with the file imports behind each, and the imports that name no file or module, by name."""
    sources = {n for n in files if n.startswith(SRC) and n.endswith(".zig")}
    names = named_modules(files)
    edges, unresolved = defaultdict(set), defaultdict(set)
    for name in sorted(sources):
        here = module_of(name)
        for target in IMPORT.findall(code(files[name])):
            if target in COMPILER:
                continue
            if "/" in target or target.endswith((".zig", ".zon")):
                path = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
                if not path.startswith(SRC):
                    continue
                if path not in sources:
                    unresolved[f"{path} (no such file)"].add(name)
                    continue
                there = module_of(path)
            elif names.get(target):
                there = names[target]
            else:
                kind = "two modules" if target in names else "no zig/src root"
                unresolved[f'@import("{target}") ({kind})'].add(name)
                continue
            if there != here:
                edges[(here, there)].add((name, target))
    return dict(edges), dict(unresolved)


def components(edges):
    """Tarjan's strongly connected components with more than one module, each sorted, iteratively."""
    succ = defaultdict(list)
    for a, b in sorted(edges):
        succ[a].append(b)
    index, low, stack, on, found = {}, {}, [], set(), []
    for start in sorted({a for a, _ in edges} | {b for _, b in edges}):
        if start in index:
            continue
        work = [(start, 0)]
        while work:
            v, i = work.pop()
            if i == 0:
                index[v] = low[v] = len(index)
                stack.append(v)
                on.add(v)
            if i < len(succ[v]):
                work.append((v, i + 1))
                w = succ[v][i]
                if w not in index:
                    work.append((w, 0))
                elif w in on:
                    low[v] = min(low[v], index[w])
                continue
            if low[v] == index[v]:
                group = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    group.append(w)
                    if w == v:
                        break
                if len(group) > 1:
                    found.append(sorted(group))
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
    return sorted(found, key=lambda g: (-len(g), g))


def cycle_edges(edges, groups):
    """The module edges inside a cycle: both ends in the same strongly connected component."""
    group_of = {m: i for i, g in enumerate(groups) for m in g}
    return {(a, b) for a, b in edges if a in group_of and group_of.get(b) == group_of[a]}


def example(edges, edge):
    name, target = min(edges[edge])
    return f"{name} imports {target}" + (f", {len(edges[edge])} imports" if len(edges[edge]) > 1 else "")


def compare(base_files, head_files):
    """ERROR: a new cycle edge that closes without the base's cycle edges, or a new module core imports; else WARN."""
    base_edges, base_unresolved = graph(base_files)
    head_edges, head_unresolved = graph(head_files)
    base_groups, head_groups = components(base_edges), components(head_edges)
    before, after = cycle_edges(base_edges, base_groups), cycle_edges(head_edges, head_groups)
    # The head graph without the base's cycle edges: a cycle left in it is one the PR closes on its own.
    rest = {e: v for e, v in head_edges.items() if e not in before}
    own_groups = components(rest)
    own = cycle_edges(rest, own_groups)
    core_new = sorted(e for e in head_edges if e[0] == CORE and e not in base_edges)
    findings, closing = [], 0
    for group in head_groups:
        inside = sorted(e for e in after if e[0] in group)
        new = [e for e in inside if e not in before]
        if not new:
            findings.append(("WARN", f"inherited import cycle of {len(group)} modules ({len(inside)} edges): "
                                     f"{', '.join(group)}"))
        for edge in new:
            how = "existing import" if edge in base_edges else "new import"
            if edge in own:
                closing += 1
                mods = next(g for g in own_groups if edge[0] in g)
                findings.append(("ERROR", f"{edge[0]} -> {edge[1]}: {how} closes a cycle without the inherited "
                                          f"one ({example(head_edges, edge)}); cycle of {len(mods)} modules: "
                                          f"{', '.join(mods)}"))
            elif edge not in core_new:
                findings.append(("WARN", f"{edge[0]} -> {edge[1]}: {how} joins the inherited cycle of "
                                         f"{len(group)} modules ({example(head_edges, edge)})"))
    for edge in (e for e in core_new if e not in own):
        findings.append(("ERROR", f"{edge[0]} -> {edge[1]}: core imports a new module ({example(head_edges, edge)})"))
    for name in sorted(set(head_unresolved) - set(base_unresolved)):
        findings.append(("WARN", f"{name}: unresolved import outside the cycle graph, in "
                                 f"{', '.join(sorted(head_unresolved[name])[:3])}"))
    counts = dict(cycle_base=len(before), cycle_head=len(after), new_cycle=len(after - before), closing_cycle=closing,
                  removed_cycle=len(before - after), unresolved=len(head_unresolved))
    return findings, counts


def checkout(root):
    """The checkout's zig/src, build.zig and zig/build sources by repo-relative name."""
    paths = [*sorted((root / "zig/src").rglob("*.zig")), root / "build.zig", *sorted((root / "zig/build").glob("*.zig"))]
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8", errors="replace") for p in paths if p.is_file()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unresolved", action="store_true", help="list every unresolved import with its file count")
    args = parser.parse_args(argv)
    edges, unresolved = graph(checkout(ROOT))
    groups = components(edges)
    inside = cycle_edges(edges, groups)
    modules = {m for e in edges for m in e}
    print(f"{len(modules)} modules, {len(edges)} module edges, {len(groups)} cycle(s), {len(inside)} cycle edges")
    for group in groups:
        print(f"cycle of {len(group)} modules: {', '.join(group)}")
        for edge in sorted(e for e in inside if e[0] in group):
            print(f"  {edge[0]} -> {edge[1]}: {example(edges, edge)}")
    print(f"{len(unresolved)} unresolved imports (outside the graph; --unresolved lists them)")
    if args.unresolved:
        for name, users in sorted(unresolved.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            print(f"  {name}: {len(users)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
