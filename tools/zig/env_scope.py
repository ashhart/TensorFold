"""Which compile steps reach each Zig source, from `zig build -Denv-graph`: the release server, tools or tests."""
import os
import re
import subprocess
import tempfile
from pathlib import Path

from env_reads import LINE_STRING, masked, test_lines

# The release server: each tensorfold-native the dist archives build, configured on any host without building it.
RELEASE = "tensorfold-native"
DIST = ("-Dversion=0.0.0", "-Ddist-host-only=true", "-Ddist-macos-sdk=/env-graph")
SCOPES = ("release", "tool", "test")
IMPORT = re.compile(r'@import\("([^"]+)"\)')


# Opt-in backends whose own servers and tools only their flag's graph has; their graphs add tool and test scopes.
OPT_IN = ("-Dhip=true",)
# The line between the default graph and each opt-in one.
SECTION = "G"


def graph_text(root: Path) -> str:
    """The build graphs `zig build -Denv-graph` writes, default then opt-in, configured without compiling anything."""
    texts = []
    with tempfile.TemporaryDirectory() as d:
        for i, extra in enumerate(((),) + tuple((flag,) for flag in OPT_IN)):
            out = Path(d) / f"graph{i}.txt"
            subprocess.run(["zig", "build", f"-Denv-graph={out}", *DIST, *extra, "-l"], cwd=root, check=True,
                           stdout=subprocess.DEVNULL)
            texts.append(out.read_text(encoding="utf-8"))
    return f"\n{SECTION}\n".join(texts)


def parse(text: str) -> tuple[dict[int, str], dict[int, dict[str, int]], list[tuple[str, str, int, bool]]]:
    """The graph's module roots, each module's imports by name, and compile steps as (kind, name, module, in dist)."""
    roots, imports, compiles = {}, {}, []
    for line in text.splitlines():
        f = line.split("\t")
        if f[0] == "M":
            roots[int(f[1])] = f[2]
            imports.setdefault(int(f[1]), {})
        elif f[0] == "I":
            imports.setdefault(int(f[1]), {})[f[2]] = int(f[3])
        elif f[0] == "C":
            compiles.append((f[1], f[2], int(f[3]), f[4] == "dist"))
    return roots, imports, compiles


def scope_of(kind: str, name: str, dist: bool) -> str:
    """A compile step's scope: a released server, another program, or a test."""
    if kind in ("test", "test_obj"):
        return "test"
    return "release" if dist and kind == "exe" and name == RELEASE else "tool"


class Imports:
    """Each source's @import targets, read once; a test block's imports count only for tests."""

    def __init__(self, root: Path):
        self.root, self.cache = root, {}

    def of(self, rel: str, tests: bool) -> list[str]:
        if rel not in self.cache:
            path = self.root / rel
            text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
            code, inside = masked_imports(text), test_lines(text)
            found = [(code.count("\n", 0, m.start()) + 1, m.group(1)) for m in IMPORT.finditer(code)]
            self.cache[rel] = ([t for _, t in found], [t for line, t in found if line not in inside])
        return self.cache[rel][0 if tests else 1]


def masked_imports(text: str) -> str:
    """`text` without comments and multi-line strings, keeping string contents so import paths stay readable."""
    out = []
    for line, code in zip(text.split("\n"), masked(text).split("\n")):
        cut = len(code.rstrip()) if "//" in line else len(line)
        out.append("" if LINE_STRING.match(line) else line[:cut])
    return "\n".join(out)


def reach(root: Path, text: str) -> dict[str, set[str]]:
    """Each source under `root` that a compile step reaches, with the scopes of the steps that reach it."""
    found: dict[str, set[str]] = {}
    files = Imports(root)
    sections = text.split(f"\n{SECTION}\n")
    for i, section in enumerate(sections):
        # an opt-in graph's release archive is not the one shipped, so it adds only tool and test scopes
        reach_section(section, files, SCOPES if i == 0 else SCOPES[1:], found)
    return found


def reach_section(text: str, files: "Imports", scopes: tuple[str, ...], found: dict[str, set[str]]) -> None:
    """Add to `found` each source one graph's compile steps of `scopes` reach."""
    roots, imports, compiles = parse(text)
    for scope in scopes:
        tests = scope == "test"
        todo = [(m, roots[m]) for kind, name, m, dist in compiles
                if scope_of(kind, name, dist) == scope and roots.get(m, "-") != "-"]
        seen = set(todo)
        while todo:
            module, rel = todo.pop()
            found.setdefault(rel, set()).add(scope)
            for target in files.of(rel, tests):
                if target.endswith(".zig"):
                    nxt = (module, os.path.normpath(os.path.join(os.path.dirname(rel), target)))
                elif target in imports.get(module, {}) and roots.get(imports[module][target], "-") != "-":
                    dep = imports[module][target]
                    nxt = (dep, roots[dep])
                else:
                    continue
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
