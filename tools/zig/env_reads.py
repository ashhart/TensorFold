"""Environment lookups in Zig source, found by the lookup call: each one's names, or the computed ones by place."""
import bisect
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from lean_check import TEST_BLOCK

LINE_STRING = re.compile(r"^\s*\\\\")
STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')
CHAR = re.compile(r"'(?:[^'\\\n]|\\.)*'")
LITERAL = r'"((?:[^"\\]|\\.)*)"'
ONE = re.compile(rf"^{LITERAL}$")
MANY = re.compile(r'^&\.\{\s*("(?:[^"\\]|\\.)*"\s*,\s*)*"(?:[^"\\]|\\.)*"\s*,?\s*\}$')
# A file-level or public string constant, so `getenv(name)` after `const name = "X";` reads X.
CONST = re.compile(rf"^(?:\s*pub\s+|)const\s+(\w+)\s*(?::\s*[^=]+)?=\s*{LITERAL}\s*;", re.M)
# A declaration whose type is an environment map, and the aliases that unwrap or copy one.
MAP_DECL = re.compile(r"\b(\w+)\s*:\s*[?*\s]*(?:const\s+)?(?:std\.process\.)?Environ\.Map\b")
MAP_ALIAS = (re.compile(r"\b(?:const|var)\s+(?P<new>\w+)\s*(?::[^=;]+)?=\s*&?(?:[\w.]+\.)?(?P<old>\w+)\s+orelse\b"),
             re.compile(r"\b(?:const|var)\s+(?P<new>\w+)\s*(?::[^=;]+)?=\s*&?(?:[\w.]+\.)?(?P<old>\w+)\s*;"),
             re.compile(r"\bif\s*\(\s*(?:[\w.]+\.)?(?P<old>\w+)\s*\)\s*\|\s*\*?(?P<new>\w+)\s*\|"))
MAP_TYPE = re.compile(r"^\s*(?:pub\s+)?const\s+(\w+)\s*=\s*(?:std\.process\.)?Environ\.Map\s*;", re.M)
# A parameter or alias typed as a function from a name to its optional value: calls through it are lookups.
GETTER_TYPE = r"\*const\s+fn\s*\(\s*\[(?:\*)?(?::0)?\]const\s+u8\s*\)\s*(?:callconv\(\.c\)\s*)?\?\["
GETTER_ALIAS = re.compile(rf"^\s*(?:pub\s+)?const\s+(\w+)\s*=\s*{GETTER_TYPE}", re.M)
MAP_GET = re.compile(r"(?<![\w.])((?:\w+\.)*)(\w+)\.get(?:Ptr)?\(")
# The C library's getenv under any receiver, sys.getenv among them, and the test runner's environment.
GETENV = re.compile(r"(?<![\w])(?<!fn )(?:[\w.]+\.)?getenv\(")
# The test runner's environment reads, by method, with each one's name argument.
TEST_ENV = (("getPosix", 0), ("getAlloc", 1))
FN = re.compile(r"^\s*(?:pub\s+)?(?:inline\s+)?fn\s+(\w+)\s*\(", re.M)
# A module alias (`@import("x.zig")`, `@import("x").member`, `b.x`), of the member too, for `a.function(` calls.
ALIAS = re.compile(r"^\s*(?:pub\s+)?const\s+(\w+)\s*=\s*"
                   r"(?:[^;]*?@import\(\"([^\"]+)\"\)((?:\.\w+)*)|[\w.]*?\.?(\w+))\s*;", re.M)


@dataclass(frozen=True)
class Read:
    """One lookup: where, its call, its names (empty when computed), its function, and whether it sweeps a table."""

    file: str
    line: int
    call: str
    names: tuple[str, ...]
    function: str
    test: bool = False
    sweep: bool = False
    arg: str = ""


def masked(text: str) -> str:
    """`text` with comments, multi-line strings, and string contents but import paths blanked, offsets kept."""
    out = []
    for line in text.split("\n"):
        if LINE_STRING.match(line):
            out.append(" " * len(line))
            continue
        line = STRING.sub(lambda m: m.group(0) if line[:m.start()].endswith("@import(") else
                          '"' + " " * (len(m.group(0)) - 2) + '"', line)
        line = CHAR.sub(lambda m: "'" + " " * (len(m.group(0)) - 2) + "'", line)
        cut = line.find("//")
        out.append(line if cut < 0 else line[:cut] + " " * (len(line) - cut))
    return "\n".join(out)


def test_lines(text: str) -> set[int]:
    """The 1-based lines inside `test` blocks, as zig fmt writes them."""
    inside, indent = set(), None
    for i, line in enumerate(text.split("\n"), 1):
        if indent is None and (block := TEST_BLOCK.match(line)):
            indent = block.group(1)
        if indent is not None:
            inside.add(i)
            if re.fullmatch(re.escape(indent) + r"\}\s*(?://.*)?", line.rstrip()):
                indent = None
    return inside


def arguments(code: str, text: str, open_at: int) -> tuple[list[str], int]:
    """The call's top-level arguments from `text`, split where `code` has commas, and where it closes."""
    depth, start, args = 0, open_at + 1, []
    for i in range(open_at, len(code)):
        ch = code[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                args.append(text[start:i].strip())
                return [a for a in args if a], i
        elif ch == "," and depth == 1:
            args.append(text[start:i].strip())
            start = i + 1
    return [], len(code)


def map_names(code: str) -> set[str]:
    """The identifiers in one file that hold an environment map or Process: declared with its type, or unwrapped."""
    held = set(MAP_DECL.findall(code)) | {"environ_map"}
    held |= set(re.findall(r"\b(\w+)\s*:\s*(?:@import\(\"sys\"\)\.|[\w.]*\.)?environment\.Process\b", code))
    for alias in MAP_TYPE.findall(code):
        held |= set(re.findall(rf"\b(\w+)\s*:\s*[?*\s]*(?:const\s+)?{alias}\b", code))
    while True:
        more = {m["new"] for pattern in MAP_ALIAS for m in pattern.finditer(code) if m["old"] in held} - held
        if not more:
            return held
        held |= more


class Source:
    """One file's text, its masked code, its test-block lines, string constants, functions and module aliases."""

    def __init__(self, rel: str, text: str):
        self.rel, self.text, self.code = rel, text, masked(text)
        self.tests = test_lines(text)
        self.consts = dict(CONST.findall(text))
        self.functions = [(self.code.count("\n", 0, m.start()) + 1, m.group(1)) for m in FN.finditer(self.code)]
        self.aliases, self.stems = {}, {}
        for m in ALIAS.finditer(text):
            target = PurePosixPath(m.group(2)).stem if m.group(2) else m.group(4)
            self.aliases.setdefault(target, set()).add(m.group(1))
            if m.group(3):
                self.aliases.setdefault(m.group(3).rsplit(".", 1)[1], set()).add(m.group(1))
            self.stems[m.group(1)] = target
        self.starts = [0] + [i + 1 for i, ch in enumerate(self.code) if ch == "\n"]

    def line(self, at: int) -> int:
        return bisect.bisect_right(self.starts, at)

    def array(self, name: str) -> tuple[str, ...] | None:
        """The strings of the array constant `name`, typed or not; None when it is missing or holds non-strings."""
        m = re.search(rf"\bconst\s+{name}\s*(?::[^=;]*)?=\s*(?:\[[^\]]*\][^{{;]*|&?\.)\{{([^}}]*)\}}\s*;", self.text)
        if not m:
            return None
        body, code = m.group(1), masked(m.group(1))
        spans = [x.span() for x in STRING.finditer(code)]
        if STRING.sub("", code).strip(" \t\n,"):
            return None
        return tuple(ONE.match(body[a:b]).group(1) for a, b in spans)

    def function(self, line: int) -> str:
        """The nearest function declared at or above `line`."""
        found = ""
        for start, name in self.functions:
            if start > line:
                break
            found = name
        return found

    def names(self, arg: str, consts: dict[str, dict[str, str]]) -> tuple[str, ...] | None:
        """The names an argument spells: strings, an array of them, or a constant naming one; None when computed."""
        if m := ONE.match(arg):
            return (m.group(1),)
        if MANY.match(arg):
            return tuple(re.findall(LITERAL, arg))
        if arg in self.consts:
            return (self.consts[arg],)
        if m := re.fullmatch(r'(?:[\w.]+\.)?(\w+)\.(\w+)|@import\("([^"]+)"\)\.(\w+)', arg):
            stem = PurePosixPath(m.group(3)).stem if m.group(3) else self.stems.get(m.group(1), m.group(1))
            name = m.group(4) or m.group(2)
            if name in consts.get(stem, {}):
                return (consts[stem][name],)
        return None


def calls(src: Source, pattern: re.Pattern, index: int, consts: dict, test: bool = False, keep=None) -> list[Read]:
    """Each call `pattern` matches, and `keep` accepts, outside test blocks and definitions, with its names."""
    found = []
    for m in pattern.finditer(src.code):
        line = src.line(m.start())
        if line in src.tests or src.code[max(0, m.start() - 3):m.start()] == "fn " or (keep and not keep(m)):
            continue
        args, end = arguments(src.code, src.text, m.end() - 1)
        if index >= len(args):
            continue
        names = src.names(args[index], consts)
        call = re.sub(r"\s+", "", src.text[m.start():end + 1])
        found.append(Read(src.rel, line, call, names or (), src.function(line), test, arg=args[index]))
    return found


def base(src: Source, fields: set[str], consts: dict) -> list[Read]:
    """getenv under any receiver, the test runner's environment, getter parameters and gets on environment maps."""
    found = calls(src, GETENV, 0, consts)
    testing = "|".join(["testing"] + sorted(src.aliases.get("testing", set())))
    for method, index in TEST_ENV:
        rx = re.compile(rf"(?<![\w])(?:[\w.]*\.)?(?:{testing})\.environ\.{method}\(")
        found += calls(src, rx, index, consts, test=True)
    held = map_names(src.code)
    types = "|".join([GETTER_TYPE] + [rf"{a}\b" for a in GETTER_ALIAS.findall(src.code)])
    for getter in sorted(set(re.findall(rf"\b(\w+)\s*:\s*(?:comptime\s+)?(?:{types})", src.code))):
        found += calls(src, re.compile(rf"(?<![\w.]){getter}\("), 0, consts)
    on_map = lambda m: (m.group(2) in held and not m.group(1)) or (m.group(1) and m.group(2) in fields)
    return found + calls(src, MAP_GET, 0, consts, keep=on_map)


def map_fields(sources: list[Source]) -> set[str]:
    """Every identifier declared with the environment map's type anywhere, for `x.field.get(` receivers."""
    return {n for s in sources for n in MAP_DECL.findall(s.code)} | {"environ_map"}


def file_consts(sources: list[Source]) -> dict[str, dict[str, str]]:
    """Each file stem's string constants, for arguments written `module.name`."""
    out: dict[str, dict[str, str]] = {}
    for s in sources:
        out.setdefault(PurePosixPath(s.rel).stem, {}).update(s.consts)
    return out
