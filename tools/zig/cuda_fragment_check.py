"""Every CUDA fragment #included inside a function body finds each name it needs declared before the include."""
import argparse
import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
KERNELS = "zig/kernels/cuda"
TOKEN = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|[A-Za-z_]\w*'
                   r'|\.?\d(?:[eEpP][+-]|[\w.])*|::|->|\\\n|\n|\S')
# Statements that open with these are never declarations.
CONTROL = set("""if else for while do switch case default return goto break continue throw delete new static_assert
sizeof alignof asm __asm__ public private protected friend operator typeid this true false nullptr static_cast
const_cast reinterpret_cast dynamic_cast co_return co_yield co_await""".split())
KEYWORDS = CONTROL | set("""auto bool char char8_t char16_t char32_t class const consteval constexpr constinit decltype
double enum explicit export extern float inline int long mutable namespace noexcept register short signed static
struct template thread_local typedef typename union unsigned using virtual void volatile wchar_t __global__ __device__
__host__ __shared__ __constant__ __managed__ __restrict__ __restrict __forceinline__ __noinline__ __inline__
__launch_bounds__ __align__ __volatile__ __attribute__ __grid_constant__ __cluster_dims__ __maxnreg__ alignas""".split())
# Specifiers written as calls; their parentheses are never a parameter list.
ATTRS = {"__launch_bounds__", "__attribute__", "__align__", "alignas", "__declspec", "decltype", "noexcept",
         "__cluster_dims__", "__maxnreg__"}
FIRST_END = ("=", ",", ";", "[", "(", "{", ":")
NEXT_END = ("=", ",", ";", "[", "(", "{")


@dataclass(frozen=True)
class Tok:
    kind: str  # id, num, str, punct, dir
    text: str
    line: int
    arg: str = ""  # a directive's include path, macro name or condition


@dataclass
class Frame:
    ch: str  # "(", "[" or "{"
    kind: str = ""  # a brace's ns, type, enum, func, block or init
    scope: set = field(default_factory=set)


@dataclass
class Site:
    path: str
    line: int
    spelling: str
    target: object  # Path of the fragment, None when it does not exist
    visible: set  # names the enclosing function declares before the include
    context: set  # file-scope names and macros of the includer and its headers so far
    switches: tuple = ()  # (macro, True/False/None) for each switch the includer sets, at the include
    need: list = field(default_factory=list)  # the fragment's required names this include must supply


@dataclass
class Branch:
    start: list  # the brace stack before #if
    now: object  # this branch's value: True, False or None when it can't be told
    after: object = None  # the brace stack after the first branch read
    taken: bool = False  # an earlier branch was true
    unsure: bool = False  # an earlier branch could not be told


@dataclass
class Unit:
    context: set
    decls: set
    uses: set
    sites: list


def lex(text):
    """C/C++ tokens without comments; each preprocessor line becomes one dir token."""
    newlines = [m.start() for m in re.finditer("\n", text)]
    ms, out, start, i = list(TOKEN.finditer(text)), [], True, 0
    while i < len(ms):
        m = ms[i]
        v, i = m.group(), i + 1
        if v == "\n":
            start = True
            continue
        if v == "\\\n" or v.startswith(("//", "/*")):
            continue
        line = bisect.bisect_left(newlines, m.start()) + 1
        if v == "#" and start:
            words = []
            while i < len(ms) and ms[i].group() != "\n":
                w, i = ms[i].group(), i + 1
                if w != "\\\n" and not w.startswith(("//", "/*")):
                    words.append(w)
            name, arg = (words[0] if words else ""), " ".join(words[1:])
            if name == "include":
                arg = arg[1:-1] if arg.startswith('"') else ""
            elif name in ("define", "undef", "ifdef", "ifndef"):
                arg = words[1] if len(words) > 1 else ""
            out.append(Tok("dir", name, line, arg))
            continue
        start = False
        kind = "id" if v[0].isalpha() or v[0] == "_" else "str" if v[0] in "\"'" else "num" if v[0].isdigit() else "punct"
        out.append(Tok(kind, v, line))
    return out


def close(ts, j):
    """Index of the delimiter matching ts[j]; strings never count."""
    left = ts[j].text
    right = {"(": ")", "[": "]", "{": "}"}[left]
    depth = 0
    for k in range(j, len(ts)):
        depth += (ts[k].text == left) - (ts[k].text == right)
        if depth == 0:
            return k
    raise ValueError(f"line {ts[j].line}: unclosed {left}")


def angle(ts, j):
    """Index of the '>' closing a template argument list at ts[j], or None when '<' is a comparison."""
    depth = paren = 0
    for k in range(j, len(ts)):
        t = ts[k].text
        if t in ("(", "["):
            paren += 1
        elif t in (")", "]"):
            paren -= 1
            if paren < 0:
                return None
        elif paren == 0 and t == "<":
            depth += 1
        elif paren == 0 and t == ">":
            depth -= 1
            if depth == 0:
                return k
        elif t in (";", "{", "}") or (paren == 0 and t in ("&&", "||")):
            return None
    return None


def typed_run(ts, j, end=None):
    """The qualified names of a type-and-declarator run from ts[j], and the index just past it."""
    end, run = len(ts) if end is None else end, []
    while j < end:
        t = ts[j]
        if t.kind == "id":
            if t.text in ATTRS and j + 1 < end and ts[j + 1].text == "(":
                j = close(ts, j + 1) + 1
                continue
            if run and ts[j - 1].text == "::":
                run[-1] += "::" + t.text
            else:
                run.append(t.text)
            j += 1
            if j < end and ts[j].text == "<" and t.text not in KEYWORDS:
                k = angle(ts, j)
                if k is None:
                    break
                j = k + 1
        elif t.kind == "str" or t.text in ("*", "&", "::"):
            j += 1
        elif t.text == "[" and j + 1 < end and ts[j + 1].text == "[":
            j = close(ts, j) + 1
        else:
            break
    return run, j


def plain(name):
    """An unqualified, non-keyword identifier."""
    return "::" not in name and name not in KEYWORDS


def declarators(ts, i):
    """The names a declaration statement starting at ts[i] declares; [] when it is not a declaration."""
    n = len(ts)
    while i + 1 < n and ts[i].text == "[" and ts[i + 1].text == "[":
        i = close(ts, i) + 1
    if i + 1 < n and ts[i].text == "template" and ts[i + 1].text == "<":
        k = angle(ts, i + 1)
        if k is None:
            return []
        i = k + 1
    if i >= n or ts[i].kind != "id" or ts[i].text in CONTROL:
        return []
    word = ts[i].text
    if word == "namespace":
        return [ts[i + 1].text] if i + 1 < n and ts[i + 1].kind == "id" else []
    if word in ("using", "typedef"):
        j = i
        while j < n and ts[j].text not in (";", "{", "}"):
            j += 1
        body = ts[i + 1:j]
        if word == "using" and body and body[0].text == "namespace":
            return []
        if word == "using" and len(body) > 1 and body[1].text == "=":
            return [body[0].text]
        return [t.text for t in body if t.kind == "id" and plain(t.text)][-1:]
    run, j = typed_run(ts, i)
    if j >= n or len(run) < 2 or not plain(run[-1]) or ts[j].text not in FIRST_END:
        return []
    names = [run[-1]]
    if ts[j].text in ("(", "{"):
        return names
    depth = 0
    while j < n:
        t = ts[j].text
        if t in ("(", "[", "{"):
            depth += 1
        elif t in (")", "]", "}"):
            depth -= 1
            if depth < 0:
                break
        elif depth == 0 and t == ";":
            break
        elif depth == 0 and t == ",":
            k = j + 1
            while k < n and ts[k].text in ("*", "&"):
                k += 1
            if k + 1 < n and ts[k].kind == "id" and plain(ts[k].text) and ts[k + 1].text in NEXT_END:
                names.append(ts[k].text)
        j += 1
    return names


def params(ts, lo, hi):
    """Named parameters between the delimiters at ts[lo] and ts[hi] (function, lambda or template)."""
    names, j = [], lo + 1
    while j < hi:
        run, k = typed_run(ts, j, hi)
        if k + 2 < hi and ts[k].text == "(" and ts[k + 1].text in ("*", "&") and ts[k + 2].kind == "id":
            names.append(ts[k + 2].text)
        elif len(run) >= 2 and plain(run[-1]) and ts[k - 1].text == run[-1] and (k == hi or ts[k].text in (",", "=", "[")):
            names.append(run[-1])
        depth = 0
        while k < hi and not (depth == 0 and ts[k].text == ","):
            depth += (ts[k].text in ("(", "[", "{")) - (ts[k].text in (")", "]", "}"))
            k += 1
        j = k + 1
    return names


def condition(words, known):
    """An #if over defined(SWITCH), !, &&, || and parentheses: True, False or None; another term is unknown, 0 false."""
    pos = 0

    def at(*want):
        return words[pos:pos + len(want)] == list(want)

    def term():
        nonlocal pos
        if at("!"):
            pos += 1
            v = term()
            return None if v is None else not v
        if at("defined"):
            name = words[pos + 2] if at("defined", "(") else words[pos + 1] if pos + 1 < len(words) else ""
            pos += 4 if at("defined", "(") else 2
            v = known.get(name)
        elif at("("):
            pos += 1
            v = either()
            pos += 1
        else:
            v = False if words[pos:pos + 1] == ["0"] else None
            pos += 1
        if pos < len(words) and not (at(")") or at("&", "&") or at("|", "|")):
            v, depth = None, 0  # part of a comparison or arithmetic: unknown, up to the next && or ||
            while pos < len(words) and not (depth == 0 and (at(")") or at("&", "&") or at("|", "|"))):
                depth += at("(") - at(")")
                pos += 1
        return v

    def chain(op, part, short):
        nonlocal pos
        v = part()
        while at(op, op):
            pos += 2
            w = part()
            v = short if short in (v, w) else None if None in (v, w) else not short
        return v

    def both():
        return chain("&", term, False)

    def either():
        return chain("|", both, True)

    return either() if words else None


def truth(t, known):
    """A fragment's #if, #ifdef, #ifndef or #elif under the includer's switches."""
    if t.text in ("ifdef", "ifndef"):
        v = known.get(t.arg)
        return v if v is None or t.text == "ifdef" else not v
    return condition(t.arg.split(), known)


def head_shape(head):
    """A namespace-level brace's kind from its statement head, and the function's parameter names."""
    if not head:
        return "block", []
    ids = [t.text for t in head]
    if "namespace" in ids[:2] or (ids[0] == "extern" and len(head) == 2 and head[1].kind == "str"):
        return "ns", []
    group, j, colon = None, 0, False
    while j < len(head):
        t, prev = head[j], head[j - 1] if j else None
        if t.text == "<" and prev is not None and prev.kind == "id" and (prev.text == "template" or prev.text not in KEYWORDS):
            k = angle(head, j)
            if k is not None:
                j = k + 1
                continue
        if t.text in ("(", "["):
            k = close(head, j)
            named = prev is not None and ((prev.kind == "id" and plain(prev.text) and prev.text not in ATTRS)
                                          or prev.text == ">" or (prev.text == ")" and "operator" in ids))
            if t.text == "(" and named and not colon:
                group = (j, k)
            j = k + 1
            continue
        if t.text == "=":
            return "init", []
        colon = colon or t.text == ":"
        j += 1
    if group is not None:
        names = params(head, *group)
        if ids[0] == "template" and len(head) > 1 and head[1].text == "<" and angle(head, 1) is not None:
            names += params(head, 1, angle(head, 1))
        return "func", names
    if "enum" in ids:
        return "enum", []
    return ("type" if {"struct", "class", "union"} & set(ids) else "init"), []


class Tree:
    """One CUDA kernel tree: files walked once each, as includers and as fragments."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.base = self.root / KERNELS
        self.units, self.busy, self.frees = {}, set(), {}

    def rel(self, p):
        return p.relative_to(self.root).as_posix()

    def resolve(self, path, spelling):
        for d in (path.parent, self.base):
            if (d / spelling).is_file():
                return (d / spelling).resolve()
        return None

    def header(self, p):
        """File-scope names and macros a header and its own headers declare; a header already open adds nothing."""
        return set() if (p, False, ()) in self.busy else self.unit(p, False).context

    def unit(self, p, fragment, switches=()):
        key = p, fragment, switches
        if key not in self.units:
            self.busy.add(key)
            try:
                self.units[key] = self.walk(p, fragment, dict(switches))
            except ValueError as exc:
                raise ValueError(str(exc) if str(exc).startswith(self.rel(p)) else f"{self.rel(p)}: {exc}") from exc
            finally:
                self.busy.discard(key)
        return self.units[key]

    def walk(self, path, fragment, known):
        """Track braces, scopes and declarations; record each quoted include inside a function body."""
        toks = lex(path.read_text())
        code = [t for t in toks if t.kind != "dir"]
        # A switch is a macro its includer both #defines and #undefs (qmmf.cu); a fragment reads only its taken branch.
        if not fragment:
            defined = {t.arg for t in toks if t.kind == "dir" and t.text == "define"}
            known = dict.fromkeys(defined & {t.arg for t in toks if t.kind == "dir" and t.text == "undef"}, False)
        stack = [Frame("{", "func")] if fragment else []
        context, decls, uses, sites, pending, conds = set(), set(), set(), [], set(), []
        at_start, head, ci, dead = True, 0, 0, False
        for t in toks:
            if t.kind == "dir":
                # A branch starts from the brace state before #if; the first branch read continues after #endif.
                if t.text in ("if", "ifdef", "ifndef"):
                    v = None if dead else truth(t, known if fragment else {})
                    conds.append(None if dead else Branch(list(stack), v, taken=v is True, unsure=v is None))
                    dead = dead or v is False
                elif t.text in ("elif", "else", "endif") and conds:
                    c = conds.pop() if t.text == "endif" else conds[-1]
                    if c is None:
                        continue
                    if not dead and c.after is None:
                        c.after = list(stack)
                    stack[:] = c.after if t.text == "endif" and c.after is not None else c.start
                    if t.text == "endif":
                        dead = False
                        continue
                    c.now = (False if c.taken else None if c.unsure else True) if t.text == "else" else \
                        False if c.taken else truth(t, known if fragment else {})
                    c.taken, c.unsure, dead = c.taken or c.now is True, c.unsure or c.now is None, c.now is False
                elif dead:
                    pass
                elif t.text in ("define", "undef") and t.arg:
                    if t.arg in known:
                        known[t.arg] = t.text == "define" if all(c.now is True for c in conds) else None
                    if t.text == "define":
                        context.add(t.arg)
                elif t.text == "include" and t.arg:
                    target = self.resolve(path, t.arg)
                    if any(f.kind == "func" for f in stack):
                        first = next(k for k, f in enumerate(stack) if f.kind == "func")
                        visible = set().union(*(f.scope for f in stack[first:]))
                        sites.append(Site(self.rel(path), t.line, t.arg, target, visible, set(context),
                                          tuple(sorted(known.items()))))
                    elif target is not None:
                        context |= self.header(target)
                continue
            i, ci = ci, ci + 1
            if dead:
                continue
            in_func = any(f.kind == "func" for f in stack)
            top = stack[-1] if stack else None
            if at_start:
                at_start, head = False, i
                names = declarators(code, i)
                if in_func:
                    top.scope.update(names)
                    decls.update(names)
                elif top is None or top.kind == "ns":
                    context.update(names)
            prev = code[i - 1].text if i else ""
            if top is not None and top.kind == "enum" and t.kind == "id" and prev in ("{", ","):
                for names in ((top.scope, decls) if in_func else (context,)):
                    names.add(t.text)
            v = t.text
            if v == "(":
                stack.append(Frame("("))
                if in_func and prev in ("]", "for"):
                    names = params(code, i, close(code, i)) if prev == "]" else declarators(code, i + 1)
                    pending |= set(names)
                    decls.update(names)
            elif v == "[":
                stack.append(Frame("["))
            elif v == "{":
                if in_func or (top is not None and top.ch != "{"):
                    kind, names = ("block" if in_func else "init"), []
                else:
                    kind, names = head_shape(code[head:i])
                    decls.update(names)
                stack.append(Frame("{", kind, set(names) | (pending if in_func else set())))
                pending, at_start = set(), True
            elif v in (")", "]", "}"):
                want = {")": "(", "]": "[", "}": "{"}[v]
                if not stack or stack[-1].ch != want or (fragment and len(stack) == 1):
                    raise ValueError(f"{self.rel(path)}:{t.line}: unbalanced {v}")
                stack.pop()
                at_start = v == "}" and (not stack or stack[-1].ch == "{")
            elif v == ";" and (top is None or top.ch == "{"):
                pending, at_start = set(), True
            elif fragment and t.kind == "id" and v not in KEYWORDS and prev not in (".", "->", "::"):
                uses.add(v)
        if len(stack) != int(fragment):
            raise ValueError(f"{self.rel(path)}: unbalanced braces at the end")
        return Unit(context, decls, uses, sites)

    def free(self, frag, switches=(), seen=()):
        """Names a fragment uses but never declares under these switches, including those its fragments leave to it."""
        if frag in seen:
            raise ValueError(f"{self.rel(frag)}: fragment includes itself")
        if (frag, switches) not in self.frees:
            u = self.unit(frag, True, switches)
            names = set(u.uses)
            for s in u.sites:
                if s.target is None:
                    raise ValueError(f"{s.path}:{s.line}: missing {s.spelling}")
                names |= self.free(s.target, s.switches, seen + (frag,)) - s.visible - s.context
            self.frees[frag, switches] = names - u.decls - u.context - KEYWORDS
        return self.frees[frag, switches]


def scan(root):
    """Each fragment with its includers and the free names a function-local declaration supplies somewhere."""
    tree = Tree(root)
    if not tree.base.is_dir():
        raise ValueError("missing CUDA kernel tree")
    files = sorted(p.resolve() for p in tree.base.rglob("*") if p.is_file() and p.suffix in (".cu", ".cuh", ".h"))
    units, failed = {}, {}
    for p in files:
        try:
            units[p] = tree.unit(p, False)
        except ValueError as exc:
            failed[p] = exc
    frags = {s.target for u in units.values() for s in u.sites}
    # A fragment is a function body, so it is read as one; read as a whole file it may not balance.
    for p, exc in failed.items():
        if p not in frags:
            raise exc
    by = defaultdict(list)
    for p, u in units.items():
        for s in u.sites if p not in frags else ():
            by[s.target].append(s)
    if None in by:
        s = by[None][0]
        raise ValueError(f"{s.path}:{s.line}: missing {s.spelling}")
    # Function-local anywhere and file-scope nowhere: such a name can only come from the includer's function.
    local = set().union(*(tree.unit(p, True).decls if p in frags else u.decls for p, u in units.items()))
    local -= set().union(*(u.context for p, u in units.items() if p not in frags))
    out = []
    for frag in sorted(by):
        required = local | set().union(*(s.visible for s in by[frag]))
        for s in by[frag]:
            s.need = sorted(tree.free(frag, s.switches) & required)
        out.append((tree.rel(frag), sorted(set().union(*(s.need for s in by[frag]))), by[frag]))
    return out


def check(root):
    """Every miss as 'path:line: fragment needs NAME'."""
    found = scan(root)
    errors = [f"{s.path}:{s.line}: {s.spelling} needs {name}"
              for _, _, sites in found for s in sites for name in s.need if name not in s.visible | s.context]
    return found, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--list", action="store_true", help="list each fragment, its required names and includers")
    args = parser.parse_args()
    try:
        found, errors = check(args.root)
        if args.list:
            for frag, need, sites in found:
                print(f"{frag}: {len(sites)} includers; needs {' '.join(need)}")
                for s in sites:
                    on = " ".join(m if v else f"{m}=?" for m, v in s.switches if v is not False)
                    print(f"  {s.path}:{s.line}" + (f" ({on})" if on else ""))
        for error in errors:
            print(error)
        print(f"cuda fragments: {len(found)} fragments, {sum(len(s) for _, _, s in found)} includes, {len(errors)} misses")
        return int(bool(errors))
    except (OSError, ValueError) as exc:
        print(f"cuda fragments: invalid input: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
