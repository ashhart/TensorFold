"""The is_past cases of zig/src/server/notes/past.zig, answered by LLWZ's notes.is_past (CPython): python3 past_cases.py LLWZ."""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ZIG = os.path.join(HERE, "..", "..", "zig", "src", "server", "notes", "past.zig")


def zig_text(lit):
    """A Zig string literal's text (the escapes past.zig uses)."""
    out = re.sub(r"\\u\{([0-9a-fA-F]+)\}", lambda m: chr(int(m.group(1), 16)), lit)
    out = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), out)
    return out.replace("\\t", "\t").replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")


def main():
    sys.path.insert(0, os.path.join(sys.argv[1], "zengine"))
    import notes  # noqa: E402  (LLWZ's)
    rows = re.findall(r'^\s*\.\{ "((?:[^"\\]|\\.)*)", (true|false) \},$', open(ZIG).read(), re.M)
    bad = [(t, w) for t, w in rows if notes.is_past(zig_text(t)) != (w == "true")]
    for t, w in bad:
        print(f"differs: {t!r}: table {w}, CPython {notes.is_past(zig_text(t))}")
    print(f"{len(rows)} cases, {len(bad)} differ from CPython's notes.is_past")
    return 1 if bad or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
