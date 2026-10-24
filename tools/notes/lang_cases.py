"""Generate zig/src/server/notes/testdata/lang_cases.json from CPython 3.12.8.

Cases follow final-multilingual.md step 1. Answers are lang_cues.cue (re.I).
Mutation: dropping each alternative must flip at least one case, except alts
the full regex already covers (schon\\s+immer is covered by immer).

  PYTHONHASHSEED=0 python -s -P tools/notes/lang_cases.py
  PYTHONHASHSEED=0 python -s -P tools/notes/lang_cases.py --check
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "zig", "src", "server", "notes", "testdata", "lang_cases.json")

sys.path.insert(0, HERE)
import lang_cues  # noqa: E402

# Same order as lang.zig de_words / fi_words. schon-immer / alun-perin / year
# are the extra plant_drop ids after the word lists.
DE_WORDS = [
    "war",
    "waren",
    "warst",
    "wart",
    "hatte",
    "hatten",
    "hattest",
    "früher",
    "vorher",
    "zuvor",
    "damals",
    "ursprünglich",
    "einst",
    "je",
    "jemals",
    "immer",
    "geändert",
    "verändert",
    "gewechselt",
    "bis",
    "seit",
    "seitdem",
]
DE_SCHON = len(DE_WORDS)
DE_VOR = DE_SCHON + 1
DE_YEAR = DE_VOR + 1
DE_ALT_COUNT = DE_YEAR + 1

FI_WORDS = [
    "oli",
    "olivat",
    "olin",
    "olit",
    "olimme",
    "olitte",
    "ollut",
    "olleet",
    "ennen",
    "aiemmin",
    "aikaisemmin",
    "alkujaan",
    "entinen",
    "entisen",
    "joskus",
    "koskaan",
    "ikinä",
    "aina",
    "muuttunut",
    "muuttuneet",
    "muuttui",
    "muuttuivat",
    "vaihtunut",
    "vaihtui",
    "vaihtanut",
    "asti",
    "saakka",
    "lähtien",
    "alkaen",
    "sitten",
]
FI_ALUN = len(FI_WORDS)
FI_YEAR = FI_ALUN + 1
FI_ALT_COUNT = FI_YEAR + 1
FI_CLITICS = ["ko", "kö", "kin", "kaan", "kään", "han", "hän"]

WAS_IST = "Was ist die Augenfarbe meiner Katze?"
WO_WOHNE = "Wo wohne ich?"
NEIGHBOURS = ["x", "4", "_", "\u0301", "é", "ẞ", "™", "Ж", "中"]
SPACES = [" ", "\t", "\u00a0", "\u3000", "  ", " \t "]


def require_runtime():
    py = sys.version.split()[0]
    if py != "3.12.8":
        sys.stderr.write(f"lang_cases.py needs Python 3.12.8, got {py}\n")
        sys.exit(2)
    ver = unicodedata.unidata_version
    if ver != "15.0.0":
        sys.stderr.write(f"lang_cases.py needs unicodedata 15.0.0, got {ver}\n")
        sys.exit(2)
    if sys.flags.hash_randomization:
        sys.stderr.write("lang_cases.py needs PYTHONHASHSEED=0 and python -s -P (not -I)\n")
        sys.exit(2)


def de_pattern(drop=None):
    parts = []
    for i, w in enumerate(DE_WORDS):
        if i == 15:
            if drop != DE_SCHON:
                parts.append(r"schon\s+immer")
        if drop != i:
            parts.append(re.escape(w))
    if drop != DE_VOR:
        parts.append(r"vor\s+\w+\s+(?:Jahren|Monaten|Wochen|Tagen)")
    body = r"\b(?:%s)\b" % "|".join(parts)
    if drop != DE_YEAR:
        body += r"|\b(?:19|20)\d\d\b"
    return re.compile(body, re.I)


def fi_pattern(drop=None):
    parts = []
    for i, w in enumerate(FI_WORDS):
        if i == 11:
            if drop != FI_ALUN:
                parts.append(r"alun\s+perin")
        if drop != i:
            parts.append(re.escape(w))
    body = r"\b(?:%s)%s\b" % ("|".join(parts), lang_cues.FI_CLIT)
    if drop != FI_YEAR:
        body += r"|\b(?:19|20)\d\d\b"
    return re.compile(body, re.I)


def casings(text: str):
    out = [text, text.upper(), text.title()]
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def collect():
    if de_pattern().pattern != lang_cues.DE_PAST_CUE.pattern:
        sys.stderr.write("de reconstruction does not match lang_cues.DE_PAST_CUE\n")
        sys.stderr.write(f" got {de_pattern().pattern!r}\n")
        sys.stderr.write(f" want {lang_cues.DE_PAST_CUE.pattern!r}\n")
        sys.exit(1)
    if fi_pattern().pattern != lang_cues.FI_PAST_CUE.pattern:
        sys.stderr.write("fi reconstruction does not match lang_cues.FI_PAST_CUE\n")
        sys.stderr.write(f" got {fi_pattern().pattern!r}\n")
        sys.stderr.write(f" want {lang_cues.FI_PAST_CUE.pattern!r}\n")
        sys.exit(1)

    cases = []
    seen = set()

    def add(lang: str, text: str):
        key = (lang, text)
        if key in seen:
            return
        seen.add(key)
        if lang == "en":
            raise AssertionError("en cases are added with an explicit want")
        want = lang_cues.cue(lang).search(text) is not None
        cases.append({"lang": lang, "text": text, "want": want})

    def add_en(text: str, want: bool):
        key = ("en", text)
        if key in seen:
            return
        seen.add(key)
        cases.append({"lang": "en", "text": text, "want": want})

    for w in DE_WORDS:
        for t in casings(w):
            add("de", t)
    for t in casings("schon immer"):
        add("de", t)
    for sp in SPACES:
        add("de", f"schon{sp}immer")
    add("de", "vor drei Jahren")
    add("de", "vor 3 Jahren")
    add("de", "vor  Jahren")
    for u in ("Jahren", "Monaten", "Wochen", "Tagen"):
        add("de", f"vor drei {u}")

    for w in FI_WORDS:
        for t in casings(w):
            add("fi", t)
        for c in FI_CLITICS:
            for t in casings(w + c):
                add("fi", t)
    for t in casings("alun perin"):
        add("fi", t)
    for sp in SPACES:
        add("fi", f"alun{sp}perin")
    for c in FI_CLITICS:
        for t in casings("alun perin" + c):
            add("fi", t)

    for lang, cue in (("de", "bis"), ("fi", "aina")):
        add(lang, cue)
        for n in NEIGHBOURS:
            add(lang, n + cue)
            add(lang, cue + n)

    for y in ("1999", "2025", "20250", "19٣٤", "１９９９"):
        add("de", y)
        add("fi", y)

    add("de", WAS_IST)
    add("de", WO_WOHNE)
    add("fi", WAS_IST)
    add("fi", WO_WOHNE)
    add_en(WAS_IST, True)

    return cases


def mutation(cases):
    de_sub = []
    fi_sub = []
    for drop in range(DE_ALT_COUNT):
        rx = de_pattern(drop)
        flipped = False
        for c in cases:
            if c["lang"] != "de" or not c["want"]:
                continue
            if rx.search(c["text"]) is None:
                flipped = True
                break
        if not flipped:
            de_sub.append(drop)
    for drop in range(FI_ALT_COUNT):
        rx = fi_pattern(drop)
        flipped = False
        for c in cases:
            if c["lang"] != "fi" or not c["want"]:
                continue
            if rx.search(c["text"]) is None:
                flipped = True
                break
        if not flipped:
            fi_sub.append(drop)
    if de_sub != [DE_SCHON]:
        sys.stderr.write(
            f"de mutation subsumed {de_sub}, expected [{DE_SCHON}] (schon immer under immer)\n"
        )
        sys.exit(1)
    if fi_sub:
        sys.stderr.write(f"fi mutation subsumed {fi_sub}, expected none\n")
        sys.exit(1)
    return de_sub, fi_sub


def document(cases, de_sub, fi_sub):
    return {
        "python": sys.version.split()[0],
        "unicode": unicodedata.unidata_version,
        "de_alt_count": DE_ALT_COUNT,
        "fi_alt_count": FI_ALT_COUNT,
        "de_subsumed": de_sub,
        "fi_subsumed": fi_sub,
        "cases": cases,
    }


def render():
    require_runtime()
    cases = collect()
    de_sub, fi_sub = mutation(cases)
    return document(cases, de_sub, fi_sub)


def main():
    doc = render()
    text = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
    check = "--check" in sys.argv[1:]
    if check:
        with open(OUT, encoding="utf-8") as f:
            old = f.read()
        if old != text:
            sys.stderr.write("lang_cases.json is stale; run tools/notes/lang_cases.py\n")
            sys.exit(1)
        print(
            f"lang_cases.json ok: {len(doc['cases'])} cases, "
            f"de_subsumed={doc['de_subsumed']}"
        )
        return 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(
        f"wrote {OUT}: {len(doc['cases'])} cases, "
        f"de_alts={doc['de_alt_count']} fi_alts={doc['fi_alt_count']} "
        f"de_subsumed={doc['de_subsumed']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
