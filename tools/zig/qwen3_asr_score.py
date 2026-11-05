"""WER (English) and CER (Chinese) of tf-qwen3-asr transcripts against the package's reference and the gold text."""
# usage: python3 -I tools/zig/qwen3_asr_score.py CLIPS_DIR REF_DIR OURS_DIR [--json OUT] (see --help)

import argparse
import json
import os
import re
import sys
import unicodedata


def body(reply):
    """The transcript after the reply's `language X<asr_text>` head."""
    return reply.split("<asr_text>", 1)[1] if "<asr_text>" in reply else reply


def words(text):
    """English words as LibriSpeech's gold has them: upper case, apostrophes kept, other punctuation dropped."""
    t = unicodedata.normalize("NFKC", text).upper().replace("-", " ")
    t = re.sub(r"[^A-Z0-9' ]+", " ", t)
    out = [w.strip("'") for w in t.split() if w.strip("'")]
    return [SPOKEN.get(w, w) for w in out]


# LibriSpeech's gold spells titles out as read; the model writes the abbreviations
SPOKEN = {"MR": "MISTER", "MRS": "MISSUS", "DR": "DOCTOR"}


def chars(text):
    """Chinese characters, letters and digits: punctuation, spaces and FLEURS's unspoken (gloss) brackets dropped."""
    t = unicodedata.normalize("NFKC", text).upper()
    t = re.sub(r"\([^()]*\)", "", t)  # NFKC turns （ ） into ( ): "埃尔多安（Recep Tayyip Erdoğan）" is read as 埃尔多安
    return [c for c in t if not (unicodedata.category(c)[0] in "PZS" or c.isspace())]


def edits(a, b):
    """Levenshtein distance between two sequences."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def ours(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("TEXT "):
                return line[5:].rstrip("\n")
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clips", help="<id>.wav and <id>.txt (gold); manifest.json lists the ids and their set")
    p.add_argument("ref", help="<id>.win/manifest.json from tools/zig/qwen3_asr_case.py (its text)")
    p.add_argument("ours", help="<id>.txt: tf-qwen3-asr's output (its TEXT line)")
    p.add_argument("--suffix", default=".txt", help="the outputs' file ending, e.g. .r1.txt for a bench round")
    p.add_argument("--json")
    a = p.parse_args()
    with open(os.path.join(a.clips, "manifest.json"), encoding="utf-8") as f:
        clips = json.load(f)
    totals = {}  # (set, against) -> [edits, length]
    rows = []
    for c in clips:
        unit = words if c["set"] == "ls" else chars
        hyp = ours(os.path.join(a.ours, c["id"] + a.suffix))
        if hyp is None:
            sys.exit(f"{c['id']}: no TEXT line in {a.ours}")
        with open(os.path.join(a.ref, c["id"] + ".win", "manifest.json"), encoding="utf-8") as f:
            ref = json.load(f)["text"]
        with open(os.path.join(a.clips, c["id"] + ".txt"), encoding="utf-8") as f:
            gold = f.read().strip()
        h = unit(body(hyp))
        row = {"id": c["id"], "set": c["set"]}
        for name, text in (("reference", body(ref)), ("gold", gold)):
            r = unit(text)
            e = edits(r, h)
            t = totals.setdefault((c["set"], name), [0, 0])
            t[0] += e
            t[1] += len(r)
            row[name] = {"edits": e, "length": len(r)}
        # the package against the gold too: our error rate beside the reference's own
        r, g = unit(body(ref)), unit(gold)
        t = totals.setdefault((c["set"], "package vs gold"), [0, 0])
        t[0] += edits(g, r)
        t[1] += len(g)
        rows.append(row)
    print(f"{'set':<8} {'against':<16} {'rate':>8} {'edits':>6} {'units':>6}")
    out = {"rows": rows, "totals": {}}
    for (s, name), (e, n) in sorted(totals.items()):
        kind = "WER" if s == "ls" else "CER"
        print(f"{s:<8} {name:<16} {kind} {100 * e / max(n, 1):5.2f}% {e:>6} {n:>6}")
        out["totals"][f"{s} {name}"] = {"metric": kind, "edits": e, "units": n, "rate": e / max(n, 1)}
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
