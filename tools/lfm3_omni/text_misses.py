"""Teacher-forced text misses with their margins: text_misses.py REF_DIR TF_DIR [OUT_JSON] [SCORES_NAME]."""
import json
import math
import re
import sys
from pathlib import Path

MISS = re.compile(r"MISS text slot (\d+) ref (\d+) pick (\d+) ref_logit (\S+) pick_logit (\S+)")


def ulp(x):
    """A bf16 unit in the last place at x (8 significant bits)."""
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x else 0.0


def misses(ref, tf, name="forced_bf16.json"):
    """Each miss of TF_DIR/<u>/forced.log beside the reference's own logits of the two tokens (reference_forced.py)."""
    out = []
    for d in sorted(p for p in tf.iterdir() if (p / "forced.log").exists()):
        slots = {s["slot"]: s for s in json.load(open(ref / d.name / name))["slots"]}
        for m in MISS.finditer(open(d / "forced.log").read()):
            slot, want, pick = (int(m.group(i)) for i in (1, 2, 3))
            mine = [float(m.group(4)), float(m.group(5))]
            top = dict(zip(slots[slot]["top_ids"], slots[slot]["top_logits"]))
            row = {"utterance": d.name, "slot": slot, "reference token": want, "engine token": pick,
                   "engine logits (reference token, its pick)": mine,
                   "engine margin, bf16 ulps": round((mine[1] - mine[0]) / ulp(mine[1]), 2)}
            if slots[slot]["ref"] != want or want not in top or pick not in top:
                row["reference"] = "the engine's pick is outside the reference's top 4"
            else:
                row["reference logits (its pick, engine token)"] = [top[want], top[pick]]
                row["reference margin, bf16 ulps"] = round((top[want] - top[pick]) / ulp(top[want]), 2)
            out.append(row)
    return out


if __name__ == "__main__":
    res = misses(Path(sys.argv[1]), Path(sys.argv[2]), *sys.argv[4:5])
    print(json.dumps(res, indent=1))
    if len(sys.argv) > 3:
        json.dump(res, open(sys.argv[3], "w"), indent=1)
