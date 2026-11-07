"""Prompt-cache series from Prometheus text exposition (name, optional labels, value), without a client library."""
import math
import re

SAMPLE = re.compile(r"([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?[ \t]+(\S+)(?:[ \t]+-?[0-9]+)?")
LABEL = re.compile(r'[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*=[ \t]*("(?:[^"\\]|\\.)*")[ \t]*(?:,[ \t]*|$)')
CACHE = re.compile(r"pr(?:ompt|efix)_?cache|prompt_tokens_cached", re.IGNORECASE)


def labels(text: str) -> str | None:
    """`{a="x",b="y"}` with the pairs sorted by name ('' for none); None when the text isn't name="value" pairs."""
    pairs, at = [], 0
    while at < len(text):
        m = LABEL.match(text, at)
        if not m:
            return None
        pairs.append(f"{m.group(1)}={m.group(2)}")
        at = m.end()
    return "{" + ",".join(sorted(pairs)) + "}" if pairs else ""


def series(text: str | None, keep: re.Pattern = CACHE) -> dict:
    """Each sample whose metric name `keep` finds, keyed by name and labels; non-finite values are None."""
    out = {}
    for line in (text or "").splitlines():
        m = SAMPLE.fullmatch(line.strip())
        tags = labels(m.group(2) or "") if m and keep.search(m.group(1)) else None
        if tags is None:
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        out[m.group(1) + tags] = None if not math.isfinite(value) else int(value) if value.is_integer() else value
    return out


def change(before: dict, after: dict) -> dict | None:
    """Both snapshots and the change of each series known in both; None when neither has a series."""
    if not before and not after:
        return None
    delta = {k: v - before[k] for k, v in after.items() if v is not None and before.get(k) is not None}
    return {"before": before, "after": after, "delta": delta}
