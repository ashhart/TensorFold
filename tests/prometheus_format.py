"""A strict reader of the Prometheus text exposition format (0.0.4) both servers publish at /metrics.

Every line must be blank, a ``# HELP <name> <text>`` line, a ``# TYPE <name> <kind>`` line, or a sample
``<name> [{labels}] <value> [timestamp]``. HELP and TYPE must name the family and precede its first sample;
a histogram's series extend the family name with ``_bucket``/``_sum``/``_count``; a label value carries no raw
quote (0.0.4 backslash escapes do); and a family's series repeat at most once.
"""

from __future__ import annotations

import re

NAME = r"[a-zA-Z_:][a-zA-Z0-9_:]*"
_QUOTED = r"\"(?:[^\"\\]|\\.)*\""        # a 0.0.4 quoted label value: backslash escapes inside
LABEL_PAIR = r"([a-zA-Z_][a-zA-Z0-9_]*)=\"((?:[^\"\\]|\\.)*)\""
LABELS = r"\{" + r"[a-zA-Z_][a-zA-Z0-9_]*=" + _QUOTED + r"(?:,[ ]?[a-zA-Z_][a-zA-Z0-9_]*=" + _QUOTED + r")*\}"
VALUE = r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[-+]Inf|NaN)"
KINDS = {"counter", "gauge", "histogram", "summary", "untyped"}
SERIES_SUFFIX = ("_bucket", "_sum", "_count")


class FormatError(AssertionError):
    """The exposition breaks the 0.0.4 format; the message names the offending line."""


def _unquote(value: str) -> str:
    """Undo the 0.0.4 escapes: a backslash stands for itself or for a quote."""

    return value.replace('\\"', '"').replace("\\\\", "\\")


def _family_of(name: str, types: dict[str, str]) -> str | None:
    if name in types:
        return name
    for suffix in SERIES_SUFFIX:
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if base in types:
                return base
    return None


def parse(text: str) -> dict[str, dict]:
    """The exposition as {family: {"type": kind, "help": str, "series": {(name, labels): value}}}."""

    help_lines: dict[str, str] = {}
    types: dict[str, str] = {}
    opened: set[str] = set()
    families: dict[str, dict] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            parts = line.split(" ")
            if len(parts) < 3 or parts[1] not in ("HELP", "TYPE"):
                raise FormatError(f"unrecognised comment line: {line!r}")
            name = parts[2]
            if not re.fullmatch(NAME, name):
                raise FormatError(f"bad metric name in {line!r}")
            if parts[1] == "HELP":
                if name in help_lines:
                    raise FormatError(f"duplicate HELP for {name}")
                if len(parts) < 4 or not " ".join(parts[3:]).strip():
                    raise FormatError(f"empty HELP for {name}")
                help_lines[name] = " ".join(parts[3:])
            else:
                if len(parts) != 4 or parts[3] not in KINDS:
                    raise FormatError(f"bad TYPE line: {line!r}")
                kind = parts[3]
                if name in types and types[name] != kind:
                    raise FormatError(f"conflicting TYPE for {name}")
                types[name] = kind
            continue
        match = re.fullmatch(rf"({NAME})(?:({LABELS}))?[ ]+({VALUE})([ ]+(?:-?\d+))?", line)
        if match is None:
            raise FormatError(f"unrecognised sample line: {line!r}")
        name, raw_labels, value, _timestamp = match.groups()
        family = _family_of(name, types)
        if family is None:
            raise FormatError(f"sample before its TYPE line: {line!r}")
        record = families.setdefault(family, {"type": types[family], "help": help_lines.get(family, ""),
                                               "series": {}, "order": []})
        if family not in opened:
            opened.add(family)
            if family not in help_lines or family not in types:
                raise FormatError(f"first sample of {family} precedes its HELP/TYPE lines")
        labels = {k: _unquote(v) for k, v in re.findall(LABEL_PAIR, raw_labels)} if raw_labels else {}
        if raw_labels and re.sub(LABEL_PAIR, "", raw_labels[1: -1]) not in ("", ",", ", "):
            raise FormatError(f"bad labelset in {line!r}")
        key = (name, tuple(sorted(labels.items())))
        if key in record["series"]:
            raise FormatError(f"repeated series: {line!r}")
        record["series"][key] = {"labels": labels, "value": float(value)}
        record["order"].append(key)
    for family, record in families.items():
        record["help"] = help_lines[family]
    return families


def series_value(parsed: dict[str, dict], name: str, labels: dict[str, str] | None = None) -> float:
    """A series' value: ``name`` the full series name (``_bucket`` etc. included), ``labels`` its labels or None."""

    wanted = labels or {}
    for record in parsed.values():
        for (series_name, label_pairs), entry in record["series"].items():
            if series_name == name and dict(label_pairs) == wanted:
                return entry["value"]
    rendered = name + ("{" + ", ".join(f'{k}="{v}"' for k, v in sorted(wanted.items())) + "}" if wanted else "")
    raise FormatError(f"series not found: {rendered}")


def check_histograms(parsed: dict[str, dict]) -> None:
    """A histogram's buckets are cumulative: their counts never drop as ``le`` rises, and the +Inf bucket
    holds every observation, which the ``_count`` series must agree with."""

    for family, record in parsed.items():
        if record["type"] != "histogram":
            continue
        counts: list[tuple[float, float]] = []
        for (series_name, label_pairs), entry in record["series"].items():
            if series_name != f"{family}_bucket":
                continue
            labels = dict(label_pairs)
            if set(labels) != {"le"}:
                raise FormatError(f"bucket series with unexpected labels: {entry!r}")
            bound = float("inf") if labels["le"] == "+Inf" else float(labels["le"])
            counts.append((bound, entry["value"]))
        if not counts:
            raise FormatError(f"histogram {family} has no bucket series")
        counts.sort()
        for (low, low_hits), (high, high_hits) in zip(counts, counts[1:]):
            assert low < high, f"bucket bounds are not ascending in {family}"
            assert low_hits <= high_hits, f"bucket counts drop from {low_hits} to {high_hits} in {family}"
        total = None
        for (series_name, label_pairs), entry in record["series"].items():
            if series_name == f"{family}_count" and not label_pairs:
                total = entry["value"]
        if total is None:
            raise FormatError(f"histogram {family} has no _count series")
        assert counts[-1][0] == float("inf"), f"the last bucket of {family} must be +Inf"
        assert counts[-1][1] == total, f"{family}'s +Inf bucket ({counts[-1][1]}) disagrees with its count ({total})"


__all__ = ["KINDS", "FormatError", "check_histograms", "parse", "series_value"]
