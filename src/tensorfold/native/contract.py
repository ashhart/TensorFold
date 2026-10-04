"""The native engine's contract: its capabilities and the gate manifest, checked against the JSON schemas here."""

from __future__ import annotations

import functools
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from tensorfold import native

PROBE_SECONDS = 15          # the contract asks for two; past this a hung binary leaves the Python engine serving
PROBE_BYTES = 1 << 20
_KEYWORDS = {"$schema", "title", "description", "type", "const", "enum", "pattern", "properties", "required",
             "additionalProperties", "propertyNames", "items", "uniqueItems"}
_TYPES = {"object": dict, "array": list, "string": str, "integer": int, "boolean": bool, "null": type(None)}


@functools.cache
def schema(name: str) -> dict[str, Any]:
    """The ``capabilities`` or ``gate`` JSON schema shipped beside this module."""

    return json.loads(Path(__file__).with_name(f"{name}.schema.json").read_text())


def check(value: Any, rules: dict[str, Any], at: str = "$") -> None:
    """Raise ValueError naming where ``value`` breaks ``rules``, a JSON schema that uses only _KEYWORDS."""

    if set(rules) - _KEYWORDS:
        raise ValueError(f"{at}: unsupported schema keywords {sorted(set(rules) - _KEYWORDS)}")
    types = [rules["type"]] if isinstance(rules.get("type"), str) else rules.get("type", [])
    if types and not any(isinstance(value, _TYPES[t]) and not (t == "integer" and isinstance(value, bool))
                         for t in types):
        raise ValueError(f"{at}: expected {' or '.join(types)}")
    if "const" in rules and not _same(value, rules["const"]):
        raise ValueError(f"{at}: expected {json.dumps(rules['const'])}")
    if "enum" in rules and not any(_same(value, option) for option in rules["enum"]):
        raise ValueError(f"{at}: expected one of {json.dumps(rules['enum'])}")
    if isinstance(value, str) and "pattern" in rules and not re.search(rules["pattern"], value):
        raise ValueError(f"{at}: {value!r} does not match {rules['pattern']}")
    if isinstance(value, dict):
        missing = [key for key in rules.get("required", []) if key not in value]
        if missing:
            raise ValueError(f"{at}: missing {', '.join(missing)}")
        for key, item in value.items():
            if "propertyNames" in rules:
                check(key, rules["propertyNames"], f"{at}.{key}")
            inner = rules.get("properties", {}).get(key, rules.get("additionalProperties", True))
            if inner is False:
                raise ValueError(f"{at}: unexpected key {key!r}")
            if isinstance(inner, dict):
                check(item, inner, f"{at}.{key}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            check(item, rules.get("items", {}), f"{at}[{index}]")
        if rules.get("uniqueItems") and len({json.dumps(item, sort_keys=True) for item in value}) < len(value):
            raise ValueError(f"{at}: repeated items")


def _same(value: Any, expected: Any) -> bool:
    return value == expected and isinstance(value, bool) == isinstance(expected, bool)    # JSON: true is not 1


def manifest() -> dict[str, Any]:
    """The gate manifest this install carries (none: no entries); ValueError when it breaks its schema."""

    path = native.ROOT / native.MANIFEST
    if not path.is_file():
        return {"schema": 1, "entries": []}
    document = json.loads(path.read_text())
    check(document, schema("gate"))
    return document


def capabilities(binary: Path) -> dict[str, Any]:
    """What ``binary capabilities --json`` reports, checked against its schema; ValueError says why it can't be used."""

    try:
        done = subprocess.run([str(binary), "capabilities", "--json"], stdin=subprocess.DEVNULL, capture_output=True,
                              timeout=PROBE_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"it could not run ({type(exc).__name__})") from None
    if done.returncode != 0:
        raise ValueError(f"it exited with status {done.returncode}")
    if len(done.stdout) > PROBE_BYTES:
        raise ValueError("its answer is over 1 MiB")
    try:
        document = json.loads(done.stdout)
    except ValueError:
        raise ValueError("its answer is not JSON") from None
    check(document, schema("capabilities"))
    return document


def weight_format(config: dict[str, Any]) -> str:
    """A gate cell's weight format: mlx-q<bits>g<group> plus its other layer widths, another method, or unquantized."""

    from tensorfold import families

    method = families.quant_method(config)
    if method is None:
        return "unquantized"
    block = next((source[key] for source in (config, config.get("text_config") or {})
                  for key in ("quantization", "quantization_config") if isinstance(source.get(key), dict)
                  and source[key]), {})
    if method == families.MLX_QUANT:
        bits, group = families.quantization(config)
        base = f"q{bits}g{group}"
        layers = {f"q{b}g{g}" if mode == "affine" else f"{mode}g{g}"
                  for b, g, mode in families.layer_quantization(config).values()}
        layers |= {"bf16"} if any(entry is False for entry in block.values()) else set()
        return "-".join(["mlx", base, *sorted(layers - {base})])
    algorithm, bits = block.get("quant_algo") or block.get("format"), block.get("bits")
    detail = [str(algorithm) if algorithm else "", f"b{bits}" if bits else ""]
    return re.sub(r"[^a-z0-9]+", "-", "-".join([method, *filter(None, detail)]).lower()).strip("-")
