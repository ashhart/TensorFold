"""Read advertised capabilities without assigning measurement grades."""
import json
import re
from pathlib import Path

MAX_BYTES = 4 * 1024 * 1024
MAX_FACTS = 4096
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}\Z")
CHIP = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+() -]{0,127}\Z")


class CapabilitiesError(ValueError):
    """A fixed diagnostic for invalid capabilities input."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CapabilitiesError("capabilities input contains duplicate JSON keys")
        result[key] = value
    return result


def _token(value):
    if not isinstance(value, str) or not TOKEN.fullmatch(value):
        raise CapabilitiesError("capabilities input contains an invalid name")
    return value


def _invalid_number(value):
    raise CapabilitiesError("capabilities input contains an invalid JSON number")


def _chip(value):
    if not isinstance(value, str) or not CHIP.fullmatch(value):
        raise CapabilitiesError("capabilities input contains an invalid chip label")
    return value


def _list(value, maximum):
    if not isinstance(value, list) or len(value) > maximum:
        raise CapabilitiesError("capabilities input contains an invalid or oversized list")
    return value


def _native(data):
    families = data.get("families")
    if not isinstance(families, dict) or len(families) > 256:
        raise CapabilitiesError("capabilities schema 1 requires a bounded family object")
    facts = []
    for family, formats in families.items():
        family = _token(family)
        for fmt in _list(formats, 64):
            facts.append((family, None, "format:" + _token(fmt)))
    for backend in _list(data.get("backends"), 64):
        facts.append((None, _token(backend), "backend"))
    if data.get("chip") is not None:
        facts.append((None, None, "chip:" + _chip(data["chip"])))
    return facts


def _legacy(families):
    facts = []
    for family in _list(families, 256):
        if not isinstance(family, dict):
            raise CapabilitiesError("capabilities input contains an invalid family")
        name = _token(family.get("family") or family.get("name"))
        backend = _token(family["backend"]) if "backend" in family else None
        for cap in _list(family.get("capabilities", []), 64):
            if not isinstance(cap, dict):
                raise CapabilitiesError("capabilities input contains an invalid capability")
            facts.append((name, backend, _token(cap.get("name"))))
    return facts


def read_facts(path):
    """Return family/backend/capability tuples from native schema 1 or the legacy list."""
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
    except OSError:
        raise CapabilitiesError("capabilities input could not be read") from None
    if len(raw) > MAX_BYTES:
        raise CapabilitiesError("capabilities input exceeds the 4 MiB limit")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_invalid_number)
    except CapabilitiesError:
        raise
    except (ValueError, RecursionError):
        raise CapabilitiesError("capabilities input is not valid UTF-8 JSON") from None
    if isinstance(data, dict) and "schema" in data:
        if type(data["schema"]) is not int or data["schema"] != 1:
            raise CapabilitiesError("capabilities input uses an unsupported schema")
        facts = _native(data)
    elif isinstance(data, list):
        facts = _legacy(data)
    elif isinstance(data, dict) and isinstance(data.get("families"), list):
        facts = _legacy(data["families"])
    else:
        raise CapabilitiesError("capabilities input requires native schema 1 or a legacy family list")
    if len(facts) > MAX_FACTS:
        raise CapabilitiesError("capabilities input exceeds the 4096 fact limit")
    return list(dict.fromkeys(facts))
