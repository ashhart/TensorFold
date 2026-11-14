"""Discover explicit thinking-off support once and label the requested benchmark mode."""
import json
import urllib.request

from bench_auth import authorize


def discover(base: str, model: str, api_key: str | None = None) -> dict:
    """Only consistent booleans for this model are trusted; older servers remain unknown."""
    out = {"thinking_switch": None, "thinking_applicable": None}
    try:
        request = urllib.request.Request(base.rstrip("/") + "/v1/models", headers={"Accept": "application/json"})
        authorize(request, api_key)
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(1_048_577)
        if len(raw) > 1_048_576:
            return out
        document = json.loads(raw)
        if not isinstance(document, dict) or not isinstance(document.get("data"), list):
            return out
        entries = [entry.get("capabilities", {}) for entry in document["data"]
                   if isinstance(entry, dict) and entry.get("id") == model]
        for field in out:
            bits = [entry.get(field) if isinstance(entry, dict) else None for entry in entries]
            if bits and all(type(bit) is bool and bit is bits[0] for bit in bits):
                out[field] = bits[0]
    except (OSError, ValueError, TypeError):
        pass
    return out


def configure(items: list[dict], capabilities: dict | bool | None) -> list[dict]:
    """Keep shared prompt definitions unchanged and retain unknown applicability for older servers."""
    values = capabilities if isinstance(capabilities, dict) else {"thinking_switch": capabilities}
    return [{**item, **values} for item in items]


def mode(item: dict) -> str:
    """Distinguish a verified off request, an unverified off request and a known unsupported switch."""
    if item["kind"] != "chat" or item.get("thinking_applicable") is False:
        return "not applicable"
    switch = item.get("thinking_switch")
    if switch is True:
        return "off"
    return "default (thinking off unsupported)" if switch is False else "off requested (unverified)"


def request_off(item: dict) -> bool:
    return item["kind"] == "chat" and item.get("thinking_applicable") is not False and item.get("thinking_switch") is not False


def receipt(item: dict) -> dict:
    return {"thinking": mode(item), "thinking_off_requested": request_off(item),
            "thinking_switch": item.get("thinking_switch"), "thinking_applicable": item.get("thinking_applicable")}


def load_comparison(path: str | None) -> list[dict]:
    if path is None:
        return []
    with open(path, encoding="utf-8") as source:
        value = json.load(source)
    rows = value.get("cells") if isinstance(value, dict) else value
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("comparison receipt must hold a row list or cells list")
    return rows


def comparison(row: dict, previous: list[dict]) -> dict:
    def key(value):
        prompt = value.get("prompt")
        if prompt == "gpu-chat":
            prompt = "gpu-chat-no-think"
        return prompt, value.get("temperature"), value.get("streams")
    def label(value):
        modes = value.get("thinking_modes")
        return ", ".join(modes) if isinstance(modes, list) else value.get("thinking", "unknown")
    matches = [old for old in previous if key(old) == key(row)]
    if not matches:
        return row
    same = all(label(old) == label(row) for old in matches)
    out = {**row, "thinking_comparable": same}
    if not same:
        out["comparison_warning"] = "Thinking labels differ; do not compare these rates"
        out["comparison_thinking"] = sorted({label(old) for old in matches})
    return out
