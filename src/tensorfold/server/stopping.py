"""Request stop strings remain active when model EOS stopping is disabled."""

from typing import Any

from tensorfold.server.errors import RequestError
from tensorfold.server.text import hide_tool_calls


def stop_options(fields: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    ignore = fields.get("ignore_eos", False)
    if not isinstance(ignore, bool):
        raise RequestError("ignore_eos must be a boolean")
    stops = fields.get("stop")
    if stops is None:
        return ignore, ()
    if isinstance(stops, str):
        stops = [stops]
    if not isinstance(stops, list) or any(not isinstance(stop, str) or not stop for stop in stops):
        raise RequestError("stop must be a nonempty string, a list of nonempty strings, or null")
    return ignore, tuple(stops)


class StopPolicy:
    def __init__(self, fields: dict[str, Any], tokenizer: Any, lock: Any, eos_ids: frozenset[int]) -> None:
        self.ignore_eos, self.strings = stop_options(fields)
        self.eos_ids = frozenset() if self.ignore_eos else eos_ids
        self.tokenizer, self.lock = tokenizer, lock
        self.tail = max((len(stop) for stop in self.strings), default=0) + 8

    def __call__(self, tokens: list[int]) -> bool:
        # checked after every token, so a new match ends in the newest tokens: decode only that tail
        with self.lock:
            text = self.tokenizer.decode(tokens[-self.tail:])
        return any(stop in text for stop in self.strings)

    def visible(self, text: str, *, partial: bool = False) -> str:
        ends = [at for stop in self.strings if (at := text.find(stop)) >= 0]
        if ends:
            return text[:min(ends)]
        if partial:
            held = max((size for stop in self.strings for size in range(1, min(len(stop), len(text) + 1))
                        if text.endswith(stop[:size])), default=0)
            return text[:len(text) - held]
        return text

    def flush(self, callback: Any, content: str, reasoning: str | None, sent: str, thought: str, tools: bool) -> None:
        """Stream what the final reply adds to what streamed: text held back as a possible stop string or tag."""

        if callback is None:
            return
        if reasoning and reasoning.startswith(thought) and len(reasoning) > len(thought):
            callback({"reasoning_content": reasoning[len(thought):]})
        shown = hide_tool_calls(content, finished=True) if tools else content
        if shown.startswith(sent) and len(shown) > len(sent):
            callback(shown[len(sent):])


def matched_stop(text: str, strings: tuple[str, ...]) -> str | None:
    """The first matched stop, ties in request order, before visible() strips it."""
    matches = [(text.find(stop), at, stop) for at, stop in enumerate(strings) if stop in text]
    return min(matches)[2] if matches else None
