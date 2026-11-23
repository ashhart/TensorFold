"""Read one streamed reply: content, reasoning and tool-call events per choice, timed on the client's clock.

Boundaries are seconds after the request was sent, and only for a reply that reached ``[DONE]``:
``headers_s`` (status and headers returned), ``first_payload_s`` (the first non-empty body line; the clock reads
lines, so this bounds the first byte from above), ``first_content_s``, ``first_reasoning_s``, ``first_tool_s``
(tool name or argument text), ``first_output_s`` (the earliest of those) and ``last_output_s`` (the last event that
carried output; a later usage-only chunk is not output). Every event of one line shares one clock reading, taken
when the line has been read and parsed (for the first line, before parsing).
``completion`` classifies how the reply ended: ``error``, ``cancelled``, ``incomplete`` (closed before ``[DONE]``),
``no_output`` (usage only), ``truncated`` (a choice finished with ``length``), ``finished`` (every choice has a
finish reason) or ``unfinished``. ``complete`` stays as strict as before: ``[DONE]`` seen and no error.
"""
import json
import time

from bench_evidence import Evidence

KINDS = ("content", "reasoning", "tool")
BOUNDARIES = ("headers_s", "first_payload_s", *(f"first_{k}_s" for k in KINDS), "first_output_s", "last_output_s")
EMPTY = {**dict.fromkeys(BOUNDARIES), "completion": "error", "finish_reasons": None, "choices": None}


class Collector:
    """One reply's events; ``read`` takes the response's lines and stops at ``[DONE]``, an error or a cancel."""

    def __init__(self, sent: float):
        self.clock = time.perf_counter
        self.sent, self.headers, self.payload = sent, None, None
        self.first = dict.fromkeys(KINDS)
        self.pieces: list[tuple[float, int]] = []    # (arrival, characters) of each text or reasoning piece
        self.outputs: list[tuple[float, int]] = []   # (arrival, choice index) of each choice delta with output
        self.text: list[str] = []
        self.evidence = Evidence()
        self.usage, self.runtime, self.speculative, self.finish = {}, {}, {}, {}
        self.end = self.error = None

    def read(self, lines, cancel=None) -> None:
        self.headers = self.clock()
        for raw in lines:
            if cancel is not None and cancel.is_set():
                self.end = "cancelled"
                return
            line = raw.decode().strip()
            if not line:
                continue
            now = None
            if self.payload is None:
                now = self.payload = self.clock()
            if line == "data: [DONE]":
                self.end = "done"
                return
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:])
            if "error" in chunk:
                self.end, self.error = "error", chunk["error"]
                return
            self.usage = chunk.get("usage") or self.usage
            self.runtime = chunk.get("tensorfold") or self.runtime
            self.speculative = chunk.get("speculative") or self.speculative
            for choice in chunk.get("choices", []):
                kinds = self.evidence.add(choice)
                index = choice.get("index", 0)
                self.finish[index] = choice.get("finish_reason") or self.finish.get(index)
                if not kinds:
                    continue
                now = now if now is not None else self.clock()
                for kind in kinds:
                    self.first[kind] = self.first[kind] if self.first[kind] is not None else now
                self.outputs.append((now, index))
                delta = choice.get("delta") or {}
                content = choice.get("text") or delta.get("content") or ""
                piece = content or delta.get("reasoning_content") or ""
                if piece:
                    self.pieces.append((now, len(piece)))
                    self.text.append(content)

    def completion(self) -> str:
        if self.end in ("error", "cancelled"):
            return self.end
        if self.end is None:
            return "incomplete"
        if not self.outputs:
            return "no_output"
        reasons = list(self.finish.values())
        if "length" in reasons:
            return "truncated"
        return "finished" if all(reasons) else "unfinished"

    def receipt(self) -> dict:
        """Boundaries, completion class and finish reasons; ``output_times`` only for a one-choice reply."""
        done = self.end == "done"
        times = [t for t, _ in self.outputs]
        at = {"headers_s": self.headers, "first_payload_s": self.payload,
              **{f"first_{k}_s": self.first[k] for k in KINDS},
              "first_output_s": times[0] if times else None, "last_output_s": times[-1] if times else None}
        return {**{k: v - self.sent if done and v is not None else None for k, v in at.items()},
                "completion": self.completion(), "choices": len(self.finish), "speculative": self.speculative,
                "finish_reasons": [self.finish[i] for i in sorted(self.finish)],
                "output_times": times if len({i for _, i in self.outputs}) <= 1 else None}


def overhead(chunks: int = 4000, delta: dict | None = None) -> float:
    """The collector's own cost in microseconds a chunk, reading a synthetic one-delta-a-line stream from memory."""
    body = json.dumps({"choices": [{"index": 0, "delta": delta or {"content": " token"}}]}).encode()
    lines = [b"data: " + body + b"\n", b"\n"] * chunks + [b"data: [DONE]\n"]
    start = time.perf_counter()
    reply = Collector(start)
    reply.read(lines)
    reply.evidence.finish({}, {}, True)
    return (time.perf_counter() - start) * 1e6 / chunks
