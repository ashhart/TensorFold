"""The parity corpus for a checkpoint's own tokenizer and chat template: rendering, token routes and decoding."""

from __future__ import annotations

from typing import Any

from corpus import TOOLS, post

HISTORY = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "find it"},
           {"role": "assistant", "content": "", "reasoning_content": "need a lookup", "tool_calls": [
               {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"query\": \"x\"}"}}]},
           {"role": "tool", "tool_call_id": "call_1", "content": "found it"}, {"role": "user", "content": "thanks @script=default"}]


def say(script: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": f"Hi there é @script={script}"}]


def cases(encode: Any) -> list[tuple[str, str, list[Any]]]:
    """Tokenize and detokenize, count_tokens and short replies decoded by the checkpoint's tokenizer."""
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    tokenize = {"prompt": {"prompt": "Hello world é ☕ 日本"}, "prompt-plain": {"prompt": "Hello", "add_special_tokens": False},
                "strs": {"prompt": "Hello world", "return_token_strs": True}, "messages": {"messages": say("default")},
                "messages-off": {"messages": say("default"), **off}, "messages-low": {"messages": say("default"), "reasoning_effort": "low"},
                "messages-high": {"messages": say("default"), "reasoning_effort": "high"}, "tools": {"messages": say("tool"), "tools": TOOLS},
                "history": {"messages": HISTORY, "tools": TOOLS}, "history-no-gen": {"messages": HISTORY, "add_generation_prompt": False}}
    out = [("real-tokens", f"tokenize-{k}", [post("/tokenize", v)]) for k, v in tokenize.items()]
    out.append(("real-tokens", "detokenize", [post("/detokenize", {"tokens": encode("café ☕ 日本語 🚀 <think>x</think>")})]))
    out.append(("real-tokens", "count", [post("/v1/messages/count_tokens", {"model": "m", "messages": say("default"), "tools": [
        {"name": "lookup", "input_schema": TOOLS[0]["function"]["parameters"]}]})]))
    for script in ("default", "think", "unicode", "tool", "stop"):
        for stream in (False, True):
            body = {"messages": say(script), "stream": stream, **({"tools": TOOLS} if script == "tool" else {}),
                    **({"stop": ["four"]} if script == "stop" else {})}
            out.append(("real-chat", f"{script}{'-stream' if stream else ''}", [post("/v1/chat/completions", body)]))
            out.append(("real-chat", f"{script}-off{'-stream' if stream else ''}", [post("/v1/chat/completions", {**body, **off})]))
    out.append(("real-chat", "history", [post("/v1/chat/completions", {"messages": HISTORY, "tools": TOOLS})]))
    out.append(("real-chat", "completion", [post("/v1/completions", {"prompt": "Once upon @script=unicode", "stream": True})]))
    out.append(("real-chat", "anthropic", [post("/v1/messages", {"model": "m", "max_tokens": 32, "messages": say("think"), "stream": True, "thinking": {"type": "adaptive"}})]))
    return out
