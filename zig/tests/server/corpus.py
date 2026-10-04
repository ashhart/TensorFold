"""The parity corpus: requests sent byte for byte to the Python and Zig servers, grouped by route and feature."""

from __future__ import annotations

import json
from typing import Any

from wire import request

TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Search",
                                            "parameters": {"type": "object", "properties": {
                                                "query": {"type": "string"}, "limit": {"type": "integer"}}}}}]


def say(script: str, text: str = "Hi") -> list[dict[str, Any]]:
    return [{"role": "user", "content": f"{text} @script={script}"}]


def chat(script: str, **fields: Any) -> dict[str, Any]:
    return {"model": "fake-model", "messages": say(script), **fields}


def post(path: str, body: Any, **kw: Any) -> bytes:
    return request("POST", path, body, {"Content-Type": "application/json"}, **kw)


def get(path: str, **headers: str) -> bytes:
    return request("GET", path, None, headers)


def chat_cases() -> list[tuple[str, str, list[Any]]]:
    out: list[tuple[str, str, list[Any]]] = []
    scripts = ["default", "plain", "think", "split_utf8", "harmony", "long", "no_end", "json_answer"]
    for s in scripts:
        for stream in (False, True):
            out.append(("chat", f"{s}{'-stream' if stream else ''}", [post("/v1/chat/completions", chat(s, stream=stream))]))
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    for s in ("think", "harmony", "default"):
        out.append(("thinking", f"{s}-off", [post("/v1/chat/completions", chat(s, **off))]))
        out.append(("thinking", f"{s}-off-stream", [post("/v1/chat/completions", chat(s, stream=True, **off))]))
    for effort in ("none", "low", "high", "max", "medium", "bogus"):
        out.append(("thinking", f"effort-{effort}", [post("/v1/chat/completions", chat("think", reasoning_effort=effort))]))
    out.append(("thinking", "deepseek-switch", [post("/v1/chat/completions", chat("think", chat_template_kwargs={"thinking": {"type": "disabled"}}))]))
    out.append(("thinking", "kwargs-effort", [post("/v1/chat/completions", chat("think", chat_template_kwargs={"reasoning_effort": "low"}))]))
    out.append(("thinking", "length-while-thinking", [post("/v1/chat/completions", chat("think_long", max_tokens=4))]))
    out.append(("thinking", "length-while-thinking-stream", [post("/v1/chat/completions", chat("think_long", max_tokens=4, stream=True))]))
    for stop in (["wor"], "STOP", ["zzz", "STOP"], ["ld STO"], ["worldx"]):
        tag = json.dumps(stop).replace('"', "")
        out.append(("stops", f"stop-{tag}", [post("/v1/chat/completions", chat("stop_words", stop=stop))]))
        out.append(("stops", f"stop-{tag}-stream", [post("/v1/chat/completions", chat("stop_words", stop=stop, stream=True))]))
    out.append(("stops", "stop-in-reasoning", [post("/v1/chat/completions", chat("think", stop=["think"]))]))
    out.append(("stops", "ignore-eos", [post("/v1/chat/completions", chat("eos_mid", ignore_eos=True))]))
    out.append(("stops", "eos-mid", [post("/v1/chat/completions", chat("eos_mid", stream=True))]))
    out.append(("usage", "include-usage", [post("/v1/chat/completions", chat("plain", stream=True, stream_options={"include_usage": True}))]))
    out.append(("usage", "max-tokens", [post("/v1/chat/completions", chat("long", max_tokens=3))]))
    out.append(("usage", "max-completion-tokens", [post("/v1/chat/completions", chat("long", max_completion_tokens=2))]))
    out.append(("usage", "max-tokens-zero", [post("/v1/chat/completions", chat("long", max_tokens=0))]))
    out.append(("usage", "max-tokens-string", [post("/v1/chat/completions", chat("long", max_tokens="5"))]))
    out.append(("usage", "sampled", [post("/v1/chat/completions", chat("plain", temperature=0.7, top_k=5))]))
    out.append(("usage", "seeded", [post("/v1/chat/completions", chat("plain", temperature=1, seed=-3, draft=False))]))
    out.append(("usage", "model-alias", [post("/v1/chat/completions", {**chat("plain"), "model": "fake-alias"})]))
    out.append(("usage", "model-unknown", [post("/v1/chat/completions", {**chat("plain"), "model": "other"})]))
    out.append(("usage", "background-title", [post("/v1/chat/completions", {"messages": [{"role": "system", "content": "Write a title"}, {"role": "user", "content": "x @script=plain"}]})]))
    return out


def tool_cases() -> list[tuple[str, str, list[Any]]]:
    out: list[tuple[str, str, list[Any]]] = []
    for s in ("qwen_tool", "qwen_tool_think", "xml_tool", "two_tools", "glm_tool", "gemma_tool", "dsml_tool", "bare_json", "bad_tool", "plain"):
        for stream in (False, True):
            out.append(("tools", f"{s}{'-stream' if stream else ''}", [post("/v1/chat/completions", chat(s, tools=TOOLS, stream=stream))]))
        out.append(("tools", f"{s}-single", [post("/v1/chat/completions", chat(s, tools=TOOLS, parallel_tool_calls=False))]))
        out.append(("tools", f"{s}-single-stream", [post("/v1/chat/completions", chat(s, tools=TOOLS, parallel_tool_calls=False, stream=True))]))
    named = {"type": "function", "function": {"name": "lookup"}}
    for choice in ("auto", "none", "required", named, {"type": "function", "function": {"name": "nope"}}, {"type": "function"}):
        out.append(("tools", f"choice-{json.dumps(choice)}", [post("/v1/chat/completions", chat("qwen_tool", tools=TOOLS, tool_choice=choice))]))
    out.append(("tools", "required-no-tools", [post("/v1/chat/completions", chat("plain", tool_choice="required"))]))
    out.append(("tools", "tools-not-list", [post("/v1/chat/completions", chat("plain", tools={"a": 1}))]))
    out.append(("tools", "tool-without-name", [post("/v1/chat/completions", chat("plain", tools=[{"type": "function", "function": {}}]))]))
    out.append(("tools", "parallel-not-bool", [post("/v1/chat/completions", chat("plain", tools=TOOLS, parallel_tool_calls="yes"))]))
    history = [{"role": "user", "content": "find it"}, {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"query\": \"x\"}"}}]},
               {"role": "tool", "tool_call_id": "call_1", "content": "found"}, {"role": "user", "content": "go @script=plain"}]
    out.append(("tools", "history-with-calls", [post("/v1/chat/completions", {"messages": history, "tools": TOOLS})]))
    bad_args = [dict(history[1], tool_calls=[{"id": "c", "type": "function", "function": {"name": "lookup", "arguments": "not json"}}]), history[3]]
    out.append(("tools", "history-bad-arguments", [post("/v1/chat/completions", {"messages": [history[0], *bad_args]})]))
    return out


def error_cases() -> list[tuple[str, str, list[Any]]]:
    bodies = {
        "not-json": b"{nope", "list-body": b"[]", "null-body": b"null", "empty-body": b"", "string-body": b'"x"',
        "bad-utf8": b'{"messages": "\xff"}', "trailing-comma": b'{"a": 1,}',
        "n-two": chat("plain", n=2), "logprobs": chat("plain", logprobs=True), "top-logprobs": chat("plain", top_logprobs=3),
        "temperature-text": chat("plain", temperature="abc"), "min-p": chat("plain", min_p=2), "top-k-float": chat("plain", top_k=1.5),
        "ignore-eos-null": chat("plain", ignore_eos=None), "stop-empty": chat("plain", stop=[""]), "stop-int": chat("plain", stop=5),
        "messages-missing": {"model": "x"}, "messages-empty": {"messages": []}, "role-robot": {"messages": [{"role": "robot", "content": "x"}]},
        "content-int": {"messages": [{"role": "user", "content": 5}]}, "content-image": {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
        "message-not-object": {"messages": ["hi"]}, "text-part-int": {"messages": [{"role": "user", "content": [{"type": "text", "text": 1}]}]},
        "modalities": chat("plain", modalities=["text", "audio"]), "audio-field": chat("plain", audio={"voice": "x"}),
        "format-json": chat("json_answer", response_format={"type": "json_object"}), "format-bogus": chat("plain", response_format={"type": "bogus"}),
        "format-text": chat("plain", response_format={"type": "text"}), "format-string": chat("plain", response_format="json"),
        "format-no-schema": chat("plain", response_format={"type": "json_schema", "json_schema": {}}),
        "guided-choice-empty": chat("plain", guided_choice=[]), "guided-json-bad": chat("plain", guided_json="{bad"),
        "structured-other": chat("plain", structured_outputs={"xml": 1}),
        "format-with-required": chat("plain", tools=TOOLS, tool_choice="required", response_format={"type": "json_object"}),
        "context-prompt": {"messages": [{"role": "user", "content": "x" * 1100}]}, "context-reply": chat("plain", max_tokens=2000),
        "system-merged": {"messages": [{"role": "system", "content": "A"}, {"role": "developer", "content": "B"}, {"role": "user", "content": "x @script=plain"}, {"role": "system", "content": "late"}]},
    }
    out = [("errors", name, [post("/v1/chat/completions", body)]) for name, body in bodies.items()]
    out.append(("errors", "stream-context", [post("/v1/chat/completions", chat("plain", max_tokens=2000, stream=True))]))
    out.append(("errors", "stream-structured", [post("/v1/chat/completions", chat("plain", response_format={"type": "json_object"}, stream=True))]))
    return out


def completion_cases(encode: Any) -> list[tuple[str, str, list[Any]]]:
    bodies = {
        "text": {"prompt": "Once @script=plain"}, "ids": {"prompt": encode("Once @script=think")},
        "bad-ids": {"prompt": [5, 99999]}, "bool-ids": {"prompt": [True, 1]}, "strings": {"prompt": ["a", "b @script=plain"]},
        "null": {"prompt": None}, "missing": {}, "messages": {"messages": say("think")}, "number": {"prompt": 12},
        "nested": {"prompt": [["x"], "y @script=default"]}, "max-tokens": {"prompt": "x @script=long", "max_tokens": 2},
    }
    out = [("completions", name, [post("/v1/completions", body)]) for name, body in bodies.items()]
    for name in ("text", "messages"):
        out.append(("completions", f"{name}-stream", [post("/v1/completions", {**bodies[name], "stream": True})]))
    out.append(("completions", "usage-chunk", [post("/v1/completions", {**bodies["text"], "stream": True, "stream_options": {"include_usage": True}})]))
    out.append(("completions", "think-script", [post("/v1/completions", {"prompt": "x @script=think", "stream": True})]))
    return out


def response_cases() -> list[tuple[str, str, list[Any]]]:
    def rid(index: int):
        return lambda replies: json.loads(replies[index]["body"])["id"]

    out = [
        ("responses", "input-string", [post("/v1/responses", {"model": "fake-model", "input": "Hi @script=think"})]),
        ("responses", "input-stream", [post("/v1/responses", {"input": "Hi @script=think", "stream": True})]),
        ("responses", "tools", [post("/v1/responses", {"input": "Hi @script=qwen_tool", "tools": [{"type": "function", "name": "lookup", "parameters": TOOLS[0]["function"]["parameters"]}]})]),
        ("responses", "tools-stream", [post("/v1/responses", {"input": "Hi @script=xml_tool", "stream": True, "tools": [{"type": "function", "name": "lookup"}]})]),
        ("responses", "incomplete", [post("/v1/responses", {"input": "Hi @script=long", "max_output_tokens": 2})]),
        ("responses", "instructions", [post("/v1/responses", {"input": [{"role": "user", "content": [{"type": "input_text", "text": "x @script=plain"}]}], "instructions": "Be brief", "metadata": {"k": "v"}, "temperature": 0, "store": False})]),
        ("responses", "items", [post("/v1/responses", {"input": [{"type": "message", "role": "user", "content": "find"}, {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "hmm"}]}, {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"}, {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_text", "text": "done @script=plain"}]}]})]),
        ("responses", "allowed-tools", [post("/v1/responses", {"input": "x @script=qwen_tool", "tools": [{"type": "function", "name": "lookup"}, {"type": "function", "name": "other"}], "tool_choice": {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "lookup"}]}})]),
        ("responses", "stored-chain", [post("/v1/responses", {"input": "one @script=plain"}),
                                       lambda r: post("/v1/responses", {"input": "two @script=think", "previous_response_id": rid(0)(r)}),
                                       lambda r: get(f"/v1/responses/{rid(1)(r)}"),
                                       lambda r: request("DELETE", f"/v1/responses/{rid(0)(r)}"),
                                       lambda r: post("/v1/responses", {"input": "three", "previous_response_id": rid(1)(r)})]),
        ("responses", "get-missing", [get("/v1/responses/resp_missing")]),
        ("responses", "delete-missing", [request("DELETE", "/v1/responses/resp_missing")]),
        ("responses", "delete-collection", [request("DELETE", "/v1/responses")]),
        ("responses", "post-with-id", [post("/v1/responses/resp_x", {"input": "x"})]),
    ]
    refusals = {"include": {"input": "x", "include": ["reasoning.encrypted_content"]}, "truncation": {"input": "x", "truncation": "auto"},
                "background": {"input": "x", "background": True}, "empty-input": {"input": []}, "item-type": {"input": [{"type": "bogus"}]},
                "tool-type": {"input": "x", "tools": [{"type": "web_search"}]}, "metadata": {"input": "x", "metadata": {str(i): "v" for i in range(17)}},
                "not-json": b"{", "list-body": b"[1]", "format": {"input": "x", "text": {"format": {"type": "bogus"}}},
                "content-part": {"input": [{"role": "user", "content": [{"type": "input_file"}]}]}, "role": {"input": [{"role": "robot", "content": "x"}]},
                "previous-missing": {"input": "x", "previous_response_id": "resp_gone"}, "logprobs": {"input": "x", "logprobs": True},
                "chat-refusal": {"input": "x", "temperature": "hot"}, "image": {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:x"}]}]}}
    out += [("responses", f"refuse-{k}", [post("/v1/responses", v)]) for k, v in refusals.items()]
    return out


def anthropic_cases() -> list[tuple[str, str, list[Any]]]:
    base = {"model": "fake-model", "max_tokens": 64, "messages": say("think")}
    tool = {"name": "lookup", "input_schema": TOOLS[0]["function"]["parameters"], "description": "Search"}
    out = [
        ("anthropic", "plain", [post("/v1/messages", base)]),
        ("anthropic", "plain-stream", [post("/v1/messages", {**base, "stream": True})]),
        ("anthropic", "thinking", [post("/v1/messages", {**base, "thinking": {"type": "enabled", "budget_tokens": 32}})]),
        ("anthropic", "thinking-stream", [post("/v1/messages", {**base, "stream": True, "thinking": {"type": "adaptive"}})]),
        ("anthropic", "system", [post("/v1/messages", {**base, "system": [{"type": "text", "text": "Be kind"}], "messages": say("plain")})]),
        ("anthropic", "tool-use", [post("/v1/messages", {**base, "messages": say("qwen_tool"), "tools": [tool]})]),
        ("anthropic", "tool-use-stream", [post("/v1/messages", {**base, "messages": say("xml_tool"), "tools": [tool], "stream": True})]),
        ("anthropic", "tool-use-think-stream", [post("/v1/messages", {**base, "messages": say("qwen_tool_think"), "tools": [tool], "stream": True, "thinking": {"type": "enabled", "budget_tokens": 8}})]),
        ("anthropic", "stop-sequence", [post("/v1/messages", {**base, "messages": say("stop_words"), "stop_sequences": ["STOP"]})]),
        ("anthropic", "stop-sequence-stream", [post("/v1/messages", {**base, "messages": say("stop_words"), "stop_sequences": ["STOP"], "stream": True})]),
        ("anthropic", "max-tokens", [post("/v1/messages", {**base, "messages": say("long"), "max_tokens": 2, "stream": True})]),
        ("anthropic", "tool-result", [post("/v1/messages", {**base, "messages": [{"role": "user", "content": "find"}, {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "tool_use", "id": "t1", "name": "lookup", "input": {"query": "é"}}]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok", "is_error": True}, {"type": "text", "text": "go @script=plain"}]}], "tools": [tool], "tool_choice": {"type": "any", "disable_parallel_tool_use": True}})]),
        ("anthropic", "count", [post("/v1/messages/count_tokens", {"model": "m", "messages": say("plain"), "tools": [tool]})]),
        ("anthropic", "count-query", [post("/messages/count_tokens?beta=true", {"model": "m", "messages": say("plain"), "thinking": {"type": "enabled", "budget_tokens": 5}})]),
    ]
    refusals = {"no-model": {"max_tokens": 5, "messages": say("plain")}, "no-max": {"model": "m", "messages": say("plain")},
                "stream-text": {**base, "stream": "yes"}, "choice": {**base, "tool_choice": {"type": "bogus"}},
                "redacted": {**base, "messages": [{"role": "assistant", "content": [{"type": "redacted_thinking", "data": "x"}]}]},
                "budget": {**base, "thinking": {"type": "enabled", "budget_tokens": 64}}, "server-tool": {**base, "tools": [{"type": "web_search_20250305", "name": "w"}]},
                "mid-system": {**base, "messages": [{"role": "user", "content": "a"}, {"role": "system", "content": "b"}, {"role": "assistant", "content": "c"}]},
                "bad-system": {**base, "messages": [{"role": "system", "content": "b"}, {"role": "user", "content": "a"}]},
                "context": {**base, "context_management": {"edits": [{"type": "other"}]}}, "service-tier": {**base, "service_tier": "priority"},
                "not-json": b"{x", "image": {**base, "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "http://x"}}]}]},
                "effort": {**base, "output_config": {"effort": "huge"}}, "format": {**base, "output_config": {"format": {"type": "json_schema", "schema": {}}}}}
    out += [("anthropic", f"refuse-{k}", [post("/v1/messages", v)]) for k, v in refusals.items()]
    return out


def token_cases() -> list[tuple[str, str, list[Any]]]:
    bodies = {"prompt": {"prompt": "Hello world é"}, "prompt-special": {"prompt": "<|im_start|>x", "add_special_tokens": False},
              "strs": {"prompt": "Hello\n", "return_token_strs": True}, "messages": {"messages": say("plain")},
              "messages-no-gen": {"messages": say("plain"), "add_generation_prompt": False, "tools": TOOLS},
              "messages-effort": {"messages": say("plain"), "reasoning_effort": "low"}, "prompt-int": {"prompt": 5},
              "flag-bad": {"prompt": "x", "return_token_strs": "yes"}, "list-body": [1], "bad-json": b"{", "tools-bad": {"messages": say("plain"), "tools": 3}}
    out = [("tokens", f"tokenize-{k}", [post("/tokenize", v)]) for k, v in bodies.items()]
    out.append(("tokens", "tokenize-v1", [post("/v1/tokenize", {"prompt": "x"})]))
    for k, v in {"ids": {"tokens": [256 + 28, 256 + 29]}, "nested": {"tokens": [[72, 105]]}, "range": {"tokens": [1, 999]},
                 "negative": {"tokens": [-1]}, "floats": {"tokens": [1.0]}, "missing": {}, "body-list": []}.items():
        out.append(("tokens", f"detokenize-{k}", [post("/v1/detokenize", v)]))
    return out


def framing_cases() -> list[tuple[str, str, list[Any], dict[str, Any]]]:
    hi = chat("plain")
    head = "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
    bad = [("Transfer-Encoding: chunked\r\nContent-Length: 2", b"0\r\n\r\n"), ("Transfer-Encoding: gzip", b""),
           ("Transfer-Encoding: gzip, chunked", b"0\r\n\r\n"), ("Transfer-Encoding: chunked, chunked", b"0\r\n\r\n"),
           ("Content-Length: bad", b""), ("Content-Length: -1", b""), ("Content-Length: 2\r\nContent-Length: 3", b"{}"),
           ("Content-Length: 20", b"{}"), ("Transfer-Encoding: chunked", b"+2\r\n{}\r\n0\r\n\r\n"),
           ("Transfer-Encoding: chunked", b"0x2\r\n{}\r\n0\r\n\r\n"), ("Transfer-Encoding: chunked", b"2\n{}\r\n0\r\n\r\n"),
           ("Transfer-Encoding: chunked", b"2\r\n{"), ("Transfer-Encoding: chunked", b"2\r\n{}XX0\r\n\r\n"),
           ("Transfer-Encoding: chunked", b"0\r\n"), ("Transfer-Encoding: chunked", b"0\r\nInvalid trailer\r\n\r\n"),
           ("Transfer-Encoding: chunked", b"2000001\r\n"), ("Content-Length: 2, 02", b"{}")]
    out: list[tuple[str, str, list[Any], dict[str, Any]]] = [
        ("framing", f"bad-{i}", [(head + h + "\r\n\r\n").encode() + body], {"half_close": True}) for i, (h, body) in enumerate(bad)]
    body = json.dumps(hi).encode()
    trailer = f"{len(body):X};name=\"x\"\r\n".encode() + body + b"\r\n0\r\nX-Test: ignored\r\n\r\n"
    out += [
        ("framing", "chunked-bytes", [post("/v1/chat/completions", hi, chunked=True), get("/v1/models")], {}),
        ("framing", "chunked-stream", [post("/v1/chat/completions", {**hi, "stream": True}, chunked=True)], {}),
        ("framing", "extensions-trailers", [(head + "Transfer-Encoding: chunked\r\n\r\n").encode() + trailer, get("/v1/models")], {}),
        ("framing", "oversized", [(head + "Content-Length: 40000000\r\n\r\n").encode()], {}),
        ("framing", "unknown-post-keepalive", [post("/v1/not-a-route", {"max_tokens": 16}), post("/v1/chat/completions", hi)], {}),
        ("framing", "keepalive-three", [get("/health"), post("/v1/chat/completions", hi), get("/v1/models?x=1")], {}),
        ("framing", "pipelined", [get("/health") + get("/v1/models"), b""], {}),
        ("framing", "http10", [post("/v1/chat/completions", hi, version="HTTP/1.0")], {}),
        ("framing", "http10-keepalive", [request("GET", "/health", None, {"Connection": "keep-alive"}, version="HTTP/1.0"), get("/health")], {}),
        ("framing", "connection-close", [request("GET", "/v1/models", None, {"Connection": "close"})], {}),
        ("framing", "expect-continue", [request("POST", "/v1/chat/completions", hi, {"Expect": "100-continue"})], {}),
        ("framing", "put", [request("PUT", "/v1/models", b"{}")], {}),
        ("framing", "options", [request("OPTIONS", "/v1/models")], {}),
        ("framing", "head", [request("HEAD", "/health")], {}),
        ("framing", "bad-version", [b"GET / HTTX/1.1\r\n\r\n"], {}),
        ("framing", "http09", [b"GET /health\r\n"], {}),
        ("framing", "http09-post", [b"POST /health\r\n"], {}),
        ("framing", "four-words", [b"GET / x HTTP/1.1\r\nHost: x\r\n\r\n"], {}),
        ("framing", "http2", [b"GET / HTTP/2.0\r\n\r\n"], {}),
        ("framing", "long-header", [b"GET / HTTP/1.1\r\nX: " + b"a" * 70000 + b"\r\n\r\n"], {}),
        ("framing", "many-headers", [b"GET / HTTP/1.1\r\n" + b"".join(b"X-%d: y\r\n" % i for i in range(105)) + b"\r\n"], {}),
        ("framing", "blank-line", [b"\r\n"], {}),
        ("framing", "double-slash", [get("//health")], {}),
        ("framing", "latin1-path", [b"GET /caf\xe9 HTTP/1.1\r\n\r\n"], {}),
        ("framing", "decisions", [post("/v1/decisions", {"input": "x"})], {}),
    ]
    return out


def status_cases() -> list[tuple[str, str, list[Any]]]:
    paths = ["/health", "/", "/health/", "/v1/models", "/models", "/v1/models?x=1", "/x/models", "/v1/health", "/nope",
             "/v1/responses", "/health?reset_peak=1"]
    return [("status", f"get{p.replace('/', '-')}", [get(p)]) for p in paths]
