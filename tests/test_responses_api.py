"""OpenAI's Responses API on both servers: a Response is its chat completion translated (same tokens), streams its
events in order, round-trips tool calls, and resumes a stored conversation by previous_response_id."""

import http.client
import json
import threading

import pytest

from tensorfold.server import responses
from tensorfold.server.errors import RequestError
from tests.test_cuda_admission import http_server
from tests.test_cuda_tool_choice import END, TOOLS, Engine, Tokens, app_for
from tests.test_server_openai_compat import FakeApp, serve_fake

FN_TOOLS = [{"type": "function", "name": t["function"]["name"], "parameters": t["function"]["parameters"]}
            for t in TOOLS]


def call(port, method, path, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request(method, path, json.dumps(body) if body is not None else None,
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read().decode()
    finally:
        connection.close()


def stream_events(text):
    """A Responses stream's events; each ``event:`` line names its data's type."""

    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        event = json.loads(lines["data"])
        assert lines["event"] == event["type"]
        out.append(event)
    return out


def valid(events):
    """The events' order as OpenAI's stream sends it; the final response, whose output is the items as done."""

    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    assert [e["type"] for e in events[:2]] == ["response.created", "response.in_progress"]
    final = events[-1]
    assert final["type"] in ("response.completed", "response.incomplete", "response.failed")
    phase, texts, done = {}, {}, []
    for e in events[2:-1]:
        kind, at = e["type"], e.get("output_index")
        if kind == "response.output_item.added":
            assert at == len(phase) and all(p == "done" for p in phase.values())
            phase[at], texts[at] = ("open" if e["item"]["type"] == "function_call" else "added"), ""
        elif kind == "response.content_part.added":
            assert phase[at] == "added" and e["content_index"] == 0
            phase[at] = "open"
        elif kind.endswith(".delta"):
            assert phase[at] == "open"
            texts[at] += e["delta"]
        elif kind in ("response.output_text.done", "response.reasoning_text.done"):
            assert phase[at] == "open" and e["text"] == texts[at]
            phase[at] = "text"
        elif kind == "response.content_part.done":
            assert phase[at] == "text" and e["part"]["text"] == texts[at]
            phase[at] = "part"
        elif kind == "response.function_call_arguments.done":
            assert phase[at] == "open" and e["arguments"] == texts[at]
            phase[at] = "part"
        elif kind == "response.output_item.done":
            assert phase[at] == "part"
            phase[at] = "done"
            done.append(e["item"])
        else:
            raise AssertionError(f"unexpected event {kind}")
    assert all(p == "done" for p in phase.values())
    assert final["response"]["output"] == done
    return final["response"]


def shape(output):
    """Output items without their ids."""

    return [{k: v for k, v in item.items() if k != "id"} for item in output]


def text_of(response):
    return "".join(p["text"] for item in response["output"] if item["type"] == "message" for p in item["content"])


# -- translation --------------------------------------------------------------------------------


def test_a_string_input_is_one_user_message_and_the_fields_carry_over():
    request = responses.translate({"input": "hi", "instructions": "be brief", "max_output_tokens": 9, "top_k": 4,
                                   "min_p": 0.1, "seed": 7, "draft": False, "reasoning": {"effort": "low"},
                                   "temperature": 0.5}, responses.Store())
    assert request.chat == {"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
                            "max_tokens": 9, "top_k": 4, "min_p": 0.1, "seed": 7, "draft": False,
                            "reasoning_effort": "low", "temperature": 0.5}
    assert request.added == [{"role": "user", "content": "hi"}]      # instructions are not carried to the next turn
    assert request.echo["instructions"] == "be brief" and request.echo["store"] is True


def test_no_effort_leaves_the_template_default():
    request = responses.translate({"input": "hi", "reasoning": {"summary": "auto"}}, responses.Store())
    assert "reasoning_effort" not in request.chat


def test_items_become_the_messages_a_chat_client_sends():
    items = [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "rules"}]},
             {"role": "user", "content": [{"type": "input_text", "text": "look "},
                                          {"type": "input_image", "image_url": "data:image/png;base64,AA"}]},
             {"type": "reasoning", "id": "rs_1", "summary": [], "content": [{"type": "reasoning_text", "text": "hm"}]},
             {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Sure."}]},
             {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": '{"city": "Oslo"}'},
             {"type": "function_call", "call_id": "c2", "name": "search", "arguments": "{}"},
             {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
             {"type": "function_call_output", "call_id": "c2", "output": [{"type": "input_text", "text": "none"}]}]
    assert responses.messages(items) == [
        {"role": "developer", "content": [{"type": "text", "text": "rules"}]},
        {"role": "user", "content": [{"type": "text", "text": "look "},
                                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Sure."}], "reasoning_content": "hm",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}},
                        {"id": "c2", "type": "function", "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "tool", "tool_call_id": "c2", "content": "none"}]


def test_tools_choices_and_formats_as_chat_completion_fields():
    store = responses.Store()
    request = responses.translate({"input": "x", "tools": FN_TOOLS, "tool_choice": {"type": "function",
                                                                                   "name": "search"}}, store)
    assert request.chat["tools"] == TOOLS
    assert request.chat["tool_choice"] == {"type": "function", "function": {"name": "search"}}
    allowed = {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "search"}]}
    request = responses.translate({"input": "x", "tools": FN_TOOLS, "tool_choice": allowed}, store)
    assert request.chat["tool_choice"] == "required" and request.chat["tools"] == TOOLS[1:]
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    request = responses.translate({"input": "x", "text": {"format": {"type": "json_schema", "name": "n",
                                                                     "schema": schema, "strict": True}}}, store)
    assert request.chat["response_format"] == {"type": "json_schema",
                                               "json_schema": {"name": "n", "schema": schema, "strict": True}}
    request = responses.translate({"input": "x", "text": {"format": {"type": "json_object"}}}, store)
    assert request.chat["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("body, words", [
    ({"tools": [{"type": "web_search"}]}, "tools of type 'web_search'"),
    ({"tools": [{"type": "file_search", "vector_store_ids": ["v"]}]}, "tools of type 'file_search'"),
    ({"tool_choice": {"type": "web_search_preview"}}, "tool_choice must be"),
    ({"background": True}, "background responses"),
    ({"include": ["reasoning.encrypted_content"]}, "include is not supported (reasoning.encrypted_content)"),
    ({"conversation": "conv_1"}, "conversations are not supported"),
    ({"prompt": {"id": "pmpt_1"}}, "prompt templates"),
    ({"truncation": "auto"}, "truncation"),
    ({"top_logprobs": 3}, "top_logprobs"),
    ({"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}]}, "'input_file'"),
    ({"input": [{"role": "user", "content": [{"type": "input_image", "file_id": "f"}]}]}, "file ids"),
    ({"input": [{"type": "item_reference", "id": "msg_1"}]}, "'item_reference'"),
    ({"input": [{"type": "web_search_call", "id": "ws_1"}]}, "'web_search_call'"),
    ({"input": [{"type": "reasoning", "summary": [], "encrypted_content": "gAAA"}]}, "encrypted reasoning"),
    ({"previous_response_id": "resp_nope"}, "previous response 'resp_nope' is not stored"),
    ({"metadata": {"k": 1}}, "metadata"),
    ({"instructions": ["x"]}, "instructions must be a string"),
    ({"input": []}, "input must be"),
    ({"text": {"format": {"type": "grammar"}}}, "text.format"),
])
def test_what_this_server_lacks_is_a_clear_400(body, words):
    with pytest.raises(RequestError, match=None) as caught:
        responses.translate({"input": "hi", **body}, responses.Store())
    assert words in str(caught.value)


# -- replies ------------------------------------------------------------------------------------


def test_a_streamed_reply_and_a_whole_one_give_the_same_output():
    events = []
    base = {"id": "resp_1", "object": "response", "output": []}
    reply = responses.Reply(base, events.append)
    reply.start()
    for payload in ({"choices": [{"delta": {"role": "assistant"}}]},
                    {"choices": [{"delta": {"reasoning_content": "think"}}]},
                    {"choices": [{"delta": {"reasoning_content": "ing"}}]},
                    {"choices": [{"delta": {"content": "Hi"}}]},
                    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                                            "function": {"name": "f", "arguments": ""}}]}}]},
                    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"a": 1}'}}]}}]},
                    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                     "usage": {"prompt_tokens": 5, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 2},
                               "completion_tokens_details": {"reasoning_tokens": 3}}},
                    None):
        reply.chunk(payload)
    streamed = valid(events)
    whole = responses.Reply(dict(base)).completion({"choices": [{"message": {
        "reasoning_content": "thinking", "content": "Hi",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]},
        "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}})
    assert shape(streamed["output"]) == shape(whole["output"]) == [
        {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "thinking"}],
         "status": "completed"},
        {"type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "Hi", "annotations": [], "logprobs": []}]},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": '{"a": 1}', "status": "completed"}]
    assert streamed["usage"] == {"input_tokens": 5, "input_tokens_details": {"cached_tokens": 2}, "output_tokens": 7,
                                 "output_tokens_details": {"reasoning_tokens": 3}, "total_tokens": 12}
    assert streamed["status"] == "completed" and streamed["completed_at"]


def test_a_response_is_stored_before_the_client_hears_it_ended():
    kept, heard = [], []
    reply = responses.Reply({"id": "r", "output": []}, lambda e: heard.append((e["type"], len(kept))), kept.append)
    reply.start()
    reply.chunk({"choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]})
    assert heard[-1] == ("response.completed", 1) and kept[0]["status"] == "completed"
    reply = responses.Reply({"id": "r", "output": []}, None, kept.append)
    reply.chunk({"error": {"message": "boom"}})
    assert len(kept) == 1                                          # a failed response is not kept


def test_a_length_stop_is_incomplete_and_an_error_fails():
    events = []
    reply = responses.Reply({"id": "r", "output": []}, events.append)
    reply.start()
    reply.chunk({"choices": [{"delta": {"content": "Hel"}}]})
    reply.chunk({"choices": [{"delta": {}, "finish_reason": "length"}]})
    final = valid(events)
    assert final["status"] == "incomplete" and final["incomplete_details"] == {"reason": "max_output_tokens"}
    assert final["output"][0]["status"] == "incomplete"
    events = []
    reply = responses.Reply({"id": "r", "output": []}, events.append)
    reply.start()
    reply.chunk({"error": {"message": "boom", "type": "server_error"}})
    failed = valid(events)
    assert failed["status"] == "failed" and failed["error"] == {"code": "server_error", "message": "boom"}


# -- the CUDA server --------------------------------------------------------------------------------


def post_response(port, **body):
    status, text = call(port, "POST", "/v1/responses", body)
    return status, (stream_events(text) if body.get("stream") and status == 200 else json.loads(text))


def test_cuda_a_response_is_its_chat_completion(tmp_path):
    engine = Engine()
    app = app_for(tmp_path, engine)
    with http_server(app) as port:
        status, chat = call(port, "POST", "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "Say hello."}], "return_token_ids": True})
        chat = json.loads(chat)
        status, got = post_response(port, input="Say hello.", return_token_ids=True)
        assert status == 200 and got["object"] == "response" and got["status"] == "completed"
        assert got["tensorfold"]["token_sha"] == chat["tensorfold"]["token_sha"]
        assert text_of(got) == chat["choices"][0]["message"]["content"] == "Hello! How can I help?"
        assert got["usage"]["input_tokens"] == chat["usage"]["prompt_tokens"]
        assert got["usage"]["output_tokens"] == chat["usage"]["completion_tokens"]
        assert engine.calls[0][0] == engine.calls[1][0]                   # the same prompt
        status, events = post_response(port, input="Say hello.", stream=True)
        streamed = valid(events)
        assert status == 200 and shape(streamed["output"]) == shape(got["output"])
        assert streamed["tensorfold"]["token_sha"] == got["tensorfold"]["token_sha"]


def test_cuda_drafted_equals_serial_and_concurrent_equals_alone(tmp_path):
    engine = Engine()
    with http_server(app_for(tmp_path, engine)) as port:
        alone = post_response(port, input="Say hello.", draft=False)[1]["tensorfold"]["token_sha"]
        assert engine.calls[-1][2] is False
        shas = []
        workers = [threading.Thread(target=lambda: shas.append(
            post_response(port, input="Say hello.")[1]["tensorfold"]["token_sha"])) for _ in range(4)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        assert shas == [alone] * 4


def test_cuda_thinking_is_a_reasoning_item_counted_in_usage(tmp_path):
    with http_server(app_for(tmp_path, Engine())) as port:
        status, got = post_response(port, input="hi", chat_template_kwargs={"enable_thinking": True})
        assert [item["type"] for item in got["output"]] == ["reasoning", "message"]
        assert got["output"][0]["content"][0]["text"] == "the user wants hi" and text_of(got) == "Hi!"
        assert got["usage"]["output_tokens_details"]["reasoning_tokens"] == len("the user wants hi") + 1  # </think>
        status, events = post_response(port, input="hi", chat_template_kwargs={"enable_thinking": True}, stream=True)
        assert shape(valid(events)["output"]) == shape(got["output"])
        assert any(e["type"] == "response.reasoning_text.delta" for e in events)


def test_cuda_a_tool_call_round_trip_resumes_the_conversation(tmp_path):
    engine = Engine()
    with http_server(app_for(tmp_path, engine)) as port:
        status, first = post_response(port, input="Say hello.", tools=FN_TOOLS, tool_choice="required")
        call_item = first["output"][-1]
        assert call_item["type"] == "function_call" and call_item["name"] == "get_weather"
        assert json.loads(call_item["arguments"]) == {"city": "Oslo"}
        output = {"type": "function_call_output", "call_id": call_item["call_id"], "output": "sunny"}
        status, chained = post_response(port, previous_response_id=first["id"], input=[output], tools=FN_TOOLS)
        assert status == 200 and text_of(chained) == "Hello! How can I help?"
        chained_prompt = engine.calls[-1][0]
        # the same turn sent whole (store false: nothing is kept), and as a chat completion
        status, whole = post_response(port, input=[{"role": "user", "content": "Say hello."}, *first["output"],
                                                   output], tools=FN_TOOLS, store=False)
        assert engine.calls[-1][0] == chained_prompt and text_of(whole) == text_of(chained)
        messages = [{"role": "user", "content": "Say hello."},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": call_item["call_id"], "type": "function",
                         "function": {"name": "get_weather", "arguments": call_item["arguments"]}}]},
                    {"role": "tool", "tool_call_id": call_item["call_id"], "content": "sunny"}]
        status, chat = call(port, "POST", "/v1/chat/completions", {"messages": messages, "tools": TOOLS})
        assert engine.calls[-1][0] == chained_prompt
        assert call(port, "GET", f"/v1/responses/{whole['id']}")[0] == 404      # store false


def test_cuda_a_length_stop_is_incomplete(tmp_path):
    with http_server(app_for(tmp_path, Engine())) as port:
        status, got = post_response(port, input="Say hello.", max_output_tokens=3)
        assert got["status"] == "incomplete" and got["incomplete_details"] == {"reason": "max_output_tokens"}
        assert text_of(got) == "Hel" and got["usage"]["output_tokens"] == 3
        status, events = post_response(port, input="Say hello.", max_output_tokens=3, stream=True)
        assert events[-1]["type"] == "response.incomplete" and text_of(valid(events)) == "Hel"


def test_cuda_get_and_delete_a_stored_response(tmp_path):
    with http_server(app_for(tmp_path, Engine())) as port:
        status, got = post_response(port, input="Say hello.", metadata={"run": "7"})
        status, found = call(port, "GET", f"/v1/responses/{got['id']}")
        assert status == 200 and json.loads(found) == got and got["metadata"] == {"run": "7"}
        status, gone = call(port, "DELETE", f"/v1/responses/{got['id']}")
        assert status == 200 and json.loads(gone) == {"id": got["id"], "object": "response", "deleted": True}
        assert call(port, "GET", f"/v1/responses/{got['id']}")[0] == 404
        assert call(port, "DELETE", f"/v1/responses/{got['id']}")[0] == 404
        status, text = post_response(port, input="again", previous_response_id=got["id"])
        assert status == 400 and "is not stored" in text["error"]["message"]


def test_cuda_the_chat_path_refuses_as_it_would_a_chat_completion(tmp_path):
    engine = Engine()
    with http_server(app_for(tmp_path, engine)) as port:
        status, chat = call(port, "POST", "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "x"}], "top_k": "many"})
        status2, got = post_response(port, input="x", top_k="many")
        assert status == status2 == 400 and got == json.loads(chat)
        status, got = post_response(port, input="x", tools=[{"type": "web_search"}], stream=True)
        assert status == 400 and "web_search" in got["error"]["message"] and not engine.calls


# -- the Mac server ----------------------------------------------------------------------------------


@pytest.fixture
def mac():
    app = FakeApp(reasoning="because")
    server = serve_fake(app)
    try:
        yield app, server.server_port
    finally:
        server.shutdown()
        server.server_close()


def test_mac_a_response_is_its_chat_completion(mac):
    app, port = mac
    status, got = post_response(port, input="Hi", instructions="Be kind.")
    assert status == 200 and text_of(got) == "Hello" and got["model"] == "fake-model"
    assert app.messages == [{"role": "system", "content": "Be kind."}, {"role": "user", "content": "Hi"}]
    status, events = post_response(port, input="Hi", stream=True)
    streamed = valid(events)
    assert [item["type"] for item in streamed["output"]] == ["reasoning", "message"]
    assert streamed["output"][0]["content"][0]["text"] == "because" and text_of(streamed) == "Hello"
    status, found = call(port, "GET", f"/v1/responses/{streamed['id']}")
    assert status == 200 and json.loads(found) == streamed
    status, again = post_response(port, input="And?", previous_response_id=streamed["id"])
    assert app.messages == [{"role": "user", "content": "Hi"},
                            {"role": "assistant", "content": "Hello", "reasoning_content": "because"},
                            {"role": "user", "content": "And?"}]
    assert call(port, "DELETE", f"/v1/responses/{streamed['id']}")[0] == 200


def test_the_store_keeps_the_newest_and_a_chain_needs_every_link():
    store = responses.Store(limit=2)
    for n in range(3):
        store.put({"id": f"r{n}", "previous_response_id": f"r{n - 1}" if n else None, "output": []},
                  [{"role": "user", "content": str(n)}])
    assert store.get("r0") is None and store.get("r2") is not None
    with pytest.raises(RequestError, match="'r0' is not stored"):
        store.conversation("r2")
    store = responses.Store()
    store.put({"id": "a", "previous_response_id": None, "output": [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "yes"}]}]},
        [{"role": "user", "content": "q"}])
    store.put({"id": "b", "previous_response_id": "a", "output": []}, [{"role": "user", "content": "q2"}])
    assert store.conversation("b") == [{"role": "user", "content": "q"},
                                       {"role": "assistant", "content": [{"type": "text", "text": "yes"}]},
                                       {"role": "user", "content": "q2"}]


def test_routes():
    assert responses.route("/v1/responses") == responses.route("/responses/") == ""
    assert responses.route("/v1/responses/resp_1?x=1") == "resp_1"
    assert responses.route("/v1/responses/resp_1/input_items") is None
    assert responses.route("/v1/chat/completions") is None


def test_reasoning_tokens_count_through_the_close():
    from tensorfold.server.text import reasoning_count

    assert reasoning_count([5, 6, END, 7], END) == 3
    assert reasoning_count([5, 6], END) == 2                       # cut while thinking: every token
    assert reasoning_count([5, 6, END, 7], -1) == reasoning_count([5, 6], None) == 0     # not thinking
    assert Tokens().encode("</think>").ids == [END]
