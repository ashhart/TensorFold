"""What the lane server streams while a reply is written: reasoning apart from the answer, tool-call
markup out of the visible text, and nothing ever taken back."""

from tensorfold.server.text import hide_tool_calls, split_thinking

REPLY = ("The user wants the jobs listed. I could call <tool_call> here, but first think.\n</think>\n\n"
         "Checking the jobs.\n<tool_call>\n<function=hub>\n<parameter=op>\njobs\n</parameter>\n"
         "</function>\n</tool_call>\nDone <b>now</b>.")


def _stream(text):
    reasoning_sent, visible_sent = "", ""
    for n in range(1, len(text) + 1):
        reasoning, answer = split_thinking(text[:n], finished=False)
        assert reasoning.startswith(reasoning_sent), f"reasoning taken back at {n}"
        reasoning_sent = reasoning
        visible = hide_tool_calls(answer, finished=False)
        assert visible.startswith(visible_sent), f"visible text taken back at {n}: {visible_sent!r} -> {visible!r}"
        visible_sent = visible
    return reasoning_sent, visible_sent


def test_streamed_parts_grow_and_add_up():
    reasoning, visible = _stream(REPLY)
    final_reasoning, final_answer = split_thinking(REPLY, finished=True)
    assert reasoning == final_reasoning
    assert "<tool_call>" in final_reasoning                     # mentioned while thinking: stays reasoning
    assert visible == hide_tool_calls(final_answer, finished=True)
    assert visible == "Checking the jobs.\n\nDone <b>now</b>."
    assert "<function" not in visible and "</parameter" not in visible


def test_unclosed_thinking_is_all_reasoning():
    reasoning, answer = split_thinking("still thinking </thi", finished=False)
    assert (reasoning, answer) == ("still thinking ", "")
    assert split_thinking("still thinking </thi", finished=True) == ("still thinking </thi", "")


def test_text_without_calls_is_unchanged():
    assert hide_tool_calls("a < b and <b>bold</b>", finished=True) == "a < b and <b>bold</b>"
    assert hide_tool_calls("ends with <tool", finished=False) == "ends with "
    assert hide_tool_calls("ends with <tool", finished=True) == "ends with <tool"


def test_gemma_thought_channel_streams_as_reasoning():
    from tensorfold.server.text import CHANNEL_MARKERS, think_markers

    reply = "<|channel>thought\nSky: Rayleigh scattering.\n<channel|>The sky is blue because air scatters blue light."
    sent = ""
    for n in range(1, len(reply) + 1):
        reasoning, answer = split_thinking(reply[:n], finished=False, markers=CHANNEL_MARKERS)
        assert reasoning.startswith(sent) and "<|channel>" not in reasoning
        sent = reasoning
    assert split_thinking(reply, finished=True, markers=CHANNEL_MARKERS) == (
        "Sky: Rayleigh scattering.\n", "The sky is blue because air scatters blue light.")
    assert split_thinking("Blue.", finished=True, markers=CHANNEL_MARKERS) == ("", "Blue.")

    class Tokens:
        unk_token_id = 3

        def __init__(self, vocab):
            self.vocab = vocab

        def convert_tokens_to_ids(self, token):
            return self.vocab.get(token, 3)

    assert think_markers(Tokens({"<channel|>": 101})) == CHANNEL_MARKERS
    assert think_markers(Tokens({"</think>": 7})) == ("", "</think>")


def test_gemma_tool_calls_are_parsed_and_kept_out_of_the_streamed_text():
    import json

    from tensorfold.server.tools import parse_tool_calls_from_content

    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
    call = ('<|tool_call>call:read_file{flags:[<|"|>x, y: {z}<|"|>],limit:20,opts:{deep:true},'
            'path:<|"|>src/a.py<|"|>}<tool_call|>')
    reply = "Reading it now." + call
    content, calls = parse_tool_calls_from_content(reply, tools)
    assert content == "Reading it now." and len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "flags": ["x, y: {z}"], "limit": 20, "opts": {"deep": True}, "path": "src/a.py"}
    shown = ""
    for n in range(1, len(reply) + 1):
        visible = hide_tool_calls(reply[:n], finished=False)
        assert visible.startswith(shown) and "<|" not in visible
        shown = visible
    assert shown == hide_tool_calls(reply, finished=True) == "Reading it now."


def test_gemma_bare_colon_tool_call_is_structured():
    """Issue 121: gemma-4-26b-a4b-it writes <|tool_call>:name{args}<tool_call|> with no call prefix."""

    import json

    from tensorfold.server.tools import parse_tool_calls_from_content

    def tool(name):
        return {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "command": {"type": "string"}, "filePath": {"type": "string"},
            "pattern": {"type": "string"}}}}}

    tools = [tool(name) for name in ("bash", "read", "write", "edit", "glob", "grep", "list")]
    leaked = '<|tool_call>:list{path:<|"|>.<|"|>}<tool_call|>'
    content, calls = parse_tool_calls_from_content(leaked, tools)
    assert content == "" and len(calls) == 1
    assert calls[0]["function"]["name"] == "list"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "."}
    prose, parsed = parse_tool_calls_from_content("Looking.\n" + leaked, tools)
    assert prose == "Looking." and json.loads(parsed[0]["function"]["arguments"]) == {"path": "."}
    stayed, missed = parse_tool_calls_from_content(leaked, [tool("bash")])
    assert missed is None and stayed == leaked


def test_glm_and_gemma_calls_parse_through_one_parser():
    """GLM's <arg_key> calls and Gemma 4's call:NAME{...} calls in one reply, each in the order written."""

    import json

    from tensorfold.server.tools import parse_tool_calls_from_content

    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "limit": {"type": "integer"}}}}},
             {"type": "function", "function": {"name": "call:search", "parameters": {"type": "object", "properties": {
                 "query": {"type": "string"}}}}}]
    glm = "<tool_call>read_file<arg_key>path</arg_key><arg_value>a.py</arg_value></tool_call>"
    gemma = '<|tool_call>call:read_file{limit:5,path:<|"|>b.py<|"|>}<tool_call|>'
    named = "<tool_call>call:search<arg_key>query</arg_key><arg_value>x</arg_value></tool_call>"
    content, calls = parse_tool_calls_from_content("Two reads." + glm + gemma + named, tools)
    got = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]
    assert content == "Two reads."
    assert got == [("read_file", {"path": "a.py"}), ("read_file", {"limit": 5, "path": "b.py"}),
                   ("call:search", {"query": "x"})]


def test_gemma_spontaneous_empty_thought_channel_is_stripped():
    """An empty thought channel is stripped, and a reply with no channel stays as written."""
    from tensorfold.server.text import CHANNEL_MARKERS

    reply = "<|channel>thought\n<channel|>The files are: a.py, b.py."
    assert split_thinking(reply, finished=True, markers=CHANNEL_MARKERS) == ("", "The files are: a.py, b.py.")
    # a plain reply (no channel) is unchanged — the strip is a no-op
    assert split_thinking("Just an answer.", finished=True, markers=CHANNEL_MARKERS) == ("", "Just an answer.")


def test_gemma_thought_channel_not_at_the_start_is_stripped():
    """A thought channel after text, a newline, or a doubled opener is stripped from the answer."""
    from tensorfold.server.text import CHANNEL_MARKERS as C

    # after visible text (block empty) -> the text stays, the block goes
    assert split_thinking("### Read the findings first.\n\n<|channel>thought\n<channel|>", finished=True,
                          markers=C) == ("", "### Read the findings first.\n\n")
    # a doubled opener -> both markers and the empty channel are dropped
    assert split_thinking("<|channel><|channel>thought\n<channel|>The findings are clear.", finished=True,
                          markers=C) == ("", "The findings are clear.")
    # a leading newline before the block
    assert split_thinking("\n\n<|channel>thought\n<channel|>Answer here.", finished=True,
                          markers=C) == ("", "\n\nAnswer here.")
    # a real (non-empty) channel after visible text: text is the answer, the channel body is reasoning
    assert split_thinking("Preamble. <|channel>thought\nquietly.\n<channel|>Done.", finished=True,
                          markers=C) == ("quietly.\n", "Preamble. Done.")
    # none of these ever leak a marker into the answer
    for reply in ("x\n<|channel>thought\n<channel|>y", "<|channel><|channel>thought\n<channel|>z",
                  "\n<|channel>thought\n<channel|>w"):
        _, answer = split_thinking(reply, finished=True, markers=C)
        assert "<|channel>" not in answer and "<channel|>" not in answer


def test_gemma_channel_after_text_streams_monotonically():
    """A streamed answer after a late thought channel only grows, and a partial opener stays hidden."""
    from tensorfold.server.text import CHANNEL_MARKERS as C

    reply = "Here is the plan.\n\n<|channel>thought\n<channel|>"
    seen = ""
    for n in range(1, len(reply) + 1):
        _, answer = split_thinking(reply[:n], finished=False, markers=C)
        assert answer.startswith(seen), f"answer taken back at {n}: {seen!r} -> {answer!r}"
        assert "<|channel>" not in answer
        seen = answer
    assert split_thinking(reply, finished=True, markers=C) == ("", "Here is the plan.\n\n")
