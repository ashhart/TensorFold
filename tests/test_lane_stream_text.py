"""What the lane server streams while a reply is written: reasoning apart from the answer, tool-call
markup out of the visible text, and nothing ever taken back."""

from tensorfold.server.app import hide_tool_calls, split_thinking

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
