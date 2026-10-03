"""A model that writes its own <think> (Kolibri 1; its template leaves the tag to the reply): the tag never shows."""

from __future__ import annotations

from tensorfold.server.text import split_thinking

REPLY = "<think>\nThe user asks for 16 states. Name three.\n</think>\n\nDeutschland hat 16 Bundesländer."


def test_the_replys_own_opener_never_reaches_the_reasoning():
    sent = ""
    for n in range(1, len(REPLY) + 1):
        reasoning, answer = split_thinking(REPLY[:n], finished=False)
        assert "<" not in reasoning and reasoning.startswith(sent)
        sent = reasoning
    assert split_thinking(REPLY, finished=True) == ("The user asks for 16 states. Name three.\n",
                                                    "Deutschland hat 16 Bundesländer.")


def test_a_reply_without_its_own_opener_is_split_as_before():
    assert split_thinking("Plan.\n</think>\n\nDone.", finished=True) == ("Plan.\n", "Done.")
