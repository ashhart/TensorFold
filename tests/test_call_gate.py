"""tool_choice "required" (#52): the answer's first word is the tool-call opener, cut identically in drafted, serial,
pipelined and shared rounds (fake models; the real model is checked in the ledger)."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from test_family_streams import END, NL, NLNL, StreamsModel, after  # noqa: E402

from tensorfold.engine.call_gate import CallGate  # noqa: E402
from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402

OPEN, BLANK = 93, {NLNL, 94}
QWEN, NO_THINK, GEMMA, NAMED = "qwen", "no-think", "gemma", "named"
# the named case's markup: the call's lead "=", tool names spelled a letter a token, ">" after a name
TEXT = {OPEN: "<tool_call>", 80: "=", 81: "a", 82: "b", 83: "c", 84: ">"}


def text(token):
    return TEXT.get(int(token), "?")


def encode(written):
    ids = {v: k for k, v in TEXT.items()}
    return [ids[c] for c in written]


def gate(kind):
    """Qwen's prompt opens the think block; Gemma 4's reply opens its own (here the chain's first two tokens)."""

    if kind == GEMMA:
        return CallGate(OPEN, BLANK.__contains__, think_open=after(8), think_end=after(after(8)))
    if kind == NAMED:
        return CallGate(OPEN, BLANK.__contains__, text=text, encode=encode, lead="=", names=["ab", "ac"], tail=">")
    return CallGate(OPEN, BLANK.__contains__, think_end=END, armed=kind != QWEN)


class ChainProposer:
    """Copies the model's own chain, as if it had seen it before: right until the gate cuts it."""

    last_match = 1 << 30

    def propose(self, context, max_draft):
        out, token = [], context[-1]
        for _ in range(max_draft):
            token = after(token)
            out.append(token)
        return out


def _stream(name, prompt, max_new, kind, *, budget=0, drafts=True, copies=False):
    return LaneStream(stream_id=name, prompt_ids=list(prompt), max_new_tokens=max_new, think_budget=budget,
                      think_close=(NL, END, NLNL), think_end=END, think_open=budget > 0, drafts=drafts,
                      proposer=ChainProposer() if copies and drafts else None,
                      call_gate=gate(kind) if kind else None)


def meant(prompt, max_new, kind, *, budget=0):
    """The serial reply by hand: the budget's close, and the gate asked one token at a time."""

    out, last, check = [], prompt[-1], gate(kind) if kind else None
    while len(out) < max_new:
        tokens = [NL, END, NLNL] if budget and len(out) + 1 == budget else [after(last)]
        hit = check.cut(tokens) if check is not None and len(tokens) == 1 else None
        for token in (hit[1] if hit else tokens):
            if check is not None:
                check.observe(token)
            out.append(token)
        last = out[-1]
    return out[:max_new]


def run(streams, **model):
    engine = LaneEngine(StreamsModel(**model))
    for stream in streams:
        engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    return [s.emitted for s in streams]


CASES = [([8, 2], 0, NO_THINK), ([3, 14, 15], 5, QWEN), ([40, 41], 1, QWEN), ([5], 0, QWEN), ([9, 8], 0, GEMMA),
         ([8, 2], 0, NAMED)]


@pytest.mark.parametrize("gpu,head", [(False, True), (True, True), (True, False), (False, False)])
@pytest.mark.parametrize("wrong_every", [0, 2])
@pytest.mark.parametrize("copies", [False, True])
@pytest.mark.parametrize("prompt,budget,kind", CASES)
def test_drafted_equals_serial_equals_the_cut(prompt, budget, kind, gpu, head, wrong_every, copies):
    want = meant(prompt, 20, kind, budget=budget)
    model = dict(gpu_tokens=gpu, head=head, wrong_every=wrong_every)
    for drafts in (True, False):
        stream = _stream("s", prompt, 20, kind, budget=budget, drafts=drafts, copies=copies)
        assert run([stream], **model) == [want]


def test_the_cases_cut_where_meant():
    assert meant([8, 2], 6, NO_THINK)[0] == OPEN and after(2) != OPEN            # thinking off: the first word
    assert meant([3, 14, 15], 12, QWEN, budget=5)[4:8] == [NL, END, NLNL, OPEN]  # after the budget's close
    assert OPEN not in meant([5], 12, QWEN)                                      # a think block never closed
    reply = meant([9, 8], 6, GEMMA)                                              # the reply's own block, then
    assert reply[:3] == [after(8), after(after(8)), OPEN]                        # the answer's first word
    assert "".join(text(t) for t in meant([8, 2], 6, NAMED)[:5]) == "<tool_call>=ab>"   # lead and name forced


def test_a_sampled_opener_or_blank_passes_through():
    first = after(2)
    assert CallGate(first, lambda t: False).cut([first, 7]) is None
    assert CallGate(first, lambda t: False).cut([7, first]) == (0, [first])
    blank = CallGate(OPEN, {first}.__contains__)
    assert blank.cut([first, 7]) == (1, [OPEN]) and blank.cut([first, OPEN, 7]) is None
    thinking = CallGate(OPEN, BLANK.__contains__, think_end=END, armed=False)
    assert thinking.cut([7, END, NLNL, 8]) == (3, [OPEN]) and thinking.cut([7, 8]) is None
    channel = CallGate(OPEN, BLANK.__contains__, think_end=END, think_open=NL)
    assert channel.cut([NLNL, NL, 7, END, 8]) == (4, [OPEN]) and channel.cut([7]) == (0, [OPEN])


def test_the_lead_and_name_hold_to_the_offered_tools():
    named = gate(NAMED)
    assert named.cut([7]) == (0, [OPEN, 80])                                     # prose: the opener and the lead
    assert named.cut([OPEN, 80, 81, 83, 84, 7]) is None                          # "=ac>": an offered name
    assert named.cut([OPEN, 80, 81, 81]) == (3, encode("b>"))                    # "aa": the first name "a" starts
    assert named.cut([OPEN, 7]) == (1, encode("="))                              # the lead broken
    assert named.cut([OPEN, 80, 7]) == (2, encode("ab>"))                        # a name no tool has
    for token in [OPEN, 80, 81, 82, 84]:
        named.observe(token)
    assert named.done and named.cut([7, 7]) is None


@pytest.mark.parametrize("wrong_every", [0, 3])
def test_streams_together_emit_what_they_emit_alone(wrong_every):
    def streams():
        return [_stream(f"s{i}", prompt, 18 + i, kind if i != 2 else None, budget=budget)
                for i, (prompt, budget, kind) in enumerate(CASES)]

    alone = [run([s], wrong_every=wrong_every)[0] for s in streams()]
    assert run(streams(), wrong_every=wrong_every) == alone
    assert alone == [meant(prompt, 18 + i, kind if i != 2 else None, budget=budget)
                     for i, (prompt, budget, kind) in enumerate(CASES)]
