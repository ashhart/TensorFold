"""response_format on the Mac lane engine: drafted, serial, tree, copied and shared rounds give the grammar's one reply,
and drafts the grammar rejects never reach a forward (fake models, toy vocabulary; real models are in the ledger)."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
xgr = pytest.importorskip("xgrammar")

from test_family_streams import END, NL, NLNL, V, StreamsModel, TreeModel, after, chain  # noqa: E402

from tensorfold.engine import grammar  # noqa: E402
from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402

STOP = 0
VOCAB = ["", *[chr(c) for c in range(0x21, 0x7f) if chr(c) not in "~|`^_"], "</think>", "\n", "\n\n",
         "a:", ",b", "1}", "c:9"]
SPEC = grammar.Spec("regex", r"\{[a-f]:[0-9](,[a-f]:[0-9]){0,3}\}")
GRAMMARS = grammar.Grammars(xgr.TokenizerInfo(VOCAB, xgr.VocabType.RAW, vocab_size=V, stop_token_ids=[STOP]))
COMPILED = GRAMMARS.compile(SPEC)


def meant(prompt, max_new, budget=0):
    """The serial reply by hand: ``after(last)`` when the grammar allows it, else its lowest allowed token; a thinking
    budget's close stops at </think> and the grammar takes the reply from there."""

    m, bits = xgr.GrammarMatcher(COMPILED), xgr.allocate_token_bitmask(1, V)
    out, active = [], not budget
    while len(out) < max_new:
        last = out[-1] if out else prompt[-1]
        if not active:
            out += [NL, END] if len(out) + 1 == budget else [after(last)]
            active = out[-1] == END
            continue
        m.fill_next_token_bitmask(bits, 0)
        allowed = [t for t in range(V) if (int(bits[0, t // 32]) >> (t % 32)) & 1]
        out.append(after(last) if after(last) in allowed else allowed[0])
        if out[-1] == STOP:
            break
        assert m.accept_token(out[-1])
    return out[:max_new]


class ReplyProposer:
    """Copies the reply it is given, one token wrong every ``wrong_every``, as if it had seen the reply before."""

    last_match = 1 << 30

    def __init__(self, prompt, reply, wrong_every=0):
        self.prompt, self.reply, self.wrong_every = len(prompt), reply, wrong_every

    def propose(self, context, max_draft):
        at = len(context) - self.prompt
        out = list(self.reply[at:at + max_draft])
        return [(t + 1) % V if self.wrong_every and (at + k) % self.wrong_every == 1 else t for k, t in enumerate(out)]


class BlankHead(StreamsModel):
    """A head that only ever drafts the blank line, which the grammar never allows."""

    def _guess(self, token):
        self.calls += 1
        return NLNL


def _stream(name, prompt, max_new, *, budget=0, constrained=True, drafts=True, copies=False, wrong_every=0):
    proposer = None
    if copies and drafts:
        reply = meant(prompt, max_new, budget) if constrained else chain(prompt[-1], budget, max_new)
        proposer = ReplyProposer(prompt, reply, wrong_every)
    constraint = GRAMMARS.constraint(COMPILED, think_end=END if budget else None) if constrained else None
    return LaneStream(stream_id=name, prompt_ids=list(prompt), max_new_tokens=max_new,
                      eos_ids=frozenset({STOP} if constrained else ()), think_budget=budget,
                      think_close=(NL, END, NLNL), think_end=END, think_open=budget > 0, drafts=drafts,
                      proposer=proposer, constraint=constraint)


def run(model, streams):
    engine = LaneEngine(model)
    for stream in streams:
        engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    return engine


def _accepted_by_the_grammar(reply, budget):
    m = xgr.GrammarMatcher(COMPILED)
    body = reply[reply.index(END) + 1:] if budget else reply
    return all(m.accept_token(t) for t in body) and m.is_terminated()


SPECS = [([3, 14, 15], 0), ([8, 2], 5), ([40, 41, 42, 43], 0), ([60, 61], 9)]


@pytest.mark.parametrize("model", [StreamsModel, TreeModel])
@pytest.mark.parametrize("wrong_every", [0, 2])
@pytest.mark.parametrize("copies", [False, True])
@pytest.mark.parametrize("gpu", [False, True])
def test_drafted_serial_and_shared_replies_are_the_grammar_s_one_reply(model, wrong_every, copies, gpu):
    wants = [meant(prompt, 40, budget) for prompt, budget in SPECS]
    for want, (_, budget) in zip(wants, SPECS):
        assert want[-1] == STOP and _accepted_by_the_grammar(want, budget)
    for drafts in (True, False):
        alone = []
        for i, (prompt, budget) in enumerate(SPECS):
            stream = _stream(f"s{i}", prompt, 40, budget=budget, drafts=drafts, copies=copies, wrong_every=wrong_every)
            run(model(wrong_every=wrong_every, gpu_tokens=gpu), [stream])
            alone.append(stream.emitted)
            assert stream.finish_reason == "stop" and stream.constraint.finished
        assert alone == wants, drafts
        together = [_stream(f"s{i}", prompt, 40, budget=budget, drafts=drafts, copies=copies, wrong_every=wrong_every)
                    for i, (prompt, budget) in enumerate(SPECS)]
        run(model(wrong_every=wrong_every, gpu_tokens=gpu), together)
        assert [s.emitted for s in together] == wants, drafts


@pytest.mark.parametrize("gpu", [False, True])
def test_a_plain_stream_beside_constrained_ones_emits_what_it_emits_alone(gpu):
    plain = _stream("p", [8, 2], 30, budget=4, constrained=False)
    shaped = [_stream(f"s{i}", prompt, 40, budget=budget) for i, (prompt, budget) in enumerate(SPECS[:2])]
    run(StreamsModel(wrong_every=3, gpu_tokens=gpu), [shaped[0], plain, shaped[1]])
    assert plain.emitted == chain(2, 4, 30) and plain.constraint is None
    assert [s.emitted for s in shaped] == [meant(p, 40, b) for p, b in SPECS[:2]]


def test_drafts_the_grammar_rejects_never_reach_a_forward():
    free = _stream("free", [3, 14, 15], 12, constrained=False)
    engine = run(BlankHead(), [free])
    assert max(r.rows for r in engine.round_stats) > 1                   # unconstrained, the blank drafts are verified
    shaped = _stream("s", [3, 14, 15], 40)
    engine = run(BlankHead(), [shaped])
    assert shaped.emitted == meant([3, 14, 15], 40) and shaped.drafted == 0
    assert {r.rows for r in engine.round_stats} == {1}                  # every blank draft cut before the forward


def test_the_thinking_budget_s_close_under_a_grammar_ends_at_think_end():
    stream = _stream("s", [8, 2], 40, budget=5, drafts=False)
    run(StreamsModel(), [stream])
    at = stream.emitted.index(END)
    assert stream.emitted[at - 1:at + 1] == [NL, END] and stream.emitted[at + 1] != NLNL
    plain = _stream("p", [8, 2], 12, budget=5, drafts=False, constrained=False)
    run(StreamsModel(), [plain])
    assert plain.emitted[4:7] == [NL, END, NLNL]                         # without a grammar the close is whole
