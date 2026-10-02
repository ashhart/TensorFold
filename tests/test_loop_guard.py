"""--loop-guard: cycle detection, commit() latch, and think-cut precedence.

Detector semantics are formula-driven: a cycle block of length L following a
non-periodic prefix fires iff L >= LOOP_MIN_RUN + period and the detected region
starts at least WARM_IN tokens into the reply. Engine tests drive a bare
LaneStream (no model): commit() latches the fire without finishing, the family
layer converts the latch (think closes, the close tokens join ``force``), and
the close drains through commit(), and the reply continues with the model's answer: the finish is the answer's own (stop/length), with the event reported on ``loop``.
"""

from __future__ import annotations

import time
from typing import Any

from tensorfold.engine.lane_engine import FIRED, LaneStream
from tensorfold.server.loop_guard import LOOP_MIN_RUN, MAX_PERIOD, WARM_IN, LoopGuard

PREFIX = 70
FIRE = PREFIX + LOOP_MIN_RUN + 1     # first emitted length at which the P=1 fixture fires


def build(prefix_len: int, period: int, cycles: int) -> list[int]:
    tokens = [10_000 + i for i in range(prefix_len)]
    block = [900 + i for i in range(period)]
    for _ in range(cycles):
        tokens += block
    return tokens


def test_fire_boundary_matrix_matches_the_closed_form() -> None:
    guard = LoopGuard()
    for prefix in (70, 63, 64, 10):
        for period in range(1, MAX_PERIOD + 1):
            below = (LOOP_MIN_RUN + period) // period
            for cycles in {below, below + 1, below + 2}:
                tokens = build(prefix, period, cycles)
                fired = guard.check(tokens) is not None
                length = period * cycles
                # the 256-equality region must lie inside the cycle, and its start must clear the warm-in
                want = length >= LOOP_MIN_RUN + period and prefix + length - LOOP_MIN_RUN - period >= WARM_IN
                assert fired == want, (prefix, period, cycles, fired, want)


def test_earliest_possible_fire_is_prefix_plus_min_run_plus_one() -> None:
    guard = LoopGuard()
    tokens = build(PREFIX, 1, 400)
    first = next(i + 1 for i in range(len(tokens)) if guard.check(tokens[: i + 1]))
    assert first == FIRE


def test_reply_younger_than_the_floor_never_fires() -> None:
    guard = LoopGuard()
    assert guard.check(build(10, 1, LOOP_MIN_RUN)) is None          # region starts ~token 10
    assert guard.check(build(10, 1, 4 * LOOP_MIN_RUN)) is not None  # same reply, grown past the floor


def test_near_miss_cycle_never_fires_and_stays_cheap() -> None:
    guard = LoopGuard()
    stream = [1000 + i for i in range(WARM_IN)]
    block, count = [9000, 9001, 9002, 9003], 0
    while len(stream) < 30_000:
        stream += block
        count += 1
        if count % 12 == 0:
            stream[-1] += 1                    # a perturbation the exact-match test must not survive
    assert guard.check(stream) is None
    started = time.perf_counter()
    per_token = LoopGuard()
    for i in range(len(stream), len(stream) + 20_000):
        value = block[i % 4] + (1 if i % 12 == 0 else 0)     # keep perturbing: a near-miss, never a run
        stream.append(value)
        per_token.check(stream)
    per_commit_us = (time.perf_counter() - started) / 20_000 * 1e6
    # generous bound: an accidentally unbounded scan would sit in the milliseconds
    assert per_commit_us < 200, per_commit_us


def test_text_gate_stands_down_on_garbled_or_empty_decodes() -> None:
    class Tok:
        def __init__(self, table: dict[int, str], default: str = "x") -> None:
            self.table, self.default = table, default

        def decode(self, ids: list[int]) -> str:
            return "".join(self.table.get(int(i), self.default) for i in ids)

    clean = LoopGuard(Tok({0: "a", 1: "b", 2: "hmm ", 3: "-"}))
    assert clean.check([0, 1] * 40 + [2, 3] * 300) is not None
    garbled = LoopGuard(Tok({0: "a", 1: "b", 4: "\ufffd\ufffd"}))
    assert garbled.check([0, 1] * 40 + [4] * 300) is None
    empty = LoopGuard(Tok({}, default=""))
    assert empty.check(build(PREFIX, 1, 300)) is None


def make_stream(**kwargs: Any) -> LaneStream:
    settings: dict[str, Any] = {
        "stream_id": "s", "prompt_ids": [1, 2, 3], "max_new_tokens": 1000,
        "think_open": True, "think_close": (7, 8, 9), "think_end": 9,
    }
    settings.update(kwargs)
    return LaneStream(**settings)


def test_commit_latches_the_fire_without_finishing() -> None:
    stream = make_stream(loop_guard=LoopGuard())
    tokens = build(PREFIX, 1, 300)
    landed = stream.commit(tokens[: FIRE - 1])
    assert len(landed) == FIRE - 1 and not stream.finished and stream.loop_stop is None
    landed = stream.commit(tokens[FIRE - 1: FIRE])
    assert landed == tokens[FIRE - 1: FIRE] and stream.loop_stop == 1
    assert stream.loop == {"period": 1}
    assert not stream.finished and stream.finish_reason == ""


def test_the_close_drains_and_the_answer_continues() -> None:
    stream = make_stream(loop_guard=LoopGuard(), eos_ids=frozenset({10_500}))
    tokens = build(PREFIX, 1, 300)
    stream.commit(tokens[:FIRE])
    assert stream.loop_stop is not None and not stream.finished
    # the family layer converts the latch, then pops the close tokens from force and
    # commits each as a normal window token; commit() only observes that force has drained
    stream.think_open = False
    stream.force = list(stream.think_close)
    for _ in stream.think_close:
        stream.commit([stream.force.pop(0)])
    # the close drained: the reply continues (the budget-cut shape) instead of ending
    assert stream.emitted[-3:] == [7, 8, 9]
    assert not stream.finished and not stream.force and stream.loop_stop == FIRED
    assert stream.loop == {"period": 1}                     # the event stays reported
    stream.commit([10_500])                                 # the answer's own end
    assert stream.finished and stream.finish_reason == "stop"


def test_an_unarmed_config_latches_for_the_family_to_degrade() -> None:
    # no close tokens (no budget, arming failed): the latch is still set; the family
    # layer's conversion ends the reply unlabelled-shaped (covered at the family level)
    stream = make_stream(loop_guard=LoopGuard(), think_close=())
    stream.commit(build(PREFIX, 1, 300)[:FIRE])
    assert stream.loop_stop is not None and stream.think_open and not stream.finished


def test_the_guard_is_inert_outside_the_think_block() -> None:
    stream = make_stream(loop_guard=LoopGuard(), think_open=False)
    stream.commit(build(PREFIX, 1, 300))
    assert not stream.finished and stream.loop_stop is None and stream.finish_reason == ""


def test_a_latched_stream_cannot_refire_and_the_answer_continues() -> None:
    stream = make_stream(loop_guard=LoopGuard())
    tokens = build(PREFIX, 1, 300)
    stream.commit(tokens[:FIRE])
    stream.think_open = False
    stream.force = list(stream.think_close)
    for _ in stream.think_close:
        stream.commit([stream.force.pop(0)])
    stream.commit(tokens[FIRE:FIRE + 40])          # still cyclic text after think: no refire
    assert not stream.finished and stream.loop_stop == FIRED
    assert stream.loop == {"period": 1}            # the event stays reported
    assert stream.emitted[FIRE:FIRE + 3] == [7, 8, 9]


def test_eos_outranks_the_guard() -> None:
    stream = make_stream(loop_guard=LoopGuard(), eos_ids=frozenset({10_069}))
    stream.commit(build(PREFIX, 1, 300))
    assert stream.finished and stream.finish_reason == "stop"


def test_a_stop_check_outranks_the_guard() -> None:
    stream = make_stream(loop_guard=LoopGuard(), stop_check=lambda ids: ids[-1] == 10_069)
    stream.commit(build(PREFIX, 1, 300))
    assert stream.finished and stream.finish_reason == "stop"


def test_a_cap_crossed_mid_drain_is_labelled_length_with_the_loop_reported() -> None:
    stream = make_stream(loop_guard=LoopGuard(), max_new_tokens=FIRE)
    stream.commit(build(PREFIX, 1, 300)[:FIRE])
    assert stream.loop_stop is not None and stream.finish_reason == ""   # latched, not capped
    stream.convert_loop_fire()
    for _ in stream.think_close:
        stream.commit([stream.force.pop(0)])
    # the cap crossed mid-drain labels "length" — the budget cut's close tokens behave
    # the same, and the cap at the drain labels "length" too (next test)
    assert stream.finished and stream.finish_reason == "length"
    assert stream.loop == {"period": 1}                                  # the event is still reported


def test_a_cap_reached_at_the_drain_labels_length_with_the_loop_reported() -> None:
    # the reply continues after the close, so the cap is the cap: a close token that
    # reaches it labels "length", the event still reported
    stream = make_stream(loop_guard=LoopGuard(), max_new_tokens=FIRE + 3)
    tokens = build(PREFIX, 1, 300)
    stream.commit(tokens[:FIRE])
    stream.convert_loop_fire()
    assert len(stream.force) == 3
    stream.commit([stream.force.pop(0)])
    stream.commit([stream.force.pop(0)])
    assert not stream.finished
    stream.commit([stream.force.pop(0)])          # this token reaches the cap too
    assert stream.finished and stream.finish_reason == "length"
    assert stream.loop == {"period": 1}


def test_a_cap_between_close_tokens_cuts_mid_drain_and_still_reports() -> None:
    # C5, differentiated from the on-the-last-token variant: the cap lands while force
    # still holds a close token — the close never finishes, the reply is capped "length",
    # and the latch still settles (no finish path leaves the transient period)
    stream = make_stream(loop_guard=LoopGuard(), max_new_tokens=FIRE + 2)
    tokens = build(PREFIX, 1, 300)
    stream.commit(tokens[:FIRE])
    stream.convert_loop_fire()
    assert len(stream.force) == 3
    stream.commit([stream.force.pop(0)])
    assert not stream.finished
    stream.commit([stream.force.pop(0)])            # this token reaches the cap; 1 close token pending
    assert stream.finished and stream.finish_reason == "length"
    assert stream.loop_stop == FIRED and stream.loop == {"period": 1}


def test_an_unarmed_stream_converts_straight_to_the_label() -> None:
    # defensive path: the family's conversion with no close tokens armed (arming pairs
    # think_close with think_end, so make_job cannot produce this; kept as a guarantee)
    stream = make_stream(loop_guard=LoopGuard(), think_close=())
    stream.commit(build(PREFIX, 1, 300)[:FIRE])
    assert stream.loop_stop is not None and stream.think_open
    stream.convert_loop_fire()
    assert stream.finished and stream.finish_reason == "loop" and not stream.force
    assert stream.loop_stop == FIRED


def test_thinking_off_never_arms_the_guard() -> None:
    # marker-less or thinking-off requests must never let the guard watch visible content
    stream = make_stream(loop_guard=LoopGuard(), think_open=False, think_close=(), think_end=-1)
    stream.commit(build(PREFIX, 1, 300))
    assert stream.loop_stop is None and not stream.finished


def test_a_required_call_fix_drains_before_the_loop_close_appends() -> None:
    # a fix landing after the latch must not clobber the pending think close: the
    # conversion defers, the fix drains through force, then the close still lands
    # and the answer continues
    stream = make_stream(loop_guard=LoopGuard(), eos_ids=frozenset({10_500}))
    stream.commit(build(PREFIX, 1, 300)[:FIRE])
    assert stream.loop_stop is not None and not stream.finished and not stream.force
    stream.force = [5, 6]                          # a required call's fix is draining
    stream.convert_loop_fire()
    assert stream.think_open and stream.force == [5, 6]   # deferred, fix untouched
    stream.force.clear()
    stream.convert_loop_fire()                     # fix drained: now it converts
    assert not stream.think_open and stream.force == [7, 8, 9]
    for _ in stream.force.copy():
        stream.commit([stream.force.pop(0)])       # drain the fix's close through commit
    assert not stream.finished and stream.loop == {"period": 1}
    stream.commit([10_500])                        # the answer's own end
    assert stream.finished and stream.finish_reason == "stop"


def test_a_reopened_think_block_never_fires_twice() -> None:
    # F1: the one-fire-per-reply invariant, driven through the worst case — after the
    # close drains and the answer continues, a think-open marker re-opens the block
    # (think_end re-armed), the answer itself cycles, and NOTHING fires again: the
    # guard runs once per reply
    stream = make_stream(loop_guard=LoopGuard(), eos_ids=frozenset({10_500}))
    tokens = build(PREFIX, 1, 300)
    stream.commit(tokens[:FIRE])
    stream.think_open = False
    stream.force = list(stream.think_close)
    for _ in stream.think_close:
        stream.commit([stream.force.pop(0)])
    assert stream.loop_stop == FIRED and not stream.finished
    stream.think_open = True                        # a re-opened block in the answer
    stream.commit(build(20, 1, 300) + [10_500])     # a second full cycle, then EOS
    assert stream.finished and stream.finish_reason == "stop"
    assert stream.loop == {"period": 1}             # one fire, reported once
    assert stream.loop_stop == FIRED and not stream.force   # the latch never reset: no second fire


def test_the_budget_rearms_for_a_block_the_answer_reopens() -> None:
    # review-2 C4 (re-arm): while the guard's close is pending (loop_stop int) the budget
    # stands down; once it has drained (FIRED) the user's budget re-arms — it bounds any
    # think block the answer re-opens, so a second cycle cannot burn to the cap
    stream = make_stream(loop_guard=LoopGuard(), think_budget=256, eos_ids=frozenset({10_500}))
    tokens = build(PREFIX, 1, 300)
    stream.commit(tokens[:FIRE])
    assert stream.think_cut(build(20, 1, 60)) is None      # pending: the guard owns the block
    stream.think_open = False
    stream.force = list(stream.think_close)
    for _ in stream.think_close:
        stream.commit([stream.force.pop(0)])
    assert stream.loop_stop == FIRED
    stream.think_open = True                                # the answer re-opens the block
    assert stream.think_cut(build(20, 1, 60)) is not None   # re-armed: the budget bounds it
    stream.commit([10_500])
    assert stream.finished and stream.finish_reason == "stop" and stream.loop_stop == FIRED


def test_the_latch_state_is_deterministic_on_every_finish_path() -> None:
    # F4: no finish path leaves an ambiguous latch — a reply that finished with the
    # close drained carries FIRED; a reply the cap cut mid-drain carries the period
    # (the close never finished). FIRED is one-per-reply either way: no second fire.
    def drained_case(kwargs: dict[str, Any]) -> LaneStream:
        stream = make_stream(loop_guard=LoopGuard(), **kwargs)
        stream.commit(build(PREFIX, 1, 300)[:FIRE])
        stream.convert_loop_fire()
        for _ in list(stream.force):
            stream.commit([stream.force.pop(0)])
        return stream

    stream = drained_case({"eos_ids": frozenset({10_500})})
    stream.commit([10_500])                          # the answer's own end
    assert stream.finish_reason == "stop" and stream.loop_stop == FIRED

    stream = drained_case({"stop_check": lambda ids: ids[-1] == 7_000})
    stream.commit([7_000, 7_001])                    # the answer's own stop string
    assert stream.finish_reason == "stop" and stream.loop_stop == FIRED

    stream = make_stream(loop_guard=LoopGuard(), max_new_tokens=FIRE + 3)
    stream.commit(build(PREFIX, 1, 300)[:FIRE])
    stream.convert_loop_fire()
    while not stream.finished:
        stream.commit([7_000 + (len(stream.emitted) % 4)])
    assert stream.finish_reason == "length" and stream.loop_stop == FIRED
    assert stream.loop == {"period": 1}


def test_conversion_is_one_shot_across_a_reopened_block() -> None:
    # review-2 F2: after the close drains and the answer re-opens think, the family
    # layer's next convert_loop_fire() is a no-op — the fire belonged to the first
    # block; a second close must not inject into the answer's own think block
    stream = make_stream(loop_guard=LoopGuard(), eos_ids=frozenset({10_500}))
    stream.commit(build(PREFIX, 1, 300)[:FIRE])
    stream.convert_loop_fire()          # the family's conversion on the fire round
    assert not stream.think_open and stream.force == list(stream.think_close)
    for _ in stream.think_close:
        stream.commit([stream.force.pop(0)])
    assert stream.loop_stop == FIRED and not stream.force
    stream.think_open = True            # the answer re-opens the block
    stream.convert_loop_fire()          # the family's next round: must not re-close
    assert stream.force == [] and stream.loop_stop == FIRED


def test_think_cut_stands_down_once_the_guard_owns_the_close() -> None:
    stream = make_stream(loop_guard=LoopGuard(), think_budget=256, eos_ids=frozenset({10_500}))
    tokens = build(PREFIX, 1, 300)
    before = stream.commit(tokens[:FIRE - 1])
    assert stream.think_cut(before) is not None    # the budget still owns the cut
    stream.commit(tokens[FIRE - 1:FIRE])
    assert stream.loop_stop is not None
    assert stream.think_cut(tokens) is None        # the guard's forced close outranks it now


def test_fire_token_semantics_match_the_length_path() -> None:
    # the fire token landed (its grammar and call gate already ran); the batch remainder is dropped
    observed: list[int] = []

    class Gate:
        watching = True

        def observe(self, token: int) -> None:
            observed.append(token)

    stream = make_stream(loop_guard=LoopGuard(), call_gate=Gate())
    tokens = build(PREFIX, 1, 300)
    landed = stream.commit(tokens[:FIRE + 5])
    assert observed[-1] == tokens[FIRE - 1]
    assert stream.emitted == tokens[:FIRE]
    assert len(landed) == FIRE


def test_batch_remainder_after_a_fire_is_not_landed() -> None:
    stream = make_stream(loop_guard=LoopGuard())
    tokens = build(PREFIX, 1, 300)
    landed = stream.commit(tokens)                 # the whole cycle in one commit batch
    assert stream.emitted == tokens[:FIRE]
    assert landed == tokens[:FIRE]
    assert stream.loop_stop is not None and not stream.finished
