"""Prefill chunk starts are a function of the prompt's tokens: the grid, and resume points far enough apart."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tensorfold.engine.prefill_plan import PrefillPlan, PromptChunks, message_markers

OPEN, ASSIST = 99, 98                    # a message's first token; the role token after it in an assistant message


def _plan(step: int, min_chunk: int) -> PrefillPlan:
    return PrefillPlan(step, (OPEN,), min_chunk, (OPEN, ASSIST))


def test_without_markers_chunks_sit_on_the_grid() -> None:
    plan = PrefillPlan(8)
    assert plan.chunks(list(range(30))).starts == [0, 8, 16, 24]
    assert plan.chunks(list(range(8))).starts == [0]
    assert plan.name == "grid8"


def test_chunks_start_at_assistant_messages_and_the_second_message_min_chunk_on_else_a_step_on() -> None:
    tokens = [5] * 44
    for at in (0, 3, 15, 30):
        tokens[at] = OPEN                                   # messages; 0 and 3 are the first two
    for at in (10, 21, 24, 40):
        tokens[at:at + 2] = [OPEN, ASSIST]                  # assistant messages
    # 3 is under 4 from 0; then 10; 15 is a user message; 21; 24 is under 4 from 21; 30 a user message; 21 + 16 = 37
    assert _plan(16, 4).chunks(tokens).starts == [0, 10, 21, 37]         # 40 is under 4 from the grid's 37
    assert _plan(16, 4).points(np.array(tokens)) == [3, 10, 21, 24, 40]
    assert _plan(16, 4).name == "grid16+msg4:99:99.98"


def test_the_second_message_starts_a_chunk_so_sessions_share_a_system_block() -> None:
    system = [OPEN, *[5] * 20]
    assert _plan(64, 4).chunks([*system, OPEN, 6, 7, OPEN, ASSIST]).starts == [0, 21]
    assert _plan(64, 4).chunks([*system, OPEN, 8, 9, 9, 9, OPEN, ASSIST]).starts == [0, 21, 26]


def _random_prompt(rng: random.Random, n: int) -> list[int]:
    out: list[int] = []
    while len(out) < n:
        r = rng.random()
        out += [OPEN, ASSIST] if r < 0.06 else [OPEN] if r < 0.1 else [ASSIST] if r < 0.12 else [rng.randrange(1, 9)]
    return out[:n]


@pytest.mark.parametrize("min_chunk", [2, 3, 7])
def test_prompts_that_agree_up_to_a_start_are_cut_alike_up_to_it(min_chunk: int) -> None:
    rng = random.Random(min_chunk)
    plan = _plan(12, min_chunk)
    for _ in range(600):
        first = _random_prompt(rng, rng.randrange(2, 90))
        shared = rng.randrange(1, len(first) + 1)
        second = first[:shared] + _random_prompt(rng, rng.randrange(0, 60))
        a, b = plan.chunks(first), plan.chunks(second)
        for start in a.starts[1:]:
            if start > shared:
                break
            if start in b:
                # a state stored at ``start`` by one resumes the other: both cut [0, start) the same way
                assert [s for s in a.starts if s <= start] == [s for s in b.starts if s <= start]
            if start + len(plan.assistant) <= min(shared, len(second) - 1):
                assert start in b          # the tokens that made it a start are shared too


def test_membership_floor_and_the_chunks_between_starts() -> None:
    tokens = [5] * 40
    for at in (10, 30):
        tokens[at:at + 2] = [OPEN, ASSIST]
    chunks = _plan(16, 4).chunks(tokens)
    assert chunks.starts == [0, 10, 26, 30]
    assert 10 in chunks and 26 in chunks and 11 not in chunks and 0 not in chunks and 40 not in chunks
    assert [chunks.floor(x) for x in (0, 9, 10, 29, 39, 99)] == [0, 0, 10, 26, 30, 30]
    assert chunks.between(0, 40) == [(0, 10), (10, 26), (26, 30), (30, 40)]
    assert chunks.between(10, 30) == [(10, 26), (26, 30)]
    assert chunks.between(30, 40) == [(30, 40)]


def test_without_a_plan_any_position_resumes_in_steps_from_it() -> None:
    anywhere = PromptChunks(None, 20, step=8)
    assert 7 in anywhere and 19 in anywhere and 0 not in anywhere and 20 not in anywhere
    assert anywhere.floor(13) == 13
    assert anywhere.between(3, 20) == [(3, 11), (11, 19), (19, 20)]


def test_the_min_chunk_is_between_the_assistant_header_and_the_step() -> None:
    for bad in (0, 1, 17):
        with pytest.raises(ValueError):
            _plan(16, bad)


class _ChatTemplate:
    """ChatML-like rendering: ``<|im_start|>`` (1) role, content, ``<|im_end|>`` (2); ids 1 and 2 are special."""

    def __init__(self, special: bool = True, opener: int = 1) -> None:
        self.opener = opener
        self.added_tokens_decoder = {1: SimpleNamespace(special=special), 2: SimpleNamespace(special=special)}

    def encode(self, text: str, **_: Any) -> list[int]:
        return [10 + ord(c) % 50 for c in text]

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
        ids: list[int] = []
        for m in messages:
            ids += [self.opener, *self.encode(m["role"] + "\n" + str(m.get("content", ""))), 2]
        if kwargs.get("add_generation_prompt", True):
            ids += [self.opener, *self.encode("assistant\n")]
        return ids


def test_message_markers_come_from_the_chat_template() -> None:
    assert message_markers(_ChatTemplate()) == ((1,), (1, 57))
    assert message_markers(_ChatTemplate(special=False)) == ((), ())   # plain text never splits prompts


def test_role_tokens_that_open_messages_are_openers_and_the_assistants_is_its_header() -> None:
    class RoleTokens(_ChatTemplate):
        """GLM-like: each role has its own special token (3 user, 4 assistant) and no shared one."""

        def __init__(self) -> None:
            super().__init__()
            self.added_tokens_decoder = {3: SimpleNamespace(special=True), 4: SimpleNamespace(special=True)}

        def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
            ids = [5, 6]                                               # a preamble, not a message
            for m in messages:
                ids += [3 if m["role"] == "user" else 4, *self.encode(str(m.get("content", "")))]
            if kwargs.get("add_generation_prompt", True):
                ids += [4]
            return ids

    assert message_markers(RoleTokens()) == ((3, 4), (4,))


def test_a_template_that_cannot_render_the_probe_gets_the_grid_alone() -> None:
    class Broken(_ChatTemplate):
        def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
            raise ValueError("no")

    assert message_markers(Broken()) == ((), ())


def test_serve_cuts_prompts_at_the_templates_markers_and_names_the_plan_in_snapshot_keys(monkeypatch) -> None:
    pytest.importorskip("mlx.core")
    from pathlib import Path

    from tensorfold import cli
    import tensorfold.server.app as app_module

    seen: dict[str, Any] = {}

    class Started(Exception):
        pass

    class App:
        def __init__(self, model: Any, tokenizer: Any, **kwargs: Any) -> None:
            seen.update(kwargs)
            raise Started

    monkeypatch.setattr(app_module, "ChatApp", App)
    package = SimpleNamespace(load=lambda model_dir, **options: (object(), _ChatTemplate()),
                              kernel_version=lambda model: "k1")
    family = SimpleNamespace(title="fake", model_type="fake", package=package)
    args = cli.build_parser().parse_args(["serve", "some/model", "--no-drafts", "--snapshot-dir", "none"])
    with pytest.raises(Started):
        cli._serve_mlx(args, family, Path("some/model"), 0, [], 1 << 30)
    plan = seen["engine_factory"].keywords["prefill_plan"]
    assert plan.openers == (1,) and plan.assistant == (1, 57) and plan.step == 2048
    assert f"|prefill={plan.name}|" in seen["model_id"]
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["serve", "some/model", "--prefill-grid", "512"])     # the plan replaced it
