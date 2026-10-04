"""Message-start resume points for CUDA engines, found by the Mac planner's rule through the CUDA chat template."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

MIN_GAP = 256            # a snapshot this close to the prompt start or to the previous snapshot saves too little
CHECKPOINT = "TF_SYS_CHECKPOINT"


def checkpoint_rows() -> int:
    """TF_SYS_CHECKPOINT: the system-block checkpoint's granule in rows (default 2048, a prompt pass); 0: none."""

    raw = os.environ.get(CHECKPOINT, "").strip()
    try:
        rows = int(raw) if raw else 2048
    except ValueError:
        raise ValueError(f"{CHECKPOINT}: rows (0: off), not {raw!r}") from None
    if rows < 0 or 0 < rows < MIN_GAP:
        raise ValueError(f"{CHECKPOINT}: 0 (off) or at least {MIN_GAP} rows, not {rows}")
    return rows


class TemplateTokens:
    """The tokenizer surface ``message_markers`` reads, over a CUDA server's tokenizer and chat template."""

    def __init__(self, tok: Any, template: Any) -> None:
        self.tok, self.template = tok, template
        self.added_tokens_decoder = tok.get_added_tokens_decoder()
        self.all_special_ids = [int(i) for i, t in self.added_tokens_decoder.items() if t.special]

    def apply_chat_template(self, messages: list[dict[str, Any]], tokenize: bool = True,
                            add_generation_prompt: bool = False, **kwargs: Any):
        text = self.template.template.render(**self.template.specials, messages=messages,
                                             add_generation_prompt=add_generation_prompt, **kwargs)
        return self.tok.encode(text, add_special_tokens=False).ids if tokenize else text


def snapshot_points(openers: Sequence[int], assistant: Sequence[int],
                    checkpoint: int = 0) -> Callable[[Sequence[int]], list[int]]:
    """Where a prefill keeps states: the second message's start (a shared system block) and the last assistant start.
    ``checkpoint`` rows (0: none): also the system-block checkpoint, which ``points.checkpoint`` returns."""

    from tensorfold.engine.prefill_plan import PrefillPlan

    plan = PrefillPlan(openers=openers, assistant=assistant, min_chunk=max(MIN_GAP, len(assistant)))

    def at_checkpoint(ids: Sequence[int]) -> int | None:
        """The last multiple of ``checkpoint`` at least MIN_GAP before the second message's start, so a system block
        that differs only near its end (a date line, a memory section) resumes there; None when there is none."""

        if not checkpoint or not plan.openers:
            return None
        starts = np.flatnonzero(np.isin(np.asarray(ids, dtype=np.int64), plan.openers))
        if len(starts) < 2:
            return None
        at = (int(starts[1]) - MIN_GAP) // checkpoint * checkpoint
        return at if at >= checkpoint else None

    def points(ids: Sequence[int]) -> list[int]:
        arr = np.asarray(ids, dtype=np.int64)
        found = plan.points(arr)
        starts = np.flatnonzero(np.isin(arr, plan.openers)) if plan.openers else []
        wanted = ([int(starts[1])] if len(starts) > 1 else []) + ([found[-1]] if found else [])
        cp = at_checkpoint(ids)
        wanted += [cp] if cp is not None else []
        out: list[int] = []
        for p in sorted(set(wanted)):
            if MIN_GAP <= p < len(arr) and (not out or p - out[-1] >= MIN_GAP):
                out.append(p)
        return out

    points.checkpoint = at_checkpoint
    return points


def resume_points(model_dir: str | Path, checkpoint: int = 0) -> Callable[[Sequence[int]], list[int]] | None:
    """``snapshot_points`` for this checkpoint's chat template, or None when it marks no message starts."""

    from tokenizers import Tokenizer

    from tensorfold.engine.prefill_plan import message_markers

    from .chat_template import ChatTemplate

    model_dir = Path(model_dir)
    if not (model_dir / "tokenizer.json").is_file():
        return None
    try:
        tokens = TemplateTokens(Tokenizer.from_file(str(model_dir / "tokenizer.json")), ChatTemplate(model_dir))
        openers, assistant = message_markers(tokens)
    except (OSError, ValueError, KeyError):
        return None
    return snapshot_points(openers, assistant, checkpoint) if openers or assistant else None
