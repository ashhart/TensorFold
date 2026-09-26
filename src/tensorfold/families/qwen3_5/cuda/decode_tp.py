"""Exact one-stream decode with the 27B split across two ranks (two Sparks).

Rank 0 drafts, samples and picks the accepted path; rank 1 follows. Each round rank 0 broadcasts
the window's tokens and parents, both ranks run the tensor-parallel forward (heads and MLP width
split, fixed-order fp32 rank sums), rank 0 samples every row and broadcasts the accepted path, and
both ranks commit it. Serial decoding is the same loop with one-row windows, so a drafted stream
matches serial on the same two ranks byte for byte.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import torch
import torch.distributed as dist

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows

from .decode import CopyIndex, DecodeResult, clone_state
from .forward import State, _paths, commit, tree_forward
from .sampling import sample_rows
from .weights import Weights


_FIRST = 256        # ints in a share's first broadcast: the length, then up to 255 values


def _share(values: Sequence[int] | None, rank: int, device: torch.device) -> list[int]:
    """Broadcast an int list from rank 0 (an empty list is the stop signal).

    One broadcast carries the length and up to 255 values, so a round's small shares cost one
    collective each; longer lists (prompts) send the rest in a second.
    """

    head = torch.zeros((_FIRST,), dtype=torch.int32, device=device)
    if rank == 0:
        first = list(values[:_FIRST - 1])
        head[:1 + len(first)] = torch.tensor([len(values)] + first, dtype=torch.int32)
    dist.broadcast(head, 0)
    got = head.tolist()
    n = got[0]
    if n == 0:
        return []
    if n <= _FIRST - 1:
        return list(values) if rank == 0 else got[1:1 + n]
    rest = (torch.tensor(list(values[_FIRST - 1:]), dtype=torch.int32, device=device) if rank == 0
            else torch.empty((n - (_FIRST - 1),), dtype=torch.int32, device=device))
    dist.broadcast(rest, 0)
    return list(values) if rank == 0 else got[1:] + rest.tolist()


def split_candidates(logits: torch.Tensor, sampling: Sampling | None, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A rank's candidates from its half of the vocabulary: the first maximum when greedy, else the
    top ``top_k + MARGIN`` (the same count ``sample_rows`` reads from the whole vocabulary)."""

    if sampling is None or sampling.temperature <= 0:
        values, ids = logits.float().max(dim=-1)
        return values[:, None].contiguous(), (ids + offset)[:, None].contiguous()
    count = min(logits.shape[1], int(sampling.top_k) + MARGIN) if sampling.top_k else logits.shape[1]
    values, ids = torch.topk(logits.float(), count, dim=-1, sorted=False)
    return values.contiguous(), (ids + offset).contiguous()


def choose_merged(values, ids, positions: Sequence[int], sampling: Sampling | None) -> list[int]:
    """Rows of candidates from both halves, rank 0's first, -> the token ``sample_rows`` would draw
    from the whole vocabulary: the union holds each half's top candidates, so the global top-k, ties
    by id included, is among them, and ``choose_rows`` orders candidates by (value, id) itself."""

    if sampling is None or sampling.temperature <= 0:
        # the whole vocabulary's first maximum: rank 0's half holds the lower ids
        return [int(ids[r, 0] if values[r, 0] >= values[r, 1] else ids[r, 1]) for r in range(ids.shape[0])]
    return choose_rows(values, ids, positions, sampling)


def _sample_split(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                  rank: int) -> list[int] | None:
    """Both ranks call this with their half of the logits; rank 0 returns the tokens, rank 1 None."""

    values, ids = split_candidates(logits, sampling, rank * logits.shape[1])
    all_values = torch.empty((2, *values.shape), dtype=values.dtype, device=values.device)
    all_ids = torch.empty((2, *ids.shape), dtype=ids.dtype, device=ids.device)
    dist.all_gather_into_tensor(all_values, values)
    dist.all_gather_into_tensor(all_ids, ids)
    if rank != 0:
        return None
    return choose_merged(torch.cat((all_values[0], all_values[1]), dim=1).cpu().numpy(),
                         torch.cat((all_ids[0], all_ids[1]), dim=1).cpu().numpy().astype("int64"),
                         positions, sampling)


def _tokens(ids: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(list(ids), dtype=torch.int32, device=device)


@torch.no_grad()
def prefill_tp(w: Weights, prompt: Sequence[int], sampling: Sampling | None, rank: int,
               draft=None, *, state: State | None = None) -> tuple[State, int]:
    """Both ranks commit the prompt in 128-row chains; rank 0 samples the first token and shares it.

    ``state``: a committed state for the prompt's first ``state.pos`` tokens (prefix reuse); only the
    rest is processed. Rows never depend on their chain-mates, so the result equals a fresh prefill.
    """

    device = w.norm.device
    split = 2 * w.head.n == w.config.vocab          # split_weights(..., split_head=True)
    st = clone_state(state) if state is not None else State(w)
    done = st.pos
    last = None
    for start in range(done, len(prompt), 128):
        chunk = list(prompt[start:start + 128])
        parents = list(range(-1, len(chunk) - 1))
        taps = draft is not None and (rank == 0 or getattr(draft, "world", 1) == 2)
        out = tree_forward(w, _tokens(chunk, device), parents, st, tp=True,
                           full_logits=split or rank == 0, capture_taps=taps)
        if taps:
            logits, record, tapped = out
            draft.add_taps(tapped)
        else:
            logits, record = out
        last = logits[-1:]
        commit(st, record, list(range(len(chunk))))
    if split:
        first = _sample_split(last, [len(prompt)], sampling, rank)
    else:
        first = [sample_rows(last, [len(prompt)], sampling)[0]] if rank == 0 else None
    return st, _share(first, rank, device)[0]


def _accept(tokens: list[int], parents: list[int], sampled: list[int], room: int,
            eos: tuple[int, ...], stop_eos: bool) -> tuple[list[int], int]:
    children: dict[tuple[int, int], int] = {}
    for row in range(1, len(tokens)):
        children.setdefault((parents[row], tokens[row]), row)
    path = [0]
    terminal = sampled[0]
    while len(path) < room:
        if stop_eos and terminal in eos:
            break
        child = children.get((path[-1], terminal))
        if child is None:
            break
        path.append(child)
        terminal = sampled[child]
    return path, terminal


@torch.no_grad()
def decode_tp(w: Weights, st: State, prompt: Sequence[int], pending: int, count: int,
              sampling: Sampling | None, rank: int, draft=None, *, max_rows: int = 16,
              allow_copy: bool = False, stop_eos: bool = True,
              on_tokens: Callable[[list[int]], bool | None] | None = None) -> DecodeResult | None:
    """Draft (rank 0 with ``draft``, or serial when ``draft`` is None), verify on both ranks, commit.

    A two-rank drafter (``DFlash2(..., world=2)``, passed on both ranks) drafts on both ranks
    together: each round rank 0 first shares whether to draft and the pending token.

    ``count`` includes the pending token. On rank 0 ``tokens`` is the pending token and every sampled
    one. Rank 1 never sees the last sampled token (it is not committed), so its ``tokens`` is the
    committed ones, read from the windows, plus -1 in that place: ``prompt + tokens[:-1]`` names the
    committed state on both ranks, which keeps their prefix caches in step.
    """

    device = w.norm.device
    split = 2 * w.head.n == w.config.vocab          # split_weights(..., split_head=True)
    st = clone_state(st)
    out = [pending]
    context = list(prompt) + out
    committed: list[int] = []
    copies = CopyIndex() if allow_copy and rank == 0 else None
    tp_draft = draft is not None and getattr(draft, "world", 1) == 2
    last, length = pending, len(context)      # rank 1's view of the pending token and context length
    stages = dict(draft=0.0, verify=0.0, sample=0.0, commit=0.0)
    rounds = drafted_rows = accepted = 0
    widths: list[int] = []
    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    eos = tuple(w.config.eos)
    stopped = False
    while True:
        stage = time.perf_counter()
        window: list[int] | None = None
        parents: list[int] = []
        if rank == 0:
            if len(out) >= count or (stop_eos and out[-1] in eos) or stopped:
                window = []
            else:
                guesses, gparents = [], []
                if draft is not None:
                    copied = copies.propose(context, max_rows - 1) if copies is not None else []
                    if tp_draft:
                        _share([0 if copied else 1, out[-1], len(context)], rank, device)
                    if copied:
                        guesses, gparents = copied, list(range(-1, len(copied) - 1))
                    else:
                        guesses, gparents = draft.propose_tree(out[-1], len(context), max_rows - 1, sampling)
                window = [out[-1]] + list(guesses)
                parents = [-1] + [0 if p < 0 else p + 1 for p in gparents]
            if tp_draft and not window:
                _share([2, 0, 0], rank, device)                     # stop
        elif tp_draft:
            mode, last, length = _share(None, rank, device)
            if mode == 1:
                draft.propose_tree(last, length, max_rows - 1, sampling)
        packed = _share((window + parents if window else []) if rank == 0 else None, rank, device)
        if not packed:
            break
        window, parents = packed[:len(packed) // 2], packed[len(packed) // 2:]
        stages["draft"] += time.perf_counter() - stage
        stage = time.perf_counter()
        taps_wanted = draft is not None and (rank == 0 or tp_draft)
        result = tree_forward(w, _tokens(window, device), parents, st, tp=True,
                              full_logits=split or rank == 0, capture_taps=taps_wanted)
        if taps_wanted:
            logits, record, taps = result
        else:
            logits, record = result
        path: list[int] | None = None
        terminal = -1
        depths, _ = _paths(parents)
        positions = [st.pos + d + 1 for d in depths]
        sampled = _sample_split(logits, positions, sampling, rank) if split else None
        if rank == 0:
            torch.cuda.synchronize()
            stages["verify"] += time.perf_counter() - stage
            stage = time.perf_counter()
            if sampled is None:
                sampled = sample_rows(logits, positions, sampling)
            path, terminal = _accept(window, parents, sampled, count - len(out), eos, stop_eos)
            stages["sample"] += time.perf_counter() - stage
            stage = time.perf_counter()
        path = _share(path, rank, device)
        commit(st, record, path)
        committed.extend(window[row] for row in path)
        if taps_wanted:
            draft.add_taps(taps[path])
        if rank == 0:
            new = [window[row] for row in path[1:]] + [terminal]
            out.extend(new)
            context.extend(new)
            torch.cuda.synchronize()
            stages["commit"] += time.perf_counter() - stage
            rounds += 1
            drafted_rows += len(window) - 1
            accepted += len(path) - 1
            widths.append(len(window))
            if on_tokens is not None:
                stopped = bool(on_tokens(new))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if rank != 0:
        return DecodeResult(committed + [-1], seconds, 0, 0, 0, state=st)
    return DecodeResult(out, seconds, rounds, drafted_rows, accepted, stages["draft"], stages["verify"],
                        stages["sample"], stages["commit"], widths, state=st)
