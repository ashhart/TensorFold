"""Two-rank decode matches serial bits through fixed-order fp32 rank sums and rank-zero broadcasts of proposals and accepted paths."""

from __future__ import annotations

import struct
import time
from typing import Callable, Sequence

import torch
import torch.distributed as dist

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows

from .decode import CopyIndex, DecodeResult, clone_state
from .forward import State, _paths, commit, tree_forward
from tensorfold.cuda.sampling import dist_gather, nucleus_rows, sample_rows
from .weights import Weights


_FIRST = 256        # ints in a share's first broadcast: the length, then up to 255 values


def _share(values: Sequence[int] | None, rank: int, device: torch.device) -> list[int]:
    """Broadcast an int list from rank zero, with an empty list signaling stop and lists over 255 values using a second broadcast."""

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


def _words(value: int) -> list[int]:
    return [(value >> (16 * i)) & 0xFFFF for i in range(4)]


def _value(words: Sequence[int]) -> int:
    return sum(int(w) << (16 * i) for i, w in enumerate(words))


SAMPLING_WORDS = 18          # pack_sampling's length: a header's fields after it start this far on


def pack_sampling(sampling: Sampling | None) -> list[int]:
    """18 ints for a share: the seed and the float settings cross as their exact bits (16-bit words)."""

    if sampling is None:
        return [0] * SAMPLING_WORDS
    bits = [struct.unpack("<Q", struct.pack("<d", float(x)))[0]
            for x in (sampling.temperature, sampling.top_p, sampling.min_p)]
    return [1, int(sampling.top_k), *_words(int(sampling.seed)), *(w for b in bits for w in _words(b))]


def unpack_sampling(words: Sequence[int]) -> Sampling | None:
    if not words[0]:
        return None
    temperature, top_p, min_p = (struct.unpack("<d", struct.pack("<Q", _value(words[i:i + 4])))[0]
                                 for i in (6, 10, 14))
    return Sampling(_value(words[2:6]), temperature, int(words[1]), top_p, min_p)


def split_candidates(logits: torch.Tensor, sampling: Sampling | None, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the first local maximum when greedy, otherwise enough local candidates to contain the global top-k."""

    if sampling is None or sampling.temperature <= 0:
        values, ids = logits.float().max(dim=-1)
        return values[:, None].contiguous(), (ids + offset)[:, None].contiguous()
    count = min(logits.shape[1], int(sampling.top_k) + MARGIN) if sampling.top_k else logits.shape[1]
    from tensorfold.cuda.sampling import top_by_value_then_id

    values, ids = top_by_value_then_id(logits, count)
    return values.contiguous(), (ids + offset).contiguous()


def choose_merged(values, ids, positions: Sequence[int], sampling: Sampling | None) -> list[int]:
    """Merge rank-zero candidates first to preserve first-maximum ties, then sample the union in value-and-id order."""

    if sampling is None or sampling.temperature <= 0:
        # the whole vocabulary's first maximum: rank 0's half holds the lower ids
        return [int(ids[r, 0] if values[r, 0] >= values[r, 1] else ids[r, 1]) for r in range(ids.shape[0])]
    return choose_rows(values, ids, positions, sampling)


def _sample_split(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                  rank: int) -> list[int] | None:
    """Both ranks call this with their half of the logits; rank 0 returns the tokens, rank 1 None."""

    if sampling is not None and sampling.temperature > 0 and not sampling.top_k:     # the shared nucleus rule
        tokens = nucleus_rows(logits, positions, sampling, offset=rank * logits.shape[1], gather=dist_gather)
        return tokens if rank == 0 else None
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


def first_token(w: Weights, normed: torch.Tensor, n: int, sampling: Sampling | None, rank: int = 0,
                world: int = 1, constraint=None) -> int:
    """The token after an ``n``-token prompt from its last row's normed state; two ranks share rank 0's draw."""

    from .forward import _mm

    if world == 1:
        logits = _mm(normed, w.head)
        if constraint is not None:              # a reply's grammar (tensorfold.engine.grammar): masked, then followed
            constraint.mask(logits)
        first = sample_rows(logits, [n], sampling)[0]
        if constraint is not None:
            constraint.advance([first])
        return first
    split = 2 * w.head.n == w.config.vocab          # split_weights(..., split_head=True)
    last = _mm(normed, w.head) if split or rank == 0 else None
    if constraint is not None and last is not None:  # each rank masks the vocabulary columns it holds
        constraint.mask(last, None, rank * w.head.n if split else 0)
    if split:
        first = _sample_split(last, [n], sampling, rank)
    else:
        first = [sample_rows(last, [n], sampling)[0]] if rank == 0 else None
    first = _share(first, rank, w.norm.device)[0]
    if constraint is not None:
        constraint.advance([first])
    return first


@torch.no_grad()
def prefill_tp(w: Weights, prompt: Sequence[int], sampling: Sampling | None, rank: int,
               draft=None, *, state: State | None = None, limit: int = 0, stops: Sequence[int] = (),
               keep: Callable | None = None, keep_at: int | None = None, vision=None, constraint=None):
    """Both ranks prefill, from a kept ``state`` with a fresh prefill's bits; rank 0 shares the first token (``keep_at``: a third item, as ``decode.prefill``'s; the split chains are each rank's own)."""

    from .decode import prefill_stops

    st = clone_state(state) if state is not None else State(w)
    if state is None:
        st.limit = limit                    # a fresh state's attention caches stop here; a resumed one keeps its own
    taps = draft is not None and (rank == 0 or getattr(draft, "world", 1) == 2)
    out = prefill_stops(w, prompt, st, draft if taps else None, stops=stops, keep=keep, tp=True, keep_at=keep_at,
                        vision=vision)
    first = first_token(w, out if keep_at is None else out[0], len(prompt), sampling, rank, 2, constraint)
    return (st, first) if keep_at is None else (st, first, out[1])


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
              on_tokens: Callable[[list[int]], bool | None] | None = None,
              inplace: bool = False, constraint=None) -> DecodeResult | None:
    """Draft on rank zero or jointly with a two-rank drafter, then verify and commit on both ranks, whose ``prompt + tokens[:-1]`` agree despite rank one storing -1 for the uncommitted last token (``inplace``: as ``draft_decode``'s)."""

    device = w.norm.device
    split = 2 * w.head.n == w.config.vocab          # split_weights(..., split_head=True)
    st = st if inplace else clone_state(st)
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
    behind = False                  # rank 1 learns a round's last token as the next window's first
    kept = None
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
                if constraint is not None:          # the drafts no accepted path can hold are cut before the share
                    kept = constraint.window(window, parents)
                    window, parents = kept.tokens, kept.parents
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
        if constraint is not None and rank != 0 and behind:
            constraint.advance(window[:1])
        masks = (kept if rank == 0 else constraint.window(window, parents)) if constraint is not None and (
            split or rank == 0) else None
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
        if masks is not None:                       # each rank masks the vocabulary columns it holds
            constraint.mask(logits, masks, rank * w.head.n if split else 0)
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
        if constraint is not None:                  # rank 1 knows the kept drafts now, the last token next round
            constraint.advance([window[r] for r in path[1:]] + ([terminal] if rank == 0 else []))
            behind = rank != 0
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
        return DecodeResult(committed + [-1], seconds, 0, 0, 0)
    return DecodeResult(out, seconds, rounds, drafted_rows, accepted, stages["draft"], stages["verify"],
                        stages["sample"], stages["commit"], widths)
