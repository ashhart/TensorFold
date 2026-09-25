"""DFlash2 block drafts for the lane engine: one small forward proposes the next 7 tokens.

``z-lab/Qwen3.8-27B-DFlash2`` (5 layers, block 8) reads the target's hidden
states at five layers ("taps") for the context it has not yet seen and
proposes a whole block after the pending token in one forward. The lane
engine verifies the block in one row-exact window (``lane_qmm``), keeps the
matching prefix and hands the kept rows' taps back, so the drafter always
conditions on exactly the committed context.

Measured on the M5 Max (2026-09-23), byte exact against serial decoding on
code, HTML and story prompts: 3.0-4.1 tokens per round; with the drafter in
8-bit a round is ~12.5 ms of drafting + ~51 ms of verification (8 rows), so
code runs ~62 tok/s against ~21 serial through the same kernel (~29 with
MLX's own one-row kernel). Draft quality does not affect the output, only
speed: every drafted token is checked against the target's own argmax.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx

# z-lab's reference MLX implementation of the DFlash2 drafter (MIT; see THIRD_PARTY_NOTICES.md), vendored verbatim
_VENDOR = Path(__file__).resolve().parent / "vendor" / "z_lab_dflash" / "model_mlx.py"


def _vendor() -> Any:
    name = "zlab_dflash_model_mlx"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _VENDOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _dflash_attend(attn: Any, x: Any, x_ctx: Any, rope: Any, cache: Any, masks: dict) -> Any:
    """``DFlashAttention.__call__`` with the same arithmetic and fewer host ops.

    [k | v] of the context rows and of the block each come from one stacked lane matmul
    (``lane_fuse.attn_kv``: a column's bits equal its separate call's), and the attention mask is
    built once a lattice per (layer kind, lengths) instead of in each of the five layers: the
    layers' graph build was ~1.9 ms a round of host time (2026-09-24 profile).
    """

    from mlx_lm.models.base import create_causal_mask

    from tensorfold.kernels.qwen.dense.v1 import lane_fuse

    B, L, _ = x.shape
    S = x_ctx.shape[1]
    if attn.is_sliding:
        keep_ctx = attn.sliding_window - 1
        if S > keep_ctx:
            skip = S - keep_ctx
            x_ctx = x_ctx[:, skip:]
            S = x_ctx.shape[1]
            cache.offset += skip
    nh, nkv = attn.n_heads, attn.n_kv_heads
    queries = attn.q_proj(x)
    kv_ctx = lane_fuse.attn_kv(attn, x_ctx)
    kv_x = lane_fuse.attn_kv(attn, x)
    if kv_ctx is None or kv_x is None:
        ctx_keys, ctx_values = attn.k_proj(x_ctx), attn.v_proj(x_ctx)
        prop_keys, prop_values = attn.k_proj(x), attn.v_proj(x)
    else:
        half = kv_x.shape[-1] // 2
        ctx_keys, ctx_values = kv_ctx[..., :half], kv_ctx[..., half:]
        prop_keys, prop_values = kv_x[..., :half], kv_x[..., half:]
    queries = attn.q_norm(queries.reshape(B, L, nh, -1)).transpose(0, 2, 1, 3)
    ctx_keys = attn.k_norm(ctx_keys.reshape(B, S, nkv, -1)).transpose(0, 2, 1, 3)
    ctx_values = ctx_values.reshape(B, S, nkv, -1).transpose(0, 2, 1, 3)
    prop_keys = attn.k_norm(prop_keys.reshape(B, L, nkv, -1)).transpose(0, 2, 1, 3)
    prop_values = prop_values.reshape(B, L, nkv, -1).transpose(0, 2, 1, 3)
    queries = rope(queries, offset=cache.offset + S)
    ctx_keys = rope(ctx_keys, offset=cache.offset)
    prop_keys = rope(prop_keys, offset=cache.offset + S)
    keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
    ctx_len = keys.shape[2]
    keys = mx.concatenate([keys, prop_keys], axis=2)
    values = mx.concatenate([values, prop_values], axis=2)
    key = (attn.is_sliding, attn.is_causal, attn.sliding_window, L, ctx_len)
    mask = masks.get(key, False)
    if mask is False:
        mask = create_causal_mask(L, offset=ctx_len) if attn.is_causal else None
        if attn.is_sliding:
            query = ctx_len + mx.arange(L)[:, None]
            k_pos = mx.arange(ctx_len + L)[None]
            context = (k_pos < ctx_len) & (query - k_pos < attn.sliding_window)
            block = k_pos >= ctx_len
            if attn.is_causal:
                block = block & (k_pos <= query)
            mask = context | block
        masks[key] = mask
    output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=attn.scale, mask=mask)
    return attn.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


def best_first_tree(cands: Any, unary: Any, hproj: Any, noise: Any, anchor: int, pred_code: Any, succ_code: Any, *,
                    temperature: float, edge: float, noise_weight: float, tau: float, children: int,
                    max_nodes: int, prior: Any = None, history: Sequence[int] = ()) -> tuple[list[int], list[int]]:
    """DFlash2's best-first draft tree over its lattice: (tokens, parents), parents index the tokens.

    ``cands`` [D, K] candidate ids per position, ``unary`` [D, K] their logits already divided by
    the temperature, ``hproj`` [D, R] the selector's projected hiddens, ``noise`` [D, K] Gumbel noise
    or None. A child's score is log-softmax over its siblings of (unary + edge * pairwise / T
    [+ noise_weight * Gumbel]) / tau; a node's value is its path's summed score.

    ``prior`` (optional, e.g. ``SessionNGram.rescorer``): (history, depth) -> an additive [K] bonus
    for the candidates at ``depth`` under a node whose path ends in ``history`` (its last three
    tokens; the context's last three, ``history``, for the root's children). A child's score is
    then log-softmax over its siblings of the score above plus the bonus. None: no bonus.
    """

    import heapq

    import numpy as np

    depth_count = int(cands.shape[0])
    count = int(children)                  # children expanded under each node
    t = max(float(temperature), 1e-6) if noise is not None else 1.0
    tables: list[Any] = [None] * depth_count

    def scores(edges: Any, d: int, bonus: Any = None) -> Any:
        s = unary[d] + edge * edges / t
        if noise is not None:
            s = s + noise_weight * noise[d]
        s = s / tau
        if bonus is not None:
            s = s + bonus
        s = s - s.max()
        return s - np.log(np.exp(s).sum())

    def expand(token: int, d: int, hist: tuple = ()) -> Any:
        # a node's children, scored only when it is expanded; a depth's successor table is built
        # once, when a node first needs it (all depths' tables up front cost 2x)
        table = tables[d]
        if table is None:
            table = tables[d] = succ_code[cands[d]].astype(np.float64)
        return scores(table @ (pred_code[token] * hproj[d]), d, None if prior is None else prior(hist, d))

    hists: list[tuple] = []                # each node's last three tokens (with a prior only)
    root_hist = tuple(([None] * 3 + [int(x) for x in list(history)[-3:]])[-3:]) if prior is not None else ()
    root = expand(anchor, 0, root_hist)
    tokens: list[int] = []
    parents: list[int] = []
    heap: list[tuple[float, int, int, int, int]] = []
    for i in np.argsort(-root)[:count]:
        heapq.heappush(heap, (-float(root[i]), -1, int(cands[0][i]), 0, int(i)))
    while heap and len(tokens) < max_nodes:
        neg, parent, token, depth, index = heapq.heappop(heap)
        tokens.append(token)
        parents.append(parent)
        me = len(tokens) - 1
        hist = ()
        if prior is not None:
            above = hists[parent] if parent >= 0 else root_hist
            hist = (above[1], above[2], token)
            hists.append(hist)
        if depth + 1 < depth_count:
            ls = expand(token, depth + 1, hist)
            for j in np.argsort(-ls)[:count]:
                heapq.heappush(heap, (neg - float(ls[j]), me, int(cands[depth + 1][j]), depth + 1, int(j)))
    return tokens, parents


def lattice_gain(cands: Any, unary: Any, hproj: Any, noise: Any, anchor: int, pred_code: Any, succ_code: Any,
                 truth: Sequence[int], prior: Any, history: Sequence[int], *, temperature: float, edge: float,
                 noise_weight: float, tau: float) -> float:
    """How much ``prior`` raised the tokens that did come (``truth``) among their lattice siblings.

    The sum, over the depths while the truth stays among the candidates, of the true token's
    log-softmax with the prior's bonus minus without it; child scores as in ``best_first_tree``.
    """

    import numpy as np

    t = max(float(temperature), 1e-6) if noise is not None else 1.0
    hist = tuple(([None] * 3 + [int(x) for x in list(history)[-3:]])[-3:])
    parent = int(anchor)
    gain = 0.0
    for d in range(min(int(cands.shape[0]), len(truth))):
        token = int(truth[d])
        hits = np.flatnonzero(cands[d] == token)
        if not hits.size:
            break
        j = int(hits[0])
        s = unary[d] + edge * (succ_code[cands[d]].astype(np.float64) @ (pred_code[parent] * hproj[d])) / t
        if noise is not None:
            s = s + noise_weight * noise[d]
        s = s / tau
        b = s + prior(hist, d)
        s_top, b_top = s.max(), b.max()
        gain += float((b[j] - b_top - np.log(np.exp(b - b_top).sum())) - (s[j] - s_top - np.log(np.exp(s - s_top).sum())))
        hist = (hist[1], hist[2], token)
        parent = token
    return gain


def resolve_draft_path(draft: str) -> str:
    """A local directory, or the newest cached snapshot of a Hugging Face repo id."""

    path = Path(draft).expanduser()
    if path.is_dir():
        return str(path)
    repo = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{draft.replace('/', '--')}" / "snapshots"
    hits = sorted(glob.glob(str(repo / "*")))
    if not hits:
        raise FileNotFoundError(f"no local DFlash drafter at {draft} (looked in {repo})")
    return hits[-1]


_CAPTURE_QUEUE: Any = None
_DRAFT_PROFILE = os.environ.get("TF_DRAFT_PROFILE", "") == "1"
_DRAFT_STAGES: dict[str, dict[str, float]] = {}
# TF_DRAFT_LEAN: "1" (default) the lean drafter attention (_dflash_attend, same bits), "0" the vendor call,
# "alt" alternate lattices between them (a fair host-time A/B in one process, with TF_DRAFT_PROFILE)
_DRAFT_LEAN = os.environ.get("TF_DRAFT_LEAN", "1")


def _capture_writer() -> Any:
    """A queue drained by one daemon thread that appends capture records to their files, in order."""

    global _CAPTURE_QUEUE
    if _CAPTURE_QUEUE is None:
        import queue
        import threading

        _CAPTURE_QUEUE = queue.Queue()

        def drain() -> None:
            while True:
                path, record = _CAPTURE_QUEUE.get()
                try:
                    with open(path, "ab") as handle:
                        handle.write(record)
                except OSError as exc:
                    print(f"[lanes] draft capture write failed: {exc}", flush=True)

        threading.Thread(target=drain, name="draft-capture", daemon=True).start()
    return _CAPTURE_QUEUE


def concat_updates(cache: list[Any]) -> list[Any]:
    """Route every update of the drafter's sliding-window caches through the concatenating path.

    mlx_lm's RotatingKVCache writes a one-row update in place, and that path grows the buffer by
    min(step, max_size - offset): negative once the offset (the absolute position, which
    ``prefill_taps`` sets past the window) exceeds the window. A round that kept only its root
    feeds the drafter one context row, so the round after it raised there and drafted nothing:
    126 of 1,494 traced agent rounds (8.4%) committed one token (2026-09-23). The concatenating
    path takes any row count.
    """

    for item in cache:
        if hasattr(item, "_update_concat"):
            item.update_and_fetch = item._update_concat
    return cache


class DFlashDrafter:
    """The shared drafter model; one ``DFlashProposer`` per stream holds that stream's cache."""

    def __init__(self, target_model: Any, draft: str, *, bits: int = 8) -> None:
        vendor = _vendor()
        path = resolve_draft_path(draft)
        vendor.snapshot_download = lambda repo_id, **_: path  # load_draft resolves ids online otherwise
        self.path = path
        self.model = vendor.load_draft(path)
        if bits:
            import mlx.nn as nn

            nn.quantize(self.model, group_size=64, bits=int(bits),
                        class_predicate=lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0)
            mx.eval(self.model.parameters())
        self.model.bind(target_model)
        vendor._patch_model(target_model, list(self.model.config.target_layer_ids))
        self.target = target_model
        self.block_size = int(self.model.config.block_size)
        self.mask_id = int(self.model.config.mask_token_id)
        window = getattr(self.model.config, "sliding_window", None)
        self.window = int(window) - 1 if window else 0
        self._trim = vendor._trim_recent_cache

    def taps(self) -> mx.array | None:
        """The last target forward's taps, [batch, rows, 5 * hidden]."""

        states = getattr(self.target, "_hidden_states", None)
        if not states or any(s is None for s in states):
            return None
        return mx.concatenate(states, axis=-1)

    def release_taps(self) -> None:
        states = getattr(self.target, "_hidden_states", None)
        if states:
            for i in range(len(states)):
                states[i] = None

    def proposer(self, copy: Any = None, sampling: Any = None) -> "DFlashProposer":
        return DFlashProposer(self, copy=copy, sampling=sampling)

    # Draft candidates come from token ids below 98,304 and the control tokens at the top of the
    # vocabulary (<|im_end|>, <tool_call>, ... at 248,044-248,076): 99.98% of the tokens committed
    # in traced agent sessions, from 40% of the head's rows (2.0 -> ~0.8 ms a round). The target
    # still verifies over the whole vocabulary, so output is unchanged. TF_DRAFT_VOCAB=full: all.
    draft_vocab: tuple[tuple[int, int], ...] = ((0, 98304), (248032, 248320))

    def candidate_logits(self, hidden: mx.array) -> tuple[mx.array, mx.array | None]:
        """The head's logits over ``draft_vocab`` and each column's token id, or (all logits, None)."""

        sub = self._sub_head()
        if sub is None:
            return self.model.compute_logits(hidden), None
        from tensorfold.kernels.qwen.dense.v1 import lane_qmm

        weight, sbt, ids, nt = sub
        logits = lane_qmm.lane_matmul(hidden, weight, sbt, tiled=True, nt=nt) * self.model.config.output_multiplier
        cap = self.model.config.final_logit_softcapping
        if cap is not None and cap > 0:
            logits = mx.tanh(logits / cap) * cap
        return logits, ids

    def _sub_head(self) -> tuple[mx.array, mx.array, mx.array, int] | None:
        """``draft_vocab``'s rows of the lane-tiled head (whole 32-row tiles), built once."""

        if getattr(self, "_sub", False) is False:
            self._sub = None
            head = self.model.lm_head
            if (os.environ.get("TF_DRAFT_VOCAB", "") != "full" and getattr(head, "_lane_tiled", False)
                    and getattr(head, "_lane_sbt", None) is not None):
                n = int(head["weight"].shape[0])
                nt = int(getattr(head, "_lane_nt", 32))
                # whole tiles of the head's layout: a span's ends round outward to its tile width
                spans = [(a - a % nt, min(-(-b // nt) * nt, n)) for a, b in self.draft_vocab if a < n]
                if all(a % nt == 0 and b % nt == 0 for a, b in spans) and sum(b - a for a, b in spans) < n:
                    weight = mx.concatenate([head["weight"][a:b] for a, b in spans], axis=0)
                    sbt = mx.concatenate([head._lane_sbt[:, a:b] for a, b in spans], axis=1)
                    ids = mx.concatenate([mx.arange(a, b, dtype=mx.int32) for a, b in spans])
                    mx.eval(weight, sbt, ids)
                    self._sub = (weight, sbt, ids, nt)
        return self._sub


class DFlashProposer:
    """Lane-engine proposer: a DFlash2 block per round, or a long verbatim copy when one is sure.

    The engine calls ``prefill_taps`` after the prompt's prefill and ``absorb`` with the
    kept rows of every precise round; ``invalidate`` when a round could not report taps
    (several streams sharing the batch), after which it proposes nothing.
    """

    name = "dflash2"

    def __init__(self, drafter: DFlashDrafter, *, copy: Any = None, sampling: Any = None) -> None:
        self.drafter = drafter
        # The stream's exact_sampling.Sampling: drafts then take the argmax of the same
        # position-keyed Gumbel noise the target samples with (coupled draws), so a draft
        # lands on the target's own sample wherever the two distributions agree.
        self.sampling = sampling
        self.cache = concat_updates(drafter.model.make_cache())
        self.context: mx.array | None = None   # taps the drafter has not read yet
        self.ready = False
        self.copy = copy
        self.last_confident = False
        self.proposals = 0
        self.proposed_tokens = 0
        self.accepted_tokens = 0
        self.copy_rounds = 0
        self.draft_ms = 0.0
        self.build_ms = self.wait_ms = self.search_ms = 0.0     # tree drafts: graph build, GPU wait, tree search
        self.skipped: dict[str, int] = {}   # rounds without a tree, by reason
        self._last_was_copy = False
        self._copy_level = 0      # a copy taken whole doubles the next copy's window (15, 31, 63, 127 rows)
        self._ngram: Any = None   # draft_ngram.SessionNGram over this stream's context (ngram_weight > 0)
        self._ngram_last: Any = None   # the last lattice, scored against the tokens that follow it
        self._ngram_gain = 0.0    # nats the prior has added to the committed tokens so far (the gate)
        self.ngram_rounds = 0     # trees shaped by the prior
        self.ngram_ms = 0.0       # n-gram upkeep, overlapped with the drafter's GPU work (inside wait_ms)

    # -- engine hooks ------------------------------------------------------------------
    def on_prefill(self, prompt_len: int) -> None:
        """The prompt's prefill forward just ran: read its taps (the new suffix's rows)."""

        taps = self.drafter.taps()
        if taps is not None:
            self.prefill_taps(prompt_len, taps)
        self.drafter.release_taps()

    def on_round(self, row: int, keep: int) -> None:
        """A precise round kept the first ``keep`` rows of batch row ``row``."""

        taps = self.drafter.taps()
        if taps is not None and keep > 0:
            self.absorb(taps[row: row + 1, :keep])

    def prefill_taps(self, prompt_len: int, taps: mx.array) -> None:
        rows = int(taps.shape[1])
        if self.drafter.window and rows > self.drafter.window:
            taps = taps[:, -self.drafter.window:]
            rows = self.drafter.window
        for item in self.cache:
            item.offset = int(prompt_len) - rows
        self.context = mx.contiguous(taps)
        mx.eval(self.context)
        self.ready = True

    def absorb(self, taps: mx.array) -> None:
        if not self.ready:
            return
        # lazy: the drafter's next forward evaluates it (one host sync fewer per round)
        self.context = taps if self.context is None else mx.concatenate([self.context, taps], axis=1)

    def invalidate(self) -> None:
        self.ready = False
        self.context = None

    # -- proposer protocol ---------------------------------------------------------------
    def propose(self, context: Sequence[int], max_draft: int) -> list[int]:
        self.last_confident = False
        self._last_was_copy = False
        if max_draft <= 0:
            return []
        if self.copy is not None:
            drafts = self.copy.propose(context, max_draft)
            if drafts and getattr(self.copy, "last_confident", False):
                self.last_confident = True
                self._last_was_copy = True
                self.copy_rounds += 1
                return drafts
        if not self.ready or self.context is None:
            return []
        block = min(self.drafter.block_size, int(max_draft) + 1)
        if block < 2:
            return []
        started = time.perf_counter()
        inputs = mx.array([[int(context[-1])] + [self.drafter.mask_id] * (block - 1)])
        if self.sampling is None:
            tokens, _, _ = self.drafter.model.propose(inputs, self.context, self.cache, 0.0, logits_start=1)
        else:
            model = self.drafter.model
            hidden = model.hidden_states(inputs, self.context, self.cache, 1)
            tokens = self._coupled(hidden, model.compute_logits(hidden), inputs[:, 0], len(context))
        anchor = len(context) - 1           # the pending token's position
        extra = int(self.cache[0].offset) - anchor
        if extra > 0:
            self.drafter._trim(self.cache, extra)
        out = [int(t) for t in tokens[0].tolist()]
        self.context = None
        self.draft_ms += (time.perf_counter() - started) * 1e3
        self.proposals += 1
        self.proposed_tokens += len(out)
        self.last_confident = True
        return out

    def _coupled(self, hidden: mx.array, logits: mx.array, anchor: mx.array, first_position: int) -> mx.array:
        """DFlash2's candidate path, each step the argmax of score / T + the target's Gumbel noise."""

        import numpy as np

        from tensorfold.engine.exact_sampling import uniform

        selector = self.drafter.model.candidate_selector
        k = int(selector.top_k)
        candidates = mx.argpartition(logits, -k, axis=-1)[..., -k:]
        unary = mx.take_along_axis(logits, candidates, axis=-1).astype(mx.float32)
        ids = np.array(candidates[0]).astype(np.int64)
        noise = np.stack([-np.log(-np.log(uniform(self.sampling.seed, first_position + j, ids[j])))
                          for j in range(ids.shape[0])]).astype(np.float32)
        noise_mx = mx.array(noise)[None]
        projected = selector.hidden_projection(hidden)
        temperature = max(float(self.sampling.temperature), 1e-6)
        predecessor = anchor
        path = []
        for position in range(int(hidden.shape[1])):
            edges = mx.sum(
                selector.predecessor_codebook(predecessor)[:, None]
                * projected[:, position, None]
                * selector.successor_codebook(candidates[:, position]),
                axis=-1,
            )
            scores = (unary[:, position] + edges.astype(mx.float32)) / temperature + noise_mx[:, position]
            chosen = mx.argmax(scores, axis=-1)
            predecessor = mx.take_along_axis(candidates[:, position], chosen[:, None], axis=-1)[:, 0]
            path.append(predecessor)
        return mx.stack(path, axis=1)

    # -- draft trees ----------------------------------------------------------------------
    tree_block = 16          # positions drafted per tree (the head's own block is 8; more is allowed)
    tree_children = 4        # candidates expanded under each node
    tree_nodes = 0           # cap on a DFlash2 tree's nodes when the round's budget is larger (0 = none)
    # Node scores: log-softmax of (unary + tree_edge * pairwise [+ tree_noise * Gumbel]) / tree_tau.
    # Fitted offline on traced lattices (12 prompts, ~1,500 rounds each for sampled and greedy
    # decoding): the head's raw scores are overconfident and its pairwise term too strong;
    # accepted tokens per round +6.5% sampled, +4.2% greedy, gains in every prompt kind.
    tree_tau = 1.5
    tree_edge = 0.6
    tree_noise = 0.7
    # A verbatim copy backed by this many matching tokens replaces the tree (a short one rides
    # in it as a branch). Traced agent rounds: a match of 8+ existed in ~25% of tree rounds and its
    # next token was right 94% of the time; tokens a round 4.49 -> 5.74 and the drafter skipped,
    # against 5.13 -> 5.08 on stories and prose (replayed exactly, 2026-09-23).
    copy_match = int(os.environ.get("TF_COPY_MATCH", "8"))
    # Session n-gram prior (draft_ngram): each child's score gets ngram_weight * log P(child | its
    # path's last 3 tokens) under the stream's own 1..4-gram counts (prompt + committed output).
    # Replayed on 703 traced agent rounds, tokens a pass: 4.494 at 0, 4.566 at 0.05, 4.615 at 0.1,
    # 4.617 at 0.11, 4.593 at 0.15, 4.528 at 0.2. 0 = off: the trees are exactly the plain ones.
    ngram_weight = float(os.environ.get("TF_NGRAM_WEIGHT", "0.1"))
    # Always on it costs short-prompt sessions (stories, code from scratch: 2,800 traced rounds)
    # 2.5-2.9%, so a stream uses it only while it has raised the log-probability of the stream's
    # committed tokens among their lattice siblings by more than this many nats, scored on its
    # earlier rounds (None = always): agent turns +2.6% (of +2.7%), short prompts -0.05% / -0.16%.
    ngram_gate: float | None = 1.0
    # layers after which the drafter's graph so far is sent to the GPU while Python builds the rest
    async_layers: tuple[int, ...] = (0, 2)
    # TF_TREE_TRACE=path appends every round's context growth and each tree's candidate lattice
    # (pickles) for offline tree-policy studies; off unless set
    trace_path = os.environ.get("TF_TREE_TRACE", "")
    # TF_DRAFT_CAPTURE=dir writes, per stream, every context row the drafter reads (its position,
    # token and hidden_norm(fc(taps)) features, bf16) plus the stream's sampling: training data for
    # the drafter on this target's own output. Off unless set; outputs are unchanged.
    capture_dir = os.environ.get("TF_DRAFT_CAPTURE", "")
    # TF_DRAFT_PROFILE=1: where the drafter's graph build (host time) goes, printed every 200 lattices

    def _trace(self, record: dict[str, Any]) -> None:
        with open(self.trace_path, "ab") as handle:
            pickle.dump({"stream": id(self), **record}, handle)

    def _codebooks(self) -> tuple[Any, Any]:
        import numpy as np

        if getattr(self.drafter, "_codes", None) is None:
            selector = self.drafter.model.candidate_selector
            self.drafter._codes = (np.array(selector.predecessor_codebook.weight.astype(mx.float32)),
                                   np.array(selector.successor_codebook.weight.astype(mx.float32)))
        return self.drafter._codes

    def propose_tree(self, context: Sequence[int], max_nodes: int) -> tuple[list[int], list[int]]:
        """A best-first draft tree: (tokens, parents), parents index the tokens (-1 = the pending token).

        Children of a node at depth d are DFlash2's top candidates at position d + 1, scored by
        the head's unary logit plus its pairwise term with the node's token; a node's value is its
        path's summed log-probability. Under exact sampling the scores carry the target's own
        position-keyed Gumbel noise (down-weighted, see ``tree_noise``), so the first child is
        close to the coupled draw.
        """

        return self._finish_tree(context, self._start_tree(context, max_nodes))

    def _start_tree(self, context: Sequence[int], max_nodes: int) -> tuple:
        """``propose_tree`` up to the lattice's submission: ("done", tokens, parents) or the lattice in flight.

        (Starting it at the end of the previous round, before that round's commit is built, measured no
        faster: 57.58 against 57.18 ms a round, 700 rounds each, 2026-09-24. The host's own work between
        forwards, not its order, bounds that phase.)"""

        self.last_confident = False
        self._last_was_copy = False
        if max_nodes <= 0:
            return ("done", [], [])
        if self.trace_path:
            seen = getattr(self, "_traced", 0)
            self._traced = len(context)
            self._trace({"n": len(context), "new": [int(t) for t in context[seen:]],
                         "seed": getattr(self.sampling, "seed", None),
                         "temperature": getattr(self.sampling, "temperature", None)})
        tree_cap = min(int(max_nodes), int(self.tree_nodes)) if self.tree_nodes else int(max_nodes)
        copy_branch: list[int] = []
        if self.copy is not None:
            # each copy taken whole doubles the next one's window, up to the round's budget
            width = min(int(max_nodes), (tree_cap + 1) * 2 ** self._copy_level - 1)
            drafts = self.copy.propose(context, width)
            backed = getattr(self.copy, "last_confident", False) or getattr(self.copy, "last_match", 0) >= self.copy_match
            if drafts and backed and len(drafts) >= min(tree_cap, width):
                self.last_confident = True
                self._last_was_copy = True
                self.copy_rounds += 1
                return ("done", drafts, list(range(-1, len(drafts) - 1)))   # a verbatim copy: a chain
            if drafts and backed:
                copy_branch = drafts[:tree_cap - 1]                  # too short to fill the window
        max_nodes = tree_cap - len(copy_branch)
        block = min(int(self.tree_block), int(max_nodes) + 1)
        if not self.ready or self.context is None or block < 2:
            why = "not_ready" if not self.ready else "no_context" if self.context is None else "no_room"
            self.skipped[why] = self.skipped.get(why, 0) + 1
            return ("done", list(copy_branch), list(range(-1, len(copy_branch) - 1)))
        started = time.perf_counter()
        cands, unary, hproj = self._lattice(context, block)       # on its way through the GPU
        return ("lattice", cands, unary, hproj, copy_branch, max_nodes, started, time.perf_counter())

    def _finish_tree(self, context: Sequence[int], state: tuple) -> tuple[list[int], list[int]]:
        """The rest of ``propose_tree``: wait for the lattice, then the best-first search."""

        import numpy as np

        if state[0] == "done":
            return state[1], state[2]
        _, cands, unary, hproj, copy_branch, max_nodes, started, built = state
        self._ngram_round(context)                                # host work while the GPU runs
        mx.eval(cands, unary, hproj)
        waited = time.perf_counter()
        if self.capture_dir and getattr(self, "_captured", None) is not None:
            self._write_capture(context)
        self.build_ms += (built - started) * 1e3
        self.wait_ms += (waited - built) * 1e3
        anchor = len(context) - 1
        extra = int(self.cache[0].offset) - anchor
        if extra > 0:
            self.drafter._trim(self.cache, extra)
        self.context = None
        # copies of the evaluated arrays (np.array of a slice of one is one more GPU round trip, ~0.15 ms)
        cands_np = np.array(cands).astype(np.int64)
        logits_np = np.array(unary).astype(np.float64)
        unary_np = logits_np
        hproj_np = np.array(hproj)[0].astype(np.float64)
        noise = None
        temp = 1.0
        if self.sampling is not None:
            from tensorfold.engine.exact_sampling import uniform_rows

            temp = max(float(self.sampling.temperature), 1e-6)
            unary_np = unary_np / temp
            noise = -np.log(-np.log(uniform_rows(self.sampling.seed, len(context) + np.arange(cands_np.shape[0]), cands_np)))
        pred_code, succ_code = self._codebooks()
        if self.trace_path:
            self._trace({"n": len(context), "lattice": True, "cands": cands_np, "unary": logits_np,
                         "hproj": hproj_np, "noise": noise, "anchor": int(context[-1])})
        prior = None
        if self._ngram is not None:
            history = [int(t) for t in context[-3:]]
            if self.ngram_gate is None or self._ngram_gain > self.ngram_gate:
                prior = self._ngram.rescorer(cands_np, self.ngram_weight)
                self.ngram_rounds += prior is not None
            if self.ngram_gate is not None:           # scored against the tokens that come, next round
                self._ngram_last = (cands_np, unary_np, hproj_np, noise, temp, int(context[-1]), len(context),
                                    history, prior)
        tokens, parents = best_first_tree(cands_np, unary_np, hproj_np, noise, int(context[-1]), pred_code, succ_code,
                                          temperature=temp, edge=self.tree_edge, noise_weight=self.tree_noise,
                                          tau=self.tree_tau, children=self.tree_children, max_nodes=max_nodes,
                                          prior=prior, history=context[-3:])
        if copy_branch:
            # the copy hangs off the root, sharing any first nodes the tree already has
            parent = -1
            for token in copy_branch:
                hit = next((i for i, (t, q) in enumerate(zip(tokens, parents)) if q == parent and t == token), None)
                if hit is None:
                    tokens.append(int(token))
                    parents.append(parent)
                    hit = len(tokens) - 1
                parent = hit
        finished = time.perf_counter()
        self.search_ms += (finished - waited) * 1e3
        self.draft_ms += (finished - started) * 1e3
        self.proposals += 1
        self.proposed_tokens += len(tokens)
        self.last_confident = True
        return tokens, parents

    def _lattice(self, context: Sequence[int], block: int) -> tuple[mx.array, mx.array, mx.array]:
        """This round's lattice, queued on the GPU: candidate ids [D, K], logits [D, K], projected hiddens [1, D, R].

        ``model.hidden_states`` op for op (the same bits), but the graph built so far goes to the
        GPU after each layer in ``async_layers`` while Python builds the rest: a single eval at
        the end left the GPU idle for the ~1 ms the whole graph takes to build.
        """

        from tensorfold.engine.topk import topk_rows

        model = self.drafter.model
        selector = model.candidate_selector
        prof = _DRAFT_PROFILE
        t0 = time.perf_counter() if prof else 0.0
        inputs = mx.array([[int(context[-1])] + [self.drafter.mask_id] * (block - 1)])
        h = model.embed_tokens(inputs) * model.embed_scale
        h_ctx = model.hidden_norm(model.fc(self.context))
        capture = None
        if self.capture_dir:
            # the rows' bits, evaluated with the lattice (a view taken after cost a GPU round trip a round)
            capture = h_ctx[0].view(mx.uint16)
            self._captured = (int(self.cache[0].offset), capture)   # written once the lattice is evaluated
        if -1 in self.async_layers:
            mx.async_eval(h, h_ctx)
        t1 = time.perf_counter() if prof else 0.0
        parts = self._compiled_parts()
        masks: dict = {}                            # one attention mask a lattice per layer kind (_dflash_attend)
        self._lattices = getattr(self, "_lattices", 0) + 1
        lean = _DRAFT_LEAN == "1" or (_DRAFT_LEAN == "alt" and self._lattices % 2 == 0)
        for i, (layer, cache) in enumerate(zip(model.layers, self.cache)):
            if parts is None:
                h = layer(h, h_ctx, model.rope, cache)
            else:                                   # DFlash2DecoderLayer.__call__, its fixed-shape parts compiled
                pre, post = parts[i]
                xn, kernel = pre(h)
                attended = (_dflash_attend(layer.self_attn, xn, h_ctx, model.rope, cache, masks) if lean
                            else layer.self_attn(xn, h_ctx, model.rope, cache))
                h = post(h, attended, kernel)
            if i in self.async_layers:
                mx.async_eval(h)
        t2 = time.perf_counter() if prof else 0.0
        hidden = model.norm(h[:, 1:])
        logits, vocab_ids = self.drafter.candidate_logits(hidden)
        cands, unary = topk_rows(logits[0], int(selector.top_k))  # radix select (argpartition took ~2 ms)
        if vocab_ids is not None:
            cands = mx.take(vocab_ids, cands)             # columns of the draft vocabulary -> token ids
        hproj = selector.hidden_projection(hidden).astype(mx.float32)
        mx.async_eval(cands, unary, hproj, *(() if capture is None else (capture,)))
        if prof:
            t3 = time.perf_counter()
            acc = _DRAFT_STAGES.setdefault("lean" if lean else "vendor", {})
            for key, value in (("context", t1 - t0), ("layers", t2 - t1), ("head+topk", t3 - t2)):
                acc[key] = acc.get(key, 0.0) + value * 1e3
            acc["n"] = acc.get("n", 0) + 1
            if acc["n"] % 200 == 0:
                print(f"[lanes] drafter build ms/round ({'lean' if lean else 'vendor'} attention): "
                      + ", ".join(f"{k} {v / acc['n']:.2f}" for k, v in acc.items() if k != "n"), flush=True)
        return cands, unary, hproj

    def _compiled_parts(self) -> list[Any] | None:
        """Per DFlash2 layer, its block-only parts under mx.compile: (norm + conv prepare) and (conv finish +
        residual + norm + conv + MLP + conv + residual). They see the same [1, 16, hidden] shapes every round,
        so each traces once; the attention (whose context grows) stays outside. Same outputs (max
        difference 0 against the vendor layer), graph build 0.8 -> 0.4 ms a forward (2026-09-23).
        """

        drafter = self.drafter
        if getattr(drafter, "_parts", None) is None:
            layers = drafter.model.layers
            if not layers or not hasattr(layers[0], "attention_conv") or not hasattr(layers[0], "mlp_conv"):
                drafter._parts = False
            else:
                def make(layer: Any) -> tuple[Any, Any]:
                    def pre(x: mx.array) -> Any:
                        return layer.attention_conv.prepare(layer.input_layernorm(x))

                    def post(residual: mx.array, attn: mx.array, kernel: mx.array) -> mx.array:
                        x = residual + layer.attention_conv.finish(attn, kernel)
                        xn, k2 = layer.mlp_conv.prepare(layer.post_attention_layernorm(x))
                        return x + layer.mlp_conv.finish(layer.mlp(xn), k2)

                    return mx.compile(pre), mx.compile(post)

                drafter._parts = [make(layer) for layer in layers]
        return drafter._parts or None

    def _write_capture(self, context: Sequence[int]) -> None:
        """Append this lattice's context rows to the stream's capture file (see ``capture_dir``).

        Record: int64 first position, int32 rows, int32 width, then the rows' features (bf16 bits)
        and their tokens (int32). The context rows sit at positions [start, start + rows): the
        tokens before the anchor, the last of which is ``context[-2]``.
        """

        import json

        import numpy as np

        start, bits = self._captured
        self._captured = None
        try:
            rows, width = int(bits.shape[0]), int(bits.shape[1])
            tokens = np.asarray([int(t) for t in context[start:start + rows]], dtype=np.int32)
            if len(tokens) != rows:
                return
            if getattr(self, "_capture_file", None) is None:
                folder = Path(self.capture_dir)
                folder.mkdir(parents=True, exist_ok=True)
                name = f"{time.strftime('%Y%m%d-%H%M%S')}-{id(self) & 0xffffff:06x}"
                self._capture_file = folder / f"{name}.bin"
                sampling = self.sampling
                meta = {"seed": getattr(sampling, "seed", None), "temperature": getattr(sampling, "temperature", None),
                        "top_k": getattr(sampling, "top_k", None), "top_p": getattr(sampling, "top_p", None),
                        "first_position": start, "width": width}
                (folder / f"{name}.json").write_text(json.dumps(meta))
            features = np.array(bits)
            record = b"".join((np.array([start], dtype=np.int64).tobytes(),
                               np.array([rows, width], dtype=np.int32).tobytes(),
                               features.tobytes(), tokens.tobytes()))
            _capture_writer().put((self._capture_file, record))   # disk I/O off the round's path
        except Exception as exc:  # noqa: BLE001 - capture must never break a stream
            print(f"[lanes] draft capture failed: {type(exc).__name__}: {exc}", flush=True)
            self.capture_dir = ""

    def capture_target(self, positions: Sequence[int], cand: Any, vals: Any) -> None:
        """Append the target's candidates behind committed tokens to the stream's ``.logits`` sidecar.

        Record: int32 count, int32 k, int64 positions [count], int32 ids [count, k], float32 logits
        [count, k]: the token at each position was drawn among these (logit / temperature plus the
        position's keyed Gumbel noise, see exact_sampling), so a drafter can be distilled on the
        target's own distribution instead of one sampled token. Only while ``capture_dir`` is set.
        """

        import numpy as np

        if not self.capture_dir or getattr(self, "_capture_file", None) is None:
            return
        try:
            ids = np.ascontiguousarray(cand, dtype=np.int32)
            logits = np.ascontiguousarray(vals, dtype=np.float32)
            count, k = ids.shape
            record = b"".join((np.array([count, k], dtype=np.int32).tobytes(),
                               np.asarray(positions, dtype=np.int64).tobytes(), ids.tobytes(), logits.tobytes()))
            _capture_writer().put((self._capture_file.with_suffix(".logits"), record))
        except Exception as exc:  # noqa: BLE001 - capture must never break a stream
            print(f"[lanes] target capture failed: {type(exc).__name__}: {exc}", flush=True)

    def _ngram_round(self, context: Sequence[int]) -> None:
        """Session n-gram upkeep, run while the drafter's forward is on the GPU: score the last lattice's
        prior against the tokens committed since (the gate), then count those tokens."""

        if not self.ngram_weight:
            return
        started = time.perf_counter()
        if self._ngram is None:
            from tensorfold.drafters.draft_ngram import SessionNGram

            self._ngram = SessionNGram(vocab=int(getattr(self.drafter.model.config, "vocab_size", 248320)))
        last, self._ngram_last = self._ngram_last, None
        if last is not None:
            cands, unary, hproj, noise, temp, anchor, n, history, prior = last
            truth = [int(t) for t in context[n: n + int(cands.shape[0])]]
            if truth and len(context) > n:
                if prior is None:           # the counts are still the ones that lattice saw
                    prior = self._ngram.rescorer(cands, self.ngram_weight)
                if prior is not None:
                    pred_code, succ_code = self._codebooks()
                    self._ngram_gain += lattice_gain(cands, unary, hproj, noise, anchor, pred_code, succ_code, truth,
                                                     prior, history, temperature=temp, edge=self.tree_edge,
                                                     noise_weight=self.tree_noise, tau=self.tree_tau)
        self._ngram.update(context)
        self.ngram_ms += (time.perf_counter() - started) * 1e3

    def on_rows(self, rows: Sequence[int]) -> None:
        """A tree round kept these window rows (root first): their taps are the next context."""

        taps = self.drafter.taps()
        if taps is not None and rows:
            self.absorb(mx.take(taps, mx.array(list(rows), dtype=mx.int32), axis=1))

    def observe(self, proposed: int, accepted: int) -> None:
        if self._last_was_copy:
            self._copy_level = min(self._copy_level + 1, 3) if proposed > 0 and accepted == proposed else 0
            observe = getattr(self.copy, "observe", None)
            if callable(observe):
                observe(proposed, accepted)
            return
        self.accepted_tokens += int(accepted)

    def telemetry(self) -> dict[str, Any]:
        out = {"dflash_proposals": self.proposals, "dflash_proposed": self.proposed_tokens,
               "dflash_accepted": self.accepted_tokens, "dflash_ms": round(self.draft_ms, 1),
               "dflash_build_ms": round(self.build_ms, 1), "dflash_wait_ms": round(self.wait_ms, 1),
               "dflash_search_ms": round(self.search_ms, 1),
               "copy_rounds": self.copy_rounds}
        if getattr(self, "skipped", None):
            out["dflash_skipped"] = dict(self.skipped)
        if self._ngram is not None:
            out.update({"ngram_rounds": self.ngram_rounds, "ngram_gain": round(self._ngram_gain, 2),
                        "ngram_ms": round(self.ngram_ms, 1)})
        if self.copy is not None and hasattr(self.copy, "telemetry"):
            out.update({f"copy_{k}": v for k, v in self.copy.telemetry().items()})
        return out


__all__ = ["DFlashDrafter", "DFlashProposer", "concat_updates", "resolve_draft_path"]
