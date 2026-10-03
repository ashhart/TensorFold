"""Per-stream DFlash2 proposals, committed taps, and draft telemetry."""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx

from .dflash_attention import _dflash_attend, concat_updates
from .dflash_capture import _capture_writer
from .dflash_tree import best_first_tree, lattice_gain

_DRAFT_PROFILE = os.environ.get("TF_DRAFT_PROFILE", "") == "1"
_DRAFT_STAGES: dict[str, dict[str, float]] = {}
# TF_DRAFT_LEAN selects lean attention with "1", vendor with "0", or alternating lattices with "alt".
_DRAFT_LEAN = os.environ.get("TF_DRAFT_LEAN", "1")


class DFlashProposer:
    """Keep per-stream committed taps; invalidation suppresses drafts until prefill restores them."""

    name = "dflash2"

    def __init__(self, drafter: DFlashDrafter, *, copy: Any = None, sampling: Any = None) -> None:
        self.drafter = drafter
        # Couple draft draws to the target's position-keyed Gumbel noise.
        self.sampling = sampling
        self.cache = concat_updates(drafter.make_cache())
        self.context: mx.array | None = None   # taps the drafter has not read yet
        self.ready = False
        self.copy = copy
        self.last_confident = False
        # most DFlash2 drafts the next proposal may hold (None: max_draft; 0: no drafter forward, copies only)
        self.model_cap: int | None = None
        # DFlash2 drafts accepted lately and chains cut short lately (decayed each drafted round)
        self.hits = self.misses = 0.0
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
        cap = int(max_draft) if self.model_cap is None else min(int(max_draft), int(self.model_cap))
        if not self.ready or self.context is None or cap <= 0:
            return []
        block = min(self.drafter.block_size, cap + 1)
        if block < 2:
            return []
        started = time.perf_counter()
        inputs = mx.array([[int(context[-1])] + [self.drafter.mask_id] * (block - 1)])
        if not hasattr(self.drafter.model, "candidate_selector"):
            # DFlash without a selector: each position's most likely token, over the draft vocabulary when it has one
            from .dflash_block import block_chain

            tokens = block_chain(self.drafter, inputs, self.context, self.cache)
        elif self.drafter._sub_head() is None and self.drafter._plain_sub_head() is not None:
            # Use the draft vocabulary and radix top-k when the head is not lane-tiled.
            hidden = self.drafter.model.hidden_states(inputs, self.context, self.cache, 1)
            tokens = self._chain_on_draft_vocab(hidden, inputs[:, 0], len(context))
        elif self.sampling is None:
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

    def _chain_on_draft_vocab(self, hidden: mx.array, anchor: mx.array, first_position: int) -> mx.array:
        """Return [1, positions] token ids using unary and pairwise scores, coupled to target position-keyed Gumbel noise."""

        import numpy as np

        from tensorfold.engine.exact_sampling import uniform
        from tensorfold.engine.topk import topk_rows

        selector = self.drafter.model.candidate_selector
        logits, vocab_ids = self.drafter.candidate_logits(hidden)
        cols, unary = topk_rows(logits[0], int(selector.top_k))
        candidates = (mx.take(vocab_ids, cols) if vocab_ids is not None else cols)[None]    # [1, P, k] token ids
        unary = unary.astype(mx.float32)[None]
        projected = selector.hidden_projection(hidden)
        noise_mx = None
        temperature = 1.0
        if self.sampling is not None:
            ids = np.array(candidates[0]).astype(np.int64)
            noise = np.stack([-np.log(-np.log(uniform(self.sampling.seed, first_position + j, ids[j])))
                              for j in range(ids.shape[0])]).astype(np.float32)
            noise_mx = mx.array(noise)[None]
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
            if noise_mx is None:
                scores = unary[:, position] + edges.astype(mx.float32)
            else:
                scores = (unary[:, position] + edges.astype(mx.float32)) / temperature + noise_mx[:, position]
            chosen = mx.argmax(scores, axis=-1)
            predecessor = mx.take_along_axis(candidates[:, position], chosen[:, None], axis=-1)[:, 0]
            path.append(predecessor)
        return mx.stack(path, axis=1)

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
    # Node scores use log-softmax of (unary + tree_edge * pairwise [+ tree_noise * Gumbel]) / tree_tau.
    tree_tau = 1.5
    tree_edge = 0.6
    tree_noise = 0.7
    # A backed copy replaces the tree when long enough; shorter copies become branches.
    copy_match = int(os.environ.get("TF_COPY_MATCH", "8"))
    # A copied token's log-probability as a tree score.
    copy_logp = -0.06
    # Path scores follow tree pop order, with copy_logp per copied token; None means no lattice.
    last_scores: list[float] | None = None
    # Add ngram_weight * log P(child | last three path tokens) from the stream's committed context.
    ngram_weight = float(os.environ.get("TF_NGRAM_WEIGHT", "0.1"))
    # Use the prior only while its cumulative true-token log-probability gain exceeds this gate; None always enables it.
    ngram_gate: float | None = 1.0
    # layers after which the drafter's graph so far is sent to the GPU while Python builds the rest
    async_layers: tuple[int, ...] = (0, 2)
    # TF_TREE_TRACE appends context growth and candidate lattices as pickles when set.
    trace_path = os.environ.get("TF_TREE_TRACE", "")
    # TF_DRAFT_CAPTURE stores context positions, tokens, bf16 projected taps, and sampling settings per stream.
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
        """Return tokens and parent indices from best-first path scores; parent -1 denotes the pending token."""

        return self._finish_tree(context, self._start_tree(context, max_nodes))

    def _start_tree(self, context: Sequence[int], max_nodes: int) -> tuple:
        """``propose_tree`` up to the lattice's submission: ("done", tokens, parents) or the lattice in flight."""

        state = self._tree_prelude(context, max_nodes)
        if state[0] == "done":
            return state
        _, block, copy_branch, max_nodes = state
        started = time.perf_counter()
        cands, unary, hproj = self._lattice(context, block)       # on its way through the GPU
        return ("lattice", cands, unary, hproj, copy_branch, max_nodes, started, time.perf_counter())

    def _tree_prelude(self, context: Sequence[int], max_nodes: int) -> tuple:
        """``_start_tree`` before its lattice: ("done", tokens, parents), or ("need", block, copy_branch, max_nodes)."""

        self.last_confident = False
        self._last_was_copy = False
        self.last_scores = None
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
        cap = getattr(self, "model_cap", None)    # the engine's paying draft count bounds the drafter's nodes
        if cap is not None:
            max_nodes = min(max_nodes, int(cap))
        block = min(int(self.tree_block), int(max_nodes) + 1)
        if not self.ready or self.context is None or block < 2:
            why = "not_ready" if not self.ready else "no_context" if self.context is None else "no_room"
            self.skipped[why] = self.skipped.get(why, 0) + 1
            return ("done", list(copy_branch), list(range(-1, len(copy_branch) - 1)))
        return ("need", block, copy_branch, max_nodes)

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
        # Copy evaluated arrays before slicing to avoid another GPU round trip.
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
        scores: list[float] = []
        tokens, parents = best_first_tree(cands_np, unary_np, hproj_np, noise, int(context[-1]), pred_code, succ_code,
                                          temperature=temp, edge=self.tree_edge, noise_weight=self.tree_noise,
                                          tau=self.tree_tau, children=self.tree_children, max_nodes=max_nodes,
                                          prior=prior, history=context[-3:], node_scores=scores)
        if copy_branch:
            # the copy hangs off the root, sharing any first nodes the tree already has
            parent = -1
            for depth, token in enumerate(copy_branch):
                hit = next((i for i, (t, q) in enumerate(zip(tokens, parents)) if q == parent and t == token), None)
                if hit is None:
                    tokens.append(int(token))
                    parents.append(parent)
                    scores.append(self.copy_logp * (depth + 1))
                    hit = len(tokens) - 1
                parent = hit
        self.last_scores = scores
        finished = time.perf_counter()
        self.search_ms += (finished - waited) * 1e3
        self.draft_ms += (finished - started) * 1e3
        self.proposals += 1
        self.proposed_tokens += len(tokens)
        self.last_confident = True
        return tokens, parents

    def _lattice(self, context: Sequence[int], block: int) -> tuple[mx.array, mx.array, mx.array]:
        """Queue candidate ids [D, K], logits [D, K], and projected hiddens [1, D, R], preserving vendor arithmetic."""

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
        cands, unary = topk_rows(logits[0], int(selector.top_k))  # radix select
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
        """Compile fixed-shape block operations once per layer; growing-context attention stays outside."""

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
        """Append int64 start, int32 rows/width, bf16 features, and int32 tokens for context rows before the anchor."""

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
        """Append int32 count/k, int64 positions, int32 candidate ids, and float32 target logits to the stream sidecar."""

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
        """Score the last prior against newly committed tokens, then count them while the drafter runs on the GPU."""

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
        if proposed > 0:
            self.hits = self.hits * 0.9 + accepted
            self.misses = self.misses * 0.9 + (1.0 if accepted < proposed else 0.0)

    def continue_rate(self) -> float:
        """Estimate continuation probability with decayed acceptance counts and a prior of three successes in five trials."""

        return (self.hits + 3.0) / (self.hits + self.misses + 5.0)

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


# Import after the proposer class so either module can resolve the shared type.
from .dflash_drafter import DFlashDrafter


__all__ = ["DFlashProposer"]
