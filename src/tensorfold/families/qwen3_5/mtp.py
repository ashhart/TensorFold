"""Qwen3.8-27B's MTP head (the checkpoint's ``mtp.*`` tensors) as the lane engine's draft proposer.

At position i the head reads the target's post-norm hidden state h_i (the rows the vocabulary head reads) and the
embedding of token i + 1:

    x = fc([norm_e(embed(t_{i+1})) | norm_h(h_i)])        concatenated embedding first, fc 2D -> D
    x = decoder layer (full attention over the head's own cache, dense MLP), as the model's attention layers
    y = norm(x)                                           logits = lm_head(y), the target's head
    draft for position i + 2 = the target's keyed draw from those logits

``y`` is the next step's "hidden" when drafts chain (step j reads the head's own output and draft j). The head's
attention cache holds one entry per absorbed position, at RoPE position = its index in that cache (MTPLX's
``mtp_position_mode: local``), so a cache rebuilt from part of a prompt stays self-consistent.

The proposer never touches the target's caches. It reads the post-norm hidden states the forward already computes
(``lane_tree.HIDDEN_SINK``: the lane decoder's windows and the prompt's prefill chunks), absorbs the kept rows with
the tokens that followed them, and chains drafts sampled with the target's keyed rule (``engine.gpu_sampling``) at
their own positions, so a draft is the target's own sample wherever the head's distribution agrees with it there.
Drafts change speed only: verification is the target's row-exact window, and every committed token is the target's
sample. Nothing on the draft path needs row invariance, so the head runs MLX's own kernels.

Depth (drafts a round): ``TF_MTP_DEPTH`` = N (fixed, capped by the round's budget) or ``auto`` (default): the depth
with the most expected tokens a millisecond, from the stream's running acceptance at each depth and the measured
wall time of rounds at each depth (the load-time window costs plus the head's step time until measured), one
deeper every ``probe_every`` rounds so the deeper estimate stays current.

The head's attention history of a prompt survives between requests in memory (``MTPDrafter.history``): a request
resumed from a prefix snapshot restores the head's entries for the shared prefix instead of drafting with the
suffix alone.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# norms stored as ``w`` for ``x * (1 + w)`` in the checkpoint (mlx_lm shifts the model's own at load)
_SHIFTED = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight",
            "pre_fc_norm_hidden.weight", "pre_fc_norm_embedding.weight", "norm.weight")


class MTPHead(nn.Module):
    """``mtp.*``: two input norms, fc (2D -> D), one full-attention decoder layer, a final norm."""

    def __init__(self, args: Any) -> None:
        super().__init__()
        from mlx_lm.models.qwen3_5 import DecoderLayer

        d = int(args.hidden_size)
        self.pre_fc_norm_embedding = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.fc = nn.Linear(2 * d, d, bias=False)
        # a full-attention layer (mlx_lm's layer index that is not linear)
        self.layers = [DecoderLayer(args, int(args.full_attention_interval) - 1)]
        self.norm = nn.RMSNorm(d, eps=args.rms_norm_eps)


def load_head(path: str | Path, args: Any) -> MTPHead:
    """The head from a file of ``mtp.*`` tensors (``tools/qwen27_mtp_head.py``: quantized linears as MLX's
    weight/scales/biases, norms in the checkpoint's 1 + w convention) or a checkpoint directory that has them."""

    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
    weights: dict[str, mx.array] = {}
    meta: dict[str, str] = {}
    for f in files:
        loaded, found = mx.load(str(f), return_metadata=True)
        weights.update({k[len("mtp."):]: v for k, v in loaded.items() if k.startswith("mtp.")})
        meta.update(found or {})
    if not weights:
        raise FileNotFoundError(f"{path}: no mtp.* tensors")
    head = MTPHead(args)
    quantized = {k[: -len(".scales")] for k in weights if k.endswith(".scales")}
    if quantized:
        # the file's metadata (tools/qwen27_mtp_head.py), else groups of 64: fc's words a row give the bits
        group_size = int(meta.get("group_size", 64))
        fc_w, fc_s = weights["fc.weight"], weights["fc.scales"]
        bits = int(meta.get("bits", 0)) or int(fc_w.shape[1]) * 32 // (int(fc_s.shape[1]) * group_size)
        nn.quantize(head, group_size=group_size, bits=bits,
                    class_predicate=lambda p, m: hasattr(m, "to_quantized") and p in quantized)
    for name in list(weights):
        if any(name.endswith(s) for s in _SHIFTED) and weights[name].ndim == 1:
            weights[name] = (weights[name].astype(mx.float32) + 1.0).astype(mx.bfloat16)
    head.load_weights(list(weights.items()), strict=True)
    mx.eval(head.parameters())
    return head


class MTPDrafter:
    """The shared head (one per server); ``proposer`` makes one ``MTPProposer`` per stream."""

    # the draft vocabulary: DFlash2's (99.98% of committed tokens in traced agent sessions, 40% of the head's rows);
    # the target still verifies over the whole vocabulary. TF_DRAFT_VOCAB=full: all rows
    draft_vocab: tuple[tuple[int, int], ...] = ((0, 98304), (248032, 248320))
    _spans: list[tuple[int, int]] = []
    # prompts whose head history is kept for resumed requests
    history_size = 4

    def __init__(self, target: Any, path: str | Path, *, most: int = 15) -> None:
        language_model = getattr(target, "language_model", target)
        self.target = language_model
        self.args = language_model.args
        self.embed = language_model.model.embed_tokens
        self.lm_head = language_model.lm_head
        self.path = str(path)
        self.head = load_head(path, self.args)
        self.most = int(most)
        self.window_costs: dict = {}         # the family hands its forward costs down when it is made
        raw = os.environ.get("TF_MTP_DEPTH", "auto").strip().lower()
        self.fixed_depth = 0 if raw in ("", "auto") else max(1, int(raw))
        # TF_MTP_WINDOW=N: the head attends to its last N entries only (0: all of them)
        self.window = int(os.environ.get("TF_MTP_WINDOW", "0") or 0)
        # TF_MTP_CUT=q: a round's drafts end before the first (after the first draft) whose probability under the
        # head is below q (0: off)
        self.cut = float(os.environ.get("TF_MTP_CUT", "0") or 0)
        self.history: list[tuple[np.ndarray, mx.array, mx.array]] = []
        self._sub = self._draft_head()
        self.step_ms = self._time_step()

    def proposer(self, copy: Any = None, sampling: Any = None) -> "MTPProposer":
        return MTPProposer(self, copy=copy, sampling=sampling)

    # -- the head ------------------------------------------------------------------------------------------------
    def _draft_head(self) -> tuple[list[tuple[mx.array, mx.array, mx.array]], mx.array, int, int] | None:
        head = self.lm_head
        if os.environ.get("TF_DRAFT_VOCAB", "") == "full" or not isinstance(head, nn.QuantizedLinear):
            return None
        n = int(head["weight"].shape[0])
        spans = [(a, min(b, n)) for a, b in self.draft_vocab if a < n]
        self._spans = spans
        parts = [(head["weight"][a:b], head["scales"][a:b], head["biases"][a:b]) for a, b in spans]
        ids = mx.concatenate([mx.arange(a, b, dtype=mx.uint32) for a, b in spans])
        mx.eval(ids, *[x for p in parts for x in p])
        return parts, ids, int(head.group_size), int(head.bits)

    def step(self, tokens: mx.array, hidden: mx.array, cache: Any) -> mx.array:
        """The head on rows (next tokens [n], hidden states [n, D]) -> its output y [n, D] (post-norm); the rows enter
        ``cache`` at its next indices."""

        head = self.head
        n = int(hidden.shape[0])
        e = head.pre_fc_norm_embedding(self.embed(tokens.reshape(1, n).astype(mx.uint32)))
        h = head.pre_fc_norm_hidden(hidden.reshape(1, n, -1).astype(e.dtype))
        x = head.fc(mx.concatenate([e, h], axis=-1))
        layer = head.layers[0]
        x = x + self._attend(layer.self_attn, layer.input_layernorm(x), cache, self.window)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        return head.norm(x)[0]

    @staticmethod
    def _attend(attn: Any, x: mx.array, cache: Any, window: int = 0) -> mx.array:
        """Qwen3-Next attention (output gate, q/k norms, partial RoPE at the cache's indices) over the head's cache,
        through MLX's fused attention (the model's own call is routed to the row-exact one, which drafts need not);
        ``window``: the last that many entries before the rows only."""

        B, L, _ = x.shape
        q = attn.q_proj(x).reshape(B, L, attn.num_attention_heads, -1)
        queries, gate = mx.split(q, 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        keys = attn.k_norm(attn.k_proj(x).reshape(B, L, attn.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = attn.v_proj(x).reshape(B, L, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
        queries = attn.rope(queries, offset=cache.offset)
        keys = attn.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
        if window and int(keys.shape[2]) > window + L:
            keys, values = keys[..., -(window + L):, :], values[..., -(window + L):, :]
        out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=attn.scale,
                                                   mask="causal" if L > 1 else None)
        return attn.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1) * mx.sigmoid(gate))

    def draw(self, y: mx.array, sampling: Any, positions: Sequence[int] | mx.array, *,
             confidence: bool = False) -> Any:
        """Drafts (uint32 [n], lazy) for the head's outputs y [n, D] at absolute ``positions``: the target's keyed
        rule over the draft vocabulary (greedy: its argmax). ``confidence``: also each draft's probability under
        the head's own softmax at the sampling temperature (fp32 [n], lazy): (drafts, probabilities)."""

        from tensorfold.engine.gpu_sampling import sample

        if self._sub is None:
            logits, ids = self.lm_head(y), None
        else:
            parts, ids, group_size, bits = self._sub
            logits = mx.concatenate([mx.quantized_matmul(y, w, s, b, transpose=True, group_size=group_size,
                                                         bits=bits) for w, s, b in parts], axis=-1)
        logits = logits.reshape(-1, logits.shape[-1])
        tokens = sample(logits, sampling, positions, ids=ids)
        if not confidence:
            return tokens
        scaled = logits.astype(mx.float32) / max(float(getattr(sampling, "temperature", 1.0) or 1.0), 1e-6)
        column = tokens.astype(mx.int32)
        if self._sub is not None:
            # the draft vocabulary's column of each id: its span's first column plus the id's offset in the span
            first, out = 0, mx.zeros_like(column)
            for a, b in self._spans:
                inside = (column >= a) & (column < b)
                out = mx.where(inside, column - a + first, out)
                first += b - a
            column = out
        chosen = mx.take_along_axis(scaled, column[:, None], axis=-1)[:, 0]
        return tokens, mx.exp(chosen - mx.logsumexp(scaled, axis=-1))

    def _time_step(self) -> float:
        """One chained draft step (the head, the draft vocabulary, a draw read back), ms, fastest of 6."""

        from mlx_lm.models.cache import KVCache

        cache = KVCache()
        d = int(self.args.hidden_size)
        vocab = int(self.args.vocab_size)
        y = self.step(mx.array([(1000 + i) % vocab for i in range(64)], dtype=mx.uint32),
                      mx.zeros((64, d), dtype=mx.bfloat16), cache)
        mx.eval(y)
        y = y[-1:]
        best = float("inf")
        for i in range(6):
            started = time.perf_counter()
            token = self.draw(y, None, [100 + i])
            y = self.step(token, y, cache)
            mx.eval(y)
            token.item()
            best = min(best, (time.perf_counter() - started) * 1e3)
        return round(best, 3)

    # -- history kept between requests -----------------------------------------------------------------------------
    def remember(self, tokens: Sequence[int], cache: Any, rows: int) -> None:
        """Keep the head's first ``rows`` entries (positions 0 .. rows - 1, which read tokens up to ``rows``) for
        prompts that share ``tokens[:rows + 1]``."""

        if rows <= 0 or cache.keys is None:
            return
        keys, values = cache.keys[..., :rows, :], cache.values[..., :rows, :]
        mx.eval(keys, values)
        entry = (np.asarray(list(tokens[: rows + 1]), dtype=np.int64), keys, values)
        self.history = [h for h in self.history if not (len(h[0]) <= len(entry[0])
                                                       and np.array_equal(h[0], entry[0][: len(h[0])]))]
        self.history.append(entry)
        del self.history[: -self.history_size]

    def recall(self, tokens: Sequence[int], rows: int) -> tuple[int, mx.array | None, mx.array | None]:
        """The most head entries (up to ``rows``) kept for a prompt that starts like ``tokens``: (count, keys,
        values)."""

        want = np.asarray(list(tokens[: rows + 1]), dtype=np.int64)
        best, hit = 0, None
        for entry in self.history:
            stored = entry[0]
            n = min(len(stored), len(want))
            diff = np.flatnonzero(stored[:n] != want[:n])
            common = int(diff[0]) if diff.size else n
            usable = min(rows, common - 1, int(entry[1].shape[2]))
            if usable > best:
                best, hit = usable, entry
        if hit is None:
            return 0, None, None
        return best, hit[1][..., :best, :], hit[2][..., :best, :]


class MTPProposer:
    """One stream's MTP drafts: the head's cache, the kept rows not yet absorbed, and the depth policy."""

    name = "mtp"
    # per-depth acceptance before the stream has any (draft j kept given drafts 1 .. j - 1 were)
    depth_prior = (0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    depth_rate = 0.15
    probe_every = 8
    cost_rate = 0.2
    prompt_chunk = 1024          # prompt rows a head call absorbs

    def __init__(self, drafter: MTPDrafter, *, copy: Any = None, sampling: Any = None) -> None:
        from mlx_lm.models.cache import KVCache

        self.drafter = drafter
        self.copy = copy
        self.sampling = sampling
        self.cache = KVCache()
        self.sink: list[Any] = []           # the forward appends its rows' post-norm hidden states here
        self.rows: list[tuple[int, mx.array]] = []   # (first position, hidden [n, D]) kept, not absorbed yet
        self.next_row = -1                  # the position of the next row the sink delivers (-1: not following)
        self.drafted = 0                    # chained draft entries at the end of the head's cache
        self.ready = False
        self.contiguous = False
        self.last_confident = False
        self._last_was_copy = False
        self._round: tuple[int, float] | None = None   # (depth, started) of the round in flight
        self._rates = [float(p) for p in self.depth_prior[: max(1, drafter.most)]]
        self._round_ms: dict[int, float] = {}
        self._rounds = 0
        self.prompt_tokens: list[int] = []
        # telemetry
        self.proposals = 0
        self.proposed_tokens = 0
        self.accepted_tokens = 0
        self.copy_rounds = 0
        self.draft_ms = 0.0
        self.build_ms = self.wait_ms = 0.0      # drafting: graph build (host) and the wait for the drafts
        self.absorb_ms = 0.0
        self.restored = 0
        self.depth_counts: dict[int, int] = {}
        self.depth_tried = [0] * len(self._rates)
        self.depth_kept = [0] * len(self._rates)
        # acceptance by the head's probability of its draft: slot b (drafts after the first) or 10 + b (first
        # drafts) for probabilities in [b / 10, (b + 1) / 10): [reached, kept]
        self.calibration: dict[int, list[int]] = {}

    # -- engine hooks --------------------------------------------------------------------------------------------
    def begin_prefill(self, prompt_ids: Sequence[int], start: int) -> None:
        """A prefill of ``prompt_ids`` from position ``start`` begins: a fresh head cache, with the entries kept
        for the shared prefix restored."""

        from mlx_lm.models.cache import KVCache

        self.cache = KVCache()
        self.sink.clear()
        self.rows = []
        self.drafted = 0
        self.prompt_tokens = [int(t) for t in prompt_ids]
        start = int(start)
        self.restored = 0
        if start > 0:
            count, keys, values = self.drafter.recall(self.prompt_tokens, start)
            if count:
                self.cache.update_and_fetch(keys, values)
                self.restored = count
        # entry i is position i only when nothing is missing before the prefilled rows (else the history is not kept)
        self.contiguous = self.restored == start
        self.next_row = start
        self.ready = True

    def on_prefill(self, prompt_len: int) -> None:
        """The prompt's rows are in the sink: absorb all but the last (its next token is the first sample)."""

        if not self.ready or self.next_row < 0:
            self.sink.clear()
            return
        self._take_sink()
        started = time.perf_counter()
        tokens = self.prompt_tokens
        pending = self.rows
        self.rows = []
        # with a window the head never attends past its last ``window`` entries: older prompt rows are skipped
        skip = len(tokens) - 1 - self.drafter.window if self.drafter.window else 0
        for first, hidden in pending:
            n = int(hidden.shape[0])
            known = max(0, min(n, len(tokens) - 1 - first))       # rows whose next token is in the prompt
            lead = min(known, max(0, skip - first))
            if lead:
                self.contiguous = False
            for begin in range(lead, known, self.prompt_chunk):
                end = min(known, begin + self.prompt_chunk)
                nxt = mx.array(tokens[first + begin + 1: first + end + 1], dtype=mx.uint32)
                y = self.drafter.step(nxt, hidden[begin:end], self.cache)
                mx.eval(y, self.cache.keys, self.cache.values)
            if known < n:
                self.rows.append((first + known, hidden[known:]))
        self.absorb_ms += (time.perf_counter() - started) * 1e3
        if self.contiguous:
            self.drafter.remember(tokens, self.cache, self.cache.offset)

    def _take_sink(self) -> None:
        """Rows the sink holds since the last read, in order, as kept rows from ``next_row`` on (prefill)."""

        for hidden in self.sink:
            h = hidden.reshape(-1, hidden.shape[-1])
            self.rows.append((self.next_row, h))
            self.next_row += int(h.shape[0])
        self.sink.clear()

    def on_rows(self, rows: Sequence[int]) -> None:
        """A tree round kept these rows of its window (root first): the last forward's hidden states of them."""

        if self._round is not None:
            depth, started = self._round
            self._round = None
            ms = (time.perf_counter() - started) * 1e3
            before = self._round_ms.get(depth)
            # a round several times its estimate (a first-use compile, a stall) says nothing about the depth
            estimate = before if before is not None else self._cost(depth)
            if estimate is None or ms < 3.0 * estimate:
                self._round_ms[depth] = ms if before is None else before + self.cost_rate * (ms - before)
        if not self.ready or not self.sink or not rows:
            self.sink.clear()
            return
        hidden = self.sink[-1].reshape(-1, self.sink[-1].shape[-1])
        self.sink.clear()
        rows = [int(r) for r in rows]
        kept = hidden[: len(rows)] if rows == list(range(len(rows))) else hidden[mx.array(rows, dtype=mx.int32)]
        self.rows.append((self.next_row, kept))
        self.next_row += len(rows)

    def on_round(self, row: int, keep: int) -> None:
        """A precise round kept the first ``keep`` rows of batch row ``row``."""

        if not self.ready or not self.sink or keep <= 0:
            self.sink.clear()
            return
        hidden = self.sink[-1][row]
        self.sink.clear()
        self.rows.append((self.next_row, hidden[:keep]))
        self.next_row += int(keep)

    def invalidate(self) -> None:
        """A round the head cannot follow (several streams sharing it): no drafts for the rest of the request."""

        self.ready = False
        self.rows = []
        self.sink.clear()

    # -- proposing -----------------------------------------------------------------------------------------------
    def propose(self, context: Sequence[int], max_draft: int) -> list[int]:
        self.last_confident = False
        self._last_was_copy = False
        self._round = None
        if max_draft <= 0:
            return []
        if self.copy is not None:
            drafts = self.copy.propose(context, max_draft)
            if drafts and getattr(self.copy, "last_confident", False):
                self.last_confident = True
                self._last_was_copy = True
                self.copy_rounds += 1
                if self.ready:
                    y = self._absorb(context)       # keep the head current, on the GPU while the round verifies
                    if y is not None:
                        mx.async_eval(y)
                return drafts
        if not self.ready:
            return []
        started = time.perf_counter()
        y = self._absorb(context)
        if y is None:
            return []
        depth = self._depth(int(max_draft))
        position = len(context)                  # the first draft's position (the pending token sits before it)
        chain, probs = [], []
        for j in range(depth):
            if j:
                y = self.drafter.step(chain[-1], y, self.cache)
                self.drafted += 1
            token, prob = self.drafter.draw(y, self.sampling, [position + j], confidence=True)
            chain.append(token)
            probs.append(prob)
            mx.async_eval(token, prob)           # the GPU runs each step while the host builds the next
        built = time.perf_counter()
        values = mx.concatenate([mx.concatenate(chain).astype(mx.float32), mx.concatenate(probs)]).tolist()
        out = [int(t) for t in values[:depth]]
        self._last_q = values[depth:]
        cut = self.drafter.cut
        if cut > 0:
            # drafts after the first the head itself doubts would mostly be verify rows spent on rejects
            keep = next((j for j, q in enumerate(self._last_q) if j and q < cut), depth)
            out = out[:keep]
        finished = time.perf_counter()
        self.build_ms += (built - started) * 1e3
        self.wait_ms += (finished - built) * 1e3
        self.draft_ms += (finished - started) * 1e3
        self.proposals += 1
        self.proposed_tokens += len(out)
        self.depth_counts[depth] = self.depth_counts.get(depth, 0) + 1
        self.last_confident = True
        self._round = (depth, started)
        return out

    def _absorb(self, context: Sequence[int]) -> mx.array | None:
        """Forget the last chain's drafts, absorb the kept rows with the tokens that followed them; the last
        row's output [1, D] (it reads the pending token), or None when the head is not at the pending token."""

        started = time.perf_counter()
        if self.drafted:
            self.cache.trim(self.drafted)
            self.drafted = 0
        if not self.rows:
            return None
        first = self.rows[0][0]
        hidden = self.rows[0][1] if len(self.rows) == 1 else mx.concatenate([h for _, h in self.rows], axis=0)
        n = int(hidden.shape[0])
        self.rows = []
        if first + n != len(context) - 1:
            # lost track of the positions (a round the hooks did not see): stop drafting for this request
            self.invalidate()
            return None
        nxt = mx.array([int(t) for t in context[first + 1: first + n + 1]], dtype=mx.uint32)
        y = self.drafter.step(nxt, hidden, self.cache)
        self.absorb_ms += (time.perf_counter() - started) * 1e3
        return y[-1:]

    # -- depth ---------------------------------------------------------------------------------------------------
    def _cost(self, depth: int) -> float | None:
        if depth in self._round_ms:
            return self._round_ms[depth]
        costs = getattr(self.drafter, "window_costs", None) or {}
        forward = costs.get(depth + 1)
        if forward is None:
            return None
        return forward + self.drafter.step_ms * depth

    def _depth(self, most: int) -> int:
        most = max(1, min(int(most), self.drafter.most, len(self._rates)))
        if self.drafter.fixed_depth:
            return min(most, self.drafter.fixed_depth)
        if self._cost(1) is None:
            # no window costs (Flash Next's rule): 1 below 80% first-draft acceptance, 2 below 90%, else 3
            rate = self._rates[0]
            return max(1, min(most, 1 if rate < 0.8 else 2 if rate < 0.9 else 3))
        best, best_rate = 1, -1.0
        expected = run = 1.0
        for d in range(1, most + 1):
            cost = self._cost(d)
            if cost is None:
                break
            run *= self._rates[d - 1]
            expected += run
            if expected / cost > best_rate:
                best, best_rate = d, expected / cost
        self._rounds += 1
        if best < most and self._rounds % self.probe_every == 0:
            best += 1
        return best

    def observe(self, proposed: int, accepted: int) -> None:
        if self._last_was_copy:
            observe = getattr(self.copy, "observe", None)
            if callable(observe):
                observe(proposed, accepted)
            return
        self.accepted_tokens += int(accepted)
        q = getattr(self, "_last_q", [])
        for j in range(min(int(proposed), len(self._rates))):
            if accepted < j:
                break
            kept = 1.0 if accepted > j else 0.0
            self._rates[j] += self.depth_rate * (kept - self._rates[j])
            self.depth_tried[j] += 1
            self.depth_kept[j] += int(kept)
            if j < len(q):
                # draft j reached (drafts before it kept): kept or not, by the head's own probability of it
                slot = self.calibration.setdefault(min(9, int(q[j] * 10)) if j else 10 + min(9, int(q[j] * 10)),
                                                   [0, 0])
                slot[0] += 1
                slot[1] += int(kept)

    def telemetry(self) -> dict[str, Any]:
        out = {"mtp_proposals": self.proposals, "mtp_proposed": self.proposed_tokens,
               "mtp_accepted": self.accepted_tokens, "mtp_ms": round(self.draft_ms, 1),
               "mtp_build_ms": round(self.build_ms, 1), "mtp_wait_ms": round(self.wait_ms, 1),
               "mtp_absorb_ms": round(self.absorb_ms, 1), "mtp_restored": self.restored,
               "mtp_depths": {str(k): v for k, v in sorted(self.depth_counts.items())},
               "mtp_accept_by_depth": [round(k / t, 3) if t else None
                                       for k, t in zip(self.depth_kept, self.depth_tried) if t],
               "mtp_tried_by_depth": [t for t in self.depth_tried if t],
               "mtp_round_ms": {str(k): round(v, 1) for k, v in sorted(self._round_ms.items())},
               "mtp_calibration": {str(k): v for k, v in sorted(self.calibration.items())},
               "copy_rounds": self.copy_rounds}
        if self.copy is not None and hasattr(self.copy, "telemetry"):
            out.update({f"copy_{k}": v for k, v in self.copy.telemetry().items()})
        return out


# ----------------------------------------------------------------------------------------------------------------------
# Upstream (0.3.6.x) family-protocol adapter: the head as a ``drafter`` for ``Qwen35Family``
# ----------------------------------------------------------------------------------------------------------------------


class MTPStream:
    """One stream's MTP drafting, on the DFlashProposer contract (``absorb``/``_start_tree``/``_finish_tree``).

    Real (hidden, next-token) pairs arrive through ``absorb``; they are consumed lazily at the next proposal —
    each row's keys/values enter the head's cache, and the last row's head output starts the chain. Drafted rows
    are trimmed off the cache after the chain, as the target has not committed them."""

    name = "mtp"
    depth_prior = MTPProposer.depth_prior

    def __init__(self, drafter: "MTPDrafter", *, copy: Any = None, sampling: Any = None) -> None:
        from mlx_lm.models.cache import KVCache

        self.drafter = drafter
        self.copy = copy
        self.sampling = sampling
        self.cache = [KVCache()]
        self.context: mx.array | None = None       # absorbed hidden rows [1, n, D] not yet consumed
        self.pending: list[int] | None = None      # each row's next token (the pair step() consumes)
        self.ready = False
        self.last_confident = False
        self.last_scores: list[float] | None = None
        self._last_was_copy = False
        self.copy_rounds = 0
        self.draft_ms = 0.0
        self.proposed_tokens = 0
        # attrs the slot/engine may poke before we set them
        self.ngram_weight = 0.0
        self.trace_path = self.capture_dir = ""
        self.tree_block = 1

    def absorb(self, taps: mx.array, next_tokens: Any = None) -> None:
        """Append kept rows' hidden states; ``next_tokens`` carries each row's following token when known."""

        rows = int(taps.shape[1])
        if next_tokens is None:
            nxt = [0] * rows
        else:
            nxt = [int(t) for t in (next_tokens.tolist() if hasattr(next_tokens, "tolist") else next_tokens)]
            if len(nxt) < rows:
                nxt += [0] * (rows - len(nxt))
        if self.context is None:
            self.context, self.pending = taps, nxt
        else:
            self.context = mx.concatenate([self.context, taps], axis=1)
            self.pending = (self.pending or []) + nxt

    def invalidate(self) -> None:
        self.ready = False
        self.context, self.pending = None, None

    def on_prefill(self, prompt_len: int) -> None:
        pass                            # consumption is lazy: ``propose`` runs the pending rows through the head

    # -- consuming the real rows --------------------------------------------------------------------------------------
    def _consume(self) -> mx.array | None:
        """Run the pending (hidden, token) pairs through the head; returns the last row's output [1, D]."""

        if self.context is None or self.pending is None:
            return None
        hidden = self.context.reshape(-1, self.context.shape[-1])
        pending = self.pending
        n = min(len(pending), int(hidden.shape[0]))
        self.context, self.pending = None, None
        if not n:
            return None
        y = None
        cache = self.cache[0]
        for begin in range(0, n, 1024):
            end = min(n, begin + 1024)
            y = self.drafter.step(mx.array(pending[begin:end], dtype=mx.uint32), hidden[begin:end], cache)
            mx.eval(y, cache.keys, cache.values)
        return y[-1:] if y is not None else None

    # -- the chain ----------------------------------------------------------------------------------------------------
    def _chain(self, anchor: int, position: int, max_draft: int) -> tuple[list[int], list[float]]:
        """Draft up to ``max_draft`` tokens after the anchor, chaining the head's own outputs; trimmed afterwards."""

        drafter, cache = self.drafter, self.cache[0]
        y = self._consume()
        if y is None:
            return [], []
        cap = max(1, min(int(max_draft), int(drafter.most)))
        if drafter.fixed_depth:
            cap = max(1, min(cap, int(drafter.fixed_depth)))
        start = int(cache.offset)
        tokens: list[int] = []
        scores: list[float] = []
        for i in range(cap):
            draft, probs = drafter.draw(y, self.sampling, [position + i + 1], confidence=True)
            token = int(draft[0].item())
            prob = float(probs[0].item())
            if i and drafter.cut and prob < drafter.cut:
                break
            tokens.append(token)
            scores.append(float(np.log(max(prob, 1e-9))))
            if i + 1 == cap:
                break
            y = drafter.step(draft, y, cache)
            mx.eval(y)
        if int(cache.offset) > start:
            cache.trim(int(cache.offset) - start)
        return tokens, scores

    # -- proposer protocol ---------------------------------------------------------------------------------------------
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
        if not self.ready:
            return []
        started = time.perf_counter()
        tokens, scores = self._chain(int(context[-1]), len(context) - 1, int(max_draft))
        self.last_scores = scores or None
        self.last_confident = bool(tokens)
        self.proposed_tokens += len(tokens)
        self.draft_ms += (time.perf_counter() - started) * 1e3
        return tokens

    def _start_tree(self, context: Any, max_nodes: int) -> tuple:
        """``propose`` as a chain-shaped tree: synchronous, so the state returned is always ``("done", ...)``."""

        tokens = self.propose(context, max(1, int(max_nodes) - 1))
        return ("done", tokens, list(range(-1, len(tokens) - 1)))

    def _finish_tree(self, context: Any, state: tuple) -> tuple[list[int], list[int]]:
        _, tokens, parents = state
        return [int(t) for t in tokens], [int(p) for p in parents]


def _dflash_head() -> Any:
    from tensorfold.families.qwen3_5.dflash_head import DFlashHead

    return DFlashHead


class MTPDrafts(_dflash_head()):
    """The ``head_drafts`` protocol over an ``MTPDrafter``: hidden taps arrive paired with their next tokens.

    The chain proposer needs no batched lattice, so ``draft_streams`` falls back to the per-stream path."""

    def absorb(self, cache: list[Any], first: int, rows: int, row: int = 0, next_tokens: Any = None) -> None:
        taps = self.drafter.taps()
        if taps is None:
            return
        taps = taps[:, row: row + rows]
        nxt = next_tokens[row: row + rows] if next_tokens is not None else None
        proposer = cache[-1].get(getattr(cache[-1].proposer, "sampling", None))
        window = self.drafter.window
        if not proposer.ready:
            if window and rows > window:
                cut = rows - window
                taps, first = taps[:, -window:], first + cut
                nxt = nxt[cut:] if nxt is not None else None
            for item in proposer.cache:
                item.offset = first
            proposer.context = mx.contiguous(taps)
            proposer.pending = [int(t) for t in nxt.tolist()] if nxt is not None else None
            proposer.ready = True
            return
        proposer.absorb(taps, nxt)
        held = int(proposer.context.shape[1])
        if window and held > window:
            cut = held - window
            proposer.context = proposer.context[:, cut:]
            proposer.pending = (proposer.pending or [])[cut:]
            for item in proposer.cache:
                item.offset += cut

    def read(self, cache: list[Any], rows: Sequence[int], follow: Sequence[int], sampling: Any) -> None:
        """Kept rows' hidden states, each paired with the token committed at its position + 1 (``follow``)."""

        proposer = cache[-1].get(sampling)
        taps = self.drafter.taps()
        if taps is not None and rows:
            proposer.absorb(mx.take(taps, mx.array([int(r) for r in rows], dtype=mx.int32), axis=1),
                            list(follow)[: len(rows)])
        cache[-1].kept = [int(t) for t in follow]
        cache[-1].anchor = cache[-1].kept[-1]

    def draft_streams(self, caches: Sequence[list[Any]], follows: Sequence[Sequence[int]],
                      rows: Sequence[Sequence[int]], positions: Sequence[int], samplings: Sequence[Any],
                      depths: Sequence[int]) -> list[Any]:
        """No batched head lattice: each stream reads its kept rows and draws its chain one at a time."""

        from tensorfold.families.qwen3_5.dflash_head import _Context

        out = []
        for cache, follow, kept, position, sampling, depth in zip(caches, follows, rows, positions, samplings, depths):
            self.read(cache, kept, follow, sampling)
            proposer = cache[-1].proposer
            context = _Context(int(position), int(follow[-1]))
            tree = proposer._finish_tree(context, proposer._start_tree(context, depth))
            out.append(self._drafts(cache[-1], tree, position, sampling, depth))
        return out


# The upstream family protocol mounts the drafter through ``Qwen35Family``: these hooks make MTPDrafter satisfy it.
def _stash_taps(self: Any, hidden: mx.array) -> None:
    self._taps = hidden


def _taps(self: Any) -> mx.array | None:
    return self._taps


def _stream_proposer(self: Any, copy: Any = None, sampling: Any = None) -> MTPStream:
    return MTPStream(self, copy=copy, sampling=sampling)


MTPDrafter._taps = None
MTPDrafter.stash_taps = _stash_taps
MTPDrafter.taps = _taps
MTPDrafter.head_class = MTPDrafts
MTPDrafter.reads_hidden = True            # the head drafts from the last hidden state
MTPDrafter.block_size = 1                     # a chain step; block size is only DFlash's tree depth
MTPDrafter.proposer = _stream_proposer        # the upstream protocol's proposer (the event-driven one is retired)


__all__ = ["MTPDrafter", "MTPDrafts", "MTPHead", "MTPProposer", "MTPStream", "load_head"]
