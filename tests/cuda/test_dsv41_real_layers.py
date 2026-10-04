"""Real weights against the fp32 reference port (``TF_DSV41_MODEL``): blocks and DSpark stages fed
TF's own stream, and a reduced model through TF's prompt path.

Blocks 0, 1, 2, 3, 8, 14, 20, 21, 24, 25, 37 and 39 each take the bf16 stream and FFN pre that TF's forward gives
them, on one rank, at contexts 100, 1,100 and 17,000 (the prompt rows that end each context, then decode windows
at even and odd positions). The reference builds its caches from every row and runs the whole block on those rows,
in mirror mode (rel-L2 2%, cosine 0.9999) and fp32 mode. Both attend TF's index lists. The fp32 and mirror references
differ most on high-norm rows, so TF's fp32 distance is gated at max(3%, the mirror reference's own + 4 x 2^-8)
over the same rows: all but those the fp32 reference routes to other experts than TF at a near-tie (reported).
TF's next pre-mix (the FFN mix's pre, which collapses the next block's stream) takes the mirror gate on those rows.
TF's routed sets, index lists and candidate blocks must equal an fp64 selection on TF's own inputs except at
near-ties.
TF runs one layer at a time with only that layer's weights on the device, so the 40 blocks fit one GPU.

Each DSpark stage runs TF's block [y, noise...] after each context's prompt and first window, its rings absorbed from
TF's taps; the reference runs the stage on TF's stage input, gated as a block. The reduced model's logits are near flat,
so TF's KL and rel-L2 against the mirror reference must not exceed the fp32 reference's own, and its top-1 must hold on
99% of the rows the reference decides (its top-1 margin above its own fp32-vs-mirror spread; at least 20% of rows).

Reported beside the gates: fp32 mode with its FP8/FP4 quantizers fed mirror mode's inputs (``fp32_qdq``: a step flip
from a 2^-9 input change is not a rounding error), and the mirror reference split over two ranks (only its fp32 sums
reordered) against itself, the reduced model's floor for any implementation that is not bit-identical.

The tokens are DeepSeek's MIT reference sources shipped in the checkpoint (``CORPUS``): paragraphs in the order a
seeded ``random.Random`` shuffles them, tokenized after BOS. ``TF_DSV41_REPORT=<path>`` writes every figure as JSON.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
MODEL = os.environ.get("TF_DSV41_MODEL", "")
if not MODEL or not Path(MODEL).is_dir():
    pytest.skip("set TF_DSV41_MODEL to the checkpoint", allow_module_level=True)

import dsv41_score as score
import torch.nn.functional as tnf
from dsv41_ref_weights import RefWeights, exl3_weight
from dsv41_reference import Mode, State, gate, hc_post, hc_pre, rms_norm
from safetensors import safe_open
from tokenizers import Tokenizer

from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import MAX_ROWS, PREFILL_ROWS, buffers, dspark, engram, loader
from tensorfold.families.deepseek_v41.cuda import forward as F
from tensorfold.families.deepseek_v41.cuda import rope as tf_rope
from tensorfold.families.deepseek_v41.cuda.weights import Weights
from tensorfold.families.glm5_next.cuda import glue

LAYERS = (0, 1, 2, 3, 8, 14, 20, 21, 24, 25, 37, 39)
CORPUS = ("inference/*.py", "inference/README.md", "encoding/*.py", "encoding/README.md")
SEED = 0
CONTEXTS = (100, 1_100, 17_000)
DECODE = ((5, False), (1, False), (1, False), (2, False))   # from an even start: rows at even and odd positions
PLAN = ((100, True), *DECODE, (991, True), *DECODE, *((PREFILL_ROWS, True),) * 7,
        (CONTEXTS[2] - CONTEXTS[1] - 9 - 7 * PREFILL_ROWS, True), *DECODE)
ROWS = sum(rows for rows, _ in PLAN)
CAP = 17_024
TAIL = 256                              # compared prompt rows that end each context
BLOCK = PREFILL_ROWS                    # rows the reference writes caches for at once
MIRROR, FP32 = Mode("mirror", world=1), Mode("fp32", world=1)     # one rank: row sums over the whole input dim
MODES = {"mirror": MIRROR, "fp32": FP32, "fp32_qdq": FP32}
REL_MIRROR, COS, REL_FP32, MARGIN = 0.02, 0.9999, 0.03, 4 * 2**-8     # the model-level gate, one block
TIE = 4                                 # a near-tie: the fp64 margin below TIE x the fp32 bound
DSPARK_AT = (99, 104, 1_099, 1_104, 16_999, 17_004)       # p after each context's prompt and its first window
REDUCED, REDUCED_ROWS = 8, 256          # reduced model: layers 0-7, norm, head over 256 positions
TOP1, DECIDED = 0.99, 0.2               # reduced model: top-1 on decided rows, the least share of rows decided


class Ref(RefWeights):
    """``RefWeights`` reading EXL3 experts through open shards into pinned buffers; ``forget`` drops a layer's dense
    tensors and closes the shards (an open shard keeps every page it was read from)."""

    RING = 6                            # buffers: two experts' w1, w2, w3, so the expert in use is never overwritten

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, experts=0, **kwargs)
        self._files: dict[Path, object] = {}
        self._pinned: list[torch.Tensor] = []
        self._turn = 0

    def _read(self, name: str) -> torch.Tensor:
        path = self.pack.path(name)
        if path not in self._files:
            self._files[path] = safe_open(str(path), framework="pt", device="cpu")
        return self._files[path].get_tensor(name)

    def exl3(self, name: str) -> torch.Tensor:
        """W [K, N] fp32 decoded on the device as ``RefWeights.exl3``, returned in a pinned host buffer."""

        codebook = next((cb for cb in ("mcg", "mul1") if self.pack.has(f"{name}.{cb}")), "3inst")
        w = exl3_weight(*(self._read(f"{name}.{p}").to(self.device) for p in ("trellis", "suh", "svh")), codebook)
        if not self._pinned or self._pinned[0].numel() < w.numel():
            self._pinned = [torch.empty(w.numel(), dtype=torch.float32, pin_memory=True) for _ in range(self.RING)]
        out = self._pinned[self._turn].narrow(0, 0, w.numel()).view(w.shape)
        self._turn = (self._turn + 1) % self.RING
        return out.copy_(w)

    def forget(self) -> None:
        self._dense.clear()
        self._files.clear()


def corpus(cfg: Config, n: int) -> list[int]:
    """BOS, then CORPUS's paragraphs in seeded shuffles (as many passes as ``n`` tokens take), cut at ``n``."""

    root = Path(MODEL)
    files = sorted({p for pattern in CORPUS for p in root.glob(pattern)})
    paragraphs = [p.strip() for f in files for p in f.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    tok, rng, ids = Tokenizer.from_file(str(root / "tokenizer.json")), random.Random(SEED), [cfg.bos_token_id]
    while len(ids) < n:
        order = paragraphs[:]
        rng.shuffle(order)
        for p in order:
            ids += tok.encode(p + "\n\n", add_special_tokens=False).ids
            if len(ids) >= n:
                break
    return ids[:n]


def _forwards() -> list[tuple[int, int, bool]]:
    out, at = [], 0
    for rows, prompt in PLAN:
        out.append((at, rows, prompt))
        at += rows
    return out


def _groups() -> list[tuple[int, int, str]]:
    """Compared rows by kind, in order: the prompt rows that end each context, then its decode windows together."""

    out: list[tuple[int, int, str]] = []
    for a, n, prompt in _forwards():
        lo = (a + n - min(TAIL, n) if a + n in CONTEXTS else a + n) if prompt else a
        kind = "prompt" if prompt else "decode"
        if lo == a + n:
            continue
        out = out[:-1] + [(out[-1][0], a + n, kind)] if out and out[-1][1:] == (lo, kind) else out + [(lo, a + n, kind)]
    return out


def _compared() -> list[tuple[int, int]]:
    """The compared row ranges with adjacent ones merged, the reference's calls."""

    spans: list[tuple[int, int]] = []
    for lo, hi, _ in _groups():
        spans = spans[:-1] + [(spans[-1][0], hi)] if spans and spans[-1][1] == lo else spans + [(lo, hi)]
    return spans


def _segments(total: int) -> list[tuple[int, int, bool]]:
    """(begin, end, compared) over all rows in order, cache-only stretches cut into BLOCK rows."""

    out, at = [], 0
    for lo, hi in _compared() + [(total, total)]:
        out += [(a, min(a + BLOCK, lo), False) for a in range(at, lo, BLOCK)]
        out += [(lo, hi, True)] if hi > lo else []
        at = hi
    return out


def _close(got: torch.Tensor, want: torch.Tensor) -> dict:
    g, x = got.double().flatten(), want.double().flatten()
    return {"rel": float((g - x).norm() / x.norm()), "cos": float(g @ x / (g.norm() * x.norm()))}


def _route(cfg: Config, xf: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, pick: torch.Tensor, k: int
           ) -> dict:
    """Per row of FFN input ``xf``: fp64 router scores (M:809-827), the sorted top-``k`` set ``pick`` made, its
    k-th to (k+1)-th margin, whether ``pick`` differs from the fp64 top-k, and whether that margin is below TIE x
    the fp32 error bound."""

    x, g = xf.double(), gate.double()
    z = x @ g.T / cfg.gate_temp
    dz = 8 * math.sqrt(x.shape[1]) * 2.0**-24 * (x.abs() @ g.abs().T) / cfg.gate_temp
    f = tnf.softplus(z).sqrt()
    s = f + bias.double()
    ds = torch.sigmoid(z) / (2 * f) * dz + 2.0**-22 * (f + bias.double().abs())
    order = torch.sort(s, dim=-1, descending=True, stable=True).indices
    kth, nxt = order[:, k - 1:k], order[:, k:k + 1]
    margin = (s.gather(1, kth) - s.gather(1, nxt))[:, 0]
    tol = TIE * (ds.gather(1, kth) + ds.gather(1, nxt))[:, 0]
    sets = pick.long().sort(-1).values
    return {"s": s, "sets": sets, "margin": margin, "ratio": margin / tol, "tie": margin < tol,
            "differ": (order[:, :k].sort(-1).values != sets).any(-1)}


def _routing(r: dict) -> dict:
    """A route's rows that differ from fp64, those of them not at a near-tie, and the least margin over its bound."""

    return {"rows": len(r["s"]), "differ": int(r["differ"].sum()), "not_ties": int((r["differ"] & ~r["tie"]).sum()),
            "least_margin": float(r["ratio"].min())}


def _rows(r: dict, lo: int, hi: int) -> dict:
    return {k: v[lo:hi] for k, v in r.items()}


def _versus(got: torch.Tensor, tf: dict, want: dict) -> dict:
    """TF's output ``got`` (route ``tf``) against each mode's (output, route) in ``want``: the distance over all rows,
    the rows that mode routes to other experts than TF, those of them at a near-tie (either side's margin under its
    bound, or the reference's under twice the largest score shift between the two inputs), and TF's and the mirror
    reference's distances over the rest (``kept``, ``mirror_ref``)."""

    out = {}
    for name, (y, r) in want.items():
        moved = (r["sets"] != tf["sets"]).any(-1)
        tie = moved & (tf["tie"] | r["tie"] | (r["margin"] <= 2 * (tf["s"] - r["s"]).abs().amax(-1)))
        keep = ~tie
        out[name] = {**_close(got, y), "rerouted": int(moved.sum()), "near_tie": int(tie.sum()),
                     "kept": _close(got[keep], y[keep])["rel"],
                     "mirror_ref": _close(want["mirror"][0][keep], y[keep])["rel"]}
    return out


def _gated(name: str, d: dict) -> bool:
    """The model-level gate of one comparison: mirror over all rows, fp32 over the kept rows; ``fp32_qdq`` is reported
    only."""

    if name == "mirror":
        return d["rel"] <= REL_MIRROR and d["cos"] >= COS
    return name != "fp32" or d["kept"] <= max(REL_FP32, d["mirror_ref"] + MARGIN)


class Stream:
    """TF's forward over PLAN one layer at a time on one rank, the next layer's weights read while one runs."""

    def __init__(self, cfg: Config, tokens: list[int], hasher, reader) -> None:
        dev = torch.device("cuda", torch.cuda.current_device())
        self.cfg, self.tokens, self.hasher, self.reader, self.dev = cfg, tokens, hasher, reader, dev
        self.pack = loader._Pack(MODEL, 0, 1, dev)
        self.build = loader._Build(self.pack, cfg)
        tables = {kind: tf_rope.tables(cfg, kind, CAP, dev) for kind in tf_rope.KINDS}
        self.w = Weights(cfg, 0, 1, None, dev, 0, self.build.t("embed.weight"), [], None, None, None, {}, tables)
        self.st = buffers.State(cfg, CAP, dev)
        self.pbuf = buffers.Buffers(cfg, 1, PREFILL_ROWS, CAP, prefill=True, device=dev)
        self.dbuf = buffers.Buffers(cfg, 1, MAX_ROWS, CAP, device=dev)
        n, S, D = ROWS, cfg.hc_mult, cfg.hidden_size
        self.X, self.Y = (torch.empty((n, S, D), dtype=torch.bfloat16, device=dev) for _ in range(2))
        self.pre, self.pre_out = (torch.empty((n, S), dtype=torch.float32, device=dev) for _ in range(2))
        self.taps = torch.zeros((n, len(cfg.dspark_target_layer_ids), D), dtype=torch.bfloat16, device=dev)
        self.lists: dict[int, tuple] = {}       # forward -> the last index layer's lists, counts
        self.cand: dict[int, tuple] = {}        # forward -> the candidate source's blocks, counts
        self.picked: dict[int, list[torch.Tensor]] = {}     # index layer -> its list of each compared row

    def layer(self, L: int) -> dict:
        """Block L over every forward: Y, pre_out take its output; returns its compared rows' FFN input and picks."""

        cfg, w, st = self.cfg, self.w, self.st
        lw, eg = self.build.layer(L)
        if L + 1 < cfg.num_hidden_layers:
            self.pack.queue(self.pack.names(f"layers.{L + 1}."))
        w.layers, w.engram = [lw], ({L: eg} if eg is not None else {})
        role, spans, seen = lw.role, _compared(), {"xf": [], "pick": []}
        own_lists = role.ratio and role.idx_src == L
        index = {k: {"picked": 0, "differ": 0, "not_ties": 0, "worst": 0.0} for k in ("lists", "blocks")}
        st.reset()
        for f, (a, n, prompt) in enumerate(_forwards()):
            b = self.pbuf if prompt else self.dbuf
            R = F.stage(w, st, b, self.tokens[a:a + n], self.hasher, self.reader)
            if L == 0:
                glue.embed(b.ids[:R], w.embed, cfg.hidden_size, cfg.hc_mult, self.X[a:a + n])
                self.pre[a:a + n].zero_()
                self.pre[a:a + n, 0].fill_(1.0)
            b.X[:R].copy_(self.X[a:a + n])
            b.pre_in[:R].copy_(self.pre[a:a + n])
            e = engram.exchange(engram.dequant_rows(b.eraw[:R], b.eloc[:R]), w.comm, b.egat, b.eng) if w.engram \
                else None
            if role.ratio and not own_lists:
                b.lists[:R].copy_(self.lists[f][0])
                b.list_n[:R].copy_(self.lists[f][1])
            if role.uses_candidates:
                b.cand[:R].copy_(self.cand[f][0])
                b.cand_n[:R].copy_(self.cand[f][1])
            F.layer(lw, w, st, b, R, e, prompt)
            if own_lists:
                self.lists[f] = (b.lists[:R].clone(), b.list_n[:R].clone())
            if role.candidate_source:
                self.cand[f] = (b.cand[:R].clone(), b.cand_n[:R].clone())
            if role.tap is not None:
                k = min(R, b.taps.shape[0])
                self.taps[a + R - k:a + R, role.tap].copy_(b.taps[:k, role.tap])
            for lo, hi in spans:
                lo, hi = max(lo, a), min(hi, a + n)
                if lo < hi:
                    seen["xf"].append(b.xn[lo - a:hi - a].cpu())
                    seen["pick"].append(b.pick[lo - a:hi - a].cpu())
                    if own_lists:
                        _index(cfg, role, b, st.index_k[role.kv_src], lo - a, hi - a, lo, index)
                        counts = b.list_n[lo - a:hi - a].tolist()
                        rows = b.lists[lo - a:hi - a].cpu().long()
                        self.picked.setdefault(L, []).extend(rows[i, :c] for i, c in enumerate(counts))
            self.Y[a:a + n].copy_(b.X[:R])
            self.pre_out[a:a + n].copy_(b.pre_in[:R])
            F.commit(w, st, b, R, R)
        seen = {k: torch.cat(v) for k, v in seen.items()}
        seen["gate"] = (lw.moe.gate.cpu(), lw.moe.bias.cpu())
        seen["route"] = _route(cfg, seen["xf"], *seen["gate"], seen["pick"], cfg.num_experts_per_tok)
        seen["routing"], seen["index"] = _routing(seen["route"]), index
        w.layers, w.engram = [], {}
        del lw, eg
        return seen

    def advance(self) -> None:
        self.X, self.Y, self.pre, self.pre_out = self.Y, self.X, self.pre_out, self.pre

    def close(self) -> None:
        self.pack.close()


def _ties(out: dict, s: torch.Tensor, tol: torch.Tensor, k: torch.Tensor, given: torch.Tensor) -> None:
    """Adds to ``out`` how many of the masks ``given`` [rows, n] differ from fp64 top-k of ``s`` (ties to the lower
    index), those not at a near-tie (|s - k-th| >= the two tolerances), and the worst |s - k-th| over its tolerance."""

    ranked = torch.sort(s, dim=-1, descending=True, stable=True)
    mine = torch.arange(s.shape[1], device=s.device) < k[:, None]
    mine = torch.zeros_like(given).scatter_(1, ranked.indices, mine)
    at = (k - 1).clamp_min(0)[:, None]
    kth = torch.where(k[:, None] > 0, ranked.values.gather(1, at), -math.inf)
    lim = tol + tol.gather(1, ranked.indices.gather(1, at))
    ratio = ((s - kth).abs() / lim).nan_to_num(math.inf)[mine ^ given]
    out["picked"] += int(k.sum())
    out["differ"] += len(ratio)
    out["not_ties"] += int((ratio >= 1).sum())
    out["worst"] = max(out["worst"], float(ratio.max()) if len(ratio) else 0.0)


def _index(cfg: Config, role, b, keys: torch.Tensor, i: int, j: int, first: int, out: dict) -> None:
    """TF's lists of forward rows i..j (at ``first`` on), and the candidate source's blocks, against an fp64
    selection on TF's own index Q, weights and keys (M:556-610): every difference must be a near-tie."""

    H, D, G = cfg.index_n_heads, cfg.index_head_dim, cfg.candidate_block_size
    bound = TIE * 8 * math.sqrt(H * D) * 2.0**-24          # the op-level fp32 bound, the scale summed below
    visible = (torch.arange(first, first + j - i, device=keys.device) + 1) // role.ratio
    K = keys[:int(visible.max())].double()
    t = torch.arange(len(K), device=K.device)
    for r in range(i, j, 8):
        n = min(8, j - r)
        q, w, v = b.qI[r:r + n].double(), b.wI[r:r + n].double(), visible[r - i:r - i + n, None]
        s = (torch.einsum("nhd,td->nht", q, K).relu() * w[..., None]).sum(1).masked_fill(t >= v, -math.inf)
        tol = bound * torch.einsum("nhd,td->nt", q.abs() * w.abs()[..., None], K.abs())
        if role.candidate_source or role.uses_candidates:
            blocks = torch.zeros((n, -(-len(K) // G)), dtype=torch.bool, device=K.device)
            for x, c in enumerate(b.cand_n[r:r + n].tolist()):
                blocks[x, b.cand[r + x, :c].long()] = True
        if role.candidate_source:
            bs = tnf.pad(s, (0, -len(K) % G), value=-math.inf).unflatten(-1, (-1, G)).amax(-1)
            bs = bs.masked_fill(torch.arange(bs.shape[1], device=K.device) == (v - 1) // G, math.inf)
            bt = tnf.pad(tol, (0, -len(K) % G)).unflatten(-1, (-1, G)).amax(-1)
            k = (bs > -math.inf).sum(-1).clamp_max(cfg.candidate_topk_blocks)
            _ties(out["blocks"], bs, bt, k, blocks)
        if role.uses_candidates:
            s = s.masked_fill(~blocks.repeat_interleave(G, -1)[:, :len(K)], -math.inf)
        given = torch.zeros_like(s, dtype=torch.bool)
        for x, c in enumerate(b.list_n[r:r + n].tolist()):
            given[x, b.lists[r + x, :c].long()] = True
        _ties(out["lists"], s, tol, (s > -math.inf).sum(-1).clamp_max(cfg.index_topk), given)


def _quantize_as_mirror(rw: Ref, ref, held: dict):
    """Points ``ref``'s window KV and compressor at a mirror reference on its state fed ``held["xa"]``, so the model's
    FP8/FP4 quantizers take mirror mode's inputs -> that mirror reference."""

    mref = rw.reference(MIRROR, ref.state)
    ref._kv = lambda p_, x, cis: mref._kv(p_, held["xa"], cis)
    ref.compress = lambda layer, x, start: mref.compress(layer, held["xa"], start)
    return mref


def ref_block(rw: Ref, L: int, name: str, state: State, stream: Stream, eng_ids) -> tuple[torch.Tensor, ...]:
    """Block L of the reference (M:968-994) in ``MODES[name]`` on TF's input stream: caches from every row, the whole
    block on the compared rows, their MoE in one call -> (their output stream fp32 [rows, hc, D], their routed experts,
    their FFN input, their next pre-mix [rows, hc]).

    ``fp32_qdq`` computes the window KV and compressed entries in mirror mode from mirror's attention input."""

    c, role, p, eps, mode = rw.cfg, rw.cfg.roles[L], f"layers.{L}", rw.cfg.rms_norm_eps, MODES[name]
    ref, given, held = rw.reference(mode, state), [], {}
    ref.indexer = lambda *_: given
    mref = _quantize_as_mirror(rw, ref, held) if name == "fp32_qdq" else None
    at, mids = 0, []
    for a, b, full in _segments(ROWS):
        X0, pre = stream.X[a:b].float().cpu(), stream.pre[a:b].cpu()
        ids = torch.as_tensor(eng_ids[a:b, c.engram_layer_ids.index(L)]) if role.engram else None
        X = ref.engram(L, X0, ids) if role.engram else X0
        xa = rms_norm(hc_pre(X, pre, mode), ref.W(f"{p}.attn_norm.weight"), eps, mode)
        if name == "fp32_qdq":
            Xm = mref.engram(L, X0, ids) if role.engram else X0
            held["xa"] = rms_norm(hc_pre(Xm, pre, MIRROR), ref.W(f"{p}.attn_norm.weight"), eps, MIRROR)
        if not full:
            ref._commit(L, ref._kv(p, xa, ref.cis(role.rope, torch.arange(a, b))), a)
            if role.ratio and role.kv_src == L:
                ref.compress(L, xa, a)
            continue
        if role.ratio:
            given[:] = stream.picked[role.idx_src][at:at + b - a]
            state.lists = list(given)
        at += b - a
        a_pre, a_post, a_comb = ref._mix(f"{p}.hc_attn", X)
        X = hc_post(ref.attention(L, xa, a), X, a_post, a_comb, mode)
        f_pre, f_post, f_comb = ref._mix(f"{p}.hc_ffn", X)
        xf = rms_norm(hc_pre(X, a_pre, mode), ref.W(f"{p}.ffn_norm.weight"), eps, mode)
        mids.append((X, xf, f_pre, f_post, f_comb))
    X, xf, pre, post, comb = (torch.cat(t) for t in zip(*mids))
    _, picked = gate(xf, ref.W(f"{p}.ffn.gate.weight"), ref.W(f"{p}.ffn.gate.bias"), c.num_experts_per_tok, c)
    return hc_post(ref.moe(L, xf), X, post, comb, mode), picked, xf, pre


def _keep_ffn_input(ref, into: dict, name: str) -> None:
    """Has ``ref.moe`` store its input in ``into[name]``."""

    moe = ref.moe

    def kept(layer: int, x: torch.Tensor) -> torch.Tensor:
        into[name] = x
        return moe(layer, x)

    ref.moe = kept


def _dspark(stream: Stream, rw: Ref) -> list[dict]:
    """Each DSpark stage of TF's block at each p of DSPARK_AT, the rings absorbed from TF's taps of the last
    min(p + 1, window) rows: the reference's stage on TF's stage input in each of MODES, and TF's routed sets."""

    cfg, st, b, B = stream.cfg, stream.st, stream.dbuf, stream.cfg.dspark_block_size
    w, n_layers = stream.w, cfg.num_hidden_layers
    stages = [stream.build.stage(s) for s in range(cfg.num_nextn_predict_layers)]
    w.layers, w.engram = [], {}
    w.dspark = SimpleNamespace(main_proj=stream.build.mx8("mtp.0.main_proj"),
                               main_norm=stream.build.t("mtp.0.main_norm.weight"), stages=stages)
    e = SimpleNamespace(w=w, st=st, dwork=dspark.Work(cfg, 1, stream.dev))
    k, out = e.dwork, []
    for p in DSPARK_AT:
        n = min(p + 1, cfg.sliding_window)
        taps = stream.taps[p + 1 - n:p + 1]
        st.set_pos(p + 1)
        stream.pbuf.taps[:n].copy_(taps)
        dspark.absorb(e, stream.pbuf, n, prompt=True)
        k.bids.copy_(torch.tensor([stream.tokens[p + 1]] + [cfg.dspark_noise_token_id] * (B - 1)))
        glue.embed(k.bids, w.embed, cfg.hidden_size, cfg.hc_mult, b.xd)
        b.pre_in[:B].zero_()
        b.pre_in[:B, 0].fill_(1.0)
        torch.sub(st.pos_dev.expand(B), 1, out=k.anchors)
        refs, held, xfs = {name: rw.reference(MODES[name], State(pos=p + 1)) for name in MODES}, {}, {}
        mref = _quantize_as_mirror(rw, refs["fp32_qdq"], held)
        for name, ref in refs.items():
            (mref if name == "fp32_qdq" else ref).absorb(taps.flatten(1).float().cpu(), p + 1 - n)
            _keep_ffn_input(ref, xfs, name)
        for sw in stages:
            X, pre = b.xd.float().cpu(), b.pre_in[:B].cpu()
            held["xa"] = rms_norm(hc_pre(X, pre, MIRROR), mref.W(f"{mref._prefix(sw.index)}.attn_norm.weight"),
                                  cfg.rms_norm_eps, MIRROR)
            dspark._stage(sw, w, st, b, k)
            got = b.xd.cpu()
            top = cfg.dspark_num_experts_per_tok            # the shared expert's slot follows the routed ones
            gw = (sw.moe.gate.cpu(), sw.moe.bias.cpu())
            tf = _route(cfg, b.xn[:B].cpu(), *gw, b.dpick[:B, :top].cpu(), top)
            want = {}
            for name, ref in refs.items():
                y = ref.block(sw.index, X, pre, p + 1)[0]
                q = mref._prefix(sw.index)
                picked = gate(xfs[name], ref.W(f"{q}.ffn.gate.weight"), ref.W(f"{q}.ffn.gate.bias"), top, cfg)[1]
                want[name] = (y, _route(cfg, xfs[name], *gw, picked, top))
            out.append({"stage": sw.index - n_layers, "p": p, "routing": _routing(tf), **_versus(got, tf, want)})
    return out


def _line(g: dict) -> str:
    m = g["mirror"]
    return f"rel-L2 {m['rel']:.3g} mirror (cosine {m['cos']:.6f}), " + ", ".join(
        f"{g[n]['rel']:.3g} {n} ({g[n]['rerouted']} rows rerouted, {g[n]['near_tie']} at a near-tie; {g[n]['kept']:.3g}"
        f" on the rest, the mirror reference {g[n]['mirror_ref']:.3g})" for n in ("fp32", "fp32_qdq"))


def _figures(cfg: Config, L: int, stream: Stream, want: dict, seen: dict) -> dict:
    """Block L's figures per compared group, ``_versus`` each mode, and TF's next pre-mix against each mode's."""

    groups, at, k = [], 0, cfg.num_experts_per_tok
    routes = {name: (y, _route(cfg, xf, *seen["gate"], picked, k)) for name, (y, picked, xf, _) in want.items()}
    for lo, hi, kind in _groups():
        n = hi - lo
        mine = {name: (y[at:at + n], _rows(r, at, at + n)) for name, (y, r) in routes.items()}
        g = {"rows": [lo, hi], "kind": kind, **_versus(stream.Y[lo:hi].cpu(), _rows(seen["route"], at, at + n), mine),
             "pre_mix": {name: _close(stream.pre_out[lo:hi].cpu(), w[3][at:at + n]) for name, w in want.items()}}
        groups.append(g)
        at += n
        print(f"layer {L} ({cfg.roles[L].mode}) {kind} rows {lo}-{hi - 1}: " + _line(g) + f"; pre-mix {g['pre_mix']}")
    print(f"layer {L}: TF routing against fp64 {seen['routing']}; TF index selection against fp64 {seen['index']}")
    return {"role": cfg.roles[L].mode, "groups": groups, "routing": seen["routing"], "index": seen["index"]}


@pytest.fixture(scope="module")
def report():
    out = {"corpus": list(CORPUS), "seed": SEED, "contexts": list(CONTEXTS), "plan": [list(p) for p in PLAN],
           "compared": [list(g) for g in _groups()], "layers": {}, "dspark": [], "reduced": {}}
    yield out
    path = os.environ.get("TF_DSV41_REPORT")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(out, indent=1) + "\n")
        print(f"report: {path}")


@pytest.fixture(scope="module")
def layers(report):
    """Every checked block's figures and DSpark's, from one pass of TF over the 40 layers."""

    cfg = Config.read(MODEL)
    rw = Ref(MODEL)
    tokens = corpus(cfg, ROWS + 1)      # one more: the token after the last row, DSpark's y there
    eng_ids = rw.hasher.ids([], tokens[:ROWS])
    stream = Stream(cfg, tokens, rw.hasher, rw.reader)
    states = {name: State() for name in MODES}
    try:
        for L in range(cfg.num_hidden_layers):
            seen = stream.layer(L)
            if L in LAYERS:
                want = {name: ref_block(rw, L, name, states[name], stream, eng_ids) for name in MODES}
                rw.forget()
                report["layers"][L] = _figures(cfg, L, stream, want, seen)
            stream.advance()
        stream.X = stream.Y = stream.pre = stream.pre_out = None
        report["dspark"] = _dspark(stream, rw)
        for d in report["dspark"]:
            print(f"DSpark stage {d['stage']} at p={d['p']}: {_line(d)}; TF routing against fp64 {d['routing']}")
    finally:
        stream.close()
    return report


@pytest.mark.parametrize("L", LAYERS)
def test_block_follows_the_reference(layers, L):
    out = layers["layers"][L]
    for g in out["groups"]:
        for name in MODES:
            assert _gated(name, g[name]), f"layer {L} rows {g['rows']}: {name} {g[name]}"
        assert _gated("mirror", g["pre_mix"]["mirror"]), f"layer {L} rows {g['rows']}: pre-mix {g['pre_mix']}"
    assert out["routing"]["not_ties"] == 0, f"layer {L}: routed sets differ away from near-ties {out['routing']}"
    for kind, d in out["index"].items():
        assert d["not_ties"] == 0, f"layer {L}: index {kind} differ away from near-ties {d}"


def test_dspark_stages_follow_the_reference(layers):
    assert len(layers["dspark"]) == 3 * len(DSPARK_AT)
    for d in layers["dspark"]:
        assert all(_gated(name, d[name]) for name in MODES), f"DSpark {d}"
        assert d["routing"]["not_ties"] == 0, f"DSpark {d}"


def _logits(got: torch.Tensor, want: torch.Tensor) -> dict:
    """Top-1 agreement, mean KL(want || got) and rel-L2 of two [rows, V] logits."""

    p = torch.softmax(want, -1)
    return {"top1": float((got.argmax(-1) == want.argmax(-1)).double().mean()),
            "kl": float((p * (torch.log_softmax(want, -1) - torch.log_softmax(got, -1))).sum(-1).mean()),
            "rel": float((got - want).norm() / want.norm())}


def test_reduced_model_follows_the_reference(report):
    cfg = Config.read(MODEL, REDUCED)
    rw = Ref(MODEL, REDUCED)
    ids = corpus(cfg, REDUCED_ROWS)
    w = loader.load(MODEL, cfg, 0, 1, None, dspark=False, capacity=REDUCED_ROWS)
    e = SimpleNamespace(w=w, st=buffers.State(cfg, REDUCED_ROWS, w.device), hasher=rw.hasher, reader=rw.reader,
                        pbuf=buffers.Buffers(cfg, 1, REDUCED_ROWS, REDUCED_ROWS, prefill=True, device=w.device))
    got = score.prompt_logits(e, ids).double()
    del e, w
    torch.cuda.empty_cache()
    mirror, fp32, split = (rw.model(ids, mode, all_logits=True)[0].double()
                           for mode in (MIRROR, FP32, Mode("mirror", world=2)))
    top2 = mirror.topk(2, -1).values
    decided = top2[:, 0] - top2[:, 1] > (fp32 - mirror).abs().amax(-1)
    out = {"layers": REDUCED, "positions": REDUCED_ROWS, "tf": _logits(got, mirror), "fp32": _logits(fp32, mirror),
           "mirror_world2": _logits(split, mirror), "decided": float(decided.double().mean()),
           "top1_decided": _logits(got[decided], mirror[decided])["top1"]}
    report["reduced"] = out
    print(f"reduced model, {REDUCED} layers over {REDUCED_ROWS} positions, against the mirror reference: TF {out['tf']}"
          f", the fp32 reference {out['fp32']}, the mirror reference over two ranks {out['mirror_world2']}; TF top-1 "
          f"{out['top1_decided']:.4f} on the {out['decided']:.3f} of rows the reference decides")
    tf, ref = out["tf"], out["fp32"]
    assert out["decided"] >= DECIDED and out["top1_decided"] >= TOP1, out
    assert tf["kl"] <= ref["kl"] and tf["rel"] <= ref["rel"], out
