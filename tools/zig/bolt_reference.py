"""Layer-by-layer fp32 PyTorch reference of Bolt's language model, streamed a layer at a time: a debugging oracle."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bolt_ggml  # noqa: E402

MODEL = "/storage/models/nemotron-bolt-74b-24gb"
FIXTURES = "/storage/models/bolt-fixtures/text.jsonl"
PRE = "language_model.backbone."


# ---------------------------------------------------------------- weight decode

_IQ2 = None


def _iq2_tables():
    global _IQ2
    if _IQ2 is None:
        signs, grid = bolt_ggml.tables()
        bits = (1 << np.arange(8)).astype(np.uint8)
        sign = np.where((signs[:, None] & bits) != 0, -1.0, 1.0).astype(np.float32)  # [128, 8]
        _IQ2 = (torch.from_numpy(grid.astype(np.float32)), torch.from_numpy(sign))
    return _IQ2


def iq2_xxs(raw):
    """U8 [rows, blocks, 66] -> f32 [rows, blocks * 256]; bit-identical to bolt_ggml.iq2_xxs."""
    grid, sign = _iq2_tables()
    rows = raw.shape[0]
    b = raw.reshape(-1, 66)
    d = b[:, :2].contiguous().view(torch.float16).float()  # [n, 1]
    qs = b[:, 2:].contiguous()  # 8 sub-blocks of (4 grid indices, u32 aux)
    pair = qs.view(-1, 8, 8)
    idx = pair[:, :, :4].long()  # [n, 8, 4]
    aux = pair[:, :, 4:].contiguous().view(torch.int32).long().squeeze(-1) & 0xFFFFFFFF  # [n, 8]
    db = d * (0.5 + (aux >> 28).float()) * 0.25  # [n, 8]
    shifts = torch.arange(4) * 7
    sidx = (aux[..., None] >> shifts) & 127  # [n, 8, 4]
    out = grid[idx] * sign[sidx] * db[..., None, None]  # [n, 8, 4, 8]
    return out.reshape(rows, -1)


def q8_0(raw):
    """U8 [rows, blocks, 34] -> f32 [rows, blocks * 32]."""
    rows = raw.shape[0]
    b = raw.reshape(-1, 34)
    d = b[:, :2].contiguous().view(torch.float16).float()
    q = b[:, 2:].contiguous().view(torch.int8).float()
    return (q * d).reshape(rows, -1)


_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


def nvfp4(packed, scale, global_scale):
    """ModelOpt NVFP4: U8 [N, K/2] (column 2j low nibble, 2j+1 high), e4m3 [N, K/16], f32 [] -> f32 [N, K]."""
    n = packed.shape[0]
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(n, -1).long()
    w = _E2M1[codes].view(n, -1, 16) * scale.float()[..., None]
    return (w * global_scale.float()).reshape(n, -1)


class Checkpoint:
    def __init__(self, model):
        self.dir = Path(model)
        self.map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        cfg = json.loads((self.dir / "config.json").read_text())
        self.algo = {k: v["quant_algo"] for k, v in cfg["quantization_config"]["quantized_layers"].items()}
        self.cfg = cfg["llm_config"]
        self.files = {}

    def raw(self, name):
        f = self.map[name]
        if f not in self.files:
            self.files[f] = safe_open(str(self.dir / f), framework="pt")
        return self.files[f].get_tensor(name)

    def has(self, name):
        return name in self.map

    def dense(self, name):
        """A Linear weight [out, in] in f32, by the format the export records for it."""
        algo = self.algo.get(name)
        if algo == "IQ2_XXS":
            return iq2_xxs(self.raw(name + ".weight"))
        if algo == "Q8_0":
            return q8_0(self.raw(name + ".weight"))
        if algo == "W4A16_NVFP4":
            return nvfp4(self.raw(name + ".weight"), self.raw(name + ".weight_scale"), self.raw(name + ".weight_scale_2"))
        if algo is None:
            return self.raw(name + ".weight").float()
        raise SystemExit(f"{name}: unsupported {algo}")

    def vec(self, name):
        return self.raw(name).float()


# ---------------------------------------------------------------- blocks


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class State:
    """Per-layer recurrent state: Mamba conv window + ssm state, attention K/V."""

    def __init__(self):
        self.conv, self.ssm, self.k, self.v = {}, {}, {}, {}
        self.pos = 0


def mamba(ck, cfg, li, x, st):
    p = f"{PRE}layers.{li}.mixer."
    H, P, N, G = cfg["mamba_num_heads"], cfg["mamba_head_dim"], cfg["ssm_state_size"], cfg["n_groups"]
    inner = H * P
    T = x.shape[0]
    zxbcdt = x @ ck.dense(p + "in_proj").T  # [T, inner + conv_dim + H]
    conv_dim = inner + 2 * G * N
    z, xbc, dt = zxbcdt.split([inner, conv_dim, H], dim=-1)

    # causal depthwise conv over [previous K-1 inputs, these T], + bias, SiLU
    cw = ck.vec(p + "conv1d.weight")[:, 0, :]  # [conv_dim, K]
    K = cw.shape[1]
    prev = st.conv.get(li, torch.zeros(K - 1, conv_dim))
    seq = torch.cat([prev, xbc], 0)  # [K-1+T, conv_dim]
    st.conv[li] = seq[-(K - 1):].clone()
    win = seq.unfold(0, K, 1)  # [T, conv_dim, K]
    xbc = F.silu((win * cw).sum(-1) + ck.vec(p + "conv1d.bias"))
    xs, B, C = xbc.split([inner, G * N, G * N], dim=-1)

    # dt = softplus(dt + dt_bias), limited to time_step_limit = (0, inf): no clamp (the kernel path)
    dt = F.softplus(dt + ck.vec(p + "dt_bias"))  # [T, H]
    lo, hi = cfg["time_step_limit"][0], cfg["time_step_limit"][1]
    hi = float(hi["__float__"]) if isinstance(hi, dict) else float(hi)
    if lo > 0 or hi != math.inf:
        dt = dt.clamp(lo, hi)
    A = -torch.exp(ck.vec(p + "A_log"))  # [H]
    D = ck.vec(p + "D")
    xs = xs.view(T, H, P)
    rep = H // G
    B = B.view(T, G, N).repeat_interleave(rep, dim=1)  # [T, H, N]
    C = C.view(T, G, N).repeat_interleave(rep, dim=1)
    dA = torch.exp(dt * A)  # [T, H]
    h = st.ssm.get(li, torch.zeros(H, P, N))
    ys = torch.empty(T, H, P)
    for t in range(T):
        h = h * dA[t][:, None, None] + (dt[t][:, None] * xs[t])[:, :, None] * B[t][:, None, :]
        ys[t] = torch.einsum("hpn,hn->hp", h, C[t])
    st.ssm[li] = h
    y = (ys + xs * D[:, None]).reshape(T, inner)

    # gated RMSNorm, norm_before_gate=False: normalize y * silu(z) per group of inner / n_groups
    g = (y * F.silu(z)).view(T, G, inner // G)
    g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + cfg["layer_norm_epsilon"])
    y = g.reshape(T, inner) * ck.vec(p + "norm.weight")
    return y @ ck.dense(p + "out_proj").T


def attention(ck, cfg, li, x, st):
    p = f"{PRE}layers.{li}.mixer."
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    T = x.shape[0]
    q = (x @ ck.dense(p + "q_proj").T).view(T, nh, hd)
    k = (x @ ck.dense(p + "k_proj").T).view(T, nkv, hd)
    v = (x @ ck.dense(p + "v_proj").T).view(T, nkv, hd)
    if li in st.k:
        k = torch.cat([st.k[li], k], 0)
        v = torch.cat([st.v[li], v], 0)
    st.k[li], st.v[li] = k, v
    S = k.shape[0]
    rep = nh // nkv
    kk = k.repeat_interleave(rep, dim=1).transpose(0, 1)  # [nh, S, hd]
    vv = v.repeat_interleave(rep, dim=1).transpose(0, 1)
    s = torch.einsum("thd,hsd->hts", q, kk) * hd**-0.5
    qpos = torch.arange(S - T, S)[:, None]
    s = s.masked_fill(torch.arange(S)[None, :] > qpos, float("-inf"))
    o = torch.einsum("hts,hsd->thd", torch.softmax(s, -1), vv).reshape(T, nh * hd)
    return o @ ck.dense(p + "o_proj").T


def moe(ck, cfg, bc, li, x, routes):
    p = f"{PRE}layers.{li}.mixer."
    E, k = bc["n_routed_experts"], bc["num_experts_per_tok"]
    T = x.shape[0]
    # router: sigmoid(x W^T); choice by score + bias (n_group 1); unbiased weights, normalized, times the scaling factor
    assert cfg["n_group"] == 1 and cfg["topk_group"] == 1
    scores = torch.sigmoid(x @ ck.vec(p + "gate.weight").T)  # [T, E]
    choice = scores + ck.vec(p + "gate.e_score_correction_bias")
    idx = torch.topk(choice, k, dim=-1).indices  # [T, k], by descending choice
    w = scores.gather(1, idx)
    if cfg["norm_topk_prob"]:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    w = w * cfg["routed_scaling_factor"]
    routes[li] = (idx.clone(), w.clone())

    lat = x @ ck.dense(p + "fc1_latent_proj").T  # [T, latent]
    acc = torch.zeros_like(lat)
    for e in torch.unique(idx).tolist():
        tok, slot = torch.where(idx == e)
        up = ck.dense(f"{p}experts.{e}.up_proj")  # [inter, latent]
        down = ck.dense(f"{p}experts.{e}.down_proj")  # [latent, inter]
        hdn = F.relu(lat[tok] @ up.T).square()
        acc.index_add_(0, tok, (hdn @ down.T) * w[tok, slot][:, None])
    routed = acc @ ck.dense(p + "fc2_latent_proj").T
    sh = F.relu(x @ ck.dense(p + "shared_experts.up_proj").T).square()
    return routed + sh @ ck.dense(p + "shared_experts.down_proj").T


def lm_head(ck, x, rows=16384):
    raw = ck.raw("language_model.lm_head.weight")
    return torch.cat([x @ q8_0(raw[i:i + rows]).T for i in range(0, raw.shape[0], rows)], -1)


def forward(ck, ids, st=None, dump=None, dump_last=8, logits_last=None, verbose=True, splice=None):
    """Run `ids` (continuing `st`) -> (logits, state, routes); `splice` = (positions, rows) overwrites embeddings."""
    cfg = ck.cfg
    st = st or State()
    eps = cfg["layer_norm_epsilon"]
    ids_t = torch.tensor(ids, dtype=torch.long)
    emb = ck.raw(PRE + "embeddings.weight")
    h = emb[ids_t].float()
    if splice is not None:
        pos, rows = splice
        h[pos] = rows.float()
    routes = {}
    keep = slice(-dump_last, None) if dump_last else slice(None)
    if dump:
        np.save(dump / "h_embed.npy", h[keep].numpy())
    t0 = time.time()
    for li, bc in enumerate(cfg["block_configs"]):
        t1 = time.time()
        x = rmsnorm(h, ck.vec(f"{PRE}layers.{li}.norm.weight"), eps)
        kind = bc["block_type"]
        if kind == "mamba":
            out = mamba(ck, cfg, li, x, st)
        elif kind == "attention":
            out = attention(ck, cfg, li, x, st)
        elif kind == "moe":
            out = moe(ck, cfg, bc, li, x, routes)
        else:
            raise SystemExit(f"layer {li}: {kind}")
        h = h + out
        if dump:
            np.save(dump / f"h_layer{li:02d}.npy", h[keep].numpy())
            np.save(dump / f"mixer_layer{li:02d}.npy", out[keep].numpy())
            if li in routes:
                np.save(dump / f"route_idx_layer{li:02d}.npy", routes[li][0][keep].numpy())
                np.save(dump / f"route_w_layer{li:02d}.npy", routes[li][1][keep].numpy())
        if verbose:
            print(f"  layer {li:2d} {kind:9s} |h|={h[-1].norm():9.3f}  {time.time() - t1:5.1f}s", file=sys.stderr, flush=True)
    st.pos += len(ids)
    h = rmsnorm(h, ck.vec(PRE + "norm_f.weight"), eps)
    if logits_last:
        h = h[-logits_last:]
    logits = lm_head(ck, h)
    if dump:
        np.save(dump / "h_final_norm.npy", h[keep].numpy())
        np.save(dump / "logits.npy", logits[keep].numpy())
    if verbose:
        print(f"  forward {len(ids)} tokens: {time.time() - t0:.1f}s", file=sys.stderr)
    return logits, st, routes


# ---------------------------------------------------------------- fixtures / scoring


def load_fixture(name, path=FIXTURES):
    for line in open(path):
        d = json.loads(line)
        if d["name"] == name:
            return d
    raise SystemExit(f"no fixture {name!r} in {path}")


def score(lp, tops, label):
    """lp [n, vocab] log-probs vs vLLM dicts (token id -> logprob) for the same n positions."""
    strict = tie = 0
    diffs = []
    rows = []
    for i, top in enumerate(tops):
        ids = torch.tensor([int(t) for t in top])
        ref = torch.tensor(list(top.values()))
        mine = int(lp[i].argmax())
        best = ref.max()
        strict += mine == int(ids[ref.argmax()])
        tie += bool((ids[ref >= best - 1e-6] == mine).any())
        err = (lp[i][ids] - ref).abs()
        d = err.max().item()
        diffs.append(d)
        j = int(err.argmax())
        rows.append((d, i, int(ids[j]), ref[j].item(), lp[i][ids[j]].item()))
    n = len(tops)
    diffs = np.array(diffs)
    print(f"{label}: top-1 {strict}/{n} = {100 * strict / n:.1f}% (counting vLLM ties: {tie}/{n} = {100 * tie / n:.1f}%) | "
          f"max|dlogprob| over vLLM top-20: mean {diffs.mean():.3f} median {np.median(diffs):.3f} max {diffs.max():.3f}")
    for d, i, t, r, m in sorted(rows, reverse=True)[:3]:
        print(f"  worst: position {i} id {t}: vLLM {r:.3f} ours {m:.3f} (|d| {d:.3f})")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ids", help="file of whitespace/comma separated token ids (or a JSON list)")
    src.add_argument("--fixture", help="fixture name in text.jsonl (short, code, think, multiturn, long, german)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fixtures", default=FIXTURES)
    ap.add_argument("--dump", help="directory for per-layer npy dumps")
    ap.add_argument("--dump-last", type=int, default=8, help="positions kept in dumps (0 = all)")
    ap.add_argument("--compare", action="store_true", help="score prompt logprobs against the fixture")
    ap.add_argument("--generate", type=int, default=0, help="greedy tokens to generate after the prompt")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    ck = Checkpoint(a.model)

    fx = None
    if a.fixture:
        fx = load_fixture(a.fixture, a.fixtures)
        ids = fx["prompt_ids"]
    else:
        text = Path(a.ids).read_text().strip()
        ids = json.loads(text) if text.startswith("[") else [int(t) for t in text.replace(",", " ").split()]
    dump = None
    if a.dump:
        dump = Path(a.dump)
        dump.mkdir(parents=True, exist_ok=True)
        (dump / "ids.json").write_text(json.dumps(ids))

    t0 = time.time()
    logits, st, _ = forward(ck, ids, dump=dump, dump_last=a.dump_last, verbose=not a.quiet)
    print(f"prompt {len(ids)} tokens in {time.time() - t0:.1f}s", flush=True)
    lp = torch.log_softmax(logits, -1)
    if a.compare:
        if fx is None:
            raise SystemExit("--compare needs --fixture")
        # prompt_top[i] is vLLM's distribution for position i given ids[:i], i.e. our logits at i-1
        score(lp[:-1], fx["prompt_top"][1:], f"{a.fixture} prompt ({len(ids) - 1} positions)")

    if a.generate:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
        out = [int(logits[-1].argmax())]
        steps = [lp[-1]]
        for _ in range(a.generate - 1):
            lg, st, _ = forward(ck, [out[-1]], st, verbose=False)
            steps.append(torch.log_softmax(lg[-1], -1))
            out.append(int(lg[-1].argmax()))
        print(f"generated {len(out)} tokens in {time.time() - t0:.1f}s total")
        print("ids:", out)
        print("text:", repr(tok.decode(out)))
        if fx is not None:
            ref = fx["gen_tokens"][: len(out)]
            mine = [tok.decode([i]) for i in out]
            match = 0
            for m, r in zip(mine, ref):
                if m != r:
                    break
                match += 1
            print(f"greedy prefix match vs vLLM: {match}/{len(ref)}")
            print("vLLM:", repr("".join(ref)))
            # logprob diffs at the generated positions (vLLM keys are token strings)
            diffs = []
            for s_lp, top in zip(steps[:match + 1], fx["gen_top"]):
                d = 0.0
                for i in torch.topk(s_lp, 40).indices.tolist():
                    t = tok.decode([i])
                    if t in top:
                        d = max(d, abs(s_lp[i].item() - top[t]))
                diffs.append(d)
            print(f"gen max|dlogprob| (shared top ids, first {len(diffs)} steps): mean {np.mean(diffs):.3f} max {np.max(diffs):.3f}")


if __name__ == "__main__":
    main()
