"""Nemotron 3.5 Bolt: each native stage against float64, from the native's own input to that stage.

tf-nemotron-forward MODEL TOKENS.npy OUT.bin --decode-from N --stages DIR dumps every stage forward.Point names, of
every layer of every forward. Here each stage is recomputed in float64 from the dequantized checkpoint weights and the
native's own input to that stage (its x for a projection, its in_proj for the conv, its conv for the scan, ...), so the
errors are each stage's own and do not compound. Prompt rows (forwards of several rows) and decode rows (one row) are
reported apart. Errors are in bf16 ulps of the float64 value; route picks are compared exactly.

usage: nemotron_stages.py MODEL STAGES_DIR [--json OUT]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

TOP_K = 9


def bf16_to_f64(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def bf16_round(v: np.ndarray) -> np.ndarray:
    return np.array(mx.array(v.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32), np.float64)


def ulp_bf16(v: np.ndarray) -> np.ndarray:
    """One bf16 ulp at |v| (the floor at the smallest normal's)."""
    e = np.floor(np.log2(np.maximum(np.abs(v), 2.0 ** -126)))
    return 2.0 ** (e - 7)


class Weights:
    """The checkpoint's tensors as float64, dequantized where {path}.scales exists (MLX's own rule)."""

    def __init__(self, model: Path):
        index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {name: model / f for name, f in index.items()}
        self.cache: dict[Path, dict] = {}
        self.group = json.loads((model / "config.json").read_text())["quantization"]["group_size"]

    def raw(self, name: str) -> mx.array:
        f = self.files[name]
        if f not in self.cache:
            self.cache = {f: mx.load(str(f))}  # one shard at a time: they hold the whole model
        return self.cache[f][name]

    def has(self, name: str) -> bool:
        return name in self.files

    def get(self, path: str, rows=None) -> np.ndarray:
        """`path`.weight as float64 [.., out, in]; `rows` takes those leading indices first (an expert's slice)."""
        w = self.raw(path + ".weight")
        if not self.has(path + ".scales"):
            w = w if rows is None else w[mx.array(rows)]
            return np.array(w.astype(mx.float32), np.float64)
        s, b = self.raw(path + ".scales"), self.raw(path + ".biases")
        if rows is not None:
            idx = mx.array(rows)
            w, s, b = w[idx], s[idx], b[idx]
        bits = w.shape[-1] * 32 // (s.shape[-1] * self.group)
        with mx.stream(mx.cpu):
            d = mx.dequantize(w, s, b, self.group, bits)
            return np.array(d.astype(mx.float32), np.float64)

    def mlx(self, path: str, x: np.ndarray, rows=None) -> np.ndarray:
        """MLX's own bf16 result for x @ `path`^T on the same input (quantized_matmul where it is quantized): the
        floor a native projection is held to."""
        xb = mx.array(x.astype(np.float32)).astype(mx.bfloat16)
        w = self.raw(path + ".weight")
        if rows is not None:
            w = w[mx.array(rows)][0]
        if not self.has(path + ".scales"):
            y = xb @ w.astype(mx.bfloat16).T
        else:
            s, b = self.raw(path + ".scales"), self.raw(path + ".biases")
            if rows is not None:
                idx = mx.array(rows)
                s, b = s[idx][0], b[idx][0]
            bits = w.shape[-1] * 32 // (s.shape[-1] * self.group)
            y = mx.quantized_matmul(xb, w, s, b, transpose=True, group_size=self.group, bits=bits)
        return np.array(y.astype(mx.float32), np.float64)

    def vec(self, name: str) -> np.ndarray:
        return np.array(self.raw(name).astype(mx.float32), np.float64)


class Dump:
    def __init__(self, d: Path):
        self.bin = np.memmap(d / "stages.bin", np.uint8, "r")
        self.rec = defaultdict(dict)  # (layer, point) -> {forward start: (rows, offset, bytes)}
        self.forwards: list[tuple[int, int]] = []
        for line in (d / "stages.jsonl").read_text().splitlines():
            r = json.loads(line)
            self.rec[(r["layer"], r["point"])][r["at"]] = (r["rows"], r["offset"], r["bytes"])
            if (r["at"], r["rows"]) not in self.forwards:
                self.forwards.append((r["at"], r["rows"]))

    def get(self, layer: int, point: str, at: int, dtype=np.uint16, width=None) -> np.ndarray:
        rows, off, n = self.rec[(layer, point)][at]
        raw = np.frombuffer(self.bin[off:off + n], dtype)
        x = bf16_to_f64(raw) if dtype is np.uint16 else raw.astype(np.float64 if dtype is np.float32 else np.int64)
        return x.reshape(rows if width is None else -1, -1)

    def has(self, layer: int, point: str) -> bool:
        return (layer, point) in self.rec


class Report:
    """Errors a stage, prompt rows and decode rows apart, the native's and (for projections) MLX's on the same input.

    An error is in bf16 ulps of its row's RMS: what a bf16 output of that row's scale can resolve, so outputs that
    cancel to near zero do not read as huge relative errors."""

    def __init__(self):
        self.err = defaultdict(list)  # (kind, stage, phase, who) -> arrays of row-ulps
        self.rel = defaultdict(lambda: [0.0, 0.0])  # sum of squared error, sum of squared reference
        self.nonfinite = defaultdict(int)
        self.picks = defaultdict(lambda: [0, 0])  # differing picks, picks
        self.pick_details = []  # each differing pick: layer, row, the two choices, float64's gap, whether fp32 ties

    def add(self, kind, stage, phase, got, want, mlx=None):
        for who, out in (("native", got), ("mlx", mlx)):
            if out is None:
                continue
            key = (kind, stage, phase, who)
            out, ref = np.atleast_2d(out), np.atleast_2d(want)
            bad = ~np.isfinite(out)
            self.nonfinite[key] += int((bad & np.isfinite(ref)).sum())
            # the row's scale: the larger of its float64 RMS and its output's (a row float64 makes exactly 0, as relu^2
            # can, is measured at what its bf16 output can resolve)
            rms_ref = np.sqrt(np.mean(np.where(np.isfinite(ref), ref, 0) ** 2, -1, keepdims=True))
            rms_out = np.sqrt(np.mean(np.where(np.isfinite(out), out, 0) ** 2, -1, keepdims=True))
            scale = ulp_bf16(np.maximum(rms_ref, rms_out))
            ok = np.isfinite(out) & np.isfinite(ref)
            d = np.where(ok, out - ref, 0)
            self.err[key].append((np.abs(d) / scale)[ok])
            self.rel[key][0] += float(np.sum(d * d))
            self.rel[key][1] += float(np.sum(np.where(ok, ref, 0) ** 2))

    def rows(self):
        out = []
        for key in sorted(self.err):
            u = np.concatenate(self.err[key])
            e2, r2 = self.rel[key]
            out.append(dict(kind=key[0], stage=key[1], phase=key[2], who=key[3], outputs=int(u.size),
                            rel_rms=(e2 / r2) ** 0.5 if r2 else 0.0, row_ulps_max=float(u.max()),
                            row_ulps_p99=float(np.quantile(u, 0.99)), row_ulps_mean=float(u.mean()),
                            nonfinite=self.nonfinite[key]))
        for key, (bad, n) in sorted(self.picks.items()):
            out.append(dict(kind=key[0], stage=key[1], phase=key[2], picks=n, differing_picks=bad))
        out.extend(dict(differing_pick=d) for d in self.pick_details)
        return out


def rms(x, w, eps):
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w


def silu(x):
    return x / (1 + np.exp(-x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("stages", type=Path)
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    cfg = json.loads((a.model / "config.json").read_text())
    kinds = cfg["layers_block_type"]
    D, eps = cfg["hidden_size"], cfg["layer_norm_epsilon"]
    H, P, G, N, K = cfg["mamba_num_heads"], cfg["mamba_head_dim"], cfg["n_groups"], cfg["ssm_state_size"], cfg["conv_kernel"]
    XD = H * P
    CD = XD + 2 * G * N
    lim = cfg["time_step_limit"]
    lo_dt = float(lim[0])
    hi_dt = float("inf") if isinstance(lim[1], dict) else float(lim[1])
    QH, KH, HD = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    scaling = cfg["routed_scaling_factor"]
    w = Weights(a.model)
    dump = Dump(a.stages)
    rep = Report()
    phase = lambda rows: "prompt" if rows > 1 else "decode"

    for i, kind in enumerate(kinds):
        pre = f"backbone.layers.{i}"
        norm_next = w.vec(f"backbone.layers.{i + 1}.norm.weight") if i + 1 < len(kinds) else w.vec("backbone.norm_f.weight")
        norm_here = w.vec(f"{pre}.norm.weight")
        if kind == "mamba":
            w_in, w_out = w.get(f"{pre}.mixer.in_proj"), w.get(f"{pre}.mixer.out_proj")
            conv_w = w.vec(f"{pre}.mixer.conv1d.weight").reshape(CD, K)
            conv_b = w.vec(f"{pre}.mixer.conv1d.bias")
            A = -np.exp(w.vec(f"{pre}.mixer.A_log"))
            Dskip, dt_bias = w.vec(f"{pre}.mixer.D"), w.vec(f"{pre}.mixer.dt_bias")
            gnorm_w = w.vec(f"{pre}.mixer.norm.weight")
            conv_hist = np.zeros((K - 1, CD))
            state = np.zeros((H, P, N))
        elif kind == "attention":
            w_qkv = np.concatenate([w.get(f"{pre}.mixer.{n}_proj") for n in "qkv"])
            w_o = w.get(f"{pre}.mixer.o_proj")
            keys, vals = [], []
        else:
            w_gate = w.get(f"{pre}.mixer.gate")
            bias = w.vec(f"{pre}.mixer.gate.e_score_correction_bias")
            w_up, w_down = w.get(f"{pre}.mixer.shared_experts.up_proj"), w.get(f"{pre}.mixer.shared_experts.down_proj")
            w_into, w_outl = w.get(f"{pre}.mixer.fc1_latent_proj"), w.get(f"{pre}.mixer.fc2_latent_proj")
            experts = {}

        for at, rows in dump.forwards:
            ph = phase(rows)
            h_in, x_in = dump.get(i, "h_in", at), dump.get(i, "x_in", at)
            rep.add("all", "input norm", ph, x_in, rms(h_in, norm_here, eps))
            if kind == "mamba":
                proj = dump.get(i, "in_proj", at)
                rep.add(kind, "in_proj", ph, proj, x_in @ w_in.T, w.mlx(f"{pre}.mixer.in_proj", x_in))
                z, xbc_in, dt_raw = proj[:, :XD], proj[:, XD:XD + CD], proj[:, XD + CD:]
                # the depthwise causal conv over the native's own in_proj rows, history carried across forwards
                seq = np.concatenate([conv_hist, xbc_in])
                conv = np.stack([np.sum(seq[t:t + K].T * conv_w, -1) for t in range(rows)]) + conv_b
                rep.add(kind, "conv", ph, dump.get(i, "conv", at), silu(conv))
                conv_hist = seq[-(K - 1):]
                xbc = dump.get(i, "conv", at)  # the scan reads the native's conv
                xs, Bm, Cm = xbc[:, :XD].reshape(rows, H, P), xbc[:, XD:XD + G * N].reshape(rows, G, N), xbc[:, XD + G * N:].reshape(rows, G, N)
                y = np.empty((rows, H, P))
                for t in range(rows):
                    dt = np.clip(np.logaddexp(0, dt_raw[t] + dt_bias), lo_dt, hi_dt)
                    g = np.repeat(np.arange(G), H // G)
                    state = state * np.exp(A * dt)[:, None, None] + (dt[:, None] * xs[t])[:, :, None] * Bm[t][g][:, None, :]
                    y[t] = np.einsum("hpn,hn->hp", state, Cm[t][g]) + Dskip[:, None] * xs[t]
                y = y.reshape(rows, XD) * silu(z)
                rep.add(kind, "scan (gated)", ph, dump.get(i, "scan", at), y)
                y_n = dump.get(i, "scan", at)
                gn = y_n.reshape(rows, G, XD // G)
                gn = (gn / np.sqrt(np.mean(gn * gn, -1, keepdims=True) + eps)).reshape(rows, XD) * gnorm_w
                rep.add(kind, "group norm", ph, dump.get(i, "gnorm", at), gn)
                rep.add(kind, "out_proj", ph, dump.get(i, "mixer_out", at), dump.get(i, "gnorm", at) @ w_out.T, w.mlx(f"{pre}.mixer.out_proj", dump.get(i, "gnorm", at)))
                h_ref = h_in + dump.get(i, "mixer_out", at)
            elif kind == "attention":
                qkv = dump.get(i, "qkv", at)
                rep.add(kind, "qkv", ph, qkv, x_in @ w_qkv.T, np.concatenate([w.mlx(f"{pre}.mixer.{n}_proj", x_in) for n in "qkv"], -1))
                q = qkv[:, :QH * HD].reshape(rows, QH, HD)
                keys.append(qkv[:, QH * HD:QH * HD + KH * HD].reshape(rows, KH, HD))
                vals.append(qkv[:, QH * HD + KH * HD:].reshape(rows, KH, HD))
                k_all, v_all = np.concatenate(keys), np.concatenate(vals)
                out = np.empty((rows, QH, HD))
                for t in range(rows):
                    n = at + t + 1  # causal: this row sees every row up to itself
                    for hq in range(QH):
                        kh = hq // (QH // KH)
                        s = k_all[:n, kh] @ q[t, hq] / np.sqrt(HD)
                        s = np.exp(s - s.max())
                        out[t, hq] = (s / s.sum()) @ v_all[:n, kh]
                rep.add(kind, "attention", ph, dump.get(i, "attn", at), out.reshape(rows, QH * HD))
                rep.add(kind, "o_proj", ph, dump.get(i, "mixer_out", at), dump.get(i, "attn", at) @ w_o.T, w.mlx(f"{pre}.mixer.o_proj", dump.get(i, "attn", at)))
                h_ref = h_in + dump.get(i, "mixer_out", at)
            else:
                logits = dump.get(i, "gate", at)
                rep.add(kind, "gate", ph, logits, x_in @ w_gate.T, w.mlx(f"{pre}.mixer.gate", x_in))
                idx = dump.get(i, "route_idx", at, np.uint32).reshape(rows, TOP_K)
                wt = dump.get(i, "route_wt", at, np.float32).reshape(rows, TOP_K)
                prob = 1 / (1 + np.exp(-logits))
                key = prob + bias
                want_idx = np.argsort(-key, axis=-1, kind="stable")[:, :TOP_K]
                bad = 0
                for r in range(rows):
                    if set(want_idx[r]) == set(idx[r]):
                        continue
                    bad += 1
                    ours = [int(e) for e in idx[r] if e not in want_idx[r]]
                    theirs = [int(e) for e in want_idx[r] if e not in idx[r]]
                    # the keys as the engine and mlx-lm both compute them: fp32 sigmoid of the bf16 logit, + the fp32 bias
                    k32 = (1 / (1 + np.exp(-logits[r].astype(np.float32)))).astype(np.float32) + bias.astype(np.float32)
                    rep.pick_details.append(dict(layer=i, row=at + r, phase=ph, native=ours, float64=theirs,
                        float64_gap=float(key[r][theirs].min() - key[r][ours].max()),
                        fp32_tie=bool(np.all(k32[ours] == k32[theirs]))))
                rep.picks[(kind, "route picks", ph)][0] += bad
                rep.picks[(kind, "route picks", ph)][1] += rows
                sel = np.take_along_axis(prob, idx, -1)  # the weights of the native's own picks
                rep.add(kind, "route weights", ph, wt, sel / (sel.sum(-1, keepdims=True) + 1e-20) * scaling)
                sh = dump.get(i, "shared", at)
                up_m = w.mlx(f"{pre}.mixer.shared_experts.up_proj", x_in)
                act_m = bf16_round(np.maximum(up_m, 0) ** 2)
                rep.add(kind, "shared expert", ph, sh, np.maximum(x_in @ w_up.T, 0) ** 2 @ w_down.T, w.mlx(f"{pre}.mixer.shared_experts.down_proj", act_m))
                lat = dump.get(i, "into", at)
                rep.add(kind, "into (fc1_latent)", ph, lat, x_in @ w_into.T, w.mlx(f"{pre}.mixer.fc1_latent_proj", x_in))
                ey = dump.get(i, "experts", at).reshape(rows, TOP_K, -1)
                want_ey = np.empty_like(ey)
                mlx_ey = np.empty_like(ey)
                for r in range(rows):
                    for j in range(TOP_K):
                        e = int(idx[r, j])
                        if e not in experts:
                            experts[e] = (w.get(f"{pre}.mixer.switch_mlp.fc1", [e])[0], w.get(f"{pre}.mixer.switch_mlp.fc2", [e])[0])
                        f1, f2 = experts[e]
                        want_ey[r, j] = np.maximum(lat[r] @ f1.T, 0) ** 2 @ f2.T
                        up_e = w.mlx(f"{pre}.mixer.switch_mlp.fc1", lat[r:r + 1], [e])
                        mlx_ey[r, j] = w.mlx(f"{pre}.mixer.switch_mlp.fc2", bf16_round(np.maximum(up_e, 0) ** 2), [e])[0]
                rep.add(kind, "routed experts", ph, ey.reshape(rows * TOP_K, -1), want_ey.reshape(rows * TOP_K, -1), mlx_ey.reshape(rows * TOP_K, -1))
                comb = dump.get(i, "combine", at)
                rep.add(kind, "combine", ph, comb, np.einsum("rk,rkd->rd", wt, ey))
                out = dump.get(i, "mixer_out", at)
                rep.add(kind, "out (fc2_latent)", ph, out, comb @ w_outl.T, w.mlx(f"{pre}.mixer.fc2_latent_proj", comb))
                h_ref = h_in + out + sh
            h_out = dump.get(i, "h_out", at)
            rep.add("all", "residual", ph, h_out, h_ref)
            rep.add("all", "next norm", ph, dump.get(i, "x_out", at), rms(h_out, norm_next, eps))
        print(f"layer {i} {kind} done", flush=True)

    rows = rep.rows()
    for r in rows:
        print(json.dumps(r))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
