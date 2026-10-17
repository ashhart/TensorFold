#!/usr/bin/env python3
"""Fish Audio Drama 3 TTS: standalone PyTorch reference ("oracle").

Follows SGLang-Omni 416a70c0 sglang_omni/models/fishaudio_drama3 (sglang_model.py,
audio_decoder.py, decoder_layers.py, sampling.py, prompt.py, weights.py) and the
HF transformers 5.15 Qwen3.5-MoE text model, with no sglang / flashinfer / fla.

Precision policy (mirrors the SGLang serving path):
  * weights bf16; routed MoE experts stay FP8 E4M3 on the GPU with 128x128 block
    scales and are dequantized per selected expert on the fly.
  * --moe-act fp8 (default) emulates SGLang's Fp8MoEMethod (block weights,
    activation_scheme=dynamic): every expert GEMM input is quantized per token
    and per 128-wide K group to E4M3 (scale = amax/448), both operands are
    dequantized exactly and multiplied in fp32.  --moe-act bf16 skips the
    activation quantization (dequantized weights, bf16-rounded activations).
  * every RMSNorm, softmax, rotary, GDN gate/decay/recurrent state and attention
    score is computed in fp32; residual streams and layer outputs are bf16.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# the codec's PyTorch modules (package fishcodec): see build_codec
if os.environ.get("FISHCODEC_DIR"):
    sys.path.insert(0, os.environ["FISHCODEC_DIR"])
from oracle_prompt import ReferencePart, build_prompt_rows  # noqa: E402

BF16 = torch.bfloat16
FP8_MAX = 448.0

# sampling.py constants (fixed for Drama 3, not request parameters)
RAS_WINDOW_FRAMES = 16
EMPTY_RAS_SLOT = -1
RAS_TEMPERATURE = 1.0
RAS_TOP_P = 0.95
RAS_TOP_K = 30
MINIMUM_TEMPERATURE = 1e-5


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    hidden: int
    vocab: int
    n_layers: int
    layer_types: list
    n_heads: int
    n_kv: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    eps: float
    n_experts: int
    top_k: int
    moe_inter: int
    shared_inter: int
    lin_k_heads: int
    lin_v_heads: int
    lin_k_dim: int
    lin_v_dim: int
    conv_k: int
    # tower
    t_dim: int
    t_heads: int
    t_kv: int
    t_head_dim: int
    t_inter: int
    t_eps: float
    t_rope_base: float
    t_rotary_dim: int
    n_shared: int
    n_codebooks: int
    codebook_size: int
    backbone_codebooks: int
    # ids
    eos: int
    sem_begin: int
    sem_end: int

    @classmethod
    def load(cls, path: str) -> "Cfg":
        c = json.load(open(os.path.join(path, "config.json")))
        t, d = c["text_config"], c["audio_decoder_config"]
        assert c["speech_template_version"] == "reference-bracket-v1"
        assert d["layernorm_zero_centered_gamma"] and t["layernorm_zero_centered_gamma"]
        assert t["attn_output_gate"] and t["use_qk_norm"] and t["shared_expert_gate"]
        assert d["vq_backbone_codebooks"] == 1 and d["decoder_type"] == "codebook_mtp_shared"
        rp = t["rope_parameters"]
        assert rp["rope_type"] == "default"
        return cls(
            hidden=t["hidden_size"], vocab=t["vocab_size"], n_layers=t["num_hidden_layers"],
            layer_types=t["layer_types"], n_heads=t["num_attention_heads"],
            n_kv=t["num_key_value_heads"], head_dim=t["head_dim"],
            rotary_dim=int(t["head_dim"] * rp["partial_rotary_factor"]),
            rope_theta=float(rp["rope_theta"]), eps=t["rms_norm_eps"],
            n_experts=t["num_experts"], top_k=t["num_experts_per_tok"],
            moe_inter=t["moe_intermediate_size"], shared_inter=t["shared_expert_intermediate_size"],
            lin_k_heads=t["linear_num_key_heads"], lin_v_heads=t["linear_num_value_heads"],
            lin_k_dim=t["linear_key_head_dim"], lin_v_dim=t["linear_value_head_dim"],
            conv_k=t["linear_conv_kernel_dim"],
            t_dim=d["dim"], t_heads=d["n_head"], t_kv=d["n_local_heads"], t_head_dim=d["head_dim"],
            t_inter=d["intermediate_size"], t_eps=d["norm_eps"], t_rope_base=float(d["rope_base"]),
            # hf_config.FishOmniAudioDecoderConfig.rotary_dim = int(head_dim * rotary_percent)
            t_rotary_dim=int(d["head_dim"] * d["rotary_percent"]),
            n_shared=d["shared_enrichment_layers"], n_codebooks=d["num_codebooks"],
            codebook_size=d["vocab_size"], backbone_codebooks=d["vq_backbone_codebooks"],
            eos=c["eos_token_id"], sem_begin=c["semantic_start_token_id"],
            sem_end=c["semantic_end_token_id"],
        )


# --------------------------------------------------------------------------- #
# Primitive ops (fp32 internals, bf16 in/out)
# --------------------------------------------------------------------------- #
def rms_zc(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Zero-centred-gamma RMSNorm (GemmaRMSNorm / Qwen3_5MoeRMSNorm): fp32, *(1+w)."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1.0 + w.float())).to(x.dtype)


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return F.linear(x, w)


def fp8_act_quant_dequant(x: torch.Tensor, group: int = 128) -> torch.Tensor:
    """per_token_group_quant_fp8 (E4M3, scale=max(amax,1e-10)/448), then exact dequant to fp32."""
    t, k = x.shape
    xg = x.float().view(t, k // group, group)
    scale = xg.abs().amax(-1, keepdim=True).clamp(min=1e-10) / FP8_MAX
    q = (xg / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q.float() * scale).view(t, k)


def dequant_block(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """FP8 [N, K] with weight_scale_inv [N/128, K/128] -> fp32 [N, K] (W = q * scale_inv)."""
    n, k = w.shape
    return (w.float().view(n // 128, 128, k // 128, 128) * s.float()[:, None, :, None]).view(n, k)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Drama3:
    def __init__(self, path: str, device: str = "cuda", moe_act: str = "fp8") -> None:
        self.cfg = cfg = Cfg.load(path)
        self.dev = torch.device(device)
        self.moe_act = moe_act
        self.W: dict[str, torch.Tensor] = {}
        self.experts: list[dict[str, torch.Tensor]] = []
        self._load(path)
        c = cfg
        # main-token candidates: sorted {eos} U [sem_begin, sem_end]  (audio_decoder.py)
        self.cand_ids = torch.tensor(
            sorted({c.eos, *range(c.sem_begin, c.sem_end + 1)}), device=self.dev
        )
        self.cb_offsets = torch.arange(c.n_codebooks, device=self.dev) * c.codebook_size
        # backbone rotary (NeoX / rotate_half on the first rotary_dim dims), fp32
        self.inv_freq = 1.0 / (c.rope_theta ** (torch.arange(0, c.rotary_dim, 2, dtype=torch.float32, device=self.dev) / c.rotary_dim))
        # tower rotary (interleaved pairs), fp32 cos/sin cache exactly as InterleavedRotaryEmbedding
        tinv = 1.0 / c.t_rope_base ** (torch.arange(0, c.t_rotary_dim, 2, dtype=torch.float32) / c.t_rotary_dim)
        self.t_inv_freq = tinv.to(self.dev)
        self.hooks: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------ load
    def _load(self, path: str) -> None:
        c = self.cfg
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        by_file: dict[str, list[str]] = {}
        for k, f in idx.items():
            by_file.setdefault(f, []).append(k)
        E, I, H = c.n_experts, c.moe_inter, c.hidden
        for _ in range(c.n_layers):
            self.experts.append({
                "w13": torch.empty(E, 2 * I, H, dtype=torch.float8_e4m3fn, device=self.dev),
                "s13": torch.empty(E, 2 * I // 128, H // 128, dtype=torch.float32, device=self.dev),
                "w2": torch.empty(E, H, I, dtype=torch.float8_e4m3fn, device=self.dev),
                "s2": torch.empty(E, H // 128, I // 128, dtype=torch.float32, device=self.dev),
            })
        filled = 0
        t0 = time.time()
        for fname in sorted(by_file):
            with safe_open(os.path.join(path, fname), framework="pt", device=str(self.dev)) as f:
                for k in by_file[fname]:
                    t = f.get_tensor(k)
                    if ".mlp.experts." in k:
                        # model.language_model.layers.{L}.mlp.experts.{e}.{proj}.{weight|weight_scale_inv}
                        p = k.split(".")
                        li, e, proj, kind = int(p[3]), int(p[6]), p[7], p[8]
                        ex = self.experts[li]
                        if kind == "weight":
                            assert t.dtype == torch.float8_e4m3fn, (k, t.dtype)
                        else:
                            t = t.float()
                        if proj == "gate_proj":
                            (ex["w13"] if kind == "weight" else ex["s13"])[e, : t.shape[0]].copy_(t)
                        elif proj == "up_proj":
                            (ex["w13"] if kind == "weight" else ex["s13"])[e, t.shape[0]:].copy_(t)
                        else:
                            (ex["w2"] if kind == "weight" else ex["s2"])[e].copy_(t)
                        filled += 1
                    else:
                        assert t.dtype in (BF16, torch.float32), (k, t.dtype)
                        self.W[k] = t if (k.endswith("A_log") or k.endswith("dt_bias")) else t.to(BF16)
                    del t
            print(f"[load] {fname} done ({time.time() - t0:.1f}s)", file=sys.stderr)
        assert filled == c.n_layers * E * 6, filled
        print(f"[load] {len(self.W)} dense tensors, {filled} expert tensors, "
              f"GPU alloc {torch.cuda.memory_allocated() / 2**30:.1f} GiB", file=sys.stderr)

    def w(self, name: str) -> torch.Tensor:
        return self.W[name]

    # ------------------------------------------------------------ embeddings
    def is_semantic(self, ids: torch.Tensor) -> torch.Tensor:
        return (ids >= self.cfg.sem_begin) & (ids <= self.cfg.sem_end)

    def codebook_sum(self, frames: torch.Tensor, first: int, end: int) -> torch.Tensor:
        """Sum of audio_decoder.codebook_embeddings rows of codes first..end-1 (fp32 accum, bf16 out)."""
        tab = self.w("audio_decoder.codebook_embeddings.weight")
        rows = frames[:, 1 + first: 1 + end].clamp(0, self.cfg.codebook_size - 1) + self.cb_offsets[first:end]
        return tab[rows].float().sum(1).to(BF16)

    def embed_backbone(self, frames: torch.Tensor) -> torch.Tensor:
        """AudioDecoder.embed_backbone: semantic rows get (tok + emb(code1)) / sqrt(2)."""
        c = self.cfg
        text = self.w("model.language_model.embed_tokens.weight")[frames[:, 0]]
        mixed = (text + self.codebook_sum(frames, 1, 1 + c.backbone_codebooks)) * (1.0 / math.sqrt(1.0 + c.backbone_codebooks))
        return torch.where(self.is_semantic(frames[:, 0]).unsqueeze(-1), mixed, text)

    def shared_input(self, frames: torch.Tensor) -> torch.Tensor:
        """Extra input of tower shared layer 0: sum of codes 2..9 of semantic frames."""
        c = self.cfg
        s = self.codebook_sum(frames, 1 + c.backbone_codebooks, c.n_codebooks)
        return torch.where(self.is_semantic(frames[:, 0]).unsqueeze(-1), s, torch.zeros_like(s))

    # --------------------------------------------------------------- state
    def new_state(self, max_len: int) -> dict:
        c = self.cfg
        st = {"bb": [], "tower_k": [], "tower_v": [], "len": 0}
        conv_dim = 2 * c.lin_k_heads * c.lin_k_dim + c.lin_v_heads * c.lin_v_dim
        for lt in c.layer_types:
            if lt == "linear_attention":
                st["bb"].append({
                    "conv": torch.zeros(c.conv_k - 1, conv_dim, dtype=BF16, device=self.dev),
                    "S": torch.zeros(c.lin_v_heads, c.lin_k_dim, c.lin_v_dim, dtype=torch.float32, device=self.dev),
                })
            else:
                st["bb"].append({
                    "k": torch.zeros(max_len, c.n_kv, c.head_dim, dtype=BF16, device=self.dev),
                    "v": torch.zeros(max_len, c.n_kv, c.head_dim, dtype=BF16, device=self.dev),
                })
        for _ in range(c.n_shared + c.n_codebooks - 1):
            st["tower_k"].append(torch.zeros(max_len, c.t_kv, c.t_head_dim, dtype=BF16, device=self.dev))
            st["tower_v"].append(torch.zeros(max_len, c.t_kv, c.t_head_dim, dtype=BF16, device=self.dev))
        return st

    def _hook(self, name: str, t: torch.Tensor) -> None:
        if self.hooks is not None:
            self.hooks[name] = t.detach().clone()

    # ---------------------------------------------------- backbone: GDN layer
    def gdn(self, li: int, x: torch.Tensor, st: dict) -> torch.Tensor:
        c = self.cfg
        p = f"model.language_model.layers.{li}.linear_attn."
        T = x.shape[0]
        kd, vd = c.lin_k_heads * c.lin_k_dim, c.lin_v_heads * c.lin_v_dim
        qkv = linear(x, self.w(p + "in_proj_qkv.weight"))              # [T, 2kd+vd] bf16
        z = linear(x, self.w(p + "in_proj_z.weight"))                  # [T, vd]
        b = linear(x, self.w(p + "in_proj_b.weight"))                  # [T, Hv]
        a = linear(x, self.w(p + "in_proj_a.weight"))                  # [T, Hv]
        # causal depthwise conv1d (kernel 4, no bias) over [conv_state; qkv], fp32, SiLU
        cw = self.w(p + "conv1d.weight").float().squeeze(1)            # [C, K]
        xin = torch.cat([st["conv"], qkv], 0)                          # [K-1+T, C] bf16
        xf = xin.float()
        out = torch.zeros(T, xin.shape[1], dtype=torch.float32, device=self.dev)
        for j in range(c.conv_k):
            out += xf[j: j + T] * cw[:, j]
        st["conv"] = xin[-(c.conv_k - 1):].clone()
        conv = F.silu(out).to(BF16)
        q, k, v = conv.split([kd, kd, vd], -1)
        q = q.float().view(T, c.lin_k_heads, c.lin_k_dim)
        k = k.float().view(T, c.lin_k_heads, c.lin_k_dim)
        v = v.float().view(T, c.lin_v_heads, c.lin_v_dim)
        beta = torch.sigmoid(b.float())                                 # [T, Hv]
        g = -self.w(p + "A_log").float().exp() * F.softplus(a.float() + self.w(p + "dt_bias").float())
        # l2norm (fla: x * rsqrt(sum x^2 + 1e-6)), GQA expand k-heads -> v-heads (head h uses h // rep)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        rep = c.lin_v_heads // c.lin_k_heads
        q = q.repeat_interleave(rep, 1) * (c.lin_k_dim ** -0.5)
        k = k.repeat_interleave(rep, 1)
        S = st["S"]
        o = torch.empty(T, c.lin_v_heads, c.lin_v_dim, dtype=torch.float32, device=self.dev)
        eg = g.exp()
        for t in range(T):
            S = S * eg[t][:, None, None]
            kv_mem = torch.einsum("hkv,hk->hv", S, k[t])
            delta = (v[t] - kv_mem) * beta[t][:, None]
            S = S + k[t][:, :, None] * delta[:, None, :]
            o[t] = torch.einsum("hkv,hk->hv", S, q[t])
        st["S"] = S
        if li == 0:
            self._hook("gdn0.conv_out", conv)
            self._hook("gdn0.g", g)
            self._hook("gdn0.beta", beta)
            self._hook("gdn0.core_out", o)
            self._hook("gdn0.state_after", S)
        # gated RMSNorm (plain gamma, NOT zero-centred), fp32: norm(o) * w * silu(z)
        nw = self.w(p + "norm.weight").float()
        zf = z.float().view(T, c.lin_v_heads, c.lin_v_dim)
        y = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + c.eps) * nw * F.silu(zf)
        return linear(y.reshape(T, vd).to(BF16), self.w(p + "out_proj.weight"))

    # ---------------------------------------------- backbone: gated attention
    def rope_neox(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x [T, H, D] bf16 -> rotate_half rope on the first rotary_dim dims (fp32 math)."""
        rd = self.cfg.rotary_dim
        fr = pos.float()[:, None] * self.inv_freq[None]                 # [T, rd/2]
        cos, sin = fr.cos()[:, None], fr.sin()[:, None]
        xr = x[..., :rd].float()
        x1, x2 = xr[..., : rd // 2], xr[..., rd // 2:]
        rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
        return torch.cat([rot.to(x.dtype), x[..., rd:]], -1)

    def full_attn(self, li: int, x: torch.Tensor, pos: torch.Tensor, st: dict) -> torch.Tensor:
        c = self.cfg
        p = f"model.language_model.layers.{li}.self_attn."
        T = x.shape[0]
        qg = linear(x, self.w(p + "q_proj.weight")).view(T, c.n_heads, 2 * c.head_dim)
        q, gate = qg[..., : c.head_dim], qg[..., c.head_dim:]          # per head [q | gate]
        k = linear(x, self.w(p + "k_proj.weight")).view(T, c.n_kv, c.head_dim)
        v = linear(x, self.w(p + "v_proj.weight")).view(T, c.n_kv, c.head_dim)
        q = rms_zc(q, self.w(p + "q_norm.weight"), c.eps)
        k = rms_zc(k, self.w(p + "k_norm.weight"), c.eps)
        q, k = self.rope_neox(q, pos), self.rope_neox(k, pos)
        st["k"][pos] = k
        st["v"][pos] = v
        L = int(pos[-1]) + 1
        o = self.attend(q, st["k"][:L], st["v"][:L], pos, c.head_dim ** -0.5)
        if li == 3:
            self._hook("attn3.q_roped", q)
            self._hook("attn3.k_roped", k)
            self._hook("attn3.v", v)
            self._hook("attn3.attn_out", o)
        o = o.reshape(T, -1) * torch.sigmoid(gate.reshape(T, -1))
        return linear(o, self.w(p + "o_proj.weight"))

    @staticmethod
    def attend(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, pos: torch.Tensor, scale: float) -> torch.Tensor:
        """Causal GQA attention by absolute position; fp32 scores/softmax/PV, bf16 out. q [T,H,D], K/V [L,Hk,D]."""
        H, Hk = q.shape[1], K.shape[1]
        Kf = K.float().repeat_interleave(H // Hk, 1)                    # [L, H, D]
        Vf = V.float().repeat_interleave(H // Hk, 1)
        s = torch.einsum("thd,lhd->htl", q.float(), Kf) * scale
        mask = torch.arange(K.shape[0], device=q.device)[None, :] > pos[:, None]  # [T, L]
        s = s.masked_fill(mask[None], float("-inf"))
        pr = torch.softmax(s, -1)
        return torch.einsum("htl,lhd->thd", pr, Vf).to(q.dtype)

    # --------------------------------------------------------- backbone: MoE
    def moe(self, li: int, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        p = f"model.language_model.layers.{li}.mlp."
        T = x.shape[0]
        logits = linear(x, self.w(p + "gate.weight"))                   # bf16 [T, E]
        probs = torch.softmax(logits.float(), -1)
        tw, ti = torch.topk(probs, c.top_k, -1)
        tw = tw / tw.sum(-1, keepdim=True)                               # norm_topk_prob
        if li == 0:
            self._hook("moe0.topk_ids", ti)
            self._hook("moe0.topk_weights", tw)
        # shared expert (bf16) with sigmoid gate
        sh = linear(F.silu(linear(x, self.w(p + "shared_expert.gate_proj.weight")).float()).to(BF16)
                    * linear(x, self.w(p + "shared_expert.up_proj.weight")),
                    self.w(p + "shared_expert.down_proj.weight"))
        sh = sh * torch.sigmoid(linear(x, self.w(p + "shared_expert_gate.weight")))
        # routed experts, fp32 accumulation
        ex = self.experts[li]
        routed = torch.zeros(T, c.hidden, dtype=torch.float32, device=self.dev)
        flat = ti.reshape(-1)
        for e in torch.unique(flat).tolist():
            rows, slot = torch.where(ti == e)
            xe = x[rows]
            w13 = dequant_block(ex["w13"][e], ex["s13"][e])            # fp32 [2I, H]
            w2 = dequant_block(ex["w2"][e], ex["s2"][e])               # fp32 [H, I]
            xin = fp8_act_quant_dequant(xe) if self.moe_act == "fp8" else xe.float()
            h = (xin @ w13.t()).to(BF16)                                # bf16 [n, 2I]
            act = (F.silu(h[:, : c.moe_inter].float()) * h[:, c.moe_inter:].float()).to(BF16)
            ain = fp8_act_quant_dequant(act) if self.moe_act == "fp8" else act.float()
            y = (ain @ w2.t()) * tw[rows, slot][:, None]
            routed.index_add_(0, rows, y)
        return routed.to(BF16) + sh

    # ----------------------------------------------------------- backbone
    def backbone(self, frames: torch.Tensor, pos: torch.Tensor, st: dict, record: dict | None = None):
        """frames [T, 11] at positions pos -> (final-norm hidden [T, H] bf16, logits [T, V] bf16)."""
        c = self.cfg
        h = self.embed_backbone(frames)
        if record is not None:
            record["backbone_input_embeds"] = h.clone()
        for li in range(c.n_layers):
            p = f"model.language_model.layers.{li}."
            xn = rms_zc(h, self.w(p + "input_layernorm.weight"), c.eps)
            if c.layer_types[li] == "linear_attention":
                a = self.gdn(li, xn, st["bb"][li])
            else:
                a = self.full_attn(li, xn, pos, st["bb"][li])
            h = h + a
            xn = rms_zc(h, self.w(p + "post_attention_layernorm.weight"), c.eps)
            h = h + self.moe(li, xn)
            if record is not None:
                record[f"backbone_hidden.layer{li:02d}"] = h.clone()
        hn = rms_zc(h, self.w("model.language_model.norm.weight"), c.eps)
        logits = linear(hn, self.w("lm_head.weight"))
        return hn, logits

    # ---------------------------------------------------------------- tower
    def tower_names(self, j: int) -> tuple[str, str]:
        c = self.cfg
        if j < c.n_shared:
            return f"audio_decoder_mtp_shared.layers.{j}.", "semantic_embedding"
        return f"audio_decoder_mtp.layers.{j - c.n_shared}.", "codebook_embedding"

    def rope_interleaved(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """InterleavedRotaryEmbedding: pairs (2i, 2i+1) of the first t_rotary_dim dims, fp32 math."""
        rd = self.cfg.t_rotary_dim
        ang = pos.float()[:, None] * self.t_inv_freq[None]              # [T, rd/2]
        cos, sin = ang.cos()[:, None], ang.sin()[:, None]
        xr = x[..., :rd].float()
        ev, od = xr[..., 0::2], xr[..., 1::2]
        rot = torch.stack((ev * cos - od * sin, od * cos + ev * sin), -1).flatten(-2)
        return torch.cat([rot.to(x.dtype), x[..., rd:]], -1)

    def tower_layer(self, j: int, h: torch.Tensor, code_ids: torch.Tensor, label_sem: torch.Tensor,
                    shared_in: torch.Tensor | None, pos: torch.Tensor, st: dict) -> torch.Tensor:
        c = self.cfg
        p, emb_name = self.tower_names(j)
        T = h.shape[0]
        tab = self.w(p + emb_name + ".weight")
        emb = torch.where(label_sem.unsqueeze(-1), tab[code_ids.clamp(0, tab.shape[0] - 1)], torch.zeros((), dtype=BF16, device=self.dev))
        if shared_in is not None:
            emb = emb + shared_in.to(BF16)
        mixed = linear(torch.cat((rms_zc(emb, self.w(p + "enorm.weight"), c.t_eps),
                                  rms_zc(h, self.w(p + "hnorm.weight"), c.t_eps)), -1),
                       self.w(p + "eh_proj.weight"))
        tl = p + "transformer_layer."
        xa = rms_zc(mixed, self.w(tl + "attention_norm.weight"), c.t_eps)
        qw = c.t_heads * c.t_head_dim
        kvw = c.t_kv * c.t_head_dim
        # checkpoint wqkv rows: [all queries | all gates | keys | values] (NOT per-head interleaved)
        q, gate, k, v = linear(xa, self.w(tl + "attention.wqkv.weight")).split([qw, qw, kvw, kvw], -1)
        q = rms_zc(q.reshape(T, c.t_heads, c.t_head_dim), self.w(tl + "attention.q_norm.weight"), c.t_eps)
        k = rms_zc(k.reshape(T, c.t_kv, c.t_head_dim), self.w(tl + "attention.k_norm.weight"), c.t_eps)
        q, k = self.rope_interleaved(q, pos), self.rope_interleaved(k, pos)
        v = v.reshape(T, c.t_kv, c.t_head_dim)
        st["tower_k"][j][pos] = k
        st["tower_v"][j][pos] = v
        L = int(pos[-1]) + 1
        o = self.attend(q, st["tower_k"][j][:L], st["tower_v"][j][:L], pos, c.t_head_dim ** -0.5)
        o = linear(o.reshape(T, qw) * torch.sigmoid(gate), self.w(tl + "attention.wo.weight"))
        resid = o + mixed                                                # fused add (bf16)
        ff_in = rms_zc(resid, self.w(tl + "ffn_norm.weight"), c.t_eps)
        ff = linear(F.silu(linear(ff_in, self.w(tl + "feed_forward.w1.weight")).float()).to(BF16)
                    * linear(ff_in, self.w(tl + "feed_forward.w3.weight")),
                    self.w(tl + "feed_forward.w2.weight"))
        return rms_zc(ff + resid, self.w(p + "final_layernorm.weight"), c.t_eps)

    def tower_prefill(self, hn: torch.Tensor, pos: torch.Tensor, in_frames: torch.Tensor,
                      next_frames: torch.Tensor, st: dict, record: dict | None = None) -> None:
        """AudioDecoder.prefill: row p is labelled with frame p+1."""
        c = self.cfg
        lab = self.is_semantic(next_frames[:, 0])
        sh = self.shared_input(in_frames)
        h = hn
        for j in range(c.n_shared + c.n_codebooks - 1):
            code = next_frames[:, 1] if j < c.n_shared else next_frames[:, 1 + j - c.n_shared]
            h = self.tower_layer(j, h, code, lab, sh if j == 0 else None, pos, st)
            if record is not None:
                record[f"tower_prefill.layer{j:02d}"] = h.clone()

    def decode_step(self, logits_row: torch.Tensor, hn_row: torch.Tensor, pos: torch.Tensor,
                    in_frame: torch.Tensor, ras_window: torch.Tensor, samp: "Sampler", st: dict,
                    record: dict | None = None) -> torch.Tensor:
        """AudioDecoder.decode_step for one row -> frame [11] int64."""
        c = self.cfg
        cand_logits = logits_row.index_select(-1, self.cand_ids).float()  # [1, 4097]
        main = samp.semantic(cand_logits, self.cand_ids, ras_window, c.sem_begin, c.codebook_size)
        code0 = (main - c.sem_begin).clamp(0, c.codebook_size - 1)
        lab = self.is_semantic(main)
        sh = self.shared_input(in_frame)
        if record is not None:
            record["tower_decode0.candidate_logits"] = cand_logits[0].clone()
            record["tower_decode0.shared_input"] = sh[0].clone()
        h = hn_row
        codes = [code0]
        for j in range(c.n_shared + c.n_codebooks - 1):
            code = code0 if j < c.n_shared else codes[-1]
            h = self.tower_layer(j, h, code, lab, sh if j == 0 else None, pos, st)
            if record is not None:
                record[f"tower_decode0.layer{j:02d}"] = h[0].clone()
            if j >= c.n_shared:
                hl = linear(h, self.w(f"audio_decoder_mtp.heads.{j - c.n_shared}.weight"))
                if record is not None:
                    record[f"tower_decode0.head{j - c.n_shared}_logits"] = hl[0].float().clone()
                codes.append(samp.plain(hl.float()))
        return torch.stack([main, *codes], -1)[0]


# --------------------------------------------------------------------------- #
# Sampling (sampling.py, torch path; optional greedy)
