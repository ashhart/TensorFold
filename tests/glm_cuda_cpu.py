"""GLM-5.3-Flash's CUDA path on the CPU: a tiny EXL3-layout checkpoint, its Triton kernels in Triton's interpreter,
torch stand-ins for the two CUDA extensions it calls (KDA's chain and replay, the routed EXL3 experts), and two ranks
as two processes joined by gloo. Nothing here runs a CUDA kernel: tests/cuda covers those on a GPU."""

from __future__ import annotations

import io
import json
import math
import os
import traceback
from types import SimpleNamespace

import numpy as np
import torch

D, V, S = 512, 1024, 4
MOE = 256                 # expert width: each rank's half (128) a whole EXL3 Hadamard block
TOPK = 16                 # index_topk: rows past 19 tokens attend to their top 4 pools and their tail
CONFIG = {
    "model_type": "glm5_next",
    "quantization_config": {"quant_method": "exl3", "bits": 4, "codebook": "mcg", "head_bits": 16},
    "text_config": {
        "hidden_size": D, "num_hidden_layers": 2, "vocab_size": V, "rms_norm_eps": 1e-5,
        "num_attention_heads": 2, "q_lora_rank": 128, "kv_lora_rank": 128, "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0, "v_head_dim": 256,
        "linear_attn_config": {"num_heads": 2, "head_dim": 128, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
        "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": MOE, "n_shared_experts": 1,
        "intermediate_size": 256, "routed_scaling_factor": 2.5, "norm_topk_prob": True, "hc_mult": S,
        "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "index_n_heads": 8, "index_head_dim": 128, "index_topk": TOPK,
        "index_kpool": 4, "swiglu_limit": 10.0, "layer_types": ["linear_attention", "full_attention"],
        "mlp_layer_types": ["dense", "sparse"], "eos_token_id": [1000], "num_nextn_predict_layers": 1,
    },
}


def write_checkpoint(path) -> None:
    """The model as GLM-5.3-Flash's EXL3 layout: routed experts as trellis tiles with their scales, the rest BF16."""

    from tensorfold.families.glm5_next.cuda import split

    rng = np.random.default_rng(3)
    tensors: list[tuple[str, str, list[int], np.ndarray]] = []

    def bf16(name: str, shape: list[int], scale: float = 0.05, offset: float = 0.0) -> None:
        x = torch.tensor(rng.standard_normal(shape) * scale + offset, dtype=torch.float32).to(torch.bfloat16)
        tensors.append((name, "BF16", shape, x.view(torch.uint16).numpy().view(np.uint8).reshape(-1)))

    def f32(name: str, shape: list[int], scale: float = 0.1, offset: float = 0.0) -> None:
        x = (rng.standard_normal(shape) * scale + offset).astype(np.float32)
        tensors.append((name, "F32", shape, x.view(np.uint8).reshape(-1)))

    def lin(name: str, n: int, k: int, scale: float = 0.046) -> None:
        bf16(name + ".weight", [n, k], scale)

    def trellis(name: str, n: int, k: int) -> None:
        t = rng.integers(-2**15, 2**15, size=(k // 16, n // 16, 64)).astype(np.int16)
        tensors.append((name + ".trellis", "I16", [k // 16, n // 16, 64], t.view(np.uint8).reshape(-1)))
        for part, size, sc in (("suh", k, 1.0), ("svh", n, 0.05)):     # expert outputs as large as the rest's
            v = (rng.standard_normal(size) * sc).astype(np.float16)
            tensors.append((name + "." + part, "F16", [size], v.view(np.uint8).reshape(-1)))
        tensors.append((name + ".mcg", "I32", [1], np.array([0xCBAC1FED], dtype=np.uint32).view(np.uint8)))

    def dsa(p: str) -> None:
        lin(p + "self_attn.q_a_proj", 128, D)
        lin(p + "self_attn.kv_a_proj_with_mqa", 128, D)
        bf16(p + "self_attn.q_a_layernorm.weight", [128], 0.05, 1.0)
        bf16(p + "self_attn.kv_a_layernorm.weight", [128], 0.05, 1.0)
        lin(p + "self_attn.q_b_proj", 2 * 256, 128)
        lin(p + "self_attn.kv_b_proj", 2 * 512, 128)
        lin(p + "self_attn.o_proj", D, 2 * 256)
        lin(p + "self_attn.indexer.wk", 128, D)
        lin(p + "self_attn.indexer.weights_proj", 8, D)              # 8 heads: few pools score exactly 0
        lin(p + "self_attn.indexer.wq_b", 8 * 128, 128)
        bf16(p + "self_attn.indexer.k_norm.weight", [128], 0.05, 1.0)
        bf16(p + "self_attn.indexer.k_norm.bias", [128])
        bf16(p + "self_attn.indexer.index_kpool_compress_gate", [128, D])
        bf16(p + "self_attn.indexer.index_kpool_compress_ape", [4, 128])

    def moe(p: str) -> None:
        bf16(p + "mlp.gate.weight", [8, D])
        f32(p + "mlp.gate.e_score_correction_bias", [8], 0.01)
        for e in range(8):
            trellis(p + f"mlp.experts.{e}.gate_proj", MOE, D)
            trellis(p + f"mlp.experts.{e}.up_proj", MOE, D)
            trellis(p + f"mlp.experts.{e}.down_proj", D, MOE)
        lin(p + "mlp.shared_experts.gate_proj", MOE, D)
        lin(p + "mlp.shared_experts.up_proj", MOE, D)
        lin(p + "mlp.shared_experts.down_proj", D, MOE)

    L = "model.language_model."
    lin(L + "embed_tokens", V, D, 0.092)
    bf16(L + "norm.weight", [D], 0.05, 1.0)
    lin("lm_head", V, D)
    for i in (0, 1):
        p = f"{L}layers.{i}."
        bf16(p + "input_layernorm.weight", [D], 0.05, 1.0)
        bf16(p + "post_attention_layernorm.weight", [D], 0.05, 1.0)
        for site in ("attn", "ffn"):
            bf16(p + f"hc_{site}_fn", [24, S * D], 0.05)
            f32(p + f"hc_{site}_base", [24])
            f32(p + f"hc_{site}_scale", [3], 0.1, 1.0)
    p = L + "layers.0.self_attn."
    for x in "qkv":
        lin(p + f"{x}_proj", 256, D)
        bf16(p + f"{x}_conv1d.weight", [256, 1, 4], 0.3)
    lin(p + "f_a_proj", 128, D)
    lin(p + "g_a_proj", 128, D)
    lin(p + "b_proj", 2, D)
    lin(p + "f_b_proj", 256, 128)
    lin(p + "g_b_proj", 256, 128)
    f32(p + "A_log", [2], 0.5)
    f32(p + "dt_bias", [256], 0.5)
    bf16(p + "o_norm.weight", [128], 0.05, 1.0)
    lin(p + "o_proj", D, 256)
    lin(L + "layers.0.mlp.gate_proj", 256, D)
    lin(L + "layers.0.mlp.up_proj", 256, D)
    lin(L + "layers.0.mlp.down_proj", D, 256)
    dsa(L + "layers.1.")
    moe(L + "layers.1.")
    m = L + "layers.2."
    bf16(m + "enorm.weight", [D], 0.05, 1.0)
    bf16(m + "hnorm.weight", [D], 0.05, 1.0)
    lin(m + "eh_proj", D, 2 * D)
    bf16(m + "shared_head.norm.weight", [D], 0.05, 1.0)
    bf16(m + "input_layernorm.weight", [D], 0.05, 1.0)
    bf16(m + "post_attention_layernorm.weight", [D], 0.05, 1.0)
    dsa(m)
    moe(m)
    path.mkdir(parents=True, exist_ok=True)
    split.write(str(path / "model-00001-of-00001.safetensors"), tensors, {"format": "pt"})
    (path / "config.json").write_text(json.dumps(CONFIG))


# -- Triton's interpreter -------------------------------------------------------------------------------------------
def fix_interpreter() -> None:
    """Triton 3.7's interpreter multiplies bf16 tl.dot operands as their raw bits and truncates fp32 -> bf16; widen
    bf16 operands to fp32 (exact, as the tensor cores multiply them) and round to nearest even, as the GPU converts."""

    import triton.language as tl
    from triton.runtime import interpreter as itp

    builder = itp.InterpreterBuilder
    if getattr(builder, "_bf16_fixed", False):
        return
    dot, cast = builder.create_dot, builder.cast_impl

    def wide(x):
        if getattr(x.dtype, "scalar", x.dtype) == tl.bfloat16:
            return itp.TensorHandle((x.data.astype(np.uint32) << 16).view(np.float32), tl.float32)
        return x

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        return dot(self, wide(a), wide(b), d, input_precision, max_num_imprecise_acc)

    def cast_impl(self, src, dst_type):
        if src.dtype.scalar == tl.float32 and dst_type.scalar == tl.bfloat16:
            data = np.ascontiguousarray(src.data, dtype=np.float32)
            bits = data.view(np.uint32).astype(np.uint64)
            rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            rounded = np.where(np.isnan(data), np.uint16(0x7FC0), rounded)
            return itp.TensorHandle(rounded.reshape(np.shape(src.data)), tl.bfloat16)
        return cast(self, src, dst_type)

    builder.create_dot, builder.cast_impl = create_dot, cast_impl
    builder.create_fp_trunc = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    builder._bf16_fixed = True


# -- stand-ins for the CUDA extensions ------------------------------------------------------------------------------
def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


class KDAExt:
    """kda.cu's chain and replay in torch, with its fp32 sums and bf16 roundings, rows one after another."""

    @staticmethod
    def _update(s, k, g, v, beta):
        """One delta-rule step of s [H, DV, DK]: decay along the key channel, read with k, correct toward v."""
        s.mul_(g[:, None, :])
        kv = (s * k[:, None, :]).sum(-1)
        s.add_(k[:, None, :] * ((v - kv) * beta[:, None])[:, :, None])

    def chain(self, p, p_stride, b_off, a, a_stride, g, g_stride, conv_state, conv_w, state_in, a_log, dt_bias,
              norm_w, eps, lower, rows, out, state_out, k_save, v_save, g_save, b_save, *_wide):
        H, dk = a_log.numel(), 128
        C = 3 * H * dk
        s = state_in.float().clone()
        window = torch.cat([conv_state.float(), p[:rows, :C].float()])
        rate = torch.exp(a_log.float())[:, None]
        for r in range(rows):
            acc = (window[r:r + 4] * conv_w.float().T).sum(0)
            act = _bf(acc * torch.sigmoid(acc)).view(3, H, dk)
            q, k, v = act[0], act[1], act[2]
            q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * dk ** -0.5
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            gate = torch.exp(lower * torch.sigmoid(rate * (a[r].float().view(H, dk) + dt_bias.view(H, dk))))
            beta = _bf(torch.sigmoid(p[r, b_off:b_off + H].float()))
            self._update(s, k, gate, v, beta)
            y = _bf((s * q[:, None, :]).sum(-1))
            if k_save is not None:
                k_save[r], v_save[r], g_save[r], b_save[r] = k, v.to(torch.bfloat16), gate, beta
            yn = y / torch.sqrt((y * y).mean(-1, keepdim=True) + eps)
            o = norm_w.float() * yn * torch.sigmoid(g[r].float().view(H, dk))
            out[r] = o.reshape(-1).to(torch.bfloat16)
        if state_out is not None:
            state_out.copy_(s)

    chain_wide = chain

    def replay(self, state_in, k, v, g, b, rows, state_out):
        s = state_in.float().clone()
        for r in range(rows):
            self._update(s, k[r], g[r], v[r].float(), b[r])
        state_out.copy_(s)

    def replay_layers(self, state_in, _stride, k, v, g, b, _kv, _b, layers, _heads, rows, state_out):
        for i in range(layers):
            self.replay(state_in[i], k[i], v[i], g[i], b[i], rows, state_out[i])


def exl3_prepare(gate, up, down, codebook, device="cpu"):
    """A rank's routed experts as fp32 matrices (x @ W), from upstream's float64 EXL3 reference decoder."""

    from tensorfold.families.glm5_next.cuda import exl3

    def mats(triples):
        return torch.stack([exl3.dequantize(t, su, sv).float() for t, su, sv in triples])

    wg, wu, wd = mats(gate), mats(up), mats(down)
    return SimpleNamespace(wg=wg, wu=wu, wd=wd, count=wg.shape[0], dims=wg.shape[1], width=wg.shape[2])


def exl3_routed(x, pick, wts, ex, s, out, R, limit=math.inf, act_mode=0, group=True):
    """Each routed pick's down projection into s.y (fp32), SwiGLU's inputs and output rounded to bf16."""

    slots = pick.shape[1]
    xf = x[:R].float()
    for r in range(R):
        for j in range(slots):
            e = int(pick[r, j])
            if e >= ex.count:
                continue
            gt, up = _bf(xf[r] @ ex.wg[e]), _bf(xf[r] @ ex.wu[e])
            gt, up = gt.clamp(max=limit), up.clamp(-limit, limit)
            s.y[r * slots + j] = _bf(gt * torch.sigmoid(gt) * up) @ ex.wd[e]
    return s.y[:R * slots]


class _Stream:
    def record_event(self):
        return SimpleNamespace(synchronize=lambda: None)


def install_cpu_path() -> None:
    """In a rank started with TRITON_INTERPRET=1, before the CUDA path is imported: the interpreter's fixes, the
    stand-ins, a no-op stream."""

    fix_interpreter()
    torch.cuda.current_stream = lambda *a, **k: _Stream()
    from tensorfold.cuda.exl3 import experts as generic
    from tensorfold.families.glm5_next.cuda import kda, sparse

    generic.prepare, generic.routed = exl3_prepare, exl3_routed
    kda._ext = lambda: KDAExt()
    sparse.TOPK_POOLS = TOPK // sparse.POOL       # the selection's 512 pools, as index_topk 16 (not 2048) gives


# -- two ranks ------------------------------------------------------------------------------------------------------
class Gloo:
    """forward.gather's all-gather over gloo: every rank's fp32 partial, in rank order."""

    def all_gather(self, src: torch.Tensor, out: torch.Tensor) -> None:
        import torch.distributed as dist

        dist.all_gather(list(out.view(dist.get_world_size(), -1).unbind(0)), src.contiguous())


def _rank_main(rank: int, port: int, folder: str, job, queue) -> None:
    try:
        import torch.distributed as dist

        install_cpu_path()
        torch.set_num_threads(2)
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
        value = io.BytesIO()
        torch.save(job(Rank(folder, rank)), value)      # bytes: the tensors outlive this process
        queue.put((rank, value.getvalue(), None))
        dist.barrier()
        dist.destroy_process_group()
    except BaseException:  # noqa: BLE001 - the parent reports it
        queue.put((rank, None, traceback.format_exc()))


def run_ranks(folder, job, timeout: float = 1800.0) -> list:
    """``job(rank)`` on ranks 0 and 1 (two spawned processes, gloo): each rank's return value, in rank order."""

    import socket

    import torch.multiprocessing as mp

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_rank_main, args=(r, port, str(folder), job, queue)) for r in (0, 1)]
    before = os.environ.get("TRITON_INTERPRET")
    os.environ["TRITON_INTERPRET"] = "1"            # set before the ranks first import Triton
    try:
        for p in procs:
            p.start()
    finally:
        if before is None:
            del os.environ["TRITON_INTERPRET"]
        else:
            os.environ["TRITON_INTERPRET"] = before
    got = {}
    try:
        for _ in procs:
            rank, value, error = queue.get(timeout=timeout)
            if error:
                raise RuntimeError(f"rank {rank} failed:\n{error}")
            got[rank] = torch.load(io.BytesIO(value), weights_only=False)
    finally:
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.kill()
    return [got[0], got[1]]


class Rank:
    """One rank's weights, decode and prompt buffers, and one sequence's state, on the CPU."""

    def __init__(self, folder: str, rank: int, capacity: int = 64) -> None:
        from tensorfold.families.glm5_next.cuda import forward as F
        from tensorfold.families.glm5_next.cuda.weights import load

        self.F, self.rank, self.capacity = F, rank, capacity
        self.w = load(folder, rank=rank, device="cpu", mtp=True)
        self.w.comm = Gloo()
        self.w.meta["long_context"] = True
        self.w.draft_head = None                  # the MTP head's logits from the BF16 head (no 4-bit CUDA matmul)
        self.buf = F.Buffers(self.w, 8, capacity)
        self.pbuf = F.Buffers(self.w, 64, capacity, prefill=True)
        self.st = F.State(self.w, capacity, 8)

    def fresh(self) -> None:
        self.st = self.F.State(self.w, self.capacity, 8)

    def _run(self, b, tokens: list[int]):
        st, R = self.st, len(tokens)
        self.F.check_room(self.w, st, R)
        b.ids[:R] = torch.tensor(tokens, dtype=torch.int32)
        return self.F.compute(self.w, st, b, R, nch=self.F.chunks_for(st, R), host_pos=st.pos)

    def prompt(self, tokens: list[int]) -> torch.Tensor:
        """A prompt chunk, committed: its last row's logits (this rank's vocabulary half), fp32."""
        out = self._run(self.pbuf, tokens).float().clone()
        self.F.commit(self.w, self.st, self.pbuf, len(tokens), len(tokens))
        return out[0]

    def window(self, tokens: list[int], keep: int | None = None) -> torch.Tensor:
        """A decode window: every row's logits; its first ``keep`` rows committed (all by default)."""
        out = self._run(self.buf, tokens).float().clone()
        self.F.commit(self.w, self.st, self.buf, len(tokens), keep or len(tokens))
        return out

    def state(self) -> dict:
        """What the committed rows leave: the KDA state and conv window, the caches below the position."""
        st, n = self.st, self.st.pos
        out = {"rec": st.rec[st.parity].clone(), "conv": st.conv.clone(), "pos": n}
        out.update({f"kc{i}": kc[:n].clone() for i, kc in enumerate(st.kc)})
        for i, (ik, ig, pk) in enumerate(st.index or []):
            out[f"ik{i}"], out[f"ig{i}"], out[f"pk{i}"] = ik[:n].clone(), ig[:n].clone(), pk[:n // 4].clone()
        return out
