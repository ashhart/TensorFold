#!/usr/bin/env python3
"""GLM-5.3-Flash Q8_0 GGUF (llama.cpp `glm5next`) -> an MLX checkpoint TensorFold's Mac engine reads, without loss.

A Q8_0 block is 32 int8 values q with one fp16 scale d, value d*q. MLX's affine 8-bit format in groups of 32 holds
it exactly: q_u = q + 128 (a uint8), scale = d, bias = -128 d, both fp16, so scale*q_u + bias == d*q for every
element. F16 / F32 tensors (norms, routers, hyper-connection mixes, the indexer and KDA low-rank projections that
llama.cpp keeps unquantised) are written as bf16 when every value survives the cast, else float32. Other layouts:
attn_k_b / attn_v_b become the absorbed embed_q [H, rank, nope] / unembed_out [H, v, rank] (same quantisation
axis, no re-blocking); ssm_a (= -exp(A_log)) becomes `A` = -ssm_a, which the loader reads instead of A_log.
The GGUF has no MTP layer; --mtp-from adds the original checkpoint's as the draft head (re-encoded, so not lossless:
see head_tensors), and --mtp-only adds it to an earlier conversion, which is otherwise left as it is.

    python tools/glm5_q8_0_gguf_to_mlx.py --gguf 'GLM-5.3-Flash-Q8_0-*.gguf' --config zai-org/config.json \\
        --tokenizer-dir zai-org/ --out GLM-5.3-Flash-MLX-q8_0 [--mtp-from zai-org/] [--verify]
    python tools/glm5_q8_0_gguf_to_mlx.py --mtp-only --mtp-from zai-org/ --out GLM-5.3-Flash-MLX-q8_0 [--verify]

--config, --tokenizer-dir and --mtp-from come from the original zai-org/GLM-5.3-Flash repository (config.json,
tokenizer files, the safetensors shards with their index). Numpy only; the tensors are streamed from memory-mapped
shards (a few GiB of RAM; about 15 GB while the MTP head is written).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import mmap
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

F32, F16, Q8_0 = 0, 1, 8
Q8_DTYPE = np.dtype([("d", "<f2"), ("q", "i1", (32,))])
P = "language_model.model."
SHARD_BYTES = 5 * 2**30

LAYER = {  # GGUF blk.N.<suffix> -> (name under layers.N., kind: q = Q8_0 linear, d = dense, neg = -x)
    "attn_norm.weight": ("input_layernorm.weight", "d"), "ffn_norm.weight": ("post_attention_layernorm.weight", "d"),
    "hc_attn_fn.weight": ("hc_attn_fn", "d"), "hc_attn_base.weight": ("hc_attn_base", "d"),
    "hc_attn_scale.weight": ("hc_attn_scale", "d"), "hc_ffn_fn.weight": ("hc_ffn_fn", "d"),
    "hc_ffn_base.weight": ("hc_ffn_base", "d"), "hc_ffn_scale.weight": ("hc_ffn_scale", "d"),
    "attn_q.weight": ("self_attn.q_proj", "q"), "attn_k.weight": ("self_attn.k_proj", "q"),
    "attn_v.weight": ("self_attn.v_proj", "q"), "attn_output.weight": ("self_attn.o_proj", "q"),
    "ssm_conv1d_q.weight": ("self_attn.q_conv1d.weight", "d"), "ssm_conv1d_k.weight": ("self_attn.k_conv1d.weight", "d"),
    "ssm_conv1d_v.weight": ("self_attn.v_conv1d.weight", "d"), "ssm_f_a.weight": ("self_attn.f_a_proj.weight", "d"),
    "ssm_f_b.weight": ("self_attn.f_b_proj.weight", "d"), "ssm_g_a.weight": ("self_attn.g_a_proj.weight", "d"),
    "ssm_g_b.weight": ("self_attn.g_b_proj.weight", "d"), "ssm_beta.weight": ("self_attn.b_proj.weight", "d"),
    "ssm_a": ("self_attn.A", "neg"), "ssm_dt.bias": ("self_attn.dt_bias", "d"),
    "ssm_norm.weight": ("self_attn.o_norm.weight", "d"),
    "attn_q_a.weight": ("self_attn.q_a_proj", "q"), "attn_q_b.weight": ("self_attn.q_b_proj", "q"),
    "attn_kv_a_mqa.weight": ("self_attn.kv_a_proj_with_mqa", "q"),
    "attn_q_a_norm.weight": ("self_attn.q_a_layernorm.weight", "d"),
    "attn_kv_a_norm.weight": ("self_attn.kv_a_layernorm.weight", "d"),
    "attn_k_b.weight": ("self_attn.embed_q", "q"), "attn_v_b.weight": ("self_attn.unembed_out", "q"),
    "indexer.attn_k.weight": ("self_attn.indexer.wk.weight", "d"),
    "indexer.attn_q_b.weight": ("self_attn.indexer.wq_b.weight", "d"),
    "indexer.proj.weight": ("self_attn.indexer.weights_proj.weight", "d"),
    "indexer.k_norm.weight": ("self_attn.indexer.k_norm.weight", "d"),
    "indexer.k_norm.bias": ("self_attn.indexer.k_norm.bias", "d"),
    "indexer_compressor_ape.weight": ("self_attn.indexer.index_kpool_compress_ape", "d"),
    "indexer_compressor_gate.weight": ("self_attn.indexer.index_kpool_compress_gate", "d"),
    "ffn_gate.weight": ("mlp.gate_proj", "q"), "ffn_up.weight": ("mlp.up_proj", "q"),
    "ffn_down.weight": ("mlp.down_proj", "q"), "ffn_gate_inp.weight": ("mlp.gate.weight", "d"),
    "exp_probs_b.bias": ("mlp.gate.e_score_correction_bias", "d"),
    "ffn_gate_shexp.weight": ("mlp.shared_experts.gate_proj", "q"), "ffn_up_shexp.weight": ("mlp.shared_experts.up_proj", "q"),
    "ffn_down_shexp.weight": ("mlp.shared_experts.down_proj", "q"),
    "ffn_gate_exps.weight": ("mlp.switch_mlp.gate_proj", "q"), "ffn_up_exps.weight": ("mlp.switch_mlp.up_proj", "q"),
    "ffn_down_exps.weight": ("mlp.switch_mlp.down_proj", "q"),
}
GLOBAL = {"token_embd.weight": (P + "embed_tokens", "q"), "output.weight": ("lm_head", "q"),
          "output_norm.weight": (P + "norm.weight", "d")}


# -- GGUF ---------------------------------------------------------------------------------------------------------

def _rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def _rs(f):
    return f.read(_rd(f, "Q")[0]).decode("utf-8", "replace")


def _rv(f, t):
    if t == 8:
        return _rs(f)
    if t == 9:
        et, n = _rd(f, "IQ")
        return [_rv(f, et) for _ in range(n)]
    return _rd(f, {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t])[0]


class GGUF:
    def __init__(self, pattern: str) -> None:
        self.kv, self.tensors, self._maps = {}, {}, {}
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise SystemExit(f"no GGUF files match {pattern!r}")
        for path in paths:
            with open(path, "rb") as f:
                if f.read(4) != b"GGUF":
                    raise SystemExit(f"{path}: not a GGUF file")
                _, nt, nkv = _rd(f, "IQQ")
                kv = {}
                for _ in range(nkv):
                    k = _rs(f)
                    kv[k] = _rv(f, _rd(f, "I")[0])
                entries = []
                for _ in range(nt):
                    name = _rs(f)
                    dims = _rd(f, "Q" * _rd(f, "I")[0])
                    t, off = _rd(f, "IQ")
                    entries.append((name, dims, t, off))
                start = f.tell()
                start += (-start) % int(kv.get("general.alignment", 32))
            for k, v in kv.items():
                self.kv.setdefault(k, v)
            for name, dims, t, off in entries:
                self.tensors[name] = (tuple(reversed(dims)), t, path, start + off)

    def array(self, name: str) -> np.ndarray:
        """F32/F16 as float arrays; Q8_0 as blocks [..., n/32] with fields d, q. Zero-copy, numpy (row-major) shape."""
        shape, t, path, off = self.tensors[name]
        if path not in self._maps:
            with open(path, "rb") as f:
                self._maps[path] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        m, n = self._maps[path], int(np.prod(shape))
        if t in (F32, F16):
            return np.frombuffer(m, dtype="<f4" if t == F32 else "<f2", count=n, offset=off).reshape(shape)
        if t == Q8_0:
            return np.frombuffer(m, dtype=Q8_DTYPE, count=n // 32, offset=off).reshape(shape[:-1] + (shape[-1] // 32,))
        raise SystemExit(f"{name}: ggml type {t} is not F32, F16 or Q8_0")


def q8_values(blocks: np.ndarray) -> np.ndarray:
    """ggml's dequantisation, d*q in float32 (exact)."""
    v = blocks["d"].astype(np.float32)[..., None] * blocks["q"].astype(np.float32)
    return v.reshape(blocks.shape[:-1] + (blocks.shape[-1] * 32,))


def q8_to_affine8(blocks: np.ndarray):
    """(weight uint32 [..., n/4], scales fp16 [..., n/32], biases fp16): q_u = q + 128, scale = d, bias = -128 d."""
    qu = blocks["q"].view(np.uint8) ^ np.uint8(0x80)
    packed = np.ascontiguousarray(qu).reshape(blocks.shape[:-1] + (blocks.shape[-1] * 32,)).view("<u4")
    d = blocks["d"]
    biases = (d.astype(np.float32) * np.float32(-128.0)).astype(np.float16)
    if not np.array_equal(biases.astype(np.float32), d.astype(np.float32) * np.float32(-128.0)):
        raise SystemExit("a Q8_0 scale too large for an exact fp16 bias (|d| > 511)")
    return packed, np.ascontiguousarray(d), np.ascontiguousarray(biases)


def dense(a: np.ndarray):
    """(bytes, safetensors dtype, shape): bf16 when every value survives, else float32 — exact either way."""
    f = np.ascontiguousarray(a, dtype=np.float32)
    bits = f.view(np.uint32)
    if not np.any(bits & np.uint32(0xFFFF)):
        return (bits >> np.uint32(16)).astype(np.uint16).tobytes(), "BF16", f.shape
    return f.tobytes(), "F32", f.shape


# -- safetensors ----------------------------------------------------------------------------------------------------

class Writer:
    def __init__(self, out: Path, stem: str = "model") -> None:
        self.out, self.stem, self.pending, self.size, self.n, self.index, self.sums = out, stem, [], 0, 0, {}, {}

    def add(self, name: str, dtype: str, shape: tuple, data: bytes) -> None:
        if self.size and self.size + len(data) > SHARD_BYTES:
            self.flush()
        self.pending.append((name, dtype, tuple(int(s) for s in shape), data))
        self.size += len(data)

    def flush(self) -> None:
        if not self.pending:
            return
        self.n += 1
        fname = f"{self.stem}-{self.n:05d}.safetensors"
        header, off = {}, 0
        for name, dtype, shape, data in self.pending:
            header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(data)]}
            off += len(data)
        header["__metadata__"] = {"format": "mlx"}
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * ((-len(hb)) % 8)
        h = hashlib.sha256()
        if (self.out / (fname + ".part")).is_symlink():
            raise SystemExit(f"{self.out / (fname + '.part')} is a symbolic link: remove it first")
        with open(self.out / (fname + ".part"), "wb") as f:
            for chunk in [len(hb).to_bytes(8, "little"), hb] + [p[3] for p in self.pending]:
                f.write(chunk)
                h.update(chunk)
        os.replace(self.out / (fname + ".part"), self.out / fname)
        self.sums[fname] = h.hexdigest()
        self.index.update({p[0]: fname for p in self.pending})
        print(f"wrote {fname} ({off / 2**30:.2f} GiB, {len(self.pending)} tensors)", flush=True)
        self.pending, self.size = [], 0


def plan(g: GGUF) -> list[tuple[str, str, str]]:
    out = []
    for gname in g.tensors:
        m = re.match(r"blk\.(\d+)\.(.*)$", gname)
        if m:
            if m.group(2) not in LAYER:
                raise SystemExit(f"unmapped tensor {gname}")
            cname, kind = LAYER[m.group(2)]
            out.append((gname, f"{P}layers.{m.group(1)}.{cname}", kind))
        elif gname in GLOBAL:
            out.append((gname, *GLOBAL[gname]))
        else:
            raise SystemExit(f"unmapped tensor {gname}")
    return sorted(out, key=lambda t: (int(re.match(r"blk\.(\d+)", t[0]).group(1)) if t[0].startswith("blk.") else 1e9, t[0]))


def convert(g: GGUF, out: Path) -> Writer:
    w = Writer(out)
    for gname, cname, kind in plan(g):
        a = g.array(gname)
        if kind == "q":
            packed, scales, biases = q8_to_affine8(a)
            w.add(cname + ".weight", "U32", packed.shape, packed.tobytes())
            w.add(cname + ".scales", "F16", scales.shape, scales.tobytes())
            w.add(cname + ".biases", "F16", biases.shape, biases.tobytes())
        elif kind == "neg":
            neg = -np.asarray(a, dtype=np.float32)
            w.add(cname, "F32", neg.shape, neg.tobytes())
        else:
            data, dt, shape = dense(a)
            w.add(cname, dt, shape, data)
    w.flush()
    return w


def verify(g: GGUF, out: Path, rows: int = 64) -> int:
    """Every tensor against the GGUF: MLX's dequantisation == d*q on sampled rows, dense values equal."""
    import mlx.core as mx

    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    loaded: dict = {}

    def get(key):
        shard = index[key]
        if shard not in loaded:
            loaded.clear()
            loaded[shard] = mx.load(str(out / shard))
        return loaded[shard][key]

    rng, bad = np.random.default_rng(0), 0
    for gname, cname, kind in plan(g):
        src = g.array(gname)
        if kind == "q":
            flat = int(np.prod(src.shape[:-1]))
            pick = np.sort(np.arange(flat) if flat <= 1024 else rng.choice(flat, rows, replace=False))
            sel = mx.array(pick)
            got = mx.dequantize(get(cname + ".weight").reshape(flat, -1)[sel],
                                get(cname + ".scales").reshape(flat, -1)[sel].astype(mx.float32),
                                get(cname + ".biases").reshape(flat, -1)[sel].astype(mx.float32), group_size=32, bits=8)
            ok = np.array_equal(np.array(got), q8_values(src.reshape(flat, src.shape[-1])[pick]))
        else:
            want = -np.asarray(src, np.float32) if kind == "neg" else np.asarray(src, np.float32)
            ok = np.array_equal(np.array(get(cname).astype(mx.float32)), want)
        if not ok:
            bad += 1
            print(f"MISMATCH {gname} -> {cname}", flush=True)
    print(f"verified {len(plan(g))} tensors, {bad} mismatches", flush=True)
    return bad


# -- the MTP layer, from the original FP8 checkpoint -------------------------------------------------------------------

# under layers.<num_hidden_layers>.: the linears the loader reads quantised, re-encoded 8-bit / 32; the rest as stored
MTP_Q = ("self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.kv_b_proj",
         "self_attn.o_proj", "mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj",
         "mlp.shared_experts.down_proj")
MTP_DENSE = ("eh_proj.weight", "enorm.weight", "hnorm.weight", "shared_head.norm.weight", "input_layernorm.weight",
             "post_attention_layernorm.weight", "mlp.gate.weight", "mlp.gate.e_score_correction_bias",
             "self_attn.q_a_layernorm.weight", "self_attn.kv_a_layernorm.weight", "self_attn.indexer.wq_b.weight",
             "self_attn.indexer.wk.weight", "self_attn.indexer.weights_proj.weight", "self_attn.indexer.k_norm.weight",
             "self_attn.indexer.k_norm.bias", "self_attn.indexer.index_kpool_compress_ape",
             "self_attn.indexer.index_kpool_compress_gate")
LAYER_PREFIX = r"(?:model\.language_model\.|language_model\.model\.|language_model\.|model\.)?layers\.%d\."
ST_TYPES = {"F8_E4M3": "u1", "BF16": "<u2", "F16": "<f2", "F32": "<f4"}


def e4m3_table() -> np.ndarray:
    """float32 of every FP8 e4m3fn byte: sign, 4 exponent bits (bias 7), 3 mantissa bits; 0x7f and 0xff are NaN."""
    b = np.arange(256)
    e, m = (b >> 3) & 15, (b & 7) / 8
    v = np.where(e == 0, m * 2.0 ** -6, (1 + m) * 2.0 ** (e - 7))
    v[(b & 0x7F) == 0x7F] = np.nan
    return np.where(b & 0x80, -v, v).astype(np.float32)


E4M3 = e4m3_table()


def widen(a: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return (a.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if dtype in ("F16", "F32"):
        return a.astype(np.float32)
    raise ValueError(f"{dtype} where BF16, F16 or F32 was expected")


def fp8_values(codes: np.ndarray, scale_inv: np.ndarray, block: int = 128) -> np.ndarray:
    """FP8 e4m3 weights [out, in] times the weight_scale_inv of their block (edge blocks partial), in float32."""
    out, ins = codes.shape
    if scale_inv.shape != (-(-out // block), -(-ins // block)):
        raise ValueError(f"block scales {scale_inv.shape} do not fit a {codes.shape} weight in {block}x{block} blocks")
    if np.any((codes & 0x7F) == 0x7F):
        raise ValueError("NaN codes")
    with np.errstate(over="ignore"):
        v = E4M3[codes] * np.repeat(np.repeat(np.asarray(scale_inv, np.float32), block, 0), block, 1)[:out, :ins]
    if not np.isfinite(v).all():
        raise ValueError("non-finite values after scaling")
    return v


def affine8(w: np.ndarray):
    """float32 [..., n] as MLX affine 8-bit in groups of 32, round to nearest: (uint32 [..., n/4], scales, biases).
    Each group's scale is (max - min) / 255 and its bias the min, both float32: every value is within scale/2."""
    if w.shape[-1] % 32:
        raise ValueError(f"{w.shape[-1]} inputs do not split into groups of 32")
    g = w.reshape(*w.shape[:-1], -1, 32)
    if not np.isfinite(g).all():
        raise ValueError("non-finite values")
    lo = g.min(-1)
    with np.errstate(over="ignore"):
        scales = (g.max(-1) - lo) / np.float32(255)
    if not np.isfinite(scales).all():
        raise ValueError("a group's range overflows float32")
    q = np.divide(g - lo[..., None], scales[..., None], out=np.zeros_like(g), where=scales[..., None] > 0)
    q = np.clip(np.rint(q), 0, 255).astype(np.uint8).reshape(w.shape)
    return q.view("<u4"), scales, lo


class Original:
    """The original checkpoint's tensors by name, found through its index and memory-mapped."""

    def __init__(self, folder: Path) -> None:
        if not (folder / "model.safetensors.index.json").is_file():
            raise SystemExit(f"{folder}: no model.safetensors.index.json")
        self.dir, self.where = folder, json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
        self._shards: dict = {}

    def raw(self, name: str) -> tuple[np.ndarray, str]:
        """(the stored values, FP8 as uint8 codes and bf16 as uint16; the safetensors dtype)."""
        if name not in self.where:
            raise SystemExit(f"{name}: not in {self.dir}'s index")
        shard = self.where[name]
        if shard not in self._shards:
            with open(self.dir / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._shards[shard] = (json.loads(f.read(n)), 8 + n, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        header, base, m = self._shards[shard]
        meta = header.get(name)
        if meta is None:
            raise SystemExit(f"{name}: not in {shard}, where the index puts it")
        if meta["dtype"] not in ST_TYPES:
            raise SystemExit(f"{name}: {meta['dtype']}, not F8_E4M3, BF16, F16 or F32")
        dt, shape, (lo, hi) = np.dtype(ST_TYPES[meta["dtype"]]), tuple(meta["shape"]), meta["data_offsets"]
        if hi - lo != int(np.prod(shape)) * dt.itemsize:
            raise SystemExit(f"{name}: {hi - lo} bytes for shape {shape}")
        return np.frombuffer(m, dtype=dt, count=int(np.prod(shape)), offset=base + lo).reshape(shape), meta["dtype"]

    def float32(self, name: str) -> np.ndarray:
        """A linear's weight in float32: FP8 times its block scales, or BF16 / F16 / F32 widened."""
        a, dt = self.raw(name + ".weight")
        try:
            if dt != "F8_E4M3":
                return widen(a, dt)
            return fp8_values(a, widen(*self.raw(name + ".weight_scale_inv")))
        except ValueError as exc:
            raise SystemExit(f"{name}: {exc}") from None


def head_plan(st: Original, cfg: dict) -> tuple[str, int, int]:
    """(the original's prefix of the MTP layer, its index, the routed experts), refusing a missing or unknown tensor."""
    text = cfg.get("text_config") or cfg
    n, experts = int(text["num_hidden_layers"]), int(text["n_routed_experts"])
    if int(text.get("num_nextn_predict_layers", 0)) < 1:
        raise SystemExit("config.json has num_nextn_predict_layers 0: the engine would not read an MTP layer")
    found = [m for m in (re.match(f"({LAYER_PREFIX % n})(.+)$", k) for k in st.where) if m]
    prefixes = sorted({m.group(1) for m in found})
    if len(prefixes) != 1:
        raise SystemExit(f"{st.dir}: the index holds layers.{n}.* (the MTP layer) under {len(prefixes)} prefixes")
    pre, shorts = prefixes[0], {m.group(2) for m in found}
    linears = list(MTP_Q) + [f"mlp.experts.{e}.{p}_proj" for e in range(experts) for p in ("gate", "up", "down")]
    missing = sorted(({f"{x}.weight" for x in linears} | set(MTP_DENSE)) - shorts)
    unknown = sorted(shorts - set(MTP_DENSE) - {f"{x}.weight{s}" for x in linears for s in ("", "_scale_inv")})
    if missing or unknown:
        raise SystemExit(f"{pre}*: " + "; ".join(f"{len(v)} {k}, {v[0]} first" for k, v in
                                                (("missing", missing), ("not mapped", unknown)) if v))
    return pre, n, experts


def head_tensors(src: Path, cfg: dict):
    """The original's MTP layer as (name, safetensors dtype, array): FP8 linears decoded (e4m3 times the
    weight_scale_inv of their 128x128 block, in float32) and, with the bf16 kv_b_proj, re-encoded 8-bit / 32 with
    float32 scales and biases; routed experts stacked as switch_mlp [n_experts, out, in]; eh_proj, the indexer, the
    router and the norms as stored. Drafts are verified against the model, so the re-encoding can change the
    acceptance rate only."""
    st = Original(src)
    pre, n, experts = head_plan(st, cfg)
    out = f"{P}layers.{n}."

    def encode(name):
        try:
            return affine8(st.float32(pre + name))
        except ValueError as exc:
            raise SystemExit(f"{pre}{name}: {exc}") from None

    def parts(name, arrays):
        return [(out + name + x, dt, a) for x, dt, a in zip((".weight", ".scales", ".biases"), ("U32", "F32", "F32"),
                                                           arrays)]

    for name in MTP_Q:
        yield from parts(name, encode(name))
    for proj in ("gate_proj", "up_proj", "down_proj"):
        stacks = None
        for e in range(experts):
            q = encode(f"mlp.experts.{e}.{proj}")
            stacks = stacks or [np.empty((experts, *a.shape), a.dtype) for a in q]
            if q[0].shape != stacks[0].shape[1:]:
                raise SystemExit(f"{pre}mlp.experts.{e}.{proj}: shape differs from expert 0's")
            for stack, a in zip(stacks, q):
                stack[e] = a
        yield from parts(f"mlp.switch_mlp.{proj}", stacks)
    for name in MTP_DENSE:
        a, dt = st.raw(pre + name)
        if dt == "F8_E4M3":
            raise SystemExit(f"{pre}{name}: FP8, where BF16, F16 or F32 was expected")
        yield out + name, dt, a


def write_index(out: Path, weight_map: dict, metadata: dict | None = None, *, swap: bool = True) -> int:
    """model.safetensors.index.json, through a .part file; ``swap``: False leaves the .part for the caller."""
    total = sum((out / f).stat().st_size for f in set(weight_map.values()))
    part = out / "model.safetensors.index.json.part"
    part.write_text(json.dumps({"metadata": {**(metadata or {}), "total_size": total},
                                "weight_map": dict(sorted(weight_map.items()))}, indent=1))
    if swap:
        os.replace(part, out / "model.safetensors.index.json")
    return total


def add_mtp(src: Path, out: Path, replace: bool) -> dict:
    """The head into a finished conversion: shards of its own, then the new index and SHA256SUMS renamed into place.
    No existing shard is written; on any failure the old index and SHA256SUMS are put back and the head's files
    removed. ``replace``: drop a head the conversion has (it must sit in shards of its own)."""
    if not (out / "model.safetensors.index.json").is_file() or not (out / "config.json").is_file():
        raise SystemExit(f"{out}: not a finished conversion (no model.safetensors.index.json or config.json)")
    index_path, sums_path = out / "model.safetensors.index.json", out / "SHA256SUMS"
    index = json.loads(index_path.read_text())
    cfg = json.loads((out / "config.json").read_text())
    quant = cfg.get("quantization") or {}
    if (quant.get("bits"), quant.get("group_size"), quant.get("mode", "affine")) != (8, 32, "affine"):
        raise SystemExit(f"{out}: config.json states {quant}; the head is 8-bit in groups of 32, this tool's output")
    n = int((cfg.get("text_config") or cfg)["num_hidden_layers"])
    weights = index["weight_map"]
    old = {k for k in weights if re.match(LAYER_PREFIX % n, k) or re.match(r"^(?:.*\.)?mtp\.\d+\.", k)}
    if old and not replace:
        raise SystemExit(f"{out} already has an MTP layer ({min(old)}, ...); --replace-mtp replaces it")
    old_shards = {weights[k] for k in old}
    if any(weights[k] in old_shards for k in set(weights) - old):
        raise SystemExit(f"{out}: its MTP layer shares shards with other tensors; this tool leaves those as they are")
    stems = ["model-mtp"] + [f"model-mtp{i}" for i in range(2, 100)]
    w = Writer(out, next(s for s in stems if not any(f.startswith(s + "-") for f in set(weights.values()))))
    before = {p: p.read_bytes() for p in (index_path, sums_path) if p.is_file()}
    for p in (index_path, sums_path):
        if p.with_name(p.name + ".part").is_symlink():
            raise SystemExit(f"{p.with_name(p.name + '.part')} is a symbolic link: remove it first")

    def strays():
        return [out / f for f in os.listdir(out) if f.endswith(".part") and not (out / f).is_symlink() and (
            f.startswith(w.stem + "-") or f in ("model.safetensors.index.json.part", "SHA256SUMS.part"))]

    try:
        for name, dtype, a in head_tensors(src, cfg):
            w.add(name, dtype, a.shape, memoryview(a).cast("B"))         # no copy of an expert stack
        w.flush()
        kept = {k: v for k, v in weights.items() if k not in old}
        write_index(out, {**kept, **w.index}, {k: v for k, v in (index.get("metadata") or {}).items()
                                               if k != "total_size"}, swap=False)
        if sums_path in before:
            sums = {f: h for h, f in (line.split("  ", 1) for line in before[sums_path].decode().splitlines() if line)}
            sums = {k: v for k, v in sums.items() if k not in old_shards} | w.sums
            (out / "SHA256SUMS.part").write_text("".join(f"{v}  {k}\n" for k, v in sorted(sums.items())))
        else:
            print(f"{out} has no SHA256SUMS: none written for the head either", flush=True)
        os.replace(out / "model.safetensors.index.json.part", index_path)
        if sums_path in before:
            os.replace(out / "SHA256SUMS.part", sums_path)
    except BaseException:
        for path, data in before.items():
            if path.read_bytes() != data:
                path.with_name(path.name + ".part").write_bytes(data)
                os.replace(path.with_name(path.name + ".part"), path)
        for path in [*(out / f for f in w.sums), *strays()]:
            path.unlink(missing_ok=True)
        raise
    for path in [*(out / f for f in old_shards - set(w.sums)), *strays()]:
        path.unlink(missing_ok=True)
    print(f"added the MTP layer to {out}: {len(w.index)} tensors in {len(w.sums)} shards"
          + (f", {len(old)} old head tensors dropped" if old else ""), flush=True)
    return cfg


def verify_mtp(src: Path, out: Path, cfg: dict, experts: int = 4) -> int:
    """The head against the original: every 8-bit value within its group's scale/2 (rounding allowed for), on all
    rows and ``experts`` sampled routed experts; the tensors kept as stored equal."""
    import mlx.core as mx

    st = Original(src)
    pre, n, count = head_plan(st, cfg)
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    loaded: dict = {}

    def get(key):
        if index[key] not in loaded:
            loaded.clear()
            loaded[index[key]] = mx.load(str(out / index[key]))
        return loaded[index[key]][key]

    picks = np.random.default_rng(0).choice(count, min(count, experts), replace=False)
    checks = [(q, pre + q, None) for q in MTP_Q] + [(f"mlp.switch_mlp.{p}", f"{pre}mlp.experts.{e}.{p}", e)
                                                   for p in ("gate_proj", "up_proj", "down_proj") for e in picks]
    bad = 0
    for name, source, e in checks:
        q, s, b = (get(f"{P}layers.{n}.{name}.{part}") for part in ("weight", "scales", "biases"))
        if e is not None:
            q, s, b = q[int(e)], s[int(e)], b[int(e)]
        got = np.array(mx.dequantize(q, s, b, group_size=32, bits=8))
        s, b = np.array(s), np.array(b)
        bound = np.repeat(0.5 * s + 2.0 ** -20 * (np.abs(b) + 255 * s), 32, axis=-1)
        if not np.all(np.abs(got - st.float32(source)) <= bound):
            bad += 1
            print(f"OUT OF BOUND {source} -> {name}", flush=True)
    for name in MTP_DENSE:
        if not np.array_equal(np.array(get(f"{P}layers.{n}.{name}").astype(mx.float32)), widen(*st.raw(pre + name))):
            bad += 1
            print(f"MISMATCH {pre}{name}", flush=True)
    print(f"verified the MTP layer: {len(checks) + len(MTP_DENSE)} tensors, {bad} mismatches", flush=True)
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    ap.add_argument("--gguf", help="glob of the Q8_0 GGUF shards")
    ap.add_argument("--config", help="the original zai-org/GLM-5.3-Flash config.json")
    ap.add_argument("--tokenizer-dir", help="folder with tokenizer.json, tokenizer_config.json, ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true", help="afterwards check every tensor against the GGUF (needs MLX)")
    ap.add_argument("--mtp-from", help="the original zai-org/GLM-5.3-Flash folder (safetensors shards and their "
                                       "index): add its MTP layer as the draft head")
    ap.add_argument("--mtp-only", action="store_true",
                    help="add the head to the earlier conversion in --out: shards of its own and a new index")
    ap.add_argument("--replace-mtp", action="store_true", help="with --mtp-only, replace a head --out already has")
    a = ap.parse_args(argv)
    gguf_args = [f"--{k.replace('_', '-')}" for k in ("gguf", "config", "tokenizer_dir") if getattr(a, k)]
    if a.mtp_only:
        if not a.mtp_from:
            ap.error("--mtp-only needs --mtp-from")
        if gguf_args:
            ap.error(f"--mtp-only reads no GGUF: drop {', '.join(gguf_args)}")
        cfg = add_mtp(Path(a.mtp_from), Path(a.out), a.replace_mtp)
        return int(a.verify and verify_mtp(Path(a.mtp_from), Path(a.out), cfg) > 0)
    if len(gguf_args) < 3:
        ap.error("the following arguments are required: --gguf, --config, --tokenizer-dir")
    if a.replace_mtp:
        ap.error("--replace-mtp goes with --mtp-only")
    cfg = json.loads(Path(a.config).read_text())
    if a.mtp_from:                                   # the whole head decoded and encoded before the GGUF's hours
        for _ in head_tensors(Path(a.mtp_from), cfg):
            pass
    g = GGUF(a.gguf)
    if g.kv.get("general.architecture") != "glm5next":
        raise SystemExit(f"architecture {g.kv.get('general.architecture')!r}: this converter reads glm5next GGUFs")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    w = convert(g, out)
    total = write_index(out, w.index)
    for holder in (cfg, cfg.get("text_config") or {}):
        holder.pop("quantization_config", None)
    cfg["quantization"] = {"group_size": 32, "bits": 8, "mode": "affine"}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
              "special_tokens_map.json"):
        if (Path(a.tokenizer_dir) / f).exists():
            (out / f).write_bytes((Path(a.tokenizer_dir) / f).read_bytes())
    (out / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sorted(w.sums.items())))
    print(f"done: {len(w.sums)} shards, {total / 2**30:.1f} GiB in {out}", flush=True)
    if a.mtp_from:                                   # a finished conversion first: --mtp-only can redo a failed head
        add_mtp(Path(a.mtp_from), out, replace=False)
    if not a.verify:
        return 0
    return int(verify(g, out) + (verify_mtp(Path(a.mtp_from), out, cfg) if a.mtp_from else 0) > 0)


if __name__ == "__main__":
    sys.exit(main())
