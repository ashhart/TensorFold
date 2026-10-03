#!/usr/bin/env python3
"""GLM-5.3-Flash Q8_0 GGUF (llama.cpp `glm5next`) -> an MLX checkpoint TensorFold's Mac engine reads, without loss.

A Q8_0 block is 32 int8 values q with one fp16 scale d, value d*q. MLX's affine 8-bit format in groups of 32 holds
it exactly: q_u = q + 128 (a uint8), scale = d, bias = -128 d, both fp16, so scale*q_u + bias == d*q for every
element. F16 / F32 tensors (norms, routers, hyper-connection mixes, the indexer and KDA low-rank projections that
llama.cpp keeps unquantised) are written as bf16 when every value survives the cast, else float32. Other layouts:
attn_k_b / attn_v_b become the absorbed embed_q [H, rank, nope] / unembed_out [H, v, rank] (same quantisation
axis, no re-blocking); ssm_a (= -exp(A_log)) becomes `A` = -ssm_a, which the loader reads instead of A_log.
The GGUF has no MTP layer: the result decodes without MTP drafts.

    python tools/glm5_q8_0_gguf_to_mlx.py --gguf 'GLM-5.3-Flash-Q8_0-*.gguf' --config zai-org/config.json \\
        --tokenizer-dir zai-org/ --out GLM-5.3-Flash-MLX-q8_0 [--verify]

--config and --tokenizer-dir come from the original zai-org/GLM-5.3-Flash repository (config.json, tokenizer
files). Numpy only; the tensors are streamed from memory-mapped shards (a few GiB of RAM).
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
    def __init__(self, out: Path) -> None:
        self.out, self.pending, self.size, self.n, self.index, self.sums = out, [], 0, 0, {}, {}

    def add(self, name: str, dtype: str, shape: tuple, data: bytes) -> None:
        if self.size and self.size + len(data) > SHARD_BYTES:
            self.flush()
        self.pending.append((name, dtype, tuple(int(s) for s in shape), data))
        self.size += len(data)

    def flush(self) -> None:
        if not self.pending:
            return
        self.n += 1
        fname = f"model-{self.n:05d}.safetensors"
        header, off = {}, 0
        for name, dtype, shape, data in self.pending:
            header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(data)]}
            off += len(data)
        header["__metadata__"] = {"format": "mlx"}
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * ((-len(hb)) % 8)
        h = hashlib.sha256()
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    ap.add_argument("--gguf", required=True, help="glob of the Q8_0 GGUF shards")
    ap.add_argument("--config", required=True, help="the original zai-org/GLM-5.3-Flash config.json")
    ap.add_argument("--tokenizer-dir", required=True, help="folder with tokenizer.json, tokenizer_config.json, ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true", help="afterwards check every tensor against the GGUF (needs MLX)")
    a = ap.parse_args()
    g = GGUF(a.gguf)
    if g.kv.get("general.architecture") != "glm5next":
        raise SystemExit(f"architecture {g.kv.get('general.architecture')!r}: this converter reads glm5next GGUFs")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    w = convert(g, out)
    total = sum((out / f).stat().st_size for f in w.sums)
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(w.index.items()))}, indent=1))
    cfg = json.loads(Path(a.config).read_text())
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
    return verify(g, out) if a.verify else 0


if __name__ == "__main__":
    sys.exit(main())
