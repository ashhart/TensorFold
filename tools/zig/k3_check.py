"""Compare tf-k3-check's outputs with k3_reference on the same weights (synthetic or the checkpoint) and row inputs."""
import argparse
import copy
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

import k3_reference as ref
from k3_synth import bf16, fnv1a, from_bf16_bits, mxfp4_decode, synthetic, values

SCHEDULE = [
    [(0, 5, True, 0), (1, 3, True, 0)],
    [(0, 4, False, 2), (1, 1, False, 1), (2, 6, True, 0)],
    [(2, 3, False, 3), (0, 3, False, 1), (1, 2, False, 0)],
    [(1, 16, False, 7), (0, 1, False, 1), (2, 40, True, 0)],
    [(0, 64, True, 0)],
    [(2, 16, False, 9), (0, 120, True, 0), (1, 120, True, 0)],
    [(0, 40, True, 0), (1, 72, True, 0), (2, 16, False, 4)],
    [(0, 16, False, 16), (1, 16, False, 3)],
]


def shape_of(c: ref.Config, name: str):
    """A tensor's shape from its checkpoint name, as zig/src/families/kimi_k3/weights.zig specifies it."""
    H, W, heads = c.hidden, c.kda_heads * c.kda_dim, c.mla_heads
    m = re.search(r"experts\.\d+\.(w\d)\.weight_(packed|scale)$", name)
    if m:
        rows, cols = (c.latent, c.moe_inter) if m.group(1) == "w2" else (c.moe_inter, c.latent)
        return (rows, cols // 2) if m.group(2) == "packed" else (rows, cols // 32)
    table = [
        (r"(input_layernorm|post_attention_layernorm|self_attention_res_norm|mlp_res_norm)\.weight$", (H,)),
        (r"res_proj\.weight$", (1, H)), (r"self_attn\.[qkvg]_proj\.weight$", None), (r"self_attn\.o_proj\.weight$", None),
        (r"f_a_proj\.weight$", (c.kda_dim, H)), (r"f_b_proj\.weight$", (W, c.kda_dim)),
        (r"self_attn\.b_proj\.weight$", (c.kda_heads, H)), (r"conv1d\.weight$", (W, 1, c.conv)), (r"A_log$", (128,)),
        (r"dt_bias$", (W,)), (r"o_norm\.weight$", (c.kda_dim,)), (r"q_a_proj\.weight$", (c.q_lora, H)),
        (r"q_a_layernorm\.weight$", (c.q_lora,)), (r"q_b_proj\.weight$", (heads * (c.nope + c.rope), c.q_lora)),
        (r"kv_a_proj_with_mqa\.weight$", (c.kv_lora + c.rope, H)), (r"kv_a_layernorm\.weight$", (c.kv_lora,)),
        (r"kv_b_proj\.weight$", (heads * (c.nope + c.v_dim), c.kv_lora)), (r"mlp\.(gate|up)_proj\.weight$", (c.dense_inter, H)),
        (r"mlp\.down_proj\.weight$", (H, c.dense_inter)), (r"moe\.gate\.weight$", (c.experts, H)),
        (r"e_score_correction_bias$", (c.experts,)), (r"routed_expert_down_proj\.weight$", (c.latent, H)),
        (r"routed_expert_norm\.weight$", (c.latent,)), (r"routed_expert_up_proj\.weight$", (H, c.latent)),
        (r"shared_experts\.(gate|up)_proj\.weight$", (c.moe_inter * c.shared, H)),
        (r"shared_experts\.down_proj\.weight$", (H, c.moe_inter * c.shared)), (r"^(embed_tokens|lm_head)\.weight$", (c.vocab, H)),
        (r"^(norm|output_attn_res_norm)\.weight$", (H,)), (r"^output_attn_res_proj\.weight$", (1, H)),
    ]
    for pat, shape in table:
        if re.search(pat, name):
            if shape is not None:
                return shape
            kda = (int(name.split(".")[1]) + 1) not in c.full_attn
            width = W if kda else heads * c.v_dim
            return (H, width) if name.endswith("o_proj.weight") else (width, H)
    raise KeyError(name)


class Checkpoint:
    """A safetensors checkpoint read in place: the index, each file's header on first use, tensors by our names."""

    def __init__(self, folder: Path):
        self.folder = folder
        index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
        self.where = {self.short(k): v for k, v in index.items() if self.short(k)}
        self.headers = {}

    @staticmethod
    def short(name: str):
        for p in ("language_model.model.", "language_model."):
            if name.startswith(p):
                return name[len(p):]
        return None

    def header(self, file: str):
        if file not in self.headers:
            with open(self.folder / file, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                self.headers[file] = (8 + n, {self.short(k): v for k, v in json.loads(f.read(n)).items() if self.short(k)})
        return self.headers[file]

    def get(self, name: str) -> np.ndarray:
        file = self.where[name]
        data, entries = self.header(file)
        e = entries[name]
        dtype = {"BF16": np.uint16, "F32": np.float32, "U8": np.uint8}[e["dtype"]]
        a = np.memmap(self.folder / file, dtype=dtype, mode="r", offset=data + e["data_offsets"][0], shape=tuple(e["shape"]))
        return from_bf16_bits(a) if e["dtype"] == "BF16" else np.array(a)


class Weights(dict):
    """Tensors by name on first use: the checkpoint's, or synthetic ones at the config's shapes."""

    def __init__(self, cfg, model=None):
        super().__init__()
        self.cfg = cfg
        self.model = model

    def __missing__(self, name):
        v = self.model.get(name) if self.model else synthetic(name, shape_of(self.cfg, name))
        self[name] = v
        return v


def row_input(cfg, key, what):
    return values(fnv1a(f"in.{key[0]}.{key[1]}.{key[2]}.{what}"), cfg.hidden, "w", 7)


@lru_cache(maxsize=48)
def _decoded(p, e, ident):
    return ref.expert_weights(_decoded.W, p, e)


def decoded(W, p, e):
    _decoded.W = W
    return _decoded(p, e, id(W))


def stats(name, ours, theirs, limit):
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    rel = np.abs(ours - theirs).max() / max(np.abs(theirs).max(), 1e-30)
    print(f"  {name:34s} rel {rel:.2e}  bf16 words equal {np.mean(ours == theirs) * 100:5.1f}%  (limit {limit:.0e})")
    return rel <= limit


def check_layer(d: Path, i: int, cfg, max_round: int, model) -> bool:
    keys = [tuple(k) for k in json.loads((d / "keys.json").read_text())]
    H = cfg.hidden
    got = from_bf16_bits(np.fromfile(d / "rows.bin", dtype=np.uint16).reshape(len(keys), 3, H))
    index = {k: n for n, k in enumerate(keys)}
    W = Weights(cfg, model)
    streams = {s: ref.Stream() for s in range(3)}
    nb = (i + cfg.block - 1) // cfg.block
    ours, theirs = {"prefix": [], "mlp_in": [], "mlp": []}, {"prefix": [], "mlp_in": [], "mlp": []}
    p = f"layers.{i}."
    for rnd, segs in enumerate(SCHEDULE[:max_round + 1]):
        for stream, rows, commit, keep in segs:
            ks = [(stream, rnd, j) for j in range(rows)]
            prefix = np.stack([row_input(cfg, k, "prefix") for k in ks])
            if i > 0:
                prefix = bf16(prefix + np.stack([row_input(cfg, k, "delta") for k in ks]))
            blocks = [np.stack([row_input(cfg, k, f"block{e}") for k in ks]) for e in range(nb)]
            before = copy.deepcopy(streams[stream])
            trace = {}
            ref.layer(i, prefix, blocks, W, cfg, streams[stream], trace, decoded)
            if not commit:
                streams[stream] = before
                if keep:
                    ref.layer(i, prefix[:keep], [b[:keep] for b in blocks], W, cfg, streams[stream], {}, decoded)
            gpu_in = np.stack([got[index[k], 1] for k in ks])
            if i == 0:
                mlp = ref.mlp(gpu_in, *(W[f"{p}mlp.{n}_proj.weight"] for n in ("gate", "up", "down")), cfg)
            else:
                mlp = ref.moe(gpu_in, W, p + "block_sparse_moe.", cfg, decoded)[0]
            for j, k in enumerate(ks):
                for n, col in (("prefix", 0), ("mlp_in", 1), ("mlp", 2)):
                    ours[n].append(got[index[k], col])
                theirs["prefix"].append(trace["prefix"][j])
                theirs["mlp_in"].append(trace["mlp_in"][j])
                theirs["mlp"].append(mlp[j])
        print(f"  round {rnd}: {sum(s[1] for s in segs)} rows")
    ok = stats(f"layer {i} prefix after attention", ours["prefix"], theirs["prefix"], 2e-2)
    stats(f"layer {i} MLP input (normed)", ours["mlp_in"], theirs["mlp_in"], 1.0)
    ok &= stats(f"layer {i} {'MoE' if i >= 1 else 'dense MLP'} on the GPU's input", ours["mlp"], theirs["mlp"], 2e-2)
    dq = d / "dequant_e5_w1.bin"
    if dq.exists():
        p = f"layers.{i}.block_sparse_moe.experts.5.w1."
        want = mxfp4_decode(W[p + "weight_packed"], W[p + "weight_scale"])
        same = np.array_equal(np.fromfile(dq, dtype=np.float32).reshape(want.shape).view(np.uint32), want.view(np.uint32))
        print(f"  MXFP4 decode of expert 5 w1 ({want.size} values): {'bit-exact' if same else 'DIFFERS'}")
        ok &= same
    return ok


def check_head(d: Path, cfg, model) -> bool:
    W = Weights(cfg, model)
    ks = [(9, 0, r) for r in range(8)]
    prefix = bf16(np.stack([row_input(cfg, k, "prefix") for k in ks]) + np.stack([row_input(cfg, k, "delta") for k in ks]))
    blocks = [np.stack([row_input(cfg, k, f"block{e}") for k in ks]) for e in range((cfg.layers - 1) // cfg.block + 1)]
    h = ref.attn_res(prefix, blocks, W["output_attn_res_norm.weight"], W["output_attn_res_proj.weight"], cfg.eps)
    x = ref.rms_norm(h, W["norm.weight"], cfg.eps).astype(np.float64)
    lm = W["lm_head.weight"]
    want = np.concatenate([bf16(x @ lm[a:a + 16384].astype(np.float64).T) for a in range(0, cfg.vocab, 16384)], 1)
    got = from_bf16_bits(np.fromfile(d / "logits.bin", dtype=np.uint16).reshape(8, cfg.vocab))
    tokens = np.fromfile(d / "tokens.bin", dtype=np.uint32)
    ok = stats("head logits (8 rows)", got, want, 2e-2)
    ok &= bool(np.all(tokens == got.argmax(1)))
    print(f"  greedy tokens are the logits' argmax: {bool(np.all(tokens == got.argmax(1)))}; "
          f"reference argmax agrees on {np.mean(want.argmax(1) == tokens) * 100:.0f}% of rows")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="tf-k3-check's output folder")
    ap.add_argument("--rounds", type=int, default=4, help="last schedule round to check (MoE layers default 1)")
    ap.add_argument("--model", help="the checkpoint tf-k3-check read (else synthetic weights)")
    a = ap.parse_args()
    cfg = ref.Config()
    model = Checkpoint(Path(a.model)) if a.model else None
    ok = True
    for d in sorted(Path(a.out).glob("layer*")):
        i = int(d.name[5:])
        print(f"layer {i}:")
        ok &= check_layer(d, i, cfg, a.rounds if i == 0 else min(a.rounds, 1), model)
    if (Path(a.out) / "head").exists():
        print("head:")
        ok &= check_head(Path(a.out) / "head", cfg, model)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
