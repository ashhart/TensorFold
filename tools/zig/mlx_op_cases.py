"""Cases for check_mlx_ops.py: Nemotron-3.5 Lightning's shapes, MLX's dispatch rule for each, our launch and MLX's."""

from __future__ import annotations

import json
import os
import math
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from check_mlx_ops import ROOT, Buf, Case, Launch

MODEL = Path(os.environ.get("TF_ZIG_MODEL", "models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit"))
D = 2688                 # hidden size
EPS = 1e-5


class Checkpoint:
    """Lazy access to the 4-bit checkpoint's tensors (only the ones asked for are read)."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
        self.files: dict[str, dict[str, mx.array]] = {}

    def get(self, name: str) -> mx.array:
        shard, key = ("mtp-4bit.safetensors", name[4:]) if name.startswith("mtp.") else (self.index[name], name)
        if shard not in self.files:
            self.files[shard] = mx.load(str(self.folder / shard))
        return self.files[shard][key]

    def linear(self, prefix: str) -> tuple[mx.array, mx.array, mx.array]:
        return self.get(prefix + ".weight"), self.get(prefix + ".scales"), self.get(prefix + ".biases")


def normal(shape: tuple[int, ...], seed: int, scale: float = 1.0) -> mx.array:
    return (mx.random.normal(shape, key=mx.random.key(seed)) * scale).astype(mx.bfloat16)


def requantize(w: mx.array, s: mx.array, b: mx.array, bits: int) -> tuple[mx.array, ...]:
    """A 4-bit weight's values quantized again at ``bits`` (groups of 64): real value distributions for 6/8-bit."""

    q = mx.quantize(mx.dequantize(w, s, b, group_size=64, bits=4), group_size=64, bits=bits)
    mx.eval(q)
    return q


def chain_for(nbytes: int) -> int:
    """Dispatches a command buffer so one runs ~20 ms at ~400 GB/s (the GPU stays at full clock)."""

    return max(16, min(4000, int(20e-3 / (nbytes / 400e9 + 4e-6))))


def past_cache(nbytes: int) -> int:
    """Copies of an input that keep its reads in DRAM while timing (192 MB of copies, at most 32)."""

    return max(2, min(32, (192 << 20) // max(1, nbytes)))


def u32(values: list[int]) -> mx.array:
    """An index table padded to 8 entries (a device buffer in MLX's signature rule)."""

    return mx.array(list(values) + [0] * max(0, 8 - len(values)), dtype=mx.uint32)


RMS_THREADS = 32 * -(-(-(-D // 4)) // 32)         # MLX's one-row kernel: 32 ceil(ceil(D / 4) / 32) threads a row


def rms_launch(x: Buf, w: mx.array, rows: int) -> Launch:
    """tf_rms_norm_bf16 over [rows, D]; x is slot 0's data or source."""

    return Launch("tf_rms_norm_bf16", [x, Buf(1, w), Buf(2, f32=[EPS]), Buf(3, i32=[D]),
                                       Buf(4, out=((rows, D), mx.bfloat16))], (RMS_THREADS * rows, 1, 1),
                  (RMS_THREADS, 1, 1), stem="embed_norm")


def embed_launch(ids: list[int], w: mx.array, s: mx.array, b: mx.array, bits: int = 4) -> Launch:
    """tf_embed_b{bits}_g64 for the rows of ids."""

    rows, per = len(ids), 4 if bits == 6 else 8 // bits
    return Launch(f"tf_embed_b{bits}_g64", [Buf(0, u32(ids)), Buf(1, w.view(mx.uint8)), Buf(2, s), Buf(3, b),
                                             Buf(4, i32=[D]), Buf(5, out=((rows, D), mx.bfloat16))],
                  (D // per, rows, 1), (256, 1, 1), stem="embed_norm")


def rms_cases(ck: Checkpoint, quick: bool) -> list[list[Case]]:
    weights = {"layers.0.norm": ck.get("backbone.layers.0.norm.weight"),
               "mtp.enorm": ck.get("mtp.layers.0.enorm.weight"), "random": normal((D,), 7, 0.3)}
    cases = []
    for wname, w in weights.items():
        for rows in ((1, 5) if quick else (1, 2, 3, 5, 8, 16, 64)):
            for seed in range(2 if quick else 4):
                x = normal((rows, D), 100 * rows + seed, (1.0, 0.01, 30.0, 300.0)[seed % 4])
                ours = rms_launch(Buf(0, x), w, rows)
                theirs = Launch("rmsbfloat16", [Buf(0, x), Buf(1, w), Buf(2, out=((rows, D), mx.bfloat16)),
                                                Buf(3, f32=[EPS]), Buf(4, i32=[D]), Buf(5, i32=[1])], ours.grid,
                                ours.threadgroup)
                cases.append(Case("rms_norm", f"{wname} R={rows} s{seed}", "rms_single_row (rmsbfloat16)",
                                  "embed_norm", [ours], [theirs], [mx.fast.rms_norm(x, w, EPS)],
                                  time=seed == 0 and rows in (1, 5, 16), chain=chain_for(4 * rows * D)))
    return [cases]


def qmv_batch_limit(k: int, n: int) -> int:
    """MLX 0.32.3 get_qmv_batch_limit on gen-17 's' GPUs (the M5 Max): from this many rows it runs qmm_splitk."""

    return 33 if (k <= 2048 and n <= 2048) else 25 if (k <= 4096 and n <= 4096) else 13


def mlx_qmv_variant(m: int, k: int, n: int, bits: int) -> tuple[str, int]:
    """(kernel, input rows a threadgroup) MLX runs for x [m, k] @ W.T, affine, transposed, on gen >= 15 GPUs."""

    if m >= qmv_batch_limit(k, n):
        return "qmm_splitk", 0
    if m >= 2:
        tiles = (m + 4) // 5
        return "qmv_wide", -(-m // tiles)
    pack = 4 if bits == 6 else 32 // bits
    return ("qmv_fast" if n % 8 == 0 and k % (pack * 2 * 32) == 0 else "qmv"), 1


def qmv_launch(w: mx.array, s: mx.array, b: mx.array, x: Buf, m: int, bits: int, copies: int = 1) -> Launch:
    """Our launch for x [m, K] @ W.T in the variant MLX would run; x is slot 3's data or source."""

    k, n = int(w.shape[1]) * 32 // bits, int(w.shape[0])
    variant, nv = mlx_qmv_variant(m, k, n, bits)
    weights = [Buf(0, w, copies=copies), Buf(1, s, copies=copies), Buf(2, b, copies=copies)]
    out = Buf(5, out=((m, n), mx.bfloat16))
    if variant == "qmv_wide":
        return Launch(f"tf_qmv_wide_b{bits}_g64_v{nv}", [*weights, x, Buf(4, i32=[k, n, m]), out],
                      (-(-m // nv), -(-n // 8), 1), (32, 2, 1), groups=True, stem="qmv")
    return Launch(f"tf_{variant}_b{bits}_g64", [*weights, x, Buf(4, i32=[k, n]), out], (m, -(-n // 8), 1),
                  (32, 2, 1), groups=True, stem="qmv")


def qmv_case(name: str, w: mx.array, s: mx.array, b: mx.array, x: mx.array, bits: int, time: bool) -> Case:
    m, k = x.shape
    n = int(w.shape[0])
    variant, nv = mlx_qmv_variant(m, k, n, bits)
    ours = qmv_launch(w, s, b, Buf(3, x), m, bits, past_cache(w.nbytes) if time else 1)
    mlx_name = (f"affine_qmv_wide_bfloat16_t_gs_64_b_{bits}_nv_{nv}_kl_8_batch_0" if variant == "qmv_wide"
                else f"affine_{variant}_bfloat16_t_gs_64_b_{bits}_batch_0")
    extra = [Buf(5, i32=[k]), Buf(6, i32=[n])] + ([Buf(7, i32=[m])] if variant == "qmv_wide" else [])
    theirs = Launch(mlx_name, [*ours.buffers[:3], Buf(3, x), Buf(4, out=((m, n), mx.bfloat16)), *extra], ours.grid,
                    ours.threadgroup, groups=True)
    ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=bits)
    return Case("qmv", f"{name} K={k} N={n} M={m} b{bits}", f"{variant}" + (f" nv{nv}" if nv > 1 else ""), "qmv",
                [ours], [theirs], [ref], time=time, chain=chain_for(w.nbytes + s.nbytes + b.nbytes))


LINEARS = {  # Nemotron's quantized linears that reach mx.quantized_matmul: the MTP head always, the rest off-M5 6/8-bit
    "in_proj": "backbone.layers.0.mixer.in_proj", "out_proj": "backbone.layers.0.mixer.out_proj",
    "q_proj": "backbone.layers.5.mixer.q_proj", "k_proj": "backbone.layers.5.mixer.k_proj",
    "o_proj": "backbone.layers.5.mixer.o_proj", "shared_up": "backbone.layers.1.mixer.shared_experts.up_proj",
    "shared_down": "backbone.layers.1.mixer.shared_experts.down_proj", "lm_head": "lm_head",
    "mtp.eh_proj": "mtp.layers.0.eh_proj", "mtp.q_proj": "mtp.layers.0.mixer.q_proj",
    "mtp.k_proj": "mtp.layers.0.mixer.k_proj", "mtp.o_proj": "mtp.layers.0.mixer.o_proj",
    "mtp.shared_up": "mtp.layers.1.mixer.shared_experts.up_proj",
    "mtp.shared_down": "mtp.layers.1.mixer.shared_experts.down_proj",
}


def qmv_cases(ck: Checkpoint, quick: bool) -> list[list[Case]]:
    names = ("in_proj", "o_proj", "lm_head", "mtp.eh_proj") if quick else tuple(LINEARS)
    weights = {n: ck.linear(LINEARS[n]) for n in names}
    if not quick:
        ids = mx.array([int(t) for t in (ROOT / "src/tensorfold/families/nemotron_h/draft_ids.txt").read_text()
                        .split()], dtype=mx.int32)
        w, s, b = ck.linear("lm_head")
        weights["draft_head"] = (w[ids], s[ids], b[ids])
        table = "backbone.layers.1.mixer.switch_mlp"
        fc1, fc2 = ck.linear(f"{table}.fc1"), ck.linear(f"{table}.fc2")
        weights["expert_fc1[17]"] = tuple(t[17] for t in fc1)
        weights["expert_fc2[17]"] = tuple(t[17] for t in fc2)
    groups = []
    for name, (w4, s4, b4) in weights.items():
        mx.eval(w4, s4, b4)
        k, n = int(w4.shape[1]) * 8, int(w4.shape[0])
        cases = []
        for bits in (4, 6, 8):
            w, s, b = (w4, s4, b4) if bits == 4 else requantize(w4, s4, b4, bits)
            for m in ((1, 2, 5) if quick else (1, 2, 3, 4, 5, 6, 8, 12)):
                if m >= qmv_batch_limit(k, n):
                    continue
                for seed in range(1 if quick else 2):
                    x = normal((m, k), 1000 * seed + 10 * m + bits, (0.05, 1.0, 6.0)[seed % 3])
                    cases.append(qmv_case(name, w, s, b, x, bits, time=seed == 0 and m in (1, 2, 5)))
        groups.append(cases)
    return groups


HEADS, KV_HEADS, HEAD_DIM = 32, 2, 128


def sdpa_blocks(n: int, rows: int) -> int:
    """MLX 0.32.3's 2-pass block count on 's' GPUs (the M5 Max) for 16 query heads a kv head."""

    if n > 1024 and 16 * rows > 4:
        return 128 if n <= 8192 else 256 if n <= 32768 else 512 if n <= 65536 else 1024
    return 64


def sdpa_case(n: int, rows: int, seed: int, time: bool) -> Case:
    cap = -(-n // 2048) * 2048 + (2048 if n % 2048 == 0 else 0)
    kc, vc = normal((1, KV_HEADS, cap, HEAD_DIM), seed, 1.5), normal((1, KV_HEADS, cap, HEAD_DIM), seed + 1)
    qraw = normal((1, rows, HEADS, HEAD_DIM), seed + 2, 2.0)
    q = qraw.transpose(0, 2, 1, 3)
    scale = HEAD_DIM ** -0.5
    ref = mx.fast.scaled_dot_product_attention(q, kc[..., :n, :], vc[..., :n, :], scale=scale,
                                               mask="causal" if rows > 1 else None)
    causal = rows > 1
    copies = past_cache(kc.nbytes) if time else 1
    kv = [Buf(1, kc, copies=copies), Buf(2, vc, copies=copies)]
    strides = [cap * HEAD_DIM, HEAD_DIM, cap * HEAD_DIM, HEAD_DIM]
    consts = [(20, "bool", 0), (21, "bool", int(causal)), (22, "bool", int(causal)), (23, "bool", 0),
              (24, "bool", 0), (25, "bool", 0)]
    out_shape = (1, HEADS, rows, HEAD_DIM)
    tail = "_cqt" if causal else ""
    if n < 1024:
        ours = [Launch(f"tf_sdpa_vec_d128{tail}", [Buf(0, qraw), *kv, Buf(3, i32=[HEADS // KV_HEADS, n]),
                                                    Buf(4, u64=strides), Buf(5, f32=[scale]),
                                                    Buf(6, out=(out_shape, mx.bfloat16))],
                       (HEADS, rows, 1), (1024, 1, 1), groups=True)]
        theirs = [Launch("sdpa_vector_bfloat16_t_128_128",
                         [Buf(0, qraw), *kv, Buf(3, out=(out_shape, mx.bfloat16)), Buf(4, i32=[HEADS // KV_HEADS]),
                          Buf(5, i32=[n]), *[Buf(6 + i, u64=[v]) for i, v in enumerate(strides)],
                          Buf(10, f32=[scale])], (HEADS, rows, 1), (1024, 1, 1), groups=True, constants=consts)]
        variant = "sdpa_vector"
    else:
        blocks = sdpa_blocks(n, rows)
        gqa = rows == 1 and n >= 8192
        part = (1, HEADS, rows, blocks, HEAD_DIM)
        outs = [Buf(6, out=(part, mx.bfloat16)), Buf(7, out=(part[:-1], mx.float32)),
                Buf(8, out=(part[:-1], mx.float32))]
        first = Launch("tf_sdpa_2p1_gqa16_d128" if gqa else f"tf_sdpa_2p1_d128{tail}",
                       [Buf(0, qraw), *kv, Buf(3, i32=[n, blocks]), Buf(4, u64=strides), Buf(5, f32=[scale]), *outs],
                       (KV_HEADS, 1, blocks), (32, 16, 1 if gqa else rows), groups=True)
        merge = Launch("tf_sdpa_2p2_d128", [Buf(0, src=(0, 0)), Buf(1, src=(0, 1)), Buf(2, src=(0, 2)),
                                            Buf(3, i32=[blocks]), Buf(4, out=(out_shape, mx.bfloat16))],
                       (HEADS, rows, 1), (1024, 1, 1), groups=True)
        ours = [first, merge]
        mlx_first = Launch(("sdpa_vector_2pass_1_gqa_16_bfloat16_t_128_128" if gqa
                            else "sdpa_vector_2pass_1_bfloat16_t_128_128"),
                           [Buf(0, qraw), *kv, Buf(3, out=(part, mx.bfloat16)), Buf(4, out=(part[:-1], mx.float32)),
                            Buf(5, out=(part[:-1], mx.float32)), Buf(7, i32=[n]),
                            *[Buf(8 + i, u64=[v]) for i, v in enumerate(strides)], Buf(12, f32=[scale])],
                           (KV_HEADS, 1, blocks), (32, 16, rows), groups=True,
                           constants=consts + [(26, "int", blocks)])
        mlx_merge = Launch("sdpa_vector_2pass_2_bfloat16_t_128",
                           [Buf(0, src=(0, 0)), Buf(1, src=(0, 1)), Buf(2, src=(0, 2)),
                            Buf(3, out=(out_shape, mx.bfloat16)), Buf(4, i32=[blocks])],
                           (HEADS, rows, 1), (1024, 1, 1), groups=True)
        theirs = [mlx_first, mlx_merge]
        variant = f"2pass{'_gqa16' if gqa else ''} blocks {blocks}"
    return Case("sdpa", f"keys={n} rows={rows} s{seed}", variant + (" causal" if causal else ""), "sdpa", ours,
                theirs, [ref], time=time, chain=chain_for(4 * kc.nbytes))


def sdpa_cases(ck: Checkpoint, quick: bool) -> list[list[Case]]:
    lengths = (300, 1500, 9000) if quick else (1, 33, 300, 1023, 1024, 1025, 1500, 4096, 8192, 8193, 20000, 40000,
                                               70000)
    groups = []
    for n in lengths:
        cases = []
        for rows in (1, 2):           # MLX fuses rows * 16 <= 32; 3-8 rows take its unfused fallback
            if rows > n:
                continue
            for seed in range(1 if quick else 2):
                cases.append(sdpa_case(n, rows, 10 * n + rows + 1000 * seed, time=seed == 0))
        groups.append(cases)
    return groups


def embed_cases(ck: Checkpoint, quick: bool) -> list[list[Case]]:
    w4, s4, b4 = ck.linear("backbone.embeddings")
    vocab = int(w4.shape[0])
    cases = []
    for bits in (4, 6, 8):
        if bits == 4:
            w, s, b, top = w4, s4, b4, vocab
        else:                      # a slice of the table re-quantized (ids stay below it)
            top = 8192
            w, s, b = requantize(w4[:top], s4[:top], b4[:top], bits)
        for rows in ((1, 5) if quick else (1, 2, 5, 8, 16, 64)):
            raw = [int((7919 * (i + 1) * (bits + 3)) % top) for i in range(rows)]
            ids = mx.array(raw, dtype=mx.uint32)
            ref = mx.dequantize(w[ids], scales=s[ids], biases=b[ids], group_size=64, bits=bits)
            cases.append(Case("embed", f"b{bits} rows={rows}", "take x3 + affine_dequantize", "embed_norm",
                              [embed_launch(raw, w, s, b, bits)], None, [ref], time=rows in (1, 16),
                              chain=chain_for(rows * D * 3)))
    return [cases]


def argmax_split(x: Any, rows: int, vocab: int, span: int = 2048) -> list[Launch]:
    """The two-pass argmax over [rows, vocab]; x is an array or a (launch, output) source, negative counts back."""

    spans = -(-vocab // span)
    first = Buf(0, src=x) if isinstance(x, tuple) else Buf(0, x)
    return [Launch("tf_argmax_part_bf16", [first, Buf(1, i32=[vocab, span]), Buf(2, out=((rows, spans), mx.float32)),
                                           Buf(3, out=((rows, spans), mx.uint32))], (spans, rows, 1), (256, 1, 1),
                   groups=True, stem="argmax"),
            Launch("tf_argmax_merge", [Buf(0, src=(-1, 0)), Buf(1, src=(-1, 1)), Buf(2, i32=[spans]),
                                       Buf(3, out=((rows,), mx.uint32))], (1, rows, 1), (32, 1, 1), groups=True,
                   stem="argmax")]


def argmax_cases(ck: Checkpoint, quick: bool) -> list[list[Case]]:
    cases = []
    for vocab in (131072, 32768):
        for rows in ((1, 5) if quick else (1, 2, 5, 16)):
            for kind in ("random", "ties", "nan", "-inf"):
                x = normal((rows, vocab), vocab + rows, 3.0)
                if kind == "ties":
                    x = mx.round(x * 4) / 4             # many equal maxima
                    x = x.astype(mx.bfloat16)
                elif kind == "nan":
                    x = mx.where(mx.arange(vocab) % 9973 == 5, mx.array(float("nan"), dtype=mx.bfloat16), x)
                elif kind == "-inf":
                    x = mx.full((rows, vocab), -math.inf, dtype=mx.bfloat16)
                ref = mx.argmax(x, axis=-1).astype(mx.uint32)
                split = argmax_split(x, rows, vocab)
                theirs = Launch("argmax_bfloat16", [Buf(0, x), Buf(1, out=((rows,), mx.uint32)), Buf(2, i32=[rows]),
                                                    Buf(3, u64=[vocab]), Buf(4, u64=[1]), Buf(5, u64=[1]),
                                                    Buf(6, u64=[1]), Buf(7, u64=[vocab])],
                                (1024, rows, 1), (1024, 1, 1))
                cases.append(Case("argmax", f"V={vocab} rows={rows} {kind}", "arg_reduce_general (argmax)", "argmax",
                                  split, [theirs], [ref], time=kind == "random", chain=chain_for(2 * rows * vocab)))
    return [cases]


FIXTURES = Path(os.environ.get("TF_ZIG_FIXTURES", "build/zig-fixtures/nemotron"))


def fixture(name: str) -> mx.array:
    """A fixture .npy (bf16 files hold uint16 bit patterns; the manifest's dtype says which)."""

    import numpy as np

    a = np.load(FIXTURES / name)
    manifest = _manifest()
    dtype = manifest.get(name)
    arr = mx.array(a)
    return arr.view(mx.bfloat16) if dtype == "bfloat16" else arr


_MANIFEST: dict[str, str] = {}


def _manifest() -> dict[str, str]:
    if not _MANIFEST:
        m = json.loads((FIXTURES / "manifest.json").read_text())
        for op in m["ops"]:
            for io in op.get("inputs", []) + op.get("outputs", []):
                if "file" in io:
                    _MANIFEST[io["file"]] = io["dtype"]
    return _MANIFEST


def decode_fixture_cases(ck: Checkpoint, quick: bool) -> list[list[Case]]:
    """The decode step's MLX built-ins on the captured r1 (serial step) and r4 (4-row window) states."""

    w4, s4, b4 = ck.linear("backbone.embeddings")
    cases = []
    for step in ("r1", "r4"):
        f = lambda name: fixture(f"{step}/{name}")                                 # noqa: E731
        ids = [int(i) for i in f("000_embed_gather/in_ids.npy").tolist()]
        rows = len(ids)
        h = f("001_embed_dequantize/out_h.npy")
        cases.append(Case("decode", f"{step} embed (gathers + dequantize)", "gather_front x3, affine_dequantize",
                          "embed_norm", [embed_launch(ids, w4, s4, b4)], None, [h], time=False))
        cases.append(Case("decode", f"{step} embed + layer 0 norm, one launch", "gather_front x3, dequantize, rms",
                          "embed_norm", [Launch("tf_embed_rms_b4_g64",
                                              [Buf(0, u32(ids)), Buf(1, w4.view(mx.uint8)), Buf(2, s4), Buf(3, b4),
                                               Buf(4, f("002_L00_input_rms_norm/in_weight.npy")), Buf(5, f32=[EPS]),
                                               Buf(6, i32=[D]), Buf(7, out=((rows, D), mx.bfloat16)),
                                               Buf(8, out=((rows, D), mx.bfloat16))],
                                              (RMS_THREADS * rows, 1, 1), (RMS_THREADS, 1, 1))], None,
                          [h, f("002_L00_input_rms_norm/out_normed.npy")], time=True, chain=64))
        cases.append(Case("decode", f"{step} layer 0 input norm", "rmsbfloat16", "embed_norm",
                          [rms_launch(Buf(0, f("002_L00_input_rms_norm/in_h.npy")),
                                      f("002_L00_input_rms_norm/in_weight.npy"), rows)], None,
                          [f("002_L00_input_rms_norm/out_normed.npy")], time=False))
        logits = f("054_argmax/in_logits.npy")
        cases.append(Case("decode", f"{step} greedy draw", "argmax_bfloat16", "argmax", argmax_split(logits, rows,
                                                                                                131072), None,
                          [f("054_argmax/out_tokens.npy")], time=False))
    return [cases]


BUILDERS: dict[str, Callable[[Checkpoint, bool], list[list[Case]]]] = {
    "rms": rms_cases, "qmv": qmv_cases, "sdpa": sdpa_cases, "embed": embed_cases, "argmax": argmax_cases,
    "decode": decode_fixture_cases,
}
