"""Cache and loader geometry shared by CUDA family constructors."""

from __future__ import annotations

import math
import re

from .capacity import Geometry, Weights, headers, itemsize

PREFILL_ROWS = 2048     # a prompt chunk's rows: Flash Next and GLM keep buffers of this many rows
PROMPT_SHARE = 32       # a dense prompt chunk's arrays take at most this fraction of the GPU's memory
PREFILL_ATT_ROWS = 256  # Flash Next's prompt attention block
MLA_PROMPT_ATT_ROWS = 512   # GLM's prompt-chunk rows one dense latent attention call takes (forward.PROMPT_ATT_ROWS)
MLA_SELECT_ROWS = 512       # GLM's prompt-chunk rows whose pool scores are held at once (sparse.SELECT_ROWS)


def indexed_prefill_rows() -> int | None:
    """Flash Next prompt-piece rows from TENSORFOLD_PREFILL_ROWS (256 to 16,384), or None: the engine's own plan."""

    import os

    value = os.environ.get("TENSORFOLD_PREFILL_ROWS", "").strip()
    if not value:
        return None
    if not value.isdecimal() or not 256 <= int(value) <= 16384:
        raise ValueError(f"TENSORFOLD_PREFILL_ROWS: 256 to 16,384 rows, not {value!r}")
    return int(value)


def size(info: dict, name: str = "tensor") -> int:
    return math.prod(info["shape"]) * itemsize(info, name)


def padded(info: dict, shape: list[int], *, float32: bool = False, name: str = "tensor") -> int:
    dims = list(shape)
    if info["dtype"] in ("U32", "I32") and len(dims) >= 2:
        dims[-2] = ((dims[-2] + 63) // 64) * 64
    return math.prod(dims) * (4 if float32 else itemsize(info, name))


def linear_weights(name: str, info: dict) -> tuple[int, int]:
    if name.startswith("vision_tower") or ".mtp." in name or name.startswith("mtp."):
        return 0, 0
    amount = padded(info, info["shape"], float32=name.endswith((".A_log", ".dt_bias")), name=name)
    return amount * (2 if "lm_head." in name else 1), 0


def exl3_weights(name: str, info: dict) -> tuple[int, int]:
    """A 27B EXL3 pack as loaded: tensors as stored, the head's words and svh once more for the drafter's slice; vision and MTP skipped."""

    if ".visual." in name or name.startswith(("model.visual.", "vision_tower", "mtp.")) or ".mtp." in name:
        return 0, 0
    amount = padded(info, info["shape"], float32=name.endswith((".A_log", ".dt_bias")), name=name)
    return (amount * 7 // 5 if name in ("lm_head.trellis", "lm_head.svh") else amount), 0


def exl3_workspace(largest: int, rows: int, width: int) -> int:
    """The EXL3 prompt matmul's decoded W_q (``largest`` weights) and rotated inputs (``rows`` of ``width``), fp16."""

    return 2 * (largest + rows * width)


def exl3_indexed_scratch(t: dict, window: int, rows: int) -> int:
    """Flash Next's EXL3 buffers: routed windows of ``window`` rows, prompt matmul workspace and n-gram staging for ``rows``."""

    d, hc, slots = int(t["hidden_size"]), int(t.get("hc_count", 1)), int(t.get("num_experts_per_tok", 1)) + 1
    width = int(t.get("moe_intermediate_size", d))
    heads, hd = int(t["num_attention_heads"]), int(t.get("head_dim") or d // int(t["num_attention_heads"]))
    conv = 2 * int(t["linear_num_key_heads"]) * int(t["linear_key_head_dim"]) + \
        int(t["linear_num_value_heads"]) * int(t["linear_value_head_dim"])
    pairs = window * slots
    moe = pairs * (4 * d + 2 * width + 4 * max(8 * width, 2 * d) + 4 * d) + (pairs + 1024) * 4
    largest = d * max(2 * heads * hd, conv, hc * d)
    ple = rows * (4 * 81 * 2 * int(t.get("heads_per_ngram", 8)) + 2 * int(t.get("ple_embed_dim") or d))
    return moe + exl3_workspace(largest, rows * hc, max(d, heads * hd)) + ple


def with_fixed(geometry: Geometry, extra: int) -> Geometry:
    """``geometry`` plus ``extra`` bytes that do not grow with the cache."""

    return Geometry(lambda slots: geometry.bytes_at(slots) + extra, geometry.reserve, geometry.minimum_slots)


def indexed_weights(world: int, mtp: bool, mapped_tables: bool = True):
    def transform(name: str, info: dict) -> tuple[int, int]:
        if "vision" in name or ".visual." in name or (not mtp and (name.startswith("mtp.") or ".mtp." in name)):
            return 0, 0
        if ".ngram_embedding.shard_" in name or name.endswith(".ngram_embedding.trellis"):     # host pages if mapped
            return 0, size(info, name) if mapped_tables else 0
        shape = list(info["shape"])
        if world > 1 and not info.get("split"):
            if ".switch_mlp." in name or ".shared_expert." in name:
                axis = -1 if ".down_proj." in name else -2
                shape[axis] //= world
            elif ".indexer." not in name and (".self_attn." in name or ".linear_attn." in name):
                if any(f".{part}." in name for part in ("q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z",
                                                        "in_proj_a", "in_proj_b", "conv1d")):
                    shape[0] //= world
                elif any(f".{part}." in name for part in ("o_proj", "out_proj")):
                    shape[-1] //= world
                elif name.endswith((".A_log", ".dt_bias")):
                    shape[0] //= world
            elif name.endswith(("lm_head.weight", "lm_head.scales", "lm_head.biases")):
                shape[0] //= world
        cast = name.endswith((".A_log", ".dt_bias", ".q_norm.weight", ".k_norm.weight", ".hc_norm.weight"))
        amount = padded(info, shape, float32=cast, name=name)
        if mtp and "lm_head." in name:
            amount *= 2  # the additional vocabulary-subset draft head
        return amount, 0
    return transform


def split_weights(rule, world: int = 2):
    def transform(name: str, info: dict) -> tuple[int, int]:
        kind = rule(name)
        if kind == "drop":
            return 0, 0
        shape = list(info["shape"])
        if not info.get("split") and kind != "rep":
            axis = {"row": 0, "col": -1, "dim1": 1}[kind]
            if shape[axis] % world:
                raise ValueError(f"checkpoint tensor does not split evenly: {name}")
            shape[axis] //= world
        if name.startswith("lm_head."):
            shape[0] //= world
        cast = name.endswith((".A_log", ".dt_bias", ".hc_attn_base", ".hc_attn_scale", ".hc_ffn_base",
                              ".hc_ffn_scale", ".e_score_correction_bias"))
        total = padded(info, shape, float32=cast, name=name)
        if name == "lm_head.weight" and info["dtype"] in ("BF16", "F16", "F32"):
            total += math.prod(shape) * 9 // 16  # the additional 4-bit draft head
        return total, 0
    return transform


def kv_bytes(head_dim: int, bits: int = 16) -> int:
    """One position's keys (or values) for one KV head: bf16, or codes plus an fp16 scale per 32 values."""

    if bits == 16:
        return 2 * head_dim
    if head_dim % 32:
        raise ValueError(f"a quantized KV cache needs a head dim that is a multiple of 32, not {head_dim}")
    return head_dim * bits // 8 + head_dim // 32 * 2


def layer_counts(t: dict) -> tuple[int, int]:
    if "layer_types" in t:
        linear = sum(kind == "linear_attention" for kind in t["layer_types"])
        return linear, len(t["layer_types"]) - linear
    layers, interval = int(t["num_hidden_layers"]), int(t.get("full_attention_interval", 4))
    return layers - layers // interval, layers // interval


def prompt_row_bytes(t: dict, world: int = 1) -> int:
    """Bytes one dense prompt row's arrays hold at once, drafter taps included (the 27B measured 306-315 KiB)."""

    return 16 * (int(t["hidden_size"]) + int(t["intermediate_size"]) // world)


def prompt_rows(total: int, row_bytes: int, most: int = 4096) -> int:
    """Prompt chunk rows: a multiple of 512 up to ``most`` whose arrays fit a PROMPT_SHARE-th of ``total``."""

    return max(512, min(most, total // PROMPT_SHARE // row_bytes // 512 * 512))


def live_kv(t: dict, world: int, window: int) -> int:
    """A dense stream's attention caches at ``window`` rows (1,024 at least) and one layer's buffer mid-grow."""

    _, attention = layer_counts(t)
    return (attention + 1) * max(1024, window) * int(t["num_key_value_heads"]) // world * int(t["head_dim"]) * 4


def gdn_geometry(t: dict, world: int, reserve: int, *, indexed: bool = False, mtp: bool = False,
                 kv_bits: int = 16, rows: int | None = None, prompt: int = 0, evicts: bool = False,
                 kept: int = 2, prefill_rows: int = PREFILL_ROWS) -> Geometry:
    """``rows``: widest verify; ``prompt``: chunk rows sharing its scratch; ``evicts``: only the live window counts."""

    linear, attention = layer_counts(t)
    d, h = int(t["hidden_size"]), int(t["num_attention_heads"]) // world
    hk = int(t["num_key_value_heads"]) // world
    hd = int(t.get("head_dim") or d // int(t["num_attention_heads"]))
    nk, nv = int(t["linear_num_key_heads"]) // world, int(t["linear_num_value_heads"]) // world
    dk, dv = int(t["linear_key_head_dim"]), int(t["linear_value_head_dim"])
    conv = int(t["linear_conv_kernel_dim"])
    streams = int(t.get("hc_count", 1))
    index_dim, ratio = int(t.get("indexer_head_dim", 128)), int(t.get("indexer_compress_ratio", 4))
    width = 2 * nk * dk + 2 * nv * dv + 2 * nv
    # Persistent state, retained recurrent prefixes, rollback and row replay inputs.
    fixed = linear * ((4 + kept if indexed else 4) * nv * dk * dv * 4 +
                      (2 + kept if indexed else 4) * (conv - 1) * (2 * nk * dk + nv * dv) * 2)
    rows = rows or (64 if indexed else 128)
    fixed += linear * rows * (width * 2 + nk * dk * 4 + nv * dv * 4 + nv * 8)
    # Bound the concurrent activation arrays, MoE expert rows, logits and split-K scratch.
    slots = int(t.get("num_experts_per_tok", 1)) + 1
    intermediate = int(t.get("moe_intermediate_size", t.get("intermediate_size", d))) // world
    extent = d * streams + int(t["vocab_size"]) // world + slots * (intermediate + d) + width + h * hd
    fixed += max(16 * rows * extent * 4, prompt * prompt_row_bytes(t, world) if prompt else 0)
    fixed += (2 if mtp else 1) * 32 * rows * 2560 * 4
    if indexed:
        fixed += ((2 + kept) * (int(t.get("ple_conv_kernel_size", 4)) - 1) * int(t.get("ngram_size", 3))
                  * streams * d * 2)
        fixed += int(mtp) * kept * streams * d * 2
        fixed += prefill_rows * _indexed_prefill_row(t, world, h, hk, hd, nv, dv, width, slots, intermediate)
    count = attention + int(mtp)
    budget = int(t.get("indexer_budget", 2048))
    row = kv_bytes(hd, kv_bits)
    def bytes_at(capacity: int) -> int:
        if indexed:
            # Separate K/V arrays in both the main state and the lazy serial-reference twin.
            cache = 4 * count * capacity * hk * row
            cache += 2 * count * (capacity + (capacity + ratio - 1) // ratio) * index_dim * 2
            # chunk partials cover the keys a row reads (at most the indexer budget and a block's tail)
            chunks = (min(capacity, budget + ratio - 1) + 511) // 512
            blocks = (capacity + ratio - 1) // ratio
            scratch = (2 if mtp else 1) * rows * (h * (hd + 2) * chunks + blocks) * 4
            scratch += PREFILL_ATT_ROWS * (h * (hd + 2) * chunks + blocks) * 4
        else:
            # Bound two retained prefixes, current KV state and a growth copy; speculative rows use separate workspace.
            rounded = 1 << (max(1024, capacity - reserve) - 1).bit_length()
            cache = live_kv(t, world, capacity - reserve) if evicts else 4 * attention * rounded * hk * hd * 4
            scratch = rows * h * (hd + 2) * ((capacity + 511) // 512) * 4
        return fixed + cache + scratch
    return Geometry(bytes_at, reserve)


def _indexed_prefill_row(t: dict, world: int, h: int, hk: int, hd: int, nv: int, dv: int, width: int, slots: int,
                         moe: int) -> int:
    """Bytes a row of Flash Next's prompt-chunk buffers holds, rounded up by group (``state.Buffers``)."""

    d, streams = int(t["hidden_size"]), int(t.get("hc_count", 1))
    heads, dim = int(t.get("indexer_n_heads", 4)), int(t.get("indexer_head_dim", 128))
    experts, low = int(t.get("num_experts", 1)), int(t.get("hc_lowrank", 320))
    ple = int(t.get("ple_embed_dim") or d)
    return (21 * streams * d + (12 + 12 * world) * d + 4 * ple + 12 * h * hd + 4 * hk * hd + 6 * heads * dim
            + 4 * experts + slots * (2 * moe + 2 * d + 24 + experts // 256) + 2 * width + 3 * nv * dv + 8 * low
            + 12 * streams + 64)


def indexed_prompt_bytes(t: dict, rows: int, world: int = 1) -> int:
    """Flash Next prompt activations and workspace for a piece, using the startup geometry's row bound."""

    _, h, hk, hd, _, nv, _, dv, width = _gdn_dims(t, world)
    slots = int(t.get("num_experts_per_tok", 1)) + 1
    moe = int(t.get("moe_intermediate_size", t.get("intermediate_size", t["hidden_size"]))) // world
    return rows * _indexed_prefill_row(t, world, h, hk, hd, nv, dv, width, slots, moe)


def exl3_expert_scratch(rows: int, slots: int, d: int, width: int) -> int:
    """GLM's ``exl3_mm.Scratch`` for ``rows`` x ``slots`` pairs: fp16 rotated inputs and the fp32 split-K sums."""

    pairs = rows * slots
    return 2 * pairs * d * 2 + pairs * width * 2 + 2 * 4 * pairs * max(width, d) * 4


def mla_geometry(t: dict, world: int, reserve: int, *, minimum_slots: int = 2560, latent: bool = False,
                 mtp: bool | None = None) -> Geometry:
    """GLM's engine; ``mtp``: whether it holds the MTP head's caches and buffers (None: when the checkpoint has one)."""
    linear, attention = layer_counts(t)
    lin = t.get("linear_attn_config") or {}
    heads = int(t["num_attention_heads"]) // world
    lh = int(lin.get("num_heads", t.get("linear_num_heads", 64))) // world
    ld = int(lin.get("head_dim", t.get("linear_head_dim", 128)))
    conv = int(lin.get("short_conv_kernel_size", t.get("linear_conv_kernel_dim", 4)))
    kd = int(t["qk_nope_head_dim"]) + int(t.get("qk_rope_head_dim", 0))
    vd, index = int(t["v_head_dim"]), int(t.get("index_head_dim", 128))
    mtp = int(t.get("num_nextn_predict_layers", 0)) > 0 if mtp is None else bool(mtp)
    rows, d, streams = 64, int(t["hidden_size"]), int(t.get("hc_mult", 4))
    fixed = linear * (4 * lh * ld * ld * 4 + 3 * (conv - 1) * 3 * lh * ld * 2)
    fixed += linear * rows * (3 * lh * ld + 2 * ld + lh) * 2
    fixed += linear * rows * lh * (12 * ld + 4)
    slots = int(t["num_experts_per_tok"]) + 1
    width = int(t["moe_intermediate_size"]) // world
    extent = d * streams + int(t["vocab_size"]) // world + slots * (d + width) + heads * (2 * kd + vd)
    extent += int(t.get("q_lora_rank", d)) * 2 + int(t.get("kv_lora_rank", d)) * 2
    extent += int(t.get("intermediate_size", width)) * 3 // world + int(t.get("index_n_heads", 32)) * index
    fixed += (2 if mtp else 1) * (16 * rows * extent * 4 + 8 * rows * 16384 * 4)
    # prompt-chunk buffers: at most 5 row extents a row without the head
    fixed += PREFILL_ROWS * 5 * (extent - int(t["vocab_size"]) // world)
    if (t.get("_quantization") or {}).get("quant_method") == "exl3":
        # EXL3 experts' scratch (decode windows, the MTP head's, a prompt chunk) and the prompt's BF16 split-K partials
        fixed += (2 if mtp else 1) * exl3_expert_scratch(rows, slots, d, width)
        fixed += exl3_expert_scratch(PREFILL_ROWS, slots, d, width) + 8 * PREFILL_ROWS * 16384 * 4
    count = attention + int(mtp)
    lw = int(t.get("kv_lora_rank", 512))
    def bytes_at(capacity: int) -> int:
        scratch = mla_chunk_scratch(t, world, capacity, latent=latent)
        if latent:
            # latent cache; a prompt chunk's partials (MLA_PROMPT_ATT_ROWS rows at a time) and absorbed rows (MTP's too)
            cache = count * capacity * lw * 2
            dense = min(capacity, minimum_slots) + PREFILL_ROWS
            scratch += (((dense + 511) // 512) * min(PREFILL_ROWS, MLA_PROMPT_ATT_ROWS) * heads * (lw + 2) * 4
                        + 4 * PREFILL_ROWS * heads * lw)
        else:
            cache = count * capacity * heads * (kd + vd) * 2
            scratch += (2 if mtp else 1) * ((capacity + rows + 511) // 512) * rows * heads * (kd + 2) * 4
        cache += count * (2 * capacity + capacity // 4 + 2) * index * 2
        return fixed + cache + scratch
    return Geometry(bytes_at, reserve, minimum_slots)


def mla_chunk_scratch(t: dict, world: int, capacity: int, *, latent: bool) -> int:
    """A prompt chunk's transient bytes: token selection (fp32 pool scores, chosen pools, token lists), then sparse attention's partials."""

    heads, topk = int(t["num_attention_heads"]) // world, int(t.get("index_topk", 2048))
    # the fp32 pool scores of at most MLA_SELECT_ROWS rows at once, the chosen pools and token lists of the chunk's
    select = min(PREFILL_ROWS, MLA_SELECT_ROWS) * 4 * ((capacity + 3) // 4) + PREFILL_ROWS * 16 * (topk + 3)
    if latent:
        return select + ((topk + 515) // 512) * PREFILL_ROWS * heads * (int(t.get("kv_lora_rank", 512)) + 2) * 4
    kd = int(t["qk_nope_head_dim"]) + int(t.get("qk_rope_head_dim", 0))
    return select + 128 * heads * (kd + 2) * 4 * ((topk + 515) // 512)


def draft_ring_rows(window: int, block: int, tile: int = 64) -> int:
    """Rows of GLM's DFlash2 context ring: window + block + a tile in whole tiles (a kept state's window fits too)."""

    return -(-(window + block + tile - 1) // tile) * tile


def draft_geometry(t: dict, world: int, reserve: int, *, bounded: bool = False, streams: int = 1,
                   kept: int = 0) -> Geometry:
    layers = int(t["num_hidden_layers"])
    heads = int(t["num_key_value_heads"]) // world
    hd = int(t["head_dim"])
    block = int((t.get("dflash_config") or {}).get("block_size", 16))
    window = int(t.get("sliding_window", 0))
    hidden = int(t["hidden_size"])
    fixed = 16 * max(64, streams * block) * (hidden + int(t["intermediate_size"])) * 4
    copies = 2 * streams + kept      # a stream's context and the one its taps replace it with; each kept prompt end
    def bytes_at(capacity: int) -> int:
        slots = min(capacity, window) if bounded and window > 0 else capacity
        return fixed + copies * 2 * layers * heads * hd * (slots + block) * 2
    return Geometry(bytes_at, reserve)


def dflash2_geometry(t: dict, world: int, reserve: int, *, ring: bool) -> Geometry:
    """GLM's DFlash2 drafter on each rank: one context (a ring, or capacity + block rows) and a block pass."""

    layers = int(t["num_hidden_layers"])
    heads = int(t["num_key_value_heads"]) // world
    hd = int(t["head_dim"])
    block = int((t.get("dflash_config") or {}).get("block_size", 16))
    window = int(t.get("sliding_window", 0))
    fixed = 16 * max(64, block) * (int(t["hidden_size"]) + int(t["intermediate_size"])) * 4
    rows = draft_ring_rows(window - 1, block) if ring and window > 0 else 0

    def bytes_at(capacity: int) -> int:
        slots = capacity + block if not rows else min(rows, capacity + block)
        return fixed + 2 * layers * heads * hd * slots * 2
    return Geometry(bytes_at, reserve)


def dflash2_weights(draft_dir, world: int) -> Weights:
    """GLM's DFlash2 drafter as held on each rank (4-bit copies, bf16 norms, fp32 codebooks), and its staging."""

    h = headers(draft_dir)
    shape = {name: [int(x) for x in info["shape"]] for name, info in h.items()}

    def q4(n: int, k: int) -> int:
        return -(-n // 128) * 128 * k * 9 // 16

    mats = [tuple(shape["fc.weight"])]
    quantized = {"fc.weight"}
    for i in sorted({int(m.group(1)) for m in map(re.compile(r"layers\.(\d+)\.").match, shape) if m}):
        p = f"layers.{i}."
        q, k, v = (shape[p + f"self_attn.{x}_proj.weight"] for x in "qkv")
        o, gate, up, down = (shape[p + x] for x in ("self_attn.o_proj.weight", "mlp.gate_proj.weight",
                                                      "mlp.up_proj.weight", "mlp.down_proj.weight"))
        d = q[1]
        mats += [((q[0] + k[0] + v[0]) // world, d), ((k[0] + v[0]) // world, d), (o[0], o[1] // world),
                 ((gate[0] + up[0]) // world, d), (down[0], down[1] // world)]
        quantized |= {p + x for x in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                                      "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                                      "mlp.down_proj.weight")}
        for conv in ("attention_conv", "mlp_conv"):
            name = p + conv + ".kernel_projection.weight"
            mats.append(tuple(shape[name]))
            quantized.add(name)
    resident = sum(q4(n, k) for n, k in mats)
    for name, dims in shape.items():
        if name not in quantized:
            resident += math.prod(dims) * (4 if name.endswith("_codebook") else 2)
    staging = max(4 * n * k + 24 * min(n, 8192) * k for n, k in mats)
    return Weights(resident, staging, 0)


def _gdn_dims(t: dict, world: int) -> tuple:
    d, heads = int(t["hidden_size"]), int(t["num_attention_heads"])
    nk, nv = int(t["linear_num_key_heads"]) // world, int(t["linear_num_value_heads"]) // world
    dk, dv = int(t["linear_key_head_dim"]), int(t["linear_value_head_dim"])
    return (d, heads // world, int(t["num_key_value_heads"]) // world, int(t.get("head_dim") or d // heads),
            nk, nv, dk, dv, 2 * nk * dk + 2 * nv * dv + 2 * nv)


def stream_geometry(t: dict, world: int, streams: int, keep: int, *, first: int | None = None) -> Geometry:
    """The 27B's concurrent decoder: live streams, ``keep`` kept prompt ends, windows; ``first``: growth on one GPU."""

    linear, attention = layer_counts(t)
    d, h, hk, hd, nk, nv, dk, dv, width = _gdn_dims(t, world)
    rows = 16 * streams
    state = linear * (nv * dk * dv * 4 + (int(t["linear_conv_kernel_dim"]) - 1) * (2 * nk * dk + nv * dv) * 2)
    # a commit writes a stream's new states before its old ones go; a cached end is added before the oldest leaves
    fixed = (2 * streams + keep + 2) * state + linear * rows * (width * 2 + nk * dk * 4 + nv * dv * 4 + nv * 8)
    slots = int(t.get("num_experts_per_tok", 1)) + 1
    intermediate = int(t.get("moe_intermediate_size", t.get("intermediate_size", d))) // world
    extent = d + int(t["vocab_size"]) // world + slots * (intermediate + d) + width + h * hd
    fixed += 16 * max(128, rows) * extent * 4 + 32 * rows * 2560 * 4
    def bytes_at(capacity: int) -> int:
        kv = attention * capacity * hk * hd * 2 * 2
        caches = (streams + keep + 1) * kv if first is None else \
            kv + (streams + keep) * attention * min(first, capacity) * hk * hd * 2 * 2
        scratch = rows * h * (hd + 2) * ((capacity + 511) // 512) * 4
        return fixed + caches + kv // max(1, attention) + scratch   # one layer's growth copy
    return Geometry(bytes_at, 1)


def indexed_stream_geometry(t: dict, streams: int, each: int, keep: int, *, mtp: bool, kv_bits: int = 16,
                            first: int = 256, prefill_rows: int = PREFILL_ROWS, world: int = 1) -> Geometry:
    """Flash Next's concurrent decoder on one GPU: per-row windows and kept snapshots sized to share one GPU."""

    linear, attention = layer_counts(t)
    d, h, hk, hd, nk, nv, dk, dv, width = _gdn_dims(t, world)
    hc = int(t.get("hc_count", 1))
    index_dim, ratio = int(t.get("indexer_head_dim", 128)), int(t.get("indexer_compress_ratio", 4))
    budget, rows = int(t.get("indexer_budget", 2048)), streams * each
    rec = linear * nv * dk * dv * 4
    conv = linear * (int(t["linear_conv_kernel_dim"]) - 1) * (2 * nk * dk + nv * dv) * 2
    tail = (int(t.get("ple_conv_kernel_size", 4)) - 1) * int(t.get("ngram_size", 3)) * hc * d * 2
    fixed = streams * (2 * rec + conv + tail + linear * each * (nk * dk * 4 + nv * dv * 4 + nv * 8))
    fixed += (keep + streams) * (rec + conv + tail + int(mtp) * hc * d * 2)  # retained plus this pass's cuts
    slots = int(t.get("num_experts_per_tok", 1)) + 1
    moe = int(t.get("moe_intermediate_size", t.get("intermediate_size", d))) // world
    extent = d * hc + int(t["vocab_size"]) // world + slots * (moe + d) + width + h * hd
    fixed += (1 + mtp) * (linear * rows * width * 2 + 32 * max(rows, 4) * 2560 * 4) + 16 * max(64, rows) * extent * 4
    fixed += prefill_rows * _indexed_prefill_row(t, world, h, hk, hd, nv, dv, width, slots, moe)
    count, row = attention + int(mtp), kv_bytes(hd, kv_bits)
    def caches(rows: int) -> int:
        return count * (2 * rows * hk * row + (rows + (rows + ratio - 1) // ratio) * index_dim * 2)

    def bytes_at(capacity: int) -> int:
        blocks = (capacity + ratio - 1) // ratio
        cache = caches(capacity) + (streams - 1) * caches(min(first, capacity))
        chunks = (min(capacity, budget + ratio - 1) + 511) // 512
        scratch = ((1 + mtp) * rows + PREFILL_ATT_ROWS) * (h * (hd + 2) * chunks + blocks + budget + ratio) * 4
        return fixed + cache + scratch
    return Geometry(bytes_at, each)


def _pattern(t: dict) -> str:
    if t.get("hybrid_override_pattern"):
        return "".join(t["hybrid_override_pattern"])
    return "".join({"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}[k] for k in t["layers_block_type"])


def hybrid_geometry(t: dict, world: int, reserve: int, *, rows: int, chunk: int, drafts: bool,
                    draft: int) -> Geometry:
    """Nemotron-H: the engine, its lazy serial twin, three prompt-end snapshots and the row buffers."""

    pattern = _pattern(t)
    nm, na = pattern.count("M"), pattern.count("*")
    d, vocab, hd = int(t["hidden_size"]), int(t["vocab_size"]), int(t.get("head_dim") or 128)
    heads, kv = int(t["num_attention_heads"]) // world, int(t["num_key_value_heads"]) // world
    mh, mhd, ms = int(t["mamba_num_heads"]) // world, int(t["mamba_head_dim"]), int(t["ssm_state_size"])
    cd = mh * mhd + 2 * (int(t["n_groups"]) // world) * ms
    proj, qkv, experts = mh * mhd + cd + mh, (heads + 2 * kv) * hd, int(t["n_routed_experts"]) + 2
    slots, width = int(t["num_experts_per_tok"]) + 2, int(t["moe_intermediate_size"])
    extent = d + proj + qkv + slots * (width + d) + experts
    state = nm * (mh * mhd * ms * 4 + (int(t["conv_kernel"]) - 1) * cd * 2
                  + 2 * rows * (2 * cd * 2 + mh * 4))
    buffers = rows * (vocab * 2 + 4 * extent * 4) + PREFILL_ROWS * (2 * d + cd + slots * (width + d) + 8 * slots) * 2
    fixed = (2 + 3) * state + 2 * buffers + 8 * max(rows, 64) * extent * 4
    fixed += PREFILL_ROWS * (d + proj + qkv + experts) * 4 * 2
    row = d // 2 + d // 64 * 4                       # a 4-bit head row with its scales and biases
    if world > 1:                                    # the rank's vocabulary scales and biases, and the partials
        fixed += vocab // world * (d // 64) * 4 + 4 * PREFILL_ROWS * d * 4
    if drafts:                                       # the draft list's tiled rows and the untile transient that builds them
        fixed += rows * d * 2 + (vocab * d * 2 if draft else 0) + (draft // world) * row
    def bytes_at(capacity: int) -> int:
        length = -(-capacity // chunk) * chunk
        cache = (2 + 3) * 2 * na * length * kv * hd * 2          # engine, serial twin and three snapshots
        cache += (1 + 3) * 2 * length * kv * hd * 2 if drafts else 0   # the MTP head's k/v and their snapshots
        scratch = (2 + int(drafts)) * rows * (length // chunk) * heads * (hd + 2) * 4
        return fixed + cache + scratch
    return Geometry(bytes_at, reserve)


def hybrid_weights(world: int):
    def transform(name: str, info: dict) -> tuple[int, int]:
        shape = list(info["shape"])
        if world > 1 and not info.get("split"):
            tiles = ".switch_mlp." in name or ".shared_experts." in name
            if tiles and (".fc1." in name or ".up_proj." in name):
                halves = 1 if ".switch_mlp." in name else 2      # the shared expert folds in as two experts
                shape[-2] = (shape[-2] // (64 * halves) + 1) // 2 * 64 * halves   # rank 0's larger tile share
            elif tiles:                                   # the down projection's input columns, by the same tiles
                unit, halves = (8 if info["dtype"] in ("U32", "I32") else 1), 1 if ".switch_mlp." in name else 2
                shape[-1] = (shape[-1] // (unit * halves) + 1) // 2 * unit * halves
            elif any(f".{p}." in name for p in ("q_proj", "k_proj", "v_proj", "in_proj", "conv1d")):
                shape[0] //= world
            elif any(f".{p}." in name for p in ("o_proj", "out_proj")):
                shape[-1] //= world
            elif name.endswith((".A_log", ".D", ".dt_bias", ".mixer.norm.weight")):
                shape[0] //= world
        cast = name.endswith((".A_log", ".D", ".dt_bias", ".e_score_correction_bias")) or ".conv1d." in name
        amount = padded(info, shape, float32=cast, name=name)
        if world > 1 and name.startswith("layers.") and shape != list(info["shape"]):
            amount += padded(info, list(info["shape"]), float32=cast, name=name)   # the MTP head is kept whole beside its split
        return amount, 0
    return transform
