# EXL3 weights on CUDA, for any family

ExLlamaV3's trellis quantization (EXL3) is what the community's best 4-bit and low-bit NVIDIA checkpoints ship as:
`mul1` codebook at 2 to 6 bits for MiMo-V2.6-Flash, Qwen3.8-27B and Flash Next, `mcg` at 4 bits for
GLM-5.3-Flash. TensorFold reads all of it with one module, so a family's engine does not need its own EXL3
kernel:

- `src/tensorfold/cuda/exl3/format.py` — the format, per-tensor metadata out of the safetensors headers, and a
  numpy reference decoder;
- `src/tensorfold/cuda/exl3/decode.cuh` — header-only device functions that decode a 16x16 tile of any codebook
  and width straight into the B fragments of two `mma.m16n8k16`;
- `src/tensorfold/cuda/exl3/linear.py`, `linear.cu` — a row-invariant linear layer for 1 to 128 rows;
- `python -m tensorfold.cuda.exl3.inspect MODEL_DIR` — what a checkpoint holds, from its headers alone.

The GLM-5.3-Flash engine keeps its own tuned `families/glm5_next/cuda/exl3.py` (4-bit `mcg`, routed experts
only); it stays as it is.

## What a family has to say

A family whose CUDA engine reads every codebook and width declares it, and TensorFold checks the checkpoint's
`config.json` before downloading anything:

```python
QUANT_METHODS = {"cuda": ("mlx", "exl3")}
EXL3_VARIANT = "any"          # any codebook and any width of BITS (the average bits may be fractional)
```

A family that reads one variant declares it as a dict instead (`families/glm5_next`) and checks it in its own
`check()`. The per-tensor widths are the module's business, not the config's: one checkpoint mixes them across
layers, projections and the experts of one MoE layer, and `config.json`'s `bits` is only the model's average
(`head_bits` is the head's, `mtp_bits` the MTP head's).

## The format, in one paragraph

A quantized linear layer with K inputs and N outputs (both multiples of 128; ExLlamaV3 pads) is a group of
tensors under one prefix: `trellis` int16 `[K/16, N/16, 16 * bits]`, `suh` fp16 `[K]` (or the packed sign words
`su` int16 `[K/16]` of older checkpoints), `svh`/`sv` for the outputs, an optional fp16 `bias`, and a zero-size
marker tensor `mcg` or `mul1` naming the codebook (no marker: the original `3inst`). Each tile holds 256 values
in a circular bitstream; the codebook maps a 16-bit state to an fp16 value; the tiles' layout is the tensor-core
permutation. The exact bit arithmetic follows; `tests/test_exl3_format.py` checks `format.py`'s decoder against
one written from this description, bit by bit, for all 27 codebook-and-width combinations.

```text
The tile. A tile holds 256 values in R = 256 * bits bits (16 * bits int16 words). Its int16 words, read in pairs as
little-endian 32-bit words, form a circular bitstream of R bits read from the most significant bit of each 32-bit
word. Value p of the tile (p = 0..255) is decoded from the 16 bits of the stream that end at bit E(p) (exclusive,
wrapping around past the tile's last bit), taken as an unsigned integer s, first bit most significant:

    E(p) = (p + 1) * bits                                   integer bits
    E(p) = ((p + 1) * (2 * KA + 1) - ((p + 1) % 2)) / 2     bits = KA + 1/2: positions alternate KA and KA + 1
                                                            new bits, the odd positions taking the extra bit

and the codebook maps the 16-bit state s to an fp16 value:

    3inst  x = (s * 89226354 + 64248484) mod 2^32; x = (x & 0x8FFF8FFF) ^ 0x3B603B60;
           value = fp16(x & 0xFFFF) + fp16(x >> 16)         one fp16 addition, rounded to nearest even
    mcg    x = s * 0xCBAC1FED mod 2^32, then as 3inst
    mul1   x = s * 0x83DCD12D mod 2^32; h = 1024 + (the sum of x's four bytes);
           value = h * fp16(0x1EEE) + fp16(0xC931)          one fused multiply-add, rounded once (1/147.7, -10.39)

Value p lands in its tile at row 2 * (l % 4) + (j & 1) + 8 * ((j >> 1) & 1), column l // 4 + 8 * (j >> 2), where
l = p // 8 and j = p % 8: lane l of a warp holds values 8l..8l+7, exactly its B fragments of the two
mma.m16n8k16 of the tile (columns 0-7 and 8-15). The tiles make W_q [K, N], the weight in the rotated domain.

The layer. With H the 128x128 Sylvester Hadamard matrix scaled by 1/sqrt(128), applied to each block of 128 inputs
or outputs,

    y = x @ W + bias,   W = diag(suh) @ H_K @ W_q @ H_N @ diag(svh),
    so  y = ((((x * suh) @ H_K) @ W_q) @ H_N) * svh + bias

(``out_scales`` in the config only says whether the quantizer folded per-column scales into svh; the formula is the
same.) Splitting a layer keeps whole tiles and whole Hadamard blocks: by outputs, take columns of tiles and of svh
and all of suh; by inputs, rows of tiles and of suh and all of svh, and add the partial outputs.
```

## Using it

```python
from tensorfold.cuda.exl3 import format as exl3_format
from tensorfold.cuda.exl3.linear import Exl3Linear

layer = Exl3Linear.load(model_dir, "model.language_model.layers.0.self_attn.q_proj")
y = layer(x)                                  # x [1..128, K] fp16/bf16/fp32 -> y, same dtype
```

`load` reads one group with safetensors and copies the trellis, never changing it, into the read order the
kernel wants (`layout="strips"`: a 128-column block's tiles in k order, so a warp walks contiguous words).
`layout="stored"` reads the checkpoint's own order instead; both give the same bits.

To serve a whole checkpoint, walk `format.scan(model_dir).groups`, build one `Exl3Linear` per group, and treat
the checkpoint's plain tensors (`checkpoint.plain`: embeddings, norms, routers, sometimes a head) as ordinary
weights of the family's engine. `python -m tensorfold.cuda.exl3.inspect MODEL_DIR` prints exactly that split,
the bits per category, and every group that does not parse.

## The layer's arithmetic

```
y = ((((x * suh) @ H_K) @ W_q) @ H_N) * svh + bias        H = the 128x128 Hadamard / sqrt(128)
```

`rot_in` rotates the input once per call (fp16 out, as ExLlamaV3 does), `linear` decodes each k tile straight
into tensor-core fragments and multiplies in fp32. Rows are independent by construction: the k ranges of warps
and of K splits depend only on (K, N) (`plan(k, n)`), every sum runs in a fixed order, and `mma.m16n8k16` keeps
its rows independent — which is the verify path's contract (`docs/recipes/cuda.md`).

## Prompts

Prompt chunks take their own arithmetic, as the MLX 4-bit path's prompt matmul does (`cuda/exl3/prefill.py`): the
input rotation is decode's, W_q is decoded once a chunk into fp16, a fixed-tile fp16 GEMM with fp32 accumulation
multiplies it, and its epilogue rotates each 128-column block (the accumulator's bf16 high and low halves times
H) before `svh` and the bias. Tiles depend on the shape alone, so a row's bits never depend on its chunk and a
resumed prompt equals a fresh one; they differ from decode's, so the engines keep prompt ends and prefill a reply
again, as for the MLX checkpoints. A family's head, read one row a prompt, keeps the decode linear.

## Numbers

`tools/bench_exl3.py` prints this table. It cycles over enough weight copies to exceed L2, so these are DRAM
figures, medians of 9 runs of a CUDA graph of back-to-back calls on an idle DGX Spark (GB10), the same tensors
through ExLlamaV3's own `LinearEXL3`:

| MiMo-V2.6-Flash 2.50bpw tensor | bits | bytes | rows | TensorFold | ExLlamaV3 |
| --- | --- | --- | --- | --- | --- |
| `layers.0.self_attn.q_proj` | 2 | 12.6 MB | 1 | 57.0 us (221 GB/s) | 59.0 us (213 GB/s) |
| `layers.0.self_attn.q_proj` | 2 | 12.6 MB | 16 | 60.0 us (210 GB/s) | 65.9 us (191 GB/s) |
| `layers.2.self_attn.o_proj` | 3 | 12.6 MB | 1 | 72.9 us (173 GB/s) | 73.0 us (172 GB/s) |
| `layers.2.self_attn.o_proj` | 3 | 12.6 MB | 16 | 68.4 us (184 GB/s) | 63.5 us (198 GB/s) |
| `layers.41.self_attn.o_proj` | 6 | 25.2 MB | 1 | 114.0 us (221 GB/s) | 113.5 us (222 GB/s) |
| `layers.41.self_attn.o_proj` | 6 | 25.2 MB | 16 | 123.7 us (203 GB/s) | 116.0 us (217 GB/s) |
| `lm_head` | 6 | 468.7 MB | 1 | 2074.5 us (226 GB/s) | 2071.8 us (226 GB/s) |
| `lm_head` | 6 | 468.7 MB | 16 | 2113.0 us (222 GB/s) | 2045.8 us (229 GB/s) |

Rows cost little: every tensor above is within 5% from 1 row to 16, because one program's weights are decoded
once and reused by all its rows. 6-bit is at ExLlamaV3's level (221-226 GB/s of a 237 GB/s streaming ceiling on
this box) and so is 2-bit at 1 row (221 vs 213 GB/s). 2 bits is where the decode itself is the work (four weights
per byte), so the loader there has a warp fetch a k step's words as one coalesced run and hand each lane its two
words of a 16x16 tile by shuffle, and keeps the next k step in flight while the current one decodes; that is worth
40% at 2 bits (q_proj 156 -> 221 GB/s at 1 row, o_proj 76 -> 187) and 10-15% at 3 to 5 bits. Above 6 bits a step
is 224 or 256 B of words a warp and prefetching it measured 1-2% slower than loading per tile, so those widths
load per tile. What is left is the small-N 3-bit shapes at 16 rows (o_proj 184 vs 198 GB/s) and, on the 27B packs,
rows 8 and 16 of the 3-bit shapes, where the split a shape alone picks is 3-19% off the best cell of the
(K split, warps) grid. An earlier version of this kernel streamed the tiles through shared memory with `cp.async`; that was
three to five times slower than these per-lane cached loads on GB10, whose L1 serves a warp's 256-byte search
from one line.

## Traps

- `config.json`'s `bits` is the average, `head_bits` the head's. Read the width from each tensor's trellis
  (`16 * bits` words per tile), never from the config.
- Half bits (1.5, 2.5, 3.5) exist only with the `mul1` codebook, and their tiles are not a whole number of
  bits per value: the bitstream alternates KA and KA + 1 bits, so the window ends are `E(p)` of `format.py`,
  not `(p + 1) * bits`.
- `suh`/`svh` are scales, not signs: the sign of a weight lives in its 16-bit state. The packed `su`/`sv` words
  of older checkpoints are signs only (`1` bit set = -1), and `format.unpack_signs` expands them.
- A model may keep some tensors unquantized (a head, an MTP head, embeddings, norms): `scan` lists them under
  `plain`.
- The Hadamard blocks run along K and N, so both must be multiples of 128. A split of K must keep whole tiles
  and whole blocks: `plan` only ever splits 128-aligned k ranges.

### Consolidated Flash Next n-grams

Flash Next EXL3 packs may store `ngram_embedding.trellis` as one int16 `[rows, words]` tensor instead of
`ngram_embedding.shard_N.trellis`. Both layouts use the same scale-plus-160-values row codec. CUDA maps either
layout read-only and gathers only requested rows; the consolidated layout does not require re-sharding or
requantization. Admission treats the packed table as reclaimable mapped pages, while head metadata remains
resident. Mapped pages still consume physical RAM when touched; keep the normal cache/workspace reserve.
