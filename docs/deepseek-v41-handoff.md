# DeepSeek-V4.1 prefill hand-off (CUDA to Metal)

A GB10 pair prefills a prompt; a Mac decodes it. The hand-off is the state a prefill leaves behind, in a neutral
bf16/f32 form that is converted to the Metal engine's own cache layout on arrival.
The library is `zig/src/families/deepseek_v41/handoff/`; it is host-only and independent of the transport.

## What travels

Everything is bf16 or f32 in a neutral form; no CUDA or Metal byte format crosses the link.

| part | layers | dtype, shape | content |
|---|---|---|---|
| `swa` | every backbone layer (0..39) | bf16 `[min(upto,128), 512]` | the last window's keys, oldest first (positions `upto-rows .. upto-1`), normed and roped, fp8 NoPE dequantised exactly |
| `carry` | the ratio-2 KV sources (2, 8, 14) | f32 `[2, 512]` | the compressor's open-group row of position `upto-1`: kv then gate |
| `ring` | the DSpark blocks (40..42) | bf16 `[rows, 512]` | the drafter's last keys, `rows = min(upto, 128)`; optional |
| `comp@P` | the KV sources (2, 8, 14, 20) | bf16 `[n, 512]` | compressed rows `P*page_rows ..` of the paged pool, exact |
| `ixk@P` | the KV sources | bf16 `[n, 128]` | index keys of the same rows, exact |

`comp`/`ixk` windows are whole pages cut to at most 2 MiB (one MKVH frame each); `P` is the first logical page. The
mailbox frame limit is the reason for the cut: a part is one frame. The manifest is one `recurrent` layer per model
layer, `parts` in the order above, dims `{tap, channel}`; `prompt_tokens` is `upto`, `tp_rank` / `tp_size` name the
shard. Nothing is added to the wire: no new dtype, dim name or protocol.

Not shipped: token ids (the decoder holds the prompt, and the manifest's `token_sha256` is verified; Engram's lookback ids
are the last 3 prompt tokens), the position (it is `upto`), the CED stash (the decoder recomputes it), fp8 bytes.

## Why bf16

CUDA stores comp rows as 584 bytes (448 e4m3 NoPE codes with a UE8M0 scale per 64 dims, 64 bf16 RoPE dims, a pad),
index keys as 132 bytes (128 e4m3 and a power-of-two fp32 scale). The Metal engines store other formats:

| engine | window | pool row | index key |
|---|---|---|---|
| release lane (`metal/kv.zig`) | bf16, mirrored over `window + 16` slots | bf16 | 68 B (E2M1 + E8M0, 32-blocks) |
| served lane (the EXL3 Metal lane) | 528 B (fp8 e4m3 + UE8M0, 32-blocks), slot `pos % 128` | 288 B (E2M1 + E4M3, 16-blocks) | 68 B |

So the conversion is not a memcpy. CUDA fp8 decodes to bf16 exactly (a product of an e4m3 value and a power of two), the
neutral form loses nothing, and the importer packs it with scalar mirrors of each Metal packer (`fp4.zig`). A packer
returns `error.NotExact` instead of a row the lane's format would hold with other values, so a hand-off lands only when
it is lossless there: an fp4 pool row, an fp8 window row and an fp4 index key each take a record whose values are
already on that format's grid, and one that is not is refused rather than rounded.

## Tests

`zig build test-dsv41-handoff` (host only): the CUDA record codec exhaustively against independent tables, the FP4 packers
against independent f64 references and the part plan's partitions. `compile-dsv41-handoff-aarch64-linux` / `-macos` compile the
tests for the targets.
