# DeepSeek-V4.1-Flash on Apple Silicon (EXL3, Metal)

TensorFold serves DeepSeek-V4.1-Flash from resident EXL3 packs on Apple Silicon through the
Metal engine. **Scope: a DeepSeek-only family.** The kernels below are specific to the EXL3
2.9 bpw pack format (trellis-coded weights the GPU decodes at use time) and are tuned on one
M5 Ultra (256 GB unified memory, 80-core GPU); receipts name that hardware class and no
other checkpoint or chip is claimed.

The lane core runs one stream per slot and windows of rows, so concurrent requests keep
their own caches and Engram cursor. The served checks hold: a drafted request equals the
no-drafts reply token for token (the 20-prompt band reads 329/329), a resumed conversation
equals the fresh one, concurrent lanes equal their solo runs (12/12 under sampled draws),
and the retained prompt cache is bounded by its own budget.

## What you need

- **256 GB machine (this lane):** the 2.9 bpw pack (~197 GB) resident plus the two Engram
  tables (~189 GB, the model's external memory) streamed from disk by address. The full
  resident admission (~386 GB) needs a 512 GB-class machine; on 256 GB the tables stream.
- **512 GB machine:** the same pack fully resident with `--engram-table-dir`; the loader's
  70%-of-RAM cap applies unless `GLM_LOAD_LIMIT_GB` opts past it.
- The original release's inference directory — `tokenizer.json`, `config.json`,
  `chat_template.jinja` and the two Engram tables — beside the pack or passed with
  `--engram-table-dir` / `--engram-stream-dir`. The weights come under DeepSeek's own
  terms; the inference code is MIT and the native reference follows it (see
  THIRD_PARTY_NOTICES).
- An EXL3 pack of the checkpoint. Pack provenance is the pack's own business: read the
  `provenance/` note that ships beside it before comparing any two packs' numbers.

## The token map

The Engram tables are keyed by a compressed vocabulary derived from the release's own
tokenizer and inference config. The native engine builds the map itself — decode each id,
fall back to the raw token when the decoder emits U+FFFD, normalize through the release's
chain (NFKD, strip marks, lowercase, the whitespace replaces), dedupe — with the multipliers
drawn from NumPy's PCG64 exactly as the source does. When `engram-token-map.bin` is missing
beside the checkpoint and the original's tokenizer sits beside the tables, the engine
builds it at open; no Python step. To build it by hand:

```sh
zig build tf-dsv41-token-map && ./zig-out/bin/tf-dsv41-token-map \
  --tokenizer <orig>/tokenizer.json --config <orig>/config.json --output <pack>/engram-token-map.bin
```

## Serve

The 256 GB streamed line the campaign receipts use (drafts on; the `TF_DSV41_` prefix on
the draft flags is mandatory — the shorthand names silently serve plain):

```sh
GLM_LOAD_LIMIT_GB=240 TF_DSV41_MTP_DRAFTS=1 TF_DSV41_TAPS=1 TF_DSV41_LOOKUP_DRAFTS=3 \
tensorfold-native serve <pack> \
  --engram-stream-dir <orig> --chat-template <pack>/chat_template.jinja \
  --context 65536 --parallel 1 --prompt-cache-gib 4 \
  --host 127.0.0.1 --port 8899 --name DeepSeek-V4.1-Flash --keep-warm 0
```

The 512 GB resident line (drafts off unless the machine has the headroom; retained caches
refuse to boot in resident mode without the streamed tables):

```sh
GLM_LOAD_LIMIT_GB=430 tensorfold-native serve <pack> \
  --engram-table-dir <orig> --chat-template <orig>/chat_template.jinja \
  --context 65536 --parallel 2 --prompt-cache-gib 0 \
  --host 127.0.0.1 --port 8099 --name DeepSeek-V4.1-Flash --keep-warm 0
```

`--parallel` is the lane count: each lane is one stream with its own caches and Engram
cursor. `--prompt-cache-gib` sizes the retained prompt cache (a resumed turn processes only
the suffix — 78 of 21,966 rows in 0.85 s on the receipt prompt).

## The kernels

The lane ships **45 Metal kernel source files** (26 new for this lane, 17 evolved from
upstream's release-checkpoint line, 2 renamed splits): 31 under
`zig/kernels/metal/deepseek_v41/` and 14 in the `core/exl3_projection*` and
`core/expert_groups` group. The kernel contract registers the
family at **89 kernel variants: 89/89 pass gates 1–4, 0 fail, 186/186 planted controls
caught red, 2 documented limit records** (`tf-kernel-contract --only dsv41` on the shipped
tree). The headline tensor-unit paths report themselves at boot: `tf_exl3_matmul_prefill_nax3` /
`tf_exl3_matmul_prefill_nax31` (prompt) and `tf_exl3_matmul_decode_fma_static_lut` /
`tf_exl3_matmul_decode_fma_dense` (decode), beside the CSA2 attention, the sparse index block, the Engram gather and the
hyper-connection boundary.

## Known issues (honest)

- **Governor first-touch.** The Engram gather keeps a 64-sample latency probe: if any
  sample exceeds 20 µs the table is called cold and the prompt reads positionally,
  uncached above 256k ids. The probe warms on its own sample byte, so the FIRST pass over
  novel text pays the cold path by construction (a 35k-token first touch measured 968
  tok/s vs ~1,300 page-warm); repeats keep the mapped path. This is the measured trade
  that removed a 24 GB page-cache pollution stall, not a defect to tune around.
- **Shared rounds decode plain.** A multi-lane batch round runs the exact plain window for
  every lane (drafted requests in a shared round answer `"draft": false` with the same
  tokens); drafting is a solo-lane property.
- **Cold concurrent prefill serializes.** Four simultaneous cold 22k prefills put up to
  ~73 s of first-token wait on the last lane; only a prompt merge would compress
  it, and the merge moves the reply, so it stays closed.
- **Long-context decode falls with context.** Drafted decode holds the 30+ tok/s bar to
  ~41k and reads ~25 tok/s at 62k; the collapse is context length, not sampling.
- **The 2.8 bpw pack's duplicated tensors** are owned by the index, so a shard contributes
  only the tensors the index maps to it.
- **The pack-format ceiling is a decision, not a kernel.** The measured path to ~1,400
  prefill / 70 decode is a 3-bit affine expert repack whose quality ratio (1.095) misses
  the 1.05 band; it ships only if the maintainer accepts that band.

## The numbers

The per-operator reference tests run every stage against an FP64 ideal at three sides — kernel, FP32
source contract, FP64 ideal — and assert rounding level. The whole-model picture is
sensitivity-dominated, and the floor-relative band is not stable across the fixed release-quality
window. On the full 128-token window (`release-quality-128.json` in the reference tests) the FP32
contract chain reads 0.1622 NRMS against FP64 and the engine 0.2671, a ratio of 1.647x: the band
fails (1.05 clears, 1.10 reviews). The 16-token blocks after the first read 1.66-2.45x and the
32-token prefix 1.180x; only the first 16 tokens clear, at 1.006x (floor 0.4074, engine 0.4100),
because that block's floor is unusually noisy. The floor is not a rounding-level bound: on the first
token, with no cache history, the FP32 contract and FP64 ideal already differ by 0.519 NRMS, while
the engine differs by 0.517 from the ideal and 0.510 from the contract. The ratios therefore measure
the model's whole-model sensitivity to any arithmetic change, not an engine-only bf16 dequant class.
The 1% absolute figure is a reported trip-wire, not the arbiter. The band is a fixed-history
receipt, and the floor manifest binds the window, the exact FP32 logits and the checkpoint metadata
identity (config, index, present shard headers, token map). The 128-token window is the
qualification measurement this record reports; the 16-token window is fixed in the tree as
`release-quality-16.json`. The streamed run's execution reads also
fail the zero-disk-read residency clause (20 KB on the FP32 half, 67.8 MB on the FP64
half); the receipt now reports the Engram stream's own logical rows and syscalls beside the
total. A cold rerun of the 16-token
window on the same tree reads 8.3 MB and fails that clause too, so the earlier 0-read
receipt was page warmth, not a property of the window. The operator self-check now runs
before the measured window (`06518d523`): a one-token qualification on that head records
`measurement_scope "model_execution"` and zero execution reads on a page-warm probe, and
the receipt names the self-check it excludes. On the same head a cold 16-token engine run
reads 24.1 MB for 1,536 stream reads — 15,691 bytes a read for a 264-byte row, the page
granularity of the stream's own positional reads, not a weight re-read. A page-warm repeat
reads 0.

Context-labelled standing (M5 Ultra, 256 GB, the streamed line above; run-to-run rates
rounded): prefill ~930 tok/s warm at 3,002 tokens; plain decode ~44 tok/s at 4.5k and ~43
at 21k, flat to 42k; drafted decode ~45 at 4.5k / ~55 at 21k; batched four-lane aggregate
59.3 tok/s; a resumed 21,966-token turn reads its 78-row suffix in 0.85 s.

## Receipts

The qualification receipts — the served parity legs, the lane bit-equality, the oracle and
margin captures — live beside the PRs on the fork; the tree carries the tests that
regenerate them: the FP64 reference suite (`test-deepseek-v41-reference`), the kernel
contract (`tf-kernel-contract`), and the ten-second select-determinism probe
(`test-deepseek-v41-index-probe`).
