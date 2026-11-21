# Ling-3.0-tiny on the CPU

This page describes the native Zig engine running inclusionAI's Ling-3.0-tiny (`bailingmoe3`) on a CPU, aimed at
aarch64 Linux boards. Ling-3.0-tiny has 24 layers: 18 of KDA linear attention (16 heads, a 4-tap convolution and a
recurrent state) and an MLA layer every fourth layer (16 heads, a 512-wide latent cache), one dense SwiGLU layer, then
23 layers of 128 routed experts (8 a token, grouped sigmoid routing) beside a shared expert. Token IDs in, greedy
token IDs out: there is no server, tokenizer or sampling here yet.

## Weights

The engine reads the community Q4_K_M GGUF published with the model: `Ling-3.0-tiny-Q4_K_M.gguf` from
`inclusionAI/Ling-3.0-tiny-GGUF`, 4,823,894,944 bytes, with 272 Q4_K, 33 Q6_K, 6 Q5_0 and 215 F32 tensors. Another
publisher's file has the same name and a different size; only this one has been checked. The reader refuses a file
whose metadata, tensor names, shapes or kinds differ from this geometry.

```bash
hf download inclusionAI/Ling-3.0-tiny-GGUF Ling-3.0-tiny-Q4_K_M.gguf --local-dir ~/models
```

The file is mapped read-only. Weights stay in their 4-bit blocks and are decoded block by block inside each
projection, never expanded to an F32 model.

## Build

With Zig 0.17.0, natively on the board:

```bash
zig build tf-ling-cpu-decode
```

or from a Mac, for any aarch64 Linux board (exact mode only):

```bash
zig build tf-ling-cpu-decode -Dtarget=aarch64-linux-gnu
```

For boards with ARM dot-product support (check `grep -o asimddp /proc/cpuinfo` on the board), add `-Dcpu=baseline+dotprod` (or the board's CPU model) so `--fast` is enabled:

```bash
zig build tf-ling-cpu-decode -Dtarget=aarch64-linux-gnu -Dcpu=baseline+dotprod
```

Either way the tool is `zig-out/bin/ling-cpu-decode`, built ReleaseSafe whatever `-Doptimize` says; copy it to the
board from a Mac. `-Dtarget=aarch64-linux-musl` gives a static binary.

## Run

Exact mode (default):

```bash
ling-cpu-decode ~/models/Ling-3.0-tiny-Q4_K_M.gguf "$PROMPT_IDS" 32 4
```

Fast mode (requires a dot-product build; uses integer SDOT, weights repacked eight rows at a time and cached next to the GGUF as `.tf_repack8.v5`, and head-parallel fast attention):

```bash
ling-cpu-decode --fast ~/models/Ling-3.0-tiny-Q4_K_M.gguf "$PROMPT_IDS" 32 4
```

On a `.tf_repack8.v5` cache hit, `--fast` verifies the cache header, entry table bounds, and the GGUF's size, inode, mtime, and ctime against the header without re-hashing the GGUF or re-scanning raw quantized scales. The cache is built from validated blocks and stores the GGUF's SHA-256 in its header; any change to the file's stat identity (or removing the cache file) triggers a rebuild.

The arguments are the GGUF, the prompt's token IDs separated by commas (`PROMPT_IDS=ID,ID,...`), the number of tokens
to generate, and the thread count (1 to 8, default 1). The output has this shape:

```
model opened in S s, 4 threads, exact
prefill P tokens in S s (R tokens/s)
generated: ID ID ID ...
decode 32 tokens in S s (R tokens/s), finish length
```

Decoding stops early, with `finish stop`, at the GGUF's end-of-sequence tokens. A prompt and its generated tokens fit
in 4,096 positions. The tool does not tokenize: take the IDs from any tokenizer for Ling-3.0-tiny, for example
llama.cpp's `llama-tokenize` on the same file.

- **Exact.** Every projection equals a scalar block-decode reference bit for bit, and threads split output rows or
  whole experts only, so 1 to 8 threads give the same token IDs. Activations round to BF16 where the reference
  model does.
- **Prompt rows.** The prompt runs 8 rows a call; whole, chunked and one-row prompts give the same logits.
- **Fast mode is not exact.** `--fast` rounds activations to 8-bit integers for SDOT, so each projection differs from
  the exact one by about 0.4% relative (1.4% at worst), and the fast attention rounds at the BF16 boundary. Its token
  IDs can differ from exact mode's after some tokens; it is opt-in for that reason.
- **Threads.** Persistent workers share each projection's rows and each token's eight routed experts, so more threads
  help up to the number of fast cores. Rows are claimed in chunks, so a slower core takes fewer, but an expert on a
  slow core is still waited for: start with the count of the board's fastest cores and try one more or fewer. Workers
  spin between projections and sleep only when idle, so while decoding each thread keeps a core busy; leave a core
  free if the board has other work. The engine does not pin threads itself. On a board with mixed cores, run it under
  `taskset -c LIST` with `LIST` set to the fast cores (on the Qualcomm board measured below, `taskset -c 0,3,5,7` with
  4 threads); the cores with the highest `/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq` are the fast ones.

## Measured

FIGURES_AT_FINAL_HEAD

## Checks

`zig build test-ling-cpu-host` runs the reader, kernel and forward checks on any host: decoded blocks against the
independent FP64 source definition, projections against expanded weights at 1 to 8 threads, and the forward on a
small synthetic model, whose whole, chunked and one-row prompts at 1, 2, 3, 4 and 8 threads give the same bytes; the
workers' rows and experts are also checked against the one-thread bytes at those thread counts.
