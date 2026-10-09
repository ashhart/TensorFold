# Intel Arc (XPU): Nemotron 3.5 Lightning

Status: experimental. One checkpoint, one card class, one request at a time. Everything below was measured on the setup in
[Tested setup](#tested-setup); nothing is claimed for other cards.

The Intel GPU backend runs OpenCL C kernels compiled to SPIR-V (`ocloc`) through Level Zero. It is a separate Linux build
(`-Dxpu`), selected with `--backend xpu` on the native server, and has no CUDA or Metal code in its path.

## Supported

| Item | State |
| --- | --- |
| Model | Nemotron 3.5 Lightning 30B-A3B, MLX affine 4-bit, group 64 (the checkpoint named in [nemotron-3.5.md](nemotron-3.5.md)) |
| Card | Intel Arc Pro B70 (Battlemage, 32 GB) |
| Not supported | NVFP4 and FP8-Mamba checkpoints: refused at load with `NVFP4 checkpoints are not supported on the XPU backend yet; use the MLX 4-bit checkpoint`. They need FP4/FP8 dequantisation kernels that do not exist yet. |
| Not supported | MTP or copy drafts (every request decodes serially), tool-call gates, grammars and structured output, prompt cache and prefix reuse, concurrent decoding (requests queue and run one at a time) |
| Sampling | Greedy uses the device argmax. A request with temperature above 0 is drawn on the host from the step's logits with the lane core's keyed sampler: temperature, `top_k`, `top_p`, `min_p` and `seed`; the draw is keyed by seed, absolute token position and candidate id. Presence and repetition penalties are not implemented. |
| Qwen3.8 | See [Qwen3.8-27B](#qwen38-27b-experimental): MLX 4-bit and EXL3 through the CLI and the server, GGUF through the CLI. |

## Build and run

Needs Zig 0.17.0, `ocloc` and the Level Zero loader with the Intel compute runtime, on Linux with the `xe` driver.
`zig build` defaults to `-Doptimize=fast` (ReleaseFast); every number on this page was measured with it, and a debug build is much slower.

```bash
zig build -Dxpu                 # tensorfold-xpu (token-id CLI) and tf-xpu-test
zig build -Dxpu native-xpu      # zig-out/native-xpu/bin/tensorfold-native with the Intel GPU engine
zig build -Dxpu xpu-tests       # tf-xpu-<name>-test programs (checks and benchmarks)

tensorfold-xpu devices
tensorfold-xpu run MODEL --tokens-file ids.txt --max-tokens 128 --prefill 512
zig-out/native-xpu/bin/tensorfold-native serve MODEL --backend xpu --context 32768
zig-out/native-xpu/bin/tensorfold-native capabilities --json
```

`--prefill ROWS` prefills the prompt in windows of ROWS tokens (16 and up); the plain loop then decodes. `--context` defaults
to the model window capped at 32768 in `serve`. `tensorfold-native capabilities --json` reports `"backends": ["xpu"]`, the
chip class `intel-<PCI device id>` (`intel-e223` here) and the family `nemotron_h` with format `mlx-q4g64`.

Run model programs one at a time: `tools/zig/xpu_guard.sh SECONDS COMMAND...` takes the GPU lock, refuses to start while other
processes hold more than 8 GB of device memory and stops the command only with SIGINT.

## Tested setup

Intel Arc Pro B70 Graphics (32.5 GB maximum single allocation), Linux 7.0.0 (`xe` driver), Zig 0.17.0, a local copy of the
Nemotron MLX 4-bit checkpoint (`TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit`, revision `d9d758fb83953437f7263256b0d96157e2a348b8`,
729 tensors, 17.27 GiB of weights). The commit under test is the head of the pull request.

## Measurements

Decode, 128 greedy tokens from a 5-token prompt: 5.72 ms/token, 175 tok/s (`tensorfold-xpu run`, five runs on different
commits: 173.6 to 174.8). Through the native server (the engine runs behind the lane core), greedy, 256 tokens, `tools/bench_openai.py`: 172.5 tok/s
(raw code prompt) and 172.5 tok/s (chat prose prompt, thinking off), first token after 0.09 to 0.10 s.

Concurrency (`tools/bench_concurrent.py --levels 1,4 --alone --serial`, 256 tokens, temperature 0): one stream 171.9 tok/s (code) and 172.3 (chat);
four streams 164.3 and 163.9 tok/s aggregate (each reply took its turn at about 171 tok/s; the last of four waited about 5 s
for its first token). All 12 four-stream replies equalled their solo run, and 3 of 3 for one stream; 0 failed.

Cold prefill through `/v1/completions` with token-id prompts (the first L tokens of a public text under the model's
tokenizer), `--prompt-cache-gib 0`, `cached_tokens` 0, 8 tokens generated:

| Prompt tokens | Prefill s | Prefill tok/s | Decode tok/s after it |
| ---: | ---: | ---: | ---: |
| 2,048 | 0.66 | 3,108 | 173.2 |
| 8,192 | 2.65 | 3,086 | 168.9 |
| 32,768 | 11.3 | 2,888 | 160.3 |
| 65,536 | 24.7 | 2,652 | 151.5 |
| 131,072 | 57.4 | 2,281 | 137.2 |
| 262,080 (the native window less a reply) | 147.5 | 1,777 | 113.6 |

Against llama.cpp on the same card (build 9c2e0e491, Vulkan, Mesa 25.2.8, Q4_0 GGUF of 17.59 GiB, `-ngl 99 -fa on`; a different
quantization from the MLX 4-bit checkpoint above, so this compares engines as users run them, not arithmetic). `llama-server`
with `-c 131584 -np 1`, f16 KV cache, prompt cache off, the same token arrays, 8 tokens generated (32 for the 64k and 128k rows; the first request after
the start of the llama.cpp server took 2.28 s at 2k tokens with warm-up):

| Prompt tokens | llama.cpp prefill s | This engine prefill s | llama.cpp decode tok/s after it | This engine decode tok/s after it |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1.57 | 0.66 | 44.5 | 173.2 |
| 8,192 | 6.93 | 2.65 | 40.5 | 168.9 |
| 32,768 | 41.8 | 11.3 | 31.0 | 160.3 |
| 65,536 | 126.5 (518 tok/s) | 24.7 | 23.4 | 151.5 |
| 131,072 | 432.2 (303 tok/s) | 57.4 | 15.6 | 137.2 |

Peak device memory of the llama.cpp server was 19.0 GB in a run at `-c 33280` and 19.70 GB over the 64k and 128k run at `-c 131584`
(one peak for the whole run); nothing failed or ran out of memory.
`llama-bench` on the same file gives prompt processing of 1,401.8 tok/s at 2,048 tokens, 1,220.2 at 8,192 and 796.4 at 32,768,
33.0 tok/s for 256 generated tokens and 24.05 tok/s for 128 tokens at 32k depth (a q8 KV cache was slower: 27.7 for 256 tokens).

Prefill throughput falls from 3.1k tok/s at 2k tokens to 2.3k at 128k and 1.8k at the native window as attention grows with the context. The prompt attention
runs on the matrix engine (`nem_attn_pfs.cl`; `NEM_OLD_ATTN=1` selects the earlier per-row kernels, which took 3.45 s, 26.6 s and 389.5 s for prompts of 8,192, 32,768 and 131,000 tokens against 2.62 s, 11.3 s and 57.5 s with the
matrix-engine kernel, in 512-row windows, with identical tokens afterwards). Against the per-row kernels its output differs by at most 5.6e-3 of the largest output magnitude (bf16
outputs), is identical for any chunking of the window, and the reply tokens of the 8k, 32k and 128k prompts did not change
(same digests as with the per-row kernels). In the CLI, an 8,192-token prompt takes 51.4 s token by token, 7.5 s in 128-row
windows and 2.65 s in 512-row windows, with identical tokens afterwards; a 32,768-token prompt takes 11.3 s in 512-row windows.

Decode attention runs on the matrix engine as well (`nem_attn_dec.cl`: split-K partials of 512 keys, then a parallel merge of the chunks;
`NEM_OLD_DEC=1` selects the earlier per-row pair). It reads the KV cache about four times faster than the earlier kernel (462 against 119 GB/s); one layer at 131,000 keys
takes 0.29 ms instead of 1.13 ms (`tf-xpu-nem_attn_dec-test`, random data, one row). A window of up to 16 rows uses the same kernel, so a window row stays bit-identical to the token decoded
alone. Against an FP64 reference on random data its error is in the same bf16 rounding class as the earlier kernel (1.8e-3 to 3.3e-3 of
the largest output magnitude against 2.1e-3 to 3.4e-3). Decode at 8,192 and 131,072 tokens of context (the needle runs of `tensorfold-xpu`) went from 146.6 and 82.6 tok/s with `NEM_OLD_DEC=1` to 171.1 and 137.8.

Long context: a needle (a secret code planted at 50% depth in public text) was found at 8,192 tokens (prefill 3,103 tok/s,
decode 171.1 tok/s) and at 131,072 tokens (prefill 2,285 tok/s, 62 s wall, decode 137.8 tok/s).

Memory, measured on the 32 GB card in a host with 62.2 GiB of RAM and 24 threads (kernel 7.0.0, `tensorfold-native serve`,
KV caches and scratch allocated at load for the chosen context): device allocation counter 17.85 GB for the weights, 18.53 GB
at `--context 4096`, 19.37 GB at 131,200 and 20.25 GB at 262,144 (the native window), unchanged by the prompts that follow.
Host process: 339 MB resident after load, 355 MB at its peak after a 262,080-token prompt (VmHWM); the engine pins only its
staging buffers. Load is ready in 7 s with the checkpoint in the page cache. After SIGINT the server exits in 0.2 s with status
0 and no device memory left. The weights alone are 17.3 GiB, so this model does not fit a 16 GB card; no 16 GB claim is made.

## Accuracy

Against the CUDA reference of the same checkpoint (teacher-forced, the reference's own tokens fed):

| Check | Result |
| --- | --- |
| 4 prompts x 48 steps, top-1 | 185 of 192 (188 with `NEM_OLD_DEC=1`; every miss is a near tie, see below) |
| 15 prompts, 1,440 positions, plain decode | top-1 1,421 (98.7%), mean abs log-probability difference 0.0233, 95th percentile 0.109, max 0.514 (1,422, 0.0232, 0.105, 0.750 with `NEM_OLD_DEC=1`) |
| same, prompt in 512-row windows | 1,420 (98.6%), mean 0.0225, max 0.417 |
| same, 128-row windows, 2 prompts (192 positions) | 187 (97.4%) |

The 19 top-1 misses of the plain run are listed with both margins by `WIDE_LIST=file tools/zig/xpu_compare_wide.py plain`:
all have a reference margin (top-1 over top-2) of at most 0.375 and a gap on our side of at most 0.625 (log-probability steps at this
precision are 0.125), and 13 of the 19 are within 0.25 on both sides. The earlier outlier of the run with the per-row decode
attention (prompt 12, step 53: reference margin 1.375, ours 0.125 the other way) is gone; with `NEM_OLD_DEC=1` it is still there. It came from the state the single-row
kernels built over the 1,000-token prompt, not from the step's own kernels, and the model is sensitive to the order of accumulation
between its decode and prefill paths at such positions; the reference engine shows the same sensitivity (see the parity results below).
On the 4-prompt gate the new decode attention has 7 misses against 4 with `NEM_OLD_DEC=1`; each of the 7 has a reference margin of
at most 0.25 and our gap at most 0.375, two of them exact ties on our side (a tie goes to the lower token id, the reference's
choice differs). The greedy reply to a test prompt differs by one word between the two kernels (the digests differ); each kernel is
deterministic.

KL divergence against BF16 (llama.cpp scoring, same public corpus, 16-row windows): the full 360-chunk run (91,800 scored tokens) gave
mean KLD 0.0775, same top-1 89.84%, perplexity ratio 1.0591; it was measured with an earlier build of the same kernels.
On this branch the first 64 chunks (16,320 scored tokens) give mean KLD
0.0864 +- 0.0011, same top-1 87.24 +- 0.26%, 334.5 tok/s over the scored windows, and every one of the 16,320 per-row KLD values
is bit-identical to the earlier build's.

Generation parity against TensorFold 0.6.4 on CUDA (41 prompts, 200 greedy tokens each, prompt in 512-row windows): 14 of 41
replies are identical token for token; 25 of the 27 first divergences are exact bf16 ties or within 0.25 logit, the other two
are 0.5 and 1.125. Teacher-forced on the reference's tokens, 5,098 of 5,159 positions agree (98.8%); 49 of the 61 mismatches
are ties or within 0.25. The other 12 (the same kind as wide prompt 12 step 53 above) have a reference margin of 0.375 to 2.5
where our gap is 0.125; feeding the same prompt and the reference's tokens to the CUDA engine as one prefill, its prefill path
sided with our order or tied at three of them (short3@54, qa2@34, qa4@87) and with its own decode order at four (short3@96,
short3@155, summ1@64, code3@142), which means the reference's own decode and prefill paths differ from each other by up to
about 1.5 logit at such positions. Our engine is deterministic (41 of 41 reruns identical).

Projection classes against an independent FP64 reference (`tf-xpu-nem_fp64-test`: real 4-bit codes dequantised exactly in
f64, random bf16 rows shaped like each layer's input, first and last layer of each kind, decode, 8-row window and 64-row
prefill kernels separately; every class is within the test's bounds):

| Output | Classes | Aggregate error | Worst column (relative to that column's max) | Notes |
| --- | --- | ---: | ---: | --- |
| bf16 | Mamba in/out, attention q/k/v/o, router gate, embed | 1.6e-3 | 3.9e-3 | at most 0.54 ulp; 99.8-100% of outputs equal the correctly rounded f64 value |
| bf16, relu2 | shared up, routed fc1 | 3.5-3.8e-3 | 1.1e-2 | 74-77% exact: the pre-activation is rounded to bf16 before squaring |
| fp32 | shared down, routed fc2, lm_head, single-row kernels | 5e-8 to 1e-7 | 6e-7 | |
| fp32 | the same classes, matrix-engine GEMMs (window, prefill) | 1.4e-6 to 6.9e-6 | 3.6e-5 | fp32 accumulation in the matrix engine |

## Exactness

- Window rows: windows of 2, 3, 4, 8 and 16 rows give bit-identical logits to one-row windows over 40 tokens, including a
  pending-then-commit case (`tf-xpu-nem_rows-test`).
- Prompt chunk size: prefill in chunks of 32 to 512 rows gives bit-identical last-token logits (`tf-xpu-nem_prefill-test`).
  The Mamba conv state, SSM state and the KV bytes below the prompt are byte-identical for chunks of 64, 128 and 512, and the
  decode logits that follow are identical (`tf-xpu-nem_seam-test`, 640 tokens, real text and random ids).
- Not identical, by design: the prefill GEMM path and the single-row path use different accumulation order. After 640 prompt
  tokens the prefilled state differs from the row-by-row state at rounding scale (for real text the worst conv difference is
  0.68 against a maximum magnitude of 33, SSM 1.95 against 5,710, KV 1.8 against 24), and the next 8 decode steps differ by up
  to 0.94 in logit with the same top-1 on real text (6 of 8 on random ids, whose distributions are nearly flat). Windows
  followed by the plain decode loop differ from windows followed by one-row windows by up to 0.38 (text) in logit.
- Queued requests: concurrent replies equal their solo runs (above). A client that disconnects mid-stream cancels its request
  and the next request gives the same tokens as before; seeded sampling repeats exactly; SIGINT or SIGTERM during a generation ends
  the running stream after its current token with an error event (`ShuttingDown`) and `[DONE]`, answers a queued
  request with the error `the server is shutting down`, exits with status 0 within 0.2 s and leaves no device memory held.
- Host-side slot ring: forwards queued without a sync equal synced forwards (`tf-xpu-nem_slot-test`).

## Tests

`tools/zig/xpu_compare_ref.py` (forced tokens against the reference), `xpu_compare_wide.py`, `xpu_nem_needle.py`,
and the `tf-xpu-*-test` programs (`runner-all`, `nem_rows`, `nem_prefill`, `nem_slot`, `nem_seam`,
`nem_fp64`, `nem_format`, `panic`). `zig build test` passes except `direct_io: direct reads return the same bytes as buffered
reads`, which fails the same way on an unmodified upstream 1.0.2 checkout on this machine.

## Known limits

- One request at a time; no drafts; about 172 tok/s at short context, 137 tok/s at 128k and 113 tok/s at 256k (decode).
- Prefill throughput falls from 3.1k tok/s at 2k tokens to 2.3k tok/s at 128k and 1.8k at the native window.
- A shutdown during a generation ends the stream with an error event rather than a finish reason.
- The 16-row and 1-row decode kernels and the prefill GEMMs are not bit-identical to each other (see Exactness).
- Single card, single process: the 17.3 GiB of weights leave about 14 GB on a 32 GB card.


## Qwen3.8-27B (experimental)

The same runtime also runs Qwen3.8-27B (Gated DeltaNet + attention, 64 layers) from three weight formats behind one linear layer: MLX affine 4-bit
and EXL3 directories (CLI and native server), and GGUF files (CLI only: the native server resolves directories, so a `.gguf` file is refused there).
Everything below was measured on the same card, host and Zig as above, with the KV cache in q8 unless a row says otherwise.

| Format | Checkpoint | Device memory at load |
| --- | --- | --- |
| MLX 4-bit | `TensorFold/Qwen3.8-27B-MLX-4bit`, revision `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c` | 15.7 GB |
| EXL3 3.00 bpw | `turboderp/Qwen3.8-27B-exl3`, branch `3.00bpw`, revision `6fe61ad620abfe97c5b49f9722c2bceeea4ccc28` | 13.2 GB |
| GGUF Q5_K_XL | `Qwen3.8-27B-UD-Q5_K_XL.gguf`, the Unsloth UD-Q5_K_XL GGUF as downloaded on 2026-09-12 (20,876,938,144 bytes, sha256 `8601193d3d5760c37fb8ce1b43afebc69df5fb24e1fbc5a547c32e2200305276`); repository and revision were not recorded | 21.1 GB |
| GGUF IQ3_S mix | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`, file `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | tested in the gates below |

- Recipe C (used below for the long-context and KV work; **not a published checkpoint**): an importance-weighted `llama-quantize` mix of the Qwen3.8-27B BF16 GGUF, 12.56 GB, built with a 1.63 MB importance matrix
  (wikitext-2 train, llama.cpp sources and docs, the Python standard library, synthetic math text) and per-tensor type overrides picked from a 62-probe sensitivity sweep (more IQ4_XS in the middle and late layer groups, the
  feed-forward tensors of the first third IQ2_S, `token_embd` IQ3_S, `ssm_alpha`/`ssm_beta` Q6_K, the output head Q4_K). The per-tensor type table and the sweep are not shipped with this tree, so the mix cannot be rebuilt from it; nothing in the tests needs it.
- GGUF block types: Q2_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS and IQ4_NL, with bf16/fp16/f32 tensors.
- Context: at most 131,072 positions. The model's configuration names 262,144; the engine does not hold that and refuses a larger `--context`
  with a message saying so. No claim is made for the 262,144 window.
- KV cache: q8 is the default (`--kv` in the CLI, the environment variable `TENSORFOLD_XPU_KV` for the server); bf16 and q4 are opt-in.
  A paired KL study of the three formats on recipe C (mixed-document corpus, 16K and 32K chunks, 5 chunks at 32K, no needle
  retrieval) gave, against the bf16 cache: q8 -0.0002 at 16K and 32K (within one standard error, saving 0.5 to 1.0 GB); q4 +0.0024 (+3.8%)
  at 16K and +0.0032 (+7.4%) at 32K, with the same-top-token rate 0.2 points lower (saving 0.8 to 1.55 GB). q4 is an opt-in mode for pushing
  context, not a free saving.
  A needle probe (a code planted at depths 10, 50 and 90% of a public text, strict string check) found the code in all 54 cells (recipe C and MLX 4-bit, three KV formats,
  prompts of 32,768, 65,536 and 122,880 tokens). Peak device memory at 32K / 64K / 120K prompts, GB: recipe C bf16 15.98 / 18.13 / 21.88, q8 15.09 / 16.23 / 18.23, q4 14.55 / 15.16 / 16.21;
  MLX 4-bit bf16 18.64 / 20.78 / 24.54, q8 17.75 / 18.89 / 20.89, q4 17.21 / 17.81 / 18.87. This is one needle at three depths, not a measure of retrieval quality in general.

  Timed cells (depth-50 needle prompts, 64 generated tokens, each run alone, prefill windows chosen automatically; measured with a prototype build whose optimize mode is being confirmed, not on this tree): prefill seconds (tok/s) / decode tok/s / peak device GB.

  | Prompt | Checkpoint | bf16 KV | q8 KV | q4 KV |
  | --- | --- | --- | --- | --- |
  | 32K | recipe C | 25.6 (1281) / 30.45 / 15.98 | 25.5 (1283) / 32.07 / 15.09 | 25.5 (1283) / 32.92 / 14.55 |
  | 32K | MLX 4-bit | 25.7 (1274) / 26.88 / 18.64 | 25.4 (1288) / 28.63 / 17.75 | 25.4 (1288) / 29.33 / 17.21 |
  | 120K | recipe C | 130.0 (945) / 22.57 / 21.88 | 127.4 (965) / 26.22 / 18.23 | 126.5 (972) / 28.35 / 16.21 |
  | 120K | MLX 4-bit | 130.0 (946) / 20.57 / 24.54 | 127.1 (967) / 24.01 / 20.89 | 126.2 (974) / 25.76 / 18.87 |

  Prefill is nearly independent of the KV format; decode is faster with a smaller cache and the gap grows with depth (recipe C at 120K: q8 +16%, q4 +26% over bf16). llama.cpp's Vulkan backend shows the opposite
  (a q8_0 KV cache halves its decode at 32K depth), so this is a property of each engine's kernels, not of KV quantization itself.
- Not supported: grammars, tool-call gates, prompt cache, sampling penalties, GGUF through the native server, other Qwen3.5 sizes (a checkpoint
  whose configuration differs is refused before it loads).

### Measurements

Decode, 128 greedy tokens from a 5-token prompt (`tensorfold-xpu run`, q8 KV; the digest is the CLI's, see Digests below):

| Format | Plain | MTP drafts, k=4 | Digest (plain and drafted equal) |
| --- | ---: | ---: | --- |
| MLX 4-bit | 30.8 tok/s (32.5 ms/token) | 77.6 tok/s, head taken from the EXL3 checkpoint | `67f96813a817` |
| EXL3 3.00 bpw | 27.0 tok/s (37.0 ms/token) | 73.7 tok/s, own head | `f579aeec19ba` |
| GGUF Q5_K_XL | 23.8 tok/s (41.9 ms/token) | not measured | `820e2a0ba74c` |

Through the native server (q8 KV, `--context 131072`, `--prompt-cache-gib 0`, greedy, 256 tokens, `tools/bench_openai.py` and
`tools/bench_concurrent.py --levels 1,4 --alone --serial`):

| | MLX 4-bit | EXL3 3.00 bpw |
| --- | ---: | ---: |
| Decode, code / chat prompt | 30.4 / 30.4 tok/s | 26.5 / 26.5 tok/s |
| First token (code / chat) | 0.46 / 0.18 s | 0.50 / 0.16 s |
| Four streams, aggregate (code / chat) | 29.1 / 29.8 tok/s | 25.5 / 26.1 tok/s |
| Replies equal to their solo run | 12 of 12 at four streams, 3 of 3 for one | 12 of 12, 3 of 3 |
| Device memory at load / after a 131k prompt | 20.7 / 21.17 GB | 18.99 GB after a 131k prompt |
| Host memory (MLX): resident after load, peak (VmHWM, at load) | 415 MB, 1.80 GB | |

Cold prefill through `/v1/completions` with token-id prompts (the first L tokens of a public text under the Qwen tokenizer),
`cached_tokens` 0, 8 tokens generated; decode tok/s after the prompt in brackets:

| Prompt tokens | MLX 4-bit s | EXL3 s |
| ---: | ---: | ---: |
| 2,048 | 1.46 (29.4) | 1.72 (25.4) |
| 8,192 | 5.86 (29.3) | 6.87 (25.1) |
| 32,768 | 25.7 (28.1) | 29.6 (24.5) |
| 65,536 | 57.4 (26.3) | 65.3 (23.4) |
| 131,008 | 139.0 (23.4) | 154.8 (21.0) |

The prompt window is the largest of 2048, 1024, 512 or 256 rows that fits the memory left. Needle (a code planted at 50% depth): found at 4,032 and at 131,008
tokens on EXL3 with q4 KV (prefill 4.7 s and 199.6 s, peak 16.23 GB at 131,008).

Against llama.cpp Vulkan (build 9c2e0e491, Mesa 25.2.8, `-ngl 99 -fa`, `-c 123392 -np 1`, f16 KV, prompt cache off, 32 decoded tokens) on two GGUF
files: recipe C (described above) and the stock IQ3_S file above. Cold prefill s / decode tok/s after the prompt:

| Prompt tokens | recipe C | stock IQ3_S file |
| ---: | ---: | ---: |
| 2,048 | 5.28 / 13.56 | 5.25 / 11.39 |
| 8,192 | 24.6 / 13.06 | 24.4 / 11.04 |
| 16,384 | 60.75 / 12.30 | 60.3 / 10.50 |
| 32,768 | 170.6 / 11.04 | 169.6 / 9.57 |
| 65,536 | 537.9 / 9.21 | 536.1 / 8.12 |
| 122,880 | 1,654.8 / 7.18 | 1,651.2 / 6.48 |

Peak device memory of that server was 20.63 GB (recipe C) and 20.26 GB (stock file). `llama-bench` with f16 KV: pp2048 408.5 / 413.1, pp8192 334.5 / 338.1,
pp32768 191.5 / 192.7, tg256 11.06 / 10.20 tok/s, tg128 at 32k depth 9.12 / 8.64; with a q8_0 KV cache tg256 is 10.84 / 9.37 and tg128 at 32k depth 4.65 / 4.36 (prefill unchanged).
These prompts are needle-style texts, not the arrays of the tables above. No engine number on prompts of those lengths has been measured on this tree (an earlier figure came from a prototype build whose optimize mode is not confirmed, and is not quoted). Treat the comparison as an order of
magnitude, not a matched benchmark.

### Prompt attention and old-vs-new

The prompt windows use matrix-engine attention (`qwen_attn_pfs`, three builds for bf16, q8 and q4 KV). `ARC_PF=old` selects the earlier bf16
kernel and `ARC_PF=1` the per-row kernels; `F16_NOPF=1` keeps the EXL3 fp16 gate projections on the per-row kernel. Same prompt (the first 8,000 and 32,768
tokens of a public text), 16 generated tokens, windows of 1,024 rows, bf16 KV; tokens were identical in every row:

| Prefill, s | default | `ARC_PF=old` | `ARC_PF=1` | `F16_NOPF=1` |
| --- | ---: | ---: | ---: | ---: |
| MLX, 8,000 tokens | 6.69 | 7.08 | 8.41 | |
| EXL3, 8,000 tokens | 7.68 | 8.08 | 9.48 | 7.94 |
| MLX, 32,768 tokens | 27.9 | 34.7 | 44.7 | |
| EXL3, 32,768 tokens | 31.9 | | 48.8 | 33.0 |

### Exactness

- Window rows: windows of 2, 3, 4, 8 and 16 rows give logits bit-identical to one-row decode over 40 tokens, including a pending-then-commit
  case, for MLX, EXL3 and Q5_K_XL (`tf-xpu-qwen_rows-test`). The delta-rule window kernel with 2 or 4 value rows a sub-group is bit-identical to the one-row version
  at 16 to 2048 rows (`tf-xpu-qwen_gdnrows-test`).
- Drafted == plain: `tools/zig/xpu_spec_identity.sh`, 7 prompts, 299 tokens, k=4, token for token: EXL3 mtp, GGUF IQ3_S mtp, MLX mtp with the EXL3 head, MLX copy drafts and EXL3 auto all pass.
- The queued-forward slot ring equals synced forwards (0 of 248,320 logits differ).
- Prompt chunk sizes (`tf-xpu-qwen_seam-test`, 640 tokens of real text): MLX chunks of 64, 128 and 512 give byte-identical conv and SSM state, KV bytes and decode
  logits. EXL3 chunks of 64 and 128 are byte-identical; chunks of 512 differ from them. The cause is a named change of GEMM path at 256 rows: below it the EXL3
  projections use the decode kernels in 16-row steps, from 256 rows they use the matrix-engine prefill GEMM, and the fp16 gate projections switch to a prefill
  GEMM at the same size. With both switches removed in a scratch build, chunks of 64, 128 and 512 agree bit for bit, so these two are the whole cause; the unified path
  was not adopted because it would change the bits of every prompt chunk above 16 rows. Measured size: decode logits after the prompt differ between chunks of
  128 and 512 by at most 2.07e-1 (largest logit 22.8), top-1 equal in 7 of 8 steps. The test accepts that difference only where told (`differ=256`) and fails otherwise.
  The same difference appeared in the earlier clean-clone runs (EXL3 chunk 64 against 512 differed there too), so it is not new.
- Prefill against row by row is not bit-identical by design (different accumulation order): after 640 tokens, 74% to 99% of the state elements differ at rounding scale
  (worst conv difference 0.31 against a maximum of 62), and the next 8 decode steps differ by up to 0.39 in logit on MLX (top-1 equal in 8 of 8) and 0.27 on EXL3 (7 of 8).
- FP64 reference per projection class (`tf-xpu-qwen_fp64-test`, MLX 4-bit, real weights read from the checkpoint file, first and last layer of each kind, 2,048 evenly spread columns at most):
  decode (one row), 8-row window and 64-row prefill kernels separately; 90 rows of results. Aggregate error 1.3e-3 to 2.0e-3 in every class and mode; worst column (relative to
  the larger of the column's maximum and 1e-4 of the class maximum) 3.9e-3 for the window and prefill kernels and up to 1.6e-2 for one-row decode against the bound of 2e-2. Errors relative to the sum of
  the absolute terms are at most 3.5e-4 (bf16 output rounding). The first run of this test failed four one-row decode rows because it divided by each column's own maximum;
  for a column whose exact value cancels to about 4e-5 the error of 2e-6 to 4e-6 is 3e-8 to 5e-8 of the term sum, below one fp32 rounding unit, so the rule was changed to
  include the floor. The failing run, the diagnosis for the four columns and the corrected run are kept in `docs/internal/receipts/qwen-fp64-bound.txt` (internal). The Nemotron
  test keeps the rule it was published with; its table was rerun and is unchanged. EXL3 and the GGUF types are covered by their own oracle tests instead: `tf-xpu-exl3-test`
  (32 cases: 27 synthetic trellises and five rows of real tensors; the fixtures are generated by `tools/zig/xpu_fixtures_exl3.py`, `synth` needs no checkpoint and `real` reads the directory named by `TF_EXL3_DIR`, and nothing from weights is committed; without the generated real slices the five real-tensor cases do not run; decode bit-exact where checked, worst relative error 5.7e-7, row invariance 0 differences) and `tf-xpu-ggml-test` (all 14 block types: dequantization and quantized-x 0
  differences against libggml, matvec within 3.4e-7 of float64 in fp32 and 3.5e-3 in bf16, window rows 0 differences, chunked prefill 0 differences).
- Fidelity: teacher-forced top-1 on 4 prompts x 48 steps (the prompts of `tools/zig/fixtures/ref_nemotron.json` tokenized with the Qwen tokenizer) against a reference made by another engine on the same checkpoint,
  not against BF16: 189 of 192 (MLX), 188 (EXL3), 190 (IQ3_S GGUF). The reference logs (`ref_qwen27.json`, `ref_qwen27_exl3.json`, `ref_qwen27_gsq.json`) are **not shipped in this tree, so these counts cannot be
  reproduced from it**. How they were made: MLX and EXL3 by the TensorFold 0.6.2 CUDA engine (its CUDA container image, qwen3_5 family) on a GB10, plain serial greedy decoding with drafting off, bf16 prompt path,
  48 tokens a prompt, the top-5 log-softmax of the bf16 logits at every step; the IQ3_S GGUF by llama.cpp upstream build 11427 (c06f84160, CUDA 13) with `llama-server -ngl 99 -fa on -c 4096 -np 1 -ctk f16 -ctv f16`,
  `/completion` with token ids, `n_predict 48`, `n_probs 5`, temperature 0, prompt cache off. A BF16 GGUF run of the same prompts exists as a separate reference. BF16 KL divergence (reference BF16 via llama.cpp, `--prefill 512`, bf16 KV, first 64 chunks only, not the full corpus, measured with a build bit-identical to this export on those chunks): recipe C 0.0437, MLX 4-bit 0.0450, EXL3 3.00 bpw 0.0436; full-corpus figures from the engine's KL harness: recipe C 0.0497, MLX 4-bit 0.0536, EXL3 3.00 bpw 0.0645.

### Digests

The CLI prints the SHA-256 of the generated token ids written as a JSON list (`[1, 2, 3]`) and `--report` stores it; the server's `token_sha` is the SHA-256 of the ids joined by commas (`1,2,3`).
The same tokens therefore give different digests. `tools/zig/xpu_token_sha.py REPORT.json` prints both: for the MLX 4-bit p0 run above the CLI digest is `67f96813a817` and the server's is `a4fda3139fa0`;
for EXL3 `f579aeec19ba` and `294fb1554dd4`. The server's completions for the same prompt hash to those values, so server and CLI produce the same tokens.

### Qwen limits

- Speculative decoding: the prompt is prefilled in the same big windows and the head then absorbs its rows one at a time, so a drafted prefill costs more than a plain one
  (EXL3: 55.6 s against 29.4 s at 32,768 tokens, 256.6 s against 137.7 s at 120,000). Drafted == plain was checked token for token at short prompts (7 prompts, k=4), with the copy, auto and mtp drafters at 8,192 and 16,384 tokens (EXL3 and GGUF IQ3_S, k=3 and 4) and with mtp at 32,768 tokens
  (EXL3, GGUF IQ3_S, MLX with the EXL3 head), with 2048-row windows on both runs; both runs must use the same prompt windows, since 16-row and 2048-row prefill are different paths that can differ at near-tie tokens.
  At 120,000 tokens on EXL3 the drafted run decoded 33.6 tok/s against 21.6 plain on one prompt (64 tokens, 40 of 86 drafts accepted); on a GGUF mix (recipe D19pm, the same family as recipe C and also unpublished) with the prototype build (arc-engine main 1c7fec1, ReleaseFast, q8 KV, needle prompts at depth 50%, 80 tokens, k=3, plain and drafted runs from the same prefill path; not rerun on this tree), decode in tok/s for plain / MTP drafts / the auto drafter: 20.94 / 41.40 / 44.94 at 65,536 tokens; 19.93 / 37.01 / 38.40 at 98,304; 19.29 / 35.88 / 31.62 at 122,880. MTP drafts are 1.86x to 1.98x faster than plain at every depth measured there; the auto drafter's figure at 122,880 used the full-history copy scan that the copy-drafter index of this PR replaces (that run is being repeated). The drafted prefill costs more than the plain one there too (plain / drafted: 57.6 / 137.0 s at 65K, 95.6 / 222.1 s at 98K, 128.3 / 293.2 s at 123K). There is no automatic fallback to plain decoding. No claim is made for every long length.
- One request at a time on the native server; no drafts there.
- 131,072 positions at most; q4 KV is a quality trade (above).
- The EXL3 256-row seam above.

