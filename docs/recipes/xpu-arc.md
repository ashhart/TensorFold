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
| Qwen3.8 | Not covered by this page. |

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
