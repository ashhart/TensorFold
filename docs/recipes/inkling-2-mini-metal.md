# Inkling 2 Mini, native Metal

This recipe serves the original [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
checkpoint, revision `ed40e7da625f594fc9c4990f5f66be24873c22d7`, with its bundled DSpark head
(`mtp.safetensors`), on native Metal. Build with Zig **0.17.0** and Xcode's Metal toolchain, as described in
[the native preview](../zig-preview.md#build). It is experimental and is not release qualification: the contract and
served runs are on M4-class chips, there is no Linux backend in this change, and the platforms table below says what
was not run.

```bash
hf download thinkingmachines/Inkling-2-Mini --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 --local-dir "$HOME/models/inkling-2-mini"
zig build native -Dcpu=apple_m1
zig-out/native/bin/tensorfold-native serve "$HOME/models/inkling-2-mini" --port 8888 --context 65552 --parallel 1 \
  --no-drafts --prompt-cache-gib 0 --no-thinking
```

Drop `--no-drafts` for DSpark. `--prompt-cache-gib N` keeps prompt states on either server; with DSpark a kept state
also holds the head's context (see Kept prompt states).
Add `--vision` for images. `--parallel N` grants as many streams as the weights, the scratch and N states at the chosen
`--context` fit in the load limit; the startup line prints the demand and a named count that does not fit is refused by
name. A 65,536-token prompt needs `--context 65552`: the receipt's longest cold row is served at that window.
`--context 65536` refused a 65,536-token prompt in an earlier run the PR description records; it was not repeated here.

## The model

The checkpoint's own `config.json` (`inkling2_loop`) and the family's `config.zig` give the shape:

- 24 physical layers, hidden 2,304, 24 query heads over 8 key heads of 128, vocabulary 201,024. Layers 5, 11, 17 and 23
  are global; the other 20 slide over 1,024 keys.
- A loop: layers 6 to 17 run in three passes, so a forward makes 48 layer visits (6 + 3 x 12 + 6) with the loop's gates
  and a cross-pass message between passes. States and taps index by visit: 8 of the 48 are global.
- 128 routed experts a layer, 8 picked a token, width 1,152, no shared expert. Layers 0 to 19 store their experts as
  NVFP4; layers 23 to 20, the last four, store MXFP8 (`hf_quant_config.json`). The weights stay packed in memory:
  18.19 GiB resident, 18.95 GiB with the dense layers and scratch (the startup line).
- A vision tower (23 layers, 20-pixel patches) and a DSpark head of three layers with a block size of 8.

## What runs

The routed experts stay packed (NVFP4 and MXFP8) and are decoded to BF16 in registers; activations are BF16 and every
sum is FP32. The 24 physical layers run as 48 visits with the loop's gates and cross-pass message. The grouped paths
produce the one-row kernels' bytes: prompt chunks of 64 rows or more group the route pairs by expert, decode each
expert's weights once for a tile and sum on simdgroup matrix multiplies in the one-row chain's order; drafted verify
rounds of 2 to 63 rows plan pairs by expert in tiles of four; the dense prompt projections and the loop gates' state and
down chains run on an MMA tile that keeps the one-row chains; prompt chunks' global layers score and weight eight rows
at a time on 8x8 matrices, each row keeping its key blocks, running max and sum. Prompt chunks of 64 rows or more are
submitted on a Metal 4 command queue where the Mac has one (`TF_METAL4=0` keeps them on Metal 3); decode and drafted
rounds stay on the Metal 3 queue. The DSpark head runs every drafting
stream of a round in one pass. Admission counts the weights, the scratch and each stream's state bytes (a cache slot per
visit, at the configured `--context`) against the load limit, and a `warning` memory-pressure level no longer serializes
streams that fit; `critical` still does. The kernel contract (`tf-kernel-contract --only inkling2`) registers every
kernel with a planted control for each rule it states. Images (PNG, JPEG) go through the vision tower.

## Platforms

| Chip | What ran | Where |
| --- | --- | --- |
| M4 Pro, 48 GiB | The served receipt below, the contract and the host tests | Here, at the heads named in the receipt and under Verification |
| M4 Max, 128 GiB | `tf-kernel-contract --only inkling2` only: 86 kernels pass, 132 controls red, 0 missed, 0 limit problems at `e6cb02e09e` | A contributor (@akol1); no served run |
| M5 Ultra | At `9a255bb79e`: the contract 86 pass, 132 controls red, 0 missed, 0 limit problems; plain decode 55.4 and 55.8 tok/s and DSpark 65.3 and 65.3 (about 1.17 times plain) over 16 chat prompts, `--parallel 1`, two blocks; first token at 6,964 prompt tokens 4.73 s (4.91 s at `e6cb02e09e`). Sampled decode and more than one stream not run | A contributor (@SashaMIT), not reproduced here; his contract line counts one control fewer than the M4 chips because the media suite plants the `bf16_nt` row control only on chips without tensor units |
| M3 Pro, 36 GB | Contract 86/86, 80 controls red, and plain decode 12.0 to 12.4 tok/s with DSpark 14.4 to 14.6, at `f3a6a992f2` | A contributor (@jschmied); not rerun at any later head |
| M1, M2 | Not run | |
| M5 tensor-unit prompt path | Not written; the text path has no tensor-unit kernel (only the vision tower's GEMMs use them) | |

The grouped MMA paths assume an 8x8x8 `simdgroup_matrix` multiply-accumulate adds its products in k order from the
accumulator. A probe on the M4 Pro found 262,144 of 262,144 bytes equal to an FMA chain, and the contract holds on the
M4 Max and the M5 Ultra rows above.

## Receipt: Apple M4 Pro, 48 GiB

Raw results, the generator of the prompt token arrays, hashes and the build's SHA-256 are in
[`receipts/inkling2-mini-m4pro`](../receipts/inkling2-mini-m4pro/README.md), which says which file each number comes
from. The decode, cold-prompt (table), concurrency and footprint numbers were not measured at the head; the contract,
the hashes and the dense-tile cold first-token pairs below were measured at `61be67917f`, and the decode-at-depth table
at `57de097873`.

| | |
| --- | --- |
| Build | Decode, cold prompts, footprint: #4337 at `a0935b5279`, the PR's recipe-time binary; concurrency: the head-pass build that became `05b6fdb5c0`, plain on its parent; decode at depth: `57de097873`, SHA-256 `e74f0e5e08aa4d1eddefe867cc286e5eccfa7ad16f72950eb14d9479b61ab2ed`. `zig build native -Dcpu=apple_m1 -Doptimize=ReleaseFast`, Zig 0.17.0, macOS 27.2 |
| Reference | `mlx_lm.server --model "$CKPT" --host 127.0.0.1 --port 8099`, mlx 0.32.4, mlx-lm `0.1.dev1006+g0c4c99192` ("Add Inkling 2 Mini"), same checkpoint, no drafter |
| Native servers | `--context 65552 --parallel 1 --no-drafts --prompt-cache-gib 0 --no-thinking` (plain decode, cold prompts); `--context 8192 --parallel 4 --prompt-cache-gib 0 --no-thinking`, with and without `--no-drafts` (concurrency, and the DSpark decode column) |
| Requests | greedy, `tools/bench_openai.py --tokens 256 --reps 3 --temperatures 0`; `tools/bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4`; cold prompts of exactly 2,048, 8,192, 32,768 and 65,536 tokens through `/v1/completions`, one token out, cached tokens 0; decode at depth: the same cold prompts, then 192 streamed tokens (16 after 65,536 on native), greedy, `decode_depth.py` |
| Conditions | one model process at a time; an unrelated CPU job used about 60 % of a core during the decode and cold runs, so read ratios between rows, not the absolute rates. The depth runs shared the machine with other jobs' builds (load average 4.5 to 6.3 when they started); every native and MLX server held the GPU alone |

Decode, one stream (tokens a second, median of three):

| Prompt | MLX | Native plain | Native DSpark |
| --- | --- | --- | --- |
| `fibonacci-raw` | 23.05 | 31.12 | 35.76 |
| `gpu-chat-no-think` | 23.69 | 31.07 | 35.01 |

Concurrency, `bench_concurrent` (aggregate tokens a second, per stream in brackets; slowest first token in seconds):

| Prompt, streams | MLX | Native plain | Native DSpark |
| --- | --- | --- | --- |
| code, 1 | 23.4, 0.25 s | 31.1, 0.12 s | 37.7, 0.13 s |
| code, 4 | 23.3 (23.6), 33.8 s | 67.0 (16.9), 0.52 s | 65.1 (16.6), 0.52 s |
| chat, 1 | 23.3, 0.37 s | 30.9, 0.23 s | 35.8, 0.23 s |
| chat, 4 | 23.3 (23.9), 33.9 s | 66.8 (17.1), 0.90 s | 66.9 (17.1), 0.91 s |

MLX answers the four requests one after another here, which is why its aggregate stays at one stream's rate and its
slowest first token is 34 s; its four-stream numbers are not a same-width comparison. Native columns are one A-B-B-A
session (plain, DSpark, DSpark, plain; the mean of each pair). A DSpark server prices its shared rounds from a table of
timed verify forwards and drafts only where a priced round beats plain. The head runs the drafting streams of a round in
one pass and a plain round runs no head work, so four DSpark streams are within noise of plain (code 3 % under inside
the 65.6 to 68.3 spread of the two plain runs, chat level) and one stream is 15 to 22 % over it.

Cold first token (seconds, tokens a second in brackets):

| Prompt tokens | MLX | Native |
| --- | --- | --- |
| 2,048 | 4.68 (437) | 5.49 (373) |
| 8,192 | 19.53 (420) | 22.53 (364) |
| 32,768 | 81.74 (401) | 108.27 (303) |
| 65,536 | 180.27 (364) | 249.21 (263) |

The same sweep on the head `4418225fb6` (the dense prompt tile and the Metal 4 prompt queue in), on a quiet machine (load
average 2 to 3; native then MLX, one fresh server each, one run per size, so not interleaved; the MLX column is also
faster than in the table above, and the cause was not isolated; `cold-sweep-metal4.*` in the receipt):

| Prompt tokens | MLX | Native | Native's rate over MLX's |
| --- | --- | --- | --- |
| 2,048 | 4.32 (474) | 5.18 (395) | 0.83 |
| 8,192 | 17.04 (481) | 21.59 (380) | 0.79 |
| 32,768 | 75.55 (434) | 97.05 (338) | 0.78 |
| 65,536 | 169.70 (386) | 225.97 (290) | 0.75 |

Decode after a cold prompt (one stream, plain, tokens a second and milliseconds a token; the mean inter-token latency of
192 streamed tokens after the first few; MLX and native cells are each the mean of two fresh-server sessions in the order
native, MLX, MLX, native, on two prompt sets; `57de097873`):

| Prompt tokens | MLX | Native plain | Native over MLX |
| --- | --- | --- | --- |
| 2,048 | 23.04 (43.41 ms) | 28.48 (35.11 ms) | 1.24 |
| 8,192 | 22.60 (44.24 ms) | 28.40 (35.22 ms) | 1.26 |
| 32,768 | 20.03 (49.94 ms) | 25.78 (38.80 ms) | 1.29 |
| 65,536 | 17.80 (56.20 ms) | 23.19 (43.11 ms) | 1.30 |

The native 65,536 row is 16 tokens after the prompt (a server at `--context 65552` takes no more), from two further
sessions taken after the others, 43.12 and 43.11 ms; a 65,344-token prompt with 192 tokens out, which also ends at
65,536, gave 43.00 and 42.96 ms. From 2,048 to 65,536 tokens native gains 8.0 ms a token (0.126 ms a 1,000 tokens) and
MLX 12.8 (0.201). The K and V that the 8 global visits read grow by 32,768 bytes a token of depth (8 key heads x 128 x 2
bytes x 2 x 8 visits), which is 0.131 ms a 1,000 tokens at 250 GB/s, so native's growth is what the bytes cost; the two
native runs at one depth differ by 0.1 to 0.5 ms. The server prints no stage times, so the split comes from timing the
decode attention kernel alone (`attn-decode-bench.txt`: one query row, eight or forty visits with their own K and V
buffers, multiplied out to 8 global and 40 sliding visits), the rest by subtraction:

| Prompt tokens | Global attention, 8 visits | Sliding attention, 40 visits | Everything else | Token |
| --- | --- | --- | --- | --- |
| 2,048 | 0.80 ms | 2.26 ms | 32.05 ms | 35.11 ms |
| 8,192 | 1.14 ms | 2.26 ms | 31.82 ms | 35.22 ms |
| 32,768 | 5.03 ms | 2.26 ms | 31.51 ms | 38.80 ms |
| 65,536 | 9.24 ms | 2.26 ms | 31.61 ms | 43.11 ms |

One global visit reads at 235 GB/s at 8,192 keys, 213 at 32,768 and 232 at 65,536 (1.15 ms); at 2,048 keys it takes 0.10
ms for 8 MB (84 GB/s, 8 threadgroups on 20 cores), about 0.5 ms a token above the 250 GB/s floor, and the 65,536-key
visit is 0.6 ms a token above it. The kernels' summation order fixes their chunking (2,048 keys, 4 simdgroups), so a
faster schedule has to keep it: hoisting the K loads, double-buffering the V rows and an `I2_V_BATCH` of 8 and 16 gave
the same bytes and at best 4 % on the 65,536-key visit (0.4 ms a token, about 1 %), so no kernel changed. A sliding visit
costs 0.057 ms against a 0.02 ms floor for its 4 MB; leaving out its V phase takes it to 0.033 ms and leaving out the K
dot product to 0.038, so it is not byte-bound, and 40 of them are about 1.6 ms a token above the floor, the one part of
the attention left well above it. The depth runs' 2,048-token rate (28.5) is below the decode table's 31.1: the head's
own short-prompt run gave 31.3 and 31.2 tokens a second (`bench_openai`, `fibonacci-raw` and `gpu-chat-no-think`, about
32.0 ms), so 2,048 tokens of depth cost 3 ms a token, which the global and sliding visits' time at that depth would
explain [INFERENCE: the short-prompt kernel times were not taken].

Where a plain decode token goes (one stream, short prompt, one command buffer of 1,560 dispatches, GPU span 31.0 ms; host
encode 0.29 ms and the gap between forwards 0.32 ms). The weights a token reads are 5,487 MB (the dense matvecs 1,904,
`wo` 680, logits 922, 48 expert visits 1,839), which is 21.95 ms at 250 GB/s. Replayed alone with rotating weight
buffers, the fused dense matvec ran at 256 GB/s, `wo` at 248 and logits at 263, so those are at the floor; the routed
experts and the router are each about 1.7 ms over, and about 1,200 small latency-bound dispatches (convs, q/rel/kv prep,
residuals, attention-residual mixes, loop gates) account for most of the remaining 4 to 6 ms. A no-op dispatch costs about
1 us, so the 1,560 launches are 1.6 to 2.3 ms. Two byte-equal attempts did not pay: matvecs with 16-byte loads ran at 0.80
to 1.00 of the baseline, and routing the 1-row attention-residual and loop mixes to their one-group kernels (312 fewer
dispatches, all four hashes equal) was 8 to 15 % slower, since a one-threadgroup kernel costs more than the launches it
saves. The family times other than the floors come from skipping dispatches in a loaded session (wrong bits, 34.3 ms
baseline) and are a ranking, not a split. Fusing the elementwise chains (conv and store with the residual adds, q/rel/kv
prep, router select) would remove about 430 dispatches, roughly 1 ms (3 %), and needs a launch record and a planted
control per new kernel; it was not written.

The dense projection tile (`61be67917f`, a different build from the table) took the first token from 23.8 s to 22.7 s at
8,192 tokens (4.7 %) and from 104.8 s to 102.0 s at 32,768 (2.7 %): one A-B-B-A session, the mean of each pair, plain
server, same hashes. The table above predates it.

Where a 512-row prompt chunk's GPU time goes (`prefill-stages.txt` in the receipt: a stamp between every two dispatches
on a Metal 4 queue, `TF_METAL4_TIMELINE`; the stage spans add to the first token, 21.7 s of 21.9 s at 8,192 tokens and
98.0 s of 98.5 s at 32,768, so the host's encoding and the gaps between kernels cost nothing there). Floors are 7.7
TFLOP/s on the matrix units and 250 GB/s; the gap of the experts is the open one the maintainer owns.

| Family (48 visits) | Chunk at position 512 | Floor | Chunk at position 31,744 | Floor |
| --- | --- | --- | --- | --- |
| Routed experts (gate/up, down, plan, combine) | 857 ms (67 %) | 406 ms | 896 ms (49 %) | 406 ms |
| Dense projections (MMA tile) | 215 ms (17 %) | 172 ms | 222 ms (12 %) | 172 ms |
| Global attention, 8 visits | 18 ms (1 %) | 5 ms | 449 ms (25 %) | 209 ms |
| Sliding attention, 40 visits | 112 ms (9 %) | 25 ms | 167 ms (9 %) | 34 ms |
| Loop gates and blend | 47 ms | not derived | 48 ms | not derived |
| Attention-residual mixes and norms | 14 ms | not derived | 14 ms | not derived |
| Conv, KV prep, q and rel clip | 14 ms | not derived | 15 ms | not derived |
| Router, residual, copies, embed | 5 ms | not derived | 5 ms | not derived |
| Chunk | 1,281 ms | | 1,815 ms | |

A 32,768-token prompt costs 0.34 ms a token more than an 8,192-token one (2.99 against 2.65 ms): 0.32 ms of it is global
attention, 0.02 ms sliding attention (its window is full after the second chunk), and the experts, dense tiles, gates and
KV writes are flat to within 0.3 %. The prompt kernels read the K and V rows of every global visit once for each tile of
eight rows and head, which is where the rate falls; the unique KV bytes of a chunk at 31,744 are 1.06 GB (4 ms at 250
GB/s), so the visit is compute and cache bound, at 3.7 of 7.7 TFLOP/s. One sliding visit takes 3.8 ms alone at a full
window (512 rows, 24 heads, the scalar kernel): 1.9 ms of it is the value phase, 1.0 ms the key dot products and 0.4 ms
the rest [INFERENCE: from leaving each phase out; the outputs of those runs differ]. Each row's window starts one key
later than the row before, so its 32-key blocks do, and rows cannot share an 8x8 key tile without changing the
running max and sum's order [INFERENCE]; no attention kernel changed.

Sending the prompt chunks through a Metal 4 queue (the same kernels, arguments and order) took the first token from 25.0
to 22.4 s at 8,192 tokens and from 116.1 to 103.9 s at 32,768 (10.4 % and 10.5 %: the mean of each pair in an A-B-B-A
session at the commit's build, plain server; an earlier session on a loaded machine gave 16.9 % and 14.8 %), with the
same token hashes on all four servers and the four chat hashes above. Why the Metal 3 queue is slower was not isolated:
its stage times were not stamped, and the Metal 4 stage spans equal the first token. Decode stays on the Metal 3 queue
and is unchanged (27.4 to 30.1 tok/s in both builds, on a machine slower than the decode table's); a build with the
whole model on a Metal 4 queue showed no decode gain in one A-P-P-A session.
With three decoding streams and a cold 4,096-token prompt arriving at 8 s (`mixed_load.py`, `--parallel 4`, A-B-B-A, same
prompt hash on all four), the prompt took 13.9 and 15.8 s to its first token on Metal 3 and 13.2 and 13.0 s on the Metal 4
queue, and the decode streams' longest gap was 12.8 and 15.6 s against 11.7 and 11.6 s: in both builds the streams wait
for the prompt, and the Metal 4 queue is not worse for them.

One global prompt visit alone (512 rows, 31,744 keys, the production dispatch, GPU timestamps, quiet machine) takes 56 to
58 ms for 201 GFLOP, 3.47 TFLOP/s, and 117 ms at 63,488 keys; the 26 ms floor at 7.7 TFLOP/s is not a byte floor (132 MB
of unique K and V, 25 GB of re-reads at about 437 GB/s). Leaving out the score MMA takes it to 27.5 ms, the value MMA to
41.2, the softmax to 45.2, the K loads to 55.2 and the V loads to 54.9 (wrong bits, timing only), so the score phase is
the excess (about 31 ms against 15 for its MMAs) and its cost is operand preparation (the transposed K fragment), not
load latency or barriers. A byte-equal variant (K as the A operand with no transpose, q as the B operand, a transposed
store, the running max and sum in threadgroup memory, the merge weights once a row) was bitwise equal to the current kernel
on 19 shapes and inputs including NaN, infinities and scale extremes (ragged, 70,000 keys, position 0) and ran the visit
in 0.82 of the time (0.85 at 63,488 keys), which is about 2.5 % of the 32,768-token first token; an A-B-B-A on a machine
with other jobs (load average 3 to 19) could not show it, so it was not applied.

Peak physical footprint (`footprint`, 1 GB resolution, sampled every 3 s through an 8,192-token prompt and a 128-token
reply): native about 24 GB, with `--parallel 1 --context 65552` and with `--parallel 4 --context 8192` and DSpark alike;
MLX about 23 GB.

Plain and drafted greedy replies are identical. These token hashes (the server's `sha=`, chat completions at
`--context 8192`) are the same on four binaries from the session: the Fibonacci question with 8 and 64 tokens out
`0b037128ee78` and `b37d8a0e39c4`, a 416-token prompt and 16 tokens `77bf7c99cc66`, a 1,036-token prompt and 24 tokens
`538175732cc7`. A drafted four-stream run with `--alone --serial` compared 30 replies with their runs alone and 4 with
`"draft": false`: all equal.

Native plain decode is ahead of the reference at every depth measured, 1.24 to 1.30 times from 2,048 to 65,536 tokens
(table above), and the drafter's decode is ahead of it at the two short prompts of the decode table; drafted decode was
not measured at depth, nor concurrent streams at depth. Cold prefill is 0.75 to 0.83 of the reference's rate on the latest
quiet session (0.72 to 0.87 in the earlier one) and falls with the prompt's length (0.83 at 2,048 tokens, 0.75 at 65,536). Not measured here: M1, M2, a 36 GB Mac, a quality
suite (benchmarks of task accuracy).

Fidelity against MLX on the original checkpoint (`fidelity/` in the receipt; plain native server, `--no-drafts`, a binary
copied from the PR checkout's `zig-out`, SHA-256 `fa04538b…` [INFERENCE: built from the head `57de097873`], one model
process at a time). Seven text prompts (Python and Zig source, prose, a maths problem, a three-language prompt, and
documents of 7,998 and 31,998 tokens), each continued by MLX's own greedy 256 tokens. The native backend has no
`prompt_logprobs` (HTTP 400), so the continuation was forced by restarts: native greedy from the prompt plus MLX's ids
so far, and at the first token that differs, MLX's token is forced and the request restarts after it (prompt token counts
checked). Result: native's top-1 equals MLX's at 1,762 of 1,792 positions (98.33 %; per prompt 95.7 % to 99.6 %, the lowest
on the Python source), the mean absolute difference of the forced token's logprob over agreeing positions is 0.005 to
0.028 (signed mean 0.0002 to 0.0021, no trend across the four 64-token blocks of the continuation, largest single difference 0.14 to 0.53), and
all 30 disagreements are near ties in MLX (top-1 minus top-2 at most 0.386 nats, median 0.090, 15 under 0.1). The first
free greedy divergence is at token 14, 43, 47, 57, 57, 0 and 1 of the seven prompts. As a control, MLX against itself with
`prefill_step_size` 256 instead of 2048 diverges at token 14, 43 and 51 on the first three prompts (and not on the other
two), at the same near ties. The comparison is agreement under BF16 rounding order, not equality; the
logits themselves, per layer, were not compared here.

Images served (`--vision`, `--context 8192 --parallel 1 --no-drafts --no-thinking`, the pinned libjpeg-turbo release built from its
release tarball, greedy, 160 tokens, streamed; PNG and JPEG; Pillow-made images, text, bar chart, shapes, a gradient
scene, a 1792 x 1344 invoice). Native read "TensorFold 4821 / Metal on Apple M4 Pro", the five bar values and "Thursday ...
9", red circle, blue square and green triangle, the scene's sun, mountains and sea, and (inside the 160-token thinking text,
cut before the final answer) "INVOICE No. 20931" and "TOTAL DUE: 1034.00 EUR", from both PNG and JPEG. `mlx_lm.server` of
the same checkpoint serves images too (no mlx_vlm needed) and read the same content; the texts differ in wording after a
few dozen characters, as near-tie noise does. Time to first token, median of two requests after a warm-up, one image (the
first three rows from the session above; the last two at `a305edd706`, quiet machine, A-B-B-A against `5e75f97db2`, which
took 15.7 to 16.2 s on the invoice and 3.0 to 3.4 s on the scene there, 17.3 to 18.1 s and 3.5 to 3.9 s in the first session;
the invoice's 6.5 s is the bf16 score build below, quiet machine, A-B-B-A against `238f688b59`, which took 6.9 s):

| Image (prompt tokens) | Native | MLX |
| --- | --- | --- |
| 320 x 120 text (56) | 0.37 s | 0.55 to 0.99 s |
| 512 x 512 shapes (197) | 0.88 s | 0.71 s |
| 640 x 420 chart (208) | 0.93 s | 0.73 s |
| 1024 x 768 scene (546) | 1.9 to 2.3 s | 1.7 s |
| 1792 x 1344 invoice (1,561) | 6.5 s | 4.7 to 4.8 s |

The image tower was most of a larger image's first token: on the invoice it took 12.5 s of the 15.7 s, 89 % of that in
the two fp32 attention dots (q k^T and p v) that ran one thread an output. They now run on 64 x 64 tiles with each output
still one fma chain over k, so the tower's output is the same bits (its hash and all 16 replies per image are equal): the
tower takes 3.2 s on the invoice and 0.64 s on the scene. The softmax between those dots was at its memory floor (0.86 ms
for a 256-row chunk of the invoice's 16 heads over 6,121 keys, 100 MB read and written, against 0.80 ms at 250 GB/s), so
the fix is bytes: q k^T stores its scores as bf16 (bf16(bf16(sum) / 8), the two roundings the softmax applied first, in
its epilogue), the softmax reads and writes bf16 with the same max, exps and sum order, and p v reads the bf16
probabilities (exact in fp32). The three kernels take 74 ms a layer on the invoice instead of 95 ms, and the first token
6.5 s instead of 6.9 s, with the tower's output hash and all 8 replies per image equal. What is left of the invoice's
6.5 s is the tower (about 2.8 s) and the text prefill of its 1,561 rows (about 3.7 s), and MLX's tower is under half of
native's (the stage tables are in `fidelity/img-ttft.txt` and `fidelity/img-softmax.txt` of the receipt). The three small
rows were not re-measured quietly: their attention is small and a 0.1 s change is inside this machine's noise.

Sampled decode, `bench_concurrent --temperatures 0.7 --seed 1234 --alone --serial --levels 1,4 --tokens 256 --reps 3`
(aggregate tokens a second), `--context 8192 --parallel 4 --prompt-cache-gib 0 --no-thinking`, on an idle machine, the
build of `61be67917f`: plain (`--no-drafts`), DSpark, DSpark again, plain again; the table gives each server's mean over
its pair of runs, the two runs in brackets:

| Prompt, streams | Plain | DSpark | DSpark against plain | Greedy plain (table above) |
| --- | --- | --- | --- | --- |
| code, 1 | 21.4 (22.0, 20.7) | 20.7 (20.9, 20.5) | 3 % under | 31.1 |
| code, 4 | 33.8 (34.6, 33.0) | 25.9 (25.9, 25.9) | 23 % under | 67.0 |
| chat, 1 | 21.5 (22.1, 20.8) | 21.3 (21.5, 21.1) | 1 % under | 30.9 |
| chat, 4 | 33.8 (35.2, 32.3) | 33.2 (34.2, 32.2) | 2 % under | 66.8 |

Two things stand out. Plain decode with temperature 0.7 runs at 21 tokens a second against 31 greedy at one stream and at
34 against 67 at four, so a sampled token costs about 15 ms more than a greedy one and the cost grows with the streams.
With the draft head on, four code streams are 23 % under plain and the other cells are within the 6 % that separates the
two plain runs. The client's check for sampled replies (each stream's seed is its index plus the base seed, and the same
request must give the same text alone and with `"draft": false`) gave all equal on both DSpark servers: 3 and 12 of
each count against the solo runs, 4 of 4 against `"draft": false`, 0 unequal, 0 failed. Drafts were accepted (467 on
code, 447 on chat), so the check was not vacuous. The raw summaries are `sampled-*.out` in the receipt.

## Kept prompt states

`--prompt-cache-gib N` keeps a stream's state at prompt boundaries (a 512-token grid, the end of each prompt) and
resumes a later prompt that extends one. A kept state is the 48 visits' keys, values and convolution histories: 32 KiB a
position over the eight global visits, 160 MiB for the 40 sliding visits' last 1,024 positions, and 19.5 MiB of
histories (273 MiB at 3,000 tokens).

With DSpark a kept state also holds the head's context. The head has no recurrence of its own, but its key and value at
a position are a projection of the target's tap rows there (the inputs to visits 5, 23 and 41 and the output after the
final norm), and a kept state holds no tap rows, so the context is copied, not rebuilt from the lane's. It is the full
layer's keys and values for every position (4 KiB a position) and the two sliding layers' last 1,024 positions (8 MiB):
19.7 MiB at 3,000 tokens, 7 % more. The head's last feature row is not kept: the next absorbed row overwrites it. A
resumed request restores the lane and the head before its first chunk, so its head drafts from the context a fresh
request's would.

Measured on the M4 Pro (`--context 8192 --parallel 4`, multi-turn chats of 2,900 to 4,500 tokens, greedy and seeded 0.7;
[drafted-resume](../receipts/inkling2-mini-m4pro/drafted-resume/summary.txt)): a DSpark server with
`--prompt-cache-gib 6` gave the same token ids as a plain `--no-drafts` server with the cache off in 56 of 56 requests,
37 of them resumed (at 2,560 to 4,541 tokens, past the 1,024-token window; four concurrent streams and a conversation
that alternates `"draft": false` and drafted turns included); single-stream resumed requests had the same drafts proposed
and accepted as cold ones, and `bench_concurrent.py --alone --serial` had 60 of 60 equal. A resumed first token took 2.3
to 2.6 s against 10 to 12 s cold at 3,000 tokens and 6 to 7 s against 16 to 17 s at 4,400, on a machine under load;
decode with no kept states is unchanged (A-B-B-A, 33.2 against 32.7 and 30.8 against 31.2 tok/s).

## Verification

Each line names the head it was taken on. The contract line was taken at `c7aa88235c` (and `test-inkling2-host` built
and passed there); the sampled-decode table at `61be67917f`; the host-test line at `e90ecce19a`.

- Kernel contract, `tf-kernel-contract --only inkling2` at `c7aa88235c`: 86 kernels pass gates 1 to 4, 133 controls red
  as planted, 0 missed, 0 limit problems, on the M4 Pro (see [the kernel contract](../kernel-contract.md#coverage)).
- Original checkpoint, device integration (`test-inkling2-metal`, 7 of 7): shared rows, rollback and restore, drafted and
  plain token ids; a 129-row trace matched every FP32 logit across 128-row and 32-row chunking, SHA-256
  `fbbb16ea5bf855e5b106ce740c32232ddb9ce7cd4f200792e3376fd4598956e7` (PR description, measured before `3439d56ae0`).
- Tiny checkpoint `hf-internal-testing/tiny-inkling2-mini@ee242a48af34ff0729518f4ac28cfcb847e470ab` against Transformers
  `6173a2f6` BF16 eager: 35 of 35 top ids, RMS 0.03390 and 0.02735 (PR description, measured before `3439d56ae0`).
- Vision, `test-inkling2-media-metal` with the pinned eager rounding: 83 of 83 (PR description, measured before
  `3439d56ae0`); the pinned public fixture matched 20 of 20 completion ids. At `db6d02593c` the suite gave 79 passed,
  9 skipped (the JPEG tests, for want of the pinned libjpeg-turbo release), 0 failed. On an M5 Ultra a contributor's run
  at `98b1e13bd0` gave 77 passed, 8 skipped, 0 failed.
- Drafted equals plain: the hashes and the `--alone --serial` counts in the receipt above; the contract's grouped kernels
  are byte-equal to the one-row kernels, including NaN, infinity and all-pairs-one-expert inputs.
- A resumed prompt equals a fresh one with the drafter on and off, greedy and sampled, at one and four streams: see
  Kept prompt states.

## Limits

- Audio parts are refused: the server answers HTTP 400 before the template runs ("content parts must be text or
  image_url; audio and video inputs are unsupported", or "this server accepts text parts only; ..." without `--vision`),
  and the checkpoint's chat template raises "Inkling 2 does not accept audio input" for `audio`, `input_audio` and
  `audio_url` parts. A host test pins both.
- JPEG images need the pinned libjpeg-turbo release (the decoder is core's picture decoder; the version and where the
  server looks are in [image input](../vision.md#jpeg-input)); without it each JPEG is refused by name and PNG and text
  still serve. Images need `--vision`.
- The native backend refuses `prompt_logprobs` (HTTP 400, "not supported by this model or backend"); `logprobs` on a
  completion or chat reply work (`fidelity/fid_native.py` uses them).
- Replies carry the model's raw delimiters (`<|message_model|><|content_thinking|>` and the like): they are not split
  into `reasoning_content`, a finished reply puts the same string in `content` as well, and `usage.reasoning_tokens` is 0
  (found by a contributor on an M3 Pro). #4124 splits Inkling 2 replies by the format's state table; merged into this
  branch it gave the split and a nonzero `reasoning_tokens` on the M4 Pro, so this limit goes when #4124 is on `main`.
- A kept state holds the DSpark head's context only when the stream that made it drafted: one made by a `"draft": false`
  request, or at the end of a prompt-only warm pass (`max_tokens` 0), holds none, and a drafted request resumed from it
  starts its head past the kept prefix. Its tokens are the same (the verify rows decide them); the head drafts from less
  context until the new rows fill it, so fewer drafts land.
- `--drafter` is refused (the head is the checkpoint's own), as are `--slide`, `--loops`, `--policy` and
  `--checkpoint-slots`.
- Cold prefill is 0.75 to 0.83 of MLX's rate (the latest quiet session, with the Metal 4 prompt queue; 0.72 to 0.87
  before it). The text path has no M5 tensor-unit kernel.
- Four drafted streams are within noise of plain, not faster; one stream is 15 to 22 % faster.
- Admission by state bytes was tested on a 48 GiB Mac only; a 36 GB Mac (where an earlier head serialized four streams
  under memory pressure) has not run it.
- With temperatures above 0 a plain token costs about 15 ms more than a greedy one (21 against 31 tokens a second at one
  stream, 34 against 67 at four), and four drafted code streams are 23 % under plain. The cost is the host sampler's
  full sort of the 201,058-token row; #4744 keeps only the best 1,024 candidates, with identical tokens, and on a shared
  machine took plain sampled code from 21 to 29 tokens a second at one stream and from 34 to 56 at four. Until it lands,
  serve sampled traffic that matters for speed with `--no-drafts`.
