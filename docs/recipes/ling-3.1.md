# Ling-3.1-flash

Work in progress on the Zig engine's CUDA side. The family is registered with the native CUDA server
(`model_type` `bailing_hybrid`) and serves requests through the lane core at TP=1, but the whole model does not fit
one GB10, so serving is verified on synthetic checkpoints and a thin real model. This page lists what exists, what is
missing, and how to test without the 572 GB checkpoint. Everything below was measured on one GB10.

Ling-3.1-flash (`BailingMoeV3ForCausalLM`, `bailing_hybrid`) has 72 layers plus one MTP layer: Kimi delta attention
(KDA) on layers with `(i + 1) % 8 != 0`, MLA on every eighth, dense MLPs on layers 0-3 and 512 routed experts (top 8,
8 groups keeping 4, sigmoid scores with a pick bias) plus one shared expert from layer 4 on. The released checkpoint is
block FP8 (e4m3 in 128 x 128 blocks with `weight_scale_inv`).

## What exists

| Part | Where | Checked by |
| --- | --- | --- |
| Group-limited top-k routing, per-layer shared SwiGLU limit (GLM's bytes unchanged) | `ling-family`, commit 51c01573 | `glm-ref`, `glm-route-old`, `glm-route-ling` |
| MLA at Ling's shape: 64 roped key columns, 128-wide heads, GPT-J rope, head-wise gate | f8e594f6, merged | `ling-mla`, `ling-mla-real`, `glm-mla-same` |
| KDA on Ling's layer 0 (q/k/v convs as one q\|k\|v conv), GLM's row path as is | b55aa13d, merged | `glm-kda-ling` |
| Config loader and weight map (all 214,025 names of the real index) | `zig/src/families/ling/config.zig`, `names.zig` | `zig build test-ling` |
| Weight loader at TP=1: bf16, FP8 block, ModelOpt NVFP4 | `zig/src/families/ling/weights.zig`, `weights_proj.zig` | `ling-weights`, `ling-weights-real` |
| Synthetic checkpoints in all three formats | `tf-cuda-test ling-synth` | its own name check |
| The TP=1 layer graph: embedding, KDA/MLA mixers, dense MLP and MoE, final norm, head; windows and commits | `zig/src/families/ling/forward.zig`, `graph.zig`, `state.zig`, `block_*.zig`, `kernels/cuda/ling_glue.cu` | `ling-forward-synth`, `ling-layer-real` |
| The MTP layer (`model.layers.72`) as the drafter: `enorm(embed(next)) \| hnorm(hidden)` through `eh_proj`, MLA + MoE on its own cache, `final_layernorm`, the shared head; chains of up to 3 | `mtp.zig`, `Engine.draft` | `ling-lanes --mtp`, `ling-mtp-real` |
| The engine and lane backend: a sequence per slot, prompt chunks cut anywhere, verify windows of up to 16 rows, shared windows of up to 8 streams (64 rows), kept prompt states, the server hooks | `engine.zig`, `lanes.zig`, `native.zig`; `native/cuda.zig` registers it | `ling-lanes` |
| SwiGLU clamped in Ling's order, `min(silu(g), L) * clip(u, -L, L)`, in the glue and the FP8/NVFP4 grouped experts (every older kernel's SASS unchanged) | `glm_glue_moe.cuh` (`tf_ling_swiglu_*`), `experts.cuh` (epilogue 4), `experts.Layer.after` | `ling-clamp`, `ling-clamp-old`, `glm-route-old` |

## The layer graph

`forward.Forward` runs rows of one or more sequences (`Seg`: a `State`, its first row, its row count) through the
embedding, every loaded layer and the head; `stack()` runs the layers alone on `Buffers.x`, `layer()` one of them.
`Buffers` are prompt rows or decode rows; each sequence's rows stay pending until `commit(state, keep)`. A prompt
keeps every row; a decode window (at most `window_rows`, 16) keeps any prefix: the KDA states are replayed from the
window's saves (`replay_layers`) and the conv windows shifted, as #804's GLM commit does. The API follows GLM's
`cuda_forward.zig` (`Rank`/`Seg`, `Buffers`/`State`, `commit`) so lanes hooks can follow.

Every projection is a `qlinear.Weight` and every routed-expert layer an `experts.Layer` in the format the checkpoint
stored; the graph only calls `qlinear.Linear.decode`/`prompt` and `experts.Grouped.decode`/`prompt`. MLA's absorbed
`kv_b` (bf16 `wk`/`wv` per head) is derived at load by running `kv_b` in its own format on identity rows, so any format
`qlinear` serves works. bf16 routed experts (no grouped bf16 kernel) run every expert on every row and the combine
reads each row's picks: correct, for tests, not for speed.

Per layer, in the HF model's order: input RMSNorm; KDA (q, k, v, beta into one `[q|k|v|b]` row, f and g rows, GLM's
chain with the safe gate, gated output norm, `o_proj`) or MLA (`q_a`, `q_a_layernorm`, `q_b`; `kv_a_proj_with_mqa`,
`kv_a_layernorm` on the 512 latent, the 64 key columns roped into the cache row; absorbed queries, roped attention
over the 576-wide cache, `expand_v`, sigmoid head gate from `g_proj`, `dense`); residual; post-attention RMSNorm;
dense SwiGLU (layers 0-3) or MoE (bf16 router dot in fp32, sigmoid + `expert_bias` with 8 groups keeping 4,
normalized weights x 2.5, grouped experts with the layer's limit, shared expert with its own limit); residual. The
residual stream, every linear's output and the MoE sums are bf16 at the points where the HF bf16 model rounds them
(`ling_glue.cu`: residual add, head gate, expert sum + shared). Decode windows run KDA's three-kernel chain, as
GLM's forward does by default (bit-equal to the fused chain, checked).

Measured on synthetic checkpoints (4 layers: KDA dense, MLA MoE, KDA MoE, MLA MoE; 80 prompt rows, 8 decode rows),
logits against `oracle/ling_forward.py` routed on the graph's own picks (with free routing, 15-21 of 264 picks
differ from float64's: group near-ties, the worst group sums 0.0014-0.008 apart):

| Format | vs float64 | vs the bf16 rounding points | bf16 points vs float64 | argmax vs float64 |
| --- | --- | --- | --- | --- |
| bf16 | 2.17e-2 | 1.31e-2 | 2.18e-2 | 86/88 |
| FP8 block | 2.12e-2 | 1.22e-2 | 2.12e-2 | 85/88 |
| NVFP4 | 2.29e-2 | 1.40e-2 | 2.25e-2 | 85/88 |

(relative L2 of the logits). The graph is as far from float64 as the HF model's own bf16 rounding is. Bit for bit, in
every format: a prompt cut at 48 equals the uncut prompt; an 8-row verify window equals 8 decode rows; keeping 3 of 8
and running the next 5 leaves every KDA state, conv window and MLA cache row equal to 8 decode rows; two sequences in
one prompt chunk and in one decode window equal each alone; the fused and three-kernel KDA chains agree.

Real layers of the FP8 release (`ling-layer-real`, the HF bf16 chain's captured input and output, 512 tokens):

| Layer | Kind | rel L2 vs HF | rows with a different top-8 | rel L2 without them | decode rows vs HF | device peak |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | KDA, dense | 3.27e-3 | - | 3.27e-3 | 3.12e-3 | 0.9 GB |
| 4 | KDA, MoE | 5.88e-3 | 21 | 3.13e-3 | 4.28e-3 | 8.5 GB |
| 7 | MLA, MoE | 4.89e-3 | 17 | 3.05e-3 | 3.57e-3 | 8.3 GB |
| 65 | KDA, MoE, limits 10 | 4.85e-3 | 14 | 3.35e-3 | 3.33e-3 | 8.5 GB |
| 71 | MLA, MoE, limits 10 | 4.13e-3 | 7 | 3.75e-3 | 3.89e-3 | 8.3 GB |

exllamav3's port measured 0.4-1.1e-3 without flips (4 flips each on layers 4 and 7) on the same chain: it keeps fp16
activations, the captured HF chain carries an fp32 residual from layer 3's output on, and this graph rounds the stream
to bf16 as the bf16 model does. On every
real layer a prompt cut at 300 equals the uncut prompt and a 16-row window equals 16 decode rows; for FP8 decode rows
equal the prompt's rows (both run the lane). Loading one MoE layer's 512 experts takes 6.1 s from
local disk and 122 s from the sshfs mount (its read alone is 92 s); see Loading below.

### Formats

The format of every tensor comes from the checkpoint, never from the layer graph:

- `quantization_config.quant_method: "fp8"` with `weight_block_size [128, 128]`: FP8 block. `modules_to_not_convert`
  names the bf16 linears (a whole name, or a tail after a dot: `mlp.gate` is the router, not `gate_proj`).
- `quant_method: "modelopt"`, `quant_algo: "NVFP4"`, 4-bit weights in groups of 16, or the same block in
  `hf_quant_config.json`: ModelOpt NVFP4. `ignore`, `exclude_modules` and `modules_to_not_convert` take names and
  `*` globs.
- No `quantization_config`: bf16.
- `quant_method: "exl3"` is refused with a message naming tracker #832. The extension point is a variant in
  `config.Quant`, its side tensors in `names.side()` and a reader in `weights_proj.Reader.linear()`.

A file tensor whose dtype or shape disagrees with the format the config gives it is refused, naming the module and
both formats. Each projection loads as a `qlinear.Weight` (`bf16`, `fp8g` or `nvfp4`) through main's packers
(`modelopt.Packer`, `fp8.packScales`); routed experts load as `experts.Layer` (`fp8g` or `nvfp4`, packed by
`Fp8Experts`/`Nvfp4Experts`'s layouts on the GPU: `Packer.fp8Experts` and `Packer.fp4Experts`). There is no grouped bf16 expert kernel, so bf16 routed experts load as one lane weight
per projection. The router must be bf16 (GLM's router kernel reads bf16); a quantized router is refused.

### Shapes

`config.parse` refuses, with a named error and the reason, every shape the kernels don't take, before anything is
allocated or launched: KDA heads other than 128 wide, short convs other than 4 taps, key/value heads other than the
query heads (`KdaGeometry`); a latent other than 512, nope or value heads other than 128, rope columns other than 64,
NeoX rope pairs, a `q_lora_rank` without a norm instance (`MlaGeometry`); a hidden width other than 512, 1536 or 4096,
MLP and expert widths outside whole 128-column blocks (`Width`); a tied head (`TiedHead`); routing past `glue.select`'s
bounds (`Routing`).

### Loading

Routed experts load in batches of `weights.Options.stage` bytes (512 MiB, 32 FP8 experts): eight threads copy the batch's
raw tensors from the mapped file into pinned memory, one upload moves them to a device stage of the same size, and the
packers lay them out in the layer's buffers; the batch's file pages are then dropped. Scales are collected on the host
and uploaded once a layer. `ling-pack DIR LAYER [EXPERTS [STAGE]]` times it and checks every packed byte against the
host packers (`Fp8Experts`/`Nvfp4Experts.packGateUp`, `packOne`). Real layer 4 (512 FP8 experts, 7.5 GiB), from local
disk / the sshfs mount:

| Loader | cold load | host peak (VmHWM) | device peak |
| --- | --- | --- | --- |
| host packers per expert (c48be499) | 14.3 s / ~157 s | 8.0 GiB (the layer's mapped pages) | 7.9 GiB |
| batches packed on the GPU | 6.1 s / 122 s | 1.3 GiB | 8.4 GiB (+ one stage) |

From local disk the load runs at the read's rate (5-7 s to read the layer once single-threaded); from the mount the
network bounds it (86 MiB/s here: ~1.9 h for the 572 GB release).

### The SwiGLU clamp

Layers 64-71 clamp the shared expert and layers 65-71 the routed experts at L = 10 as inclusionAI's vLLM
(`SwigluStepAndMul`) and SGLang (`BailingMoeV3MLP`) do: `min(silu(g), L) * clip(u, -L, L)`, the limit on SiLU's output.
GLM's glue and the grouped expert kernels clamp the gate before SiLU, so Ling launches its own instances: the glue's
`tf_ling_swiglu_512`/`_64` (`glue.swigluWith`) for the shared expert, dense and bf16 routed experts, and epilogue 4 of
the FP8 and NVFP4 grouped kernels (decode and prompt), chosen by `experts.Layer.after`, which the Ling loader sets.
Every kernel that existed before has the same SASS and resource usage as at a33dde0c (`cuobjdump`, all 40 fatbins);
`ling-clamp-old` runs the kept FP8 (12 calls) and NVFP4 (13 calls) instances against a33dde0c's fatbins byte for byte.

At L = 10 the two orders give the same bf16 bytes: the kernels (and HF's bf16 `act_fn`) round SiLU to bf16 first, and
bf16(silu(g)) is 10 for every bf16 g >= 10. They differ at small limits (L = 2: 16,255 of the 65,536 bf16 gates).
`ling-clamp DIR` checks Ling's order against float64 at L = 10 and L = 2 on gates and ups straddling L: the glue at
widths 1024 and 704 (worst 0.93 of 2^-7 relative; 0 bytes differ from GLM's order at 10, 6,794 at 2) and the clamped
layer's grouped gate-up, decode and prompt rows (FP8 worst 0.090, NVFP4 0.102 of the bound). a33dde0c's binary and
this one print the same digits on the real layers (71: rel L2 4.125e-3, 3.747e-3 without 7 flipped rows, decode rows
3.893e-3; 65: 4.851e-3, 3.347e-3 without 14, 3.325e-3) and on the synthetic forward in all three formats, whose
float64 references do not change either (the synthetic clamped layer never reaches g > 10). HF's clamp itself moves
layer 71's output by 3.7e-3 (rel L2 against the unclamped layer) and layer 65's by 0 on these 512 tokens.

## Serving

`engine.Engine` owns the weights (every backbone layer, or `Options.layers` for a thin model), the graph, three
`Buffers` (prompt chunks, one stream's window, shared windows) and one `State` per slot. `Lanes` is the lane backend
(GLM's `cuda_lanes.zig` without the ranks): `prefill`/`prefill_step` (a chunk at a time, other streams' rounds
between), `first`, `verify` (one stream's window, or every stream's in one forward through `verifyShared`), `keep`
(a window's prefix, the KDA states replayed), `release`. Greedy draws take GLM's `tf_glm_top1`, sampled ones
upstream's `sample.cu` through GLM's sampler at TP=1. `native.zig` gives the server `open`, the facts (exact windows of
16 rows, shared rounds unless `TF_LING_SHARED=0`; `TF_LING_PROMPT_STEPS=0` passes prompts whole) and the prompt
cache's `snap_*` hooks: a kept state is the sequence's arena copied whole at a mark, and the lane backend restores it
inside its own prompt pass. Ling's prompt arithmetic does not depend on where a chunk ends, so a mark may fall on any
row (no `snap_grid`): the pass cuts its chunk there. There is no recorder hook (`features`): the drafter recorder
(#848) is not on this branch.

`ling-lanes DIR OUT [REF] [--layers a,b]` drives the backend under upstream's round loop as the server's lane host
does, prompts in 37-row chunks; every check compares tokens, and the engine-level ones bytes:

- chunked == whole: chunks of 512, 37 and 16 rows and the whole `prefill` give the same reply; the states after
  37- and 16-row chunks equal the whole pass's byte for byte (KDA states, conv windows, MLA rows).
- drafted == plain: drafts from the plain reply (2 to 15 a round, most rounds one of them wrong, so windows of 3 to
  16 rows are cut anywhere), greedy and sampled; a 9-row window kept to 4 then a 5-row window equal 9 single rows in
  tokens, logits and state bytes.
- resumed == fresh: states kept at marks 37, 101, 150, 263, 349 (off the chunk grid, not multiples of 16), each
  resumed pass equals the fresh one, and a different tail after a kept 263; resumed states equal the whole pass's.
- concurrent == solo: nine cases of three streams (at once, staggered, reversed, the same prompt twice, sampled,
  one discarded, prompts a chunk at a time between the others' rounds, drafted or not) equal their solo replies;
  shared windows of 2 streams over 8 rounds (1 to 16 rows each) equal lone twins in logits, tokens and states.
- cancels in the prompt pass, discards between rounds, eos and `max_new` 1-3.
- the reply equals the serial forward's argmax at every row; OUT.ids and OUT.picks feed
  `ling_forward.py --ids OUT.ids --picks OUT.picks`, and with REF the served logits are checked against float64:

| Format | served vs float64 (same picks) | vs the bf16 points | bf16 points vs float64 | argmax vs float64 |
| --- | --- | --- | --- | --- |
| bf16 | 2.24e-2 | 1.27e-2 | 2.25e-2 | 301/309 |
| FP8 block | 2.09e-2 | 1.18e-2 | 2.10e-2 | 302/309 |
| NVFP4 | 2.21e-2 | 1.38e-2 | 2.22e-2 | 294/309 |

92 checks pass in each format. On the real FP8 release, `--layers 0,1,2,3` (the real embedding, layers 0-3, final
norm and head; KDA and dense MLP only, 6 GB) passes the same 90 exactness checks; it has no float64 reference (an
MLA + MoE layer adds 8 GB). Mutations caught: committing every row on `keep`, a restore that keeps the fresh KDA copy
bits, chunks not cut at marks, shared draws keyed one position early, a lane commit that keeps one row too many.

## MTP drafting

The HF model (`BailingMoeV3MTPLayer`) feeds the MTP layer the backbone's rows after `model.norm` and the next
token's embedding: `eh_proj(enorm(embed(t+1)) | hnorm(h_t))`, then input norm, MLA, residual, post-attention norm,
MoE, residual, `final_layernorm`, and `lm_head` reads that. The layer runs unclamped (the SwiGLU limit lists cover
the backbone). `mtp.Head` holds a view of the weights (the MTP layer alone, `final_layernorm` as its `norm`), its
own graph and the concatenated input rows; each sequence gets a second `State` (one MLA cache, its own position).
A prompt pass feeds every row whose next token is known; `Engine.draft` absorbs a verify's kept rows (their
final-norm rows and the tokens after them), then chains up to 3 drafts, each step's hidden row the MTP layer's own
`final_layernorm` output, as GLM's `decode.draft` does; the chain's rows are dropped before the next absorb. Kept
prompt states copy the MTP cache too. The server loads it unless `--no-drafts`; the lane facts then say `mtp`.

Checked (`ling-lanes ... --mtp`, 104 checks a format; acceptance on random weights says nothing about quality and is
not reported as such): MTP-drafted replies, greedy and sampled, equal the plain ones; so do MTP-drafted passes
resumed at 53 and 211 and every concurrent case; a prompt pass's MTP cache in 37-row chunks equals the serial MTP
rows byte for byte; twins drafting chains of 3 and of 1 agree on tokens and first drafts over 6 windows, each
chain's second draft equals one computed by hand, and after one more window both states are equal byte for byte.
The MTP logits at every prompt row against `ling_forward.py --mtp` (float64 from the file, picks free):

| Format | vs float64, rows routed alike | rows whose MTP picks differ | the bf16 points vs float64 on those rows | argmax |
| --- | --- | --- | --- | --- |
| bf16 | 1.72e-2 | 14 of 308 | 2.47e-2 | 293/308 |
| FP8 block | 1.65e-2 | 5 | 2.38e-2 | 300/308 |
| NVFP4 | 3.20e-2 | 11 | 7.81e-2 | 283/308 |

The real layer (`ling-mtp-real`, fixture from `oracle/ling_mtp.py`: HF's `BailingMoeV3MTPLayer` in bf16 on 511 real
tokens' embeddings and `model.norm` of layer 0's output): `final_layernorm` rows at rel L2 7.28e-3 of HF on the 488
rows whose top-8 agrees (4.56e-2 with the 23 others), worst row 1.61e-2, decode rows 6.74e-3; cut at 300 == uncut and
a 16-row window == 16 decode rows byte for byte; 8.4 GB peak, 153 s to load from the mount. Mutations caught: no
rollback of the chain's rows, `hnorm` with `enorm`'s weight, prompt rows absorbing their own token, kept states
without the MTP cache, the chain's hidden row taken from row 0.

## What is missing

- A full-checkpoint run: 574 GB in FP8 needs five Sparks and tensor parallelism.
- Speed: nothing is tuned. Projections land in their columns through a byte copy, bf16 routed experts run densely,
  and the RMS norms take hidden and `q_lora_rank` widths 512, 1536 and 4096 only (`glue.rmsnorm`).
- EXL3, through ashhart's EXL3 module (tracker #832).
- Tensor parallelism: the loader is TP=1; shares and transport would come from #804's `cuda_shard.zig` and #855.

## Testing without the full model

```bash
zig build test-ling                                    # config, name map, tiny configs (host only)
TF_LING_INDEX=/path/to/model.safetensors.index.json zig build test-ling   # adds the real index
zig build -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc
zig-out/bin/tf-cuda-test ling-synth /tmp/ling-fp8 --format fp8      # bf16 | fp8 | nvfp4, --layers N (default 4)
zig-out/bin/tf-cuda-test ling-weights /tmp/ling-fp8
python zig/tests/cuda/oracle/ling_forward.py /tmp/ling-fp8 /tmp/ref                 # float64 reference, free routing
zig-out/bin/tf-cuda-test ling-forward-synth /tmp/ling-fp8 /tmp/ref /tmp/picks       # exactness; writes the picks
python zig/tests/cuda/oracle/ling_forward.py /tmp/ling-fp8 /tmp/fref --picks /tmp/picks
zig-out/bin/tf-cuda-test ling-forward-synth /tmp/ling-fp8 /tmp/fref                 # same picks: logits bounded
zig-out/bin/tf-cuda-test ling-lanes /tmp/ling-fp8 /tmp/served                    # serving checks; writes .ids/.picks
python zig/tests/cuda/oracle/ling_forward.py /tmp/ling-fp8 /tmp/sref --ids /tmp/served.ids --picks /tmp/served.picks
zig-out/bin/tf-cuda-test ling-lanes /tmp/ling-fp8 /tmp/served /tmp/sref          # and the served logits vs float64
zig-out/bin/tf-cuda-test ling-lanes /tmp/ling-fp8 /tmp/m --mtp                     # the same with MTP drafts
python zig/tests/cuda/oracle/ling_forward.py /tmp/ling-fp8 /tmp/mref --ids /tmp/m.ids --picks /tmp/m.picks --mtp /tmp/mref.mtp
zig-out/bin/tf-cuda-test ling-lanes /tmp/ling-fp8 /tmp/m /tmp/mref --mtp           # and the MTP logits vs float64
zig-out/bin/tf-cuda-test ling-clamp /tmp/ling-fp8                  # Ling's SwiGLU order vs float64, L 10 and 2
zig-out/bin/tf-cuda-test ling-clamp-old /tmp/ling-fp8 OLD/fatbin   # kept expert instances vs an older build's bytes
```

`ling-synth` writes `config.json` (the real file's schema) and `model.safetensors` with the real key names: hidden 512,
4 heads, 16 experts in 4 groups keeping 2, `q_lora_rank` 512 (a width the norm kernel takes), vocabulary 1,024, layer
groups of 2 (KDA, MLA, KDA, MLA), layer 0 dense, the last layer clamped, plus the MTP layer. KDA's head dim 128 and
MLA's 512 latent, 64 rope and 128-wide heads stay as the kernels take them. The FP8 file carries the real exception
list's pattern; the NVFP4 file uses globs and leaves layer 0's dense MLP in bf16, so one checkpoint mixes formats. It
checks its names against the weight map before it passes. FP8 codes stay within 2^-3 to 3.75 so a forward stays in
range. Sizes: bf16 282 tensors, 45 MiB; FP8 517, 24 MiB; NVFP4 978, 16 MiB.

`ling-weights` loads every layer and checks every projection (decode and prompt rows, 3 rows) against float64 from
the file, the grouped experts' gate-up (SwiGLU) and down, and each small tensor (norms, convs as fp32 q|k|v, `A_log`,
`dt_bias`, router, bias) byte for byte. Measured: bf16 85 projections, FP8 and NVFP4 61 each, worst 0.433 of the bound
(2^-7 of |y| plus 2^-12 of the sum of |x w|); 37 small tensors equal.

`ling-init-faults DIR [--mtp]` opens the backend (engine, sequence 0, lane backend) on `fake_driver.zig`, a host
stand-in for the driver, once cleanly and then with each of its driver calls, and each of its host allocations,
failing in turn: every failed open must leave nothing allocated, held or freed twice (no GPU work; FP8 with `--mtp`
is 869 calls and 394 allocations). `ling-slots` checks the lane backend's slot choice in the release build.

With the real checkpoint:

```bash
python zig/tests/cuda/oracle/ling_weights.py /path/to/Ling-3.1-flash /tmp/ling-ref   # needs torch, safetensors
zig-out/bin/tf-cuda-test ling-weights-real /path/to/Ling-3.1-flash /tmp/ling-ref
```

`ling-ckpt-check DIR [INDEX]` reads every shard's header and nothing else: each of the release's 214,025 tensors is
one the weight map names, at the dtype and shape the loader reads (`names.dims`, the loader's own table, and each
format's side tensors), none missing, and their bytes add up to the index's `total_size` (573,753,281,152). Over a
network mount it takes seconds.

`ling-mtp-real DIR FIXTURE` runs the MTP layer (fixture from `oracle/ling_mtp.py SRC MODELING_PKG ROWS.pt OUT`).
`ling-layer-real DIR FIXTURE` runs one whole layer (fixture from `oracle/ling_layer.py CHAIN L OUT`, the captured
HF chain's `h<L>.pt`, `h<L+1>.pt`, `r<L>.pt`). `ling-weights-real` loads layers 0, 4 and 7 with experts 0-3 (769 MiB on the device) and compares 35 products with torch's float64
on the same 8 rows: worst 0.414 of the bound, relative L2 at most 1.7e-3 for bf16 outputs and 6.9e-8 for the
experts' fp32 down rows.

## Memory

Computed from `config.json` (572.1 B parameters; routed experts are 97.1 %). Quantized formats keep the FP8 release's
bf16 modules (embedding, head, routers, KDA `b_proj`, MLA `g_proj`, MTP `eh_proj`). The FP8 figure equals the
checkpoint's shard total. EXL3 assumes every other linear at the given rate.

| Format | Weights | Sparks, 128 GB each | Sparks, 100 GB of weights each |
| --- | --- | --- | --- |
| bf16 | 1,144 GB | 9 | 12 |
| FP8 block (the release) | 574 GB | 5 | 6 |
| ModelOpt NVFP4 | 324 GB | 3 | 4 |
| EXL3 2.5 bpw | 181 GB | 2 | 2 |
| EXL3 1.25 bpw | 92 GB | 1 | 1 |

The first count is weights alone; the second leaves room for KV cache, activations and the system. Loaded, the FP8
lane weights add 0.94 GB of fp32 scale tiles (one per 64 inputs) on the non-expert linears; NVFP4 loads at its stored
size.

## How to help

Each item stands alone.

1. **Recorder.** Give `Lanes` a `features` hook (the final-norm rows `Engine.verify` already keeps) once #848 lands.
2. **MTP acceptance.** Measure the MTP layer's acceptance on real text once a whole model runs (TP).
3. **Fused projections.** Same-format q|k|v (and gate|up) rows as one weight per format, without the column copies.
4. **Bf16 routed experts.** Add a grouped bf16 expert kernel, or quantize at load, and drop the per-expert lanes.
5. **TP.** Cut `weights.load` by heads and widths as `glm/cuda_shard.zig` does; FP8 scale blocks span 128 rows and
   128 inputs, so FP8 cuts fall on multiples of 128.
6. **EXL3.** Wire #832's reader behind the `config.Quant` extension point and lift the refusal.
7. **An NVFP4 checkpoint.** Quantize the release with ModelOpt, keep the routers in bf16, and run `ling-weights` on
   it; the loader expects `weight`, `weight_scale`, `weight_scale_2` and `input_scale` per linear.
8. **GPU packing of FP8 experts.** They pack on the host today (`Fp8Experts.packGateUp`); `modelopt.Packer` has the
   NVFP4 experts' GPU packer to follow.
