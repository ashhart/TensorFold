# Ling-3.1-flash

Work in progress on the Zig engine's CUDA side: the model is **not served yet**. This page lists what exists, what is
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
| KDA on Ling's layer 0 (q/k/v convs as one q\|k\|v conv), opt-in chunked prompt path | b55aa13d, merged | `glm-kda-ling`, `glm-kda-chunk` |
| Config loader and weight map (all 214,025 names of the real index) | `zig/src/families/ling/config.zig`, `names.zig` | `zig build test-ling` |
| Weight loader at TP=1: bf16, FP8 block, ModelOpt NVFP4 | `zig/src/families/ling/weights.zig`, `weights_proj.zig` | `ling-weights`, `ling-weights-real` |
| Synthetic checkpoints in all three formats | `tf-cuda-test ling-synth` | its own name check |
| The TP=1 layer graph: embedding, KDA/MLA mixers, dense MLP and MoE, final norm, head; windows and commits | `zig/src/families/ling/forward.zig`, `graph.zig`, `state.zig`, `block_*.zig`, `kernels/cuda/ling_glue.cu` | `ling-forward-synth`, `ling-layer-real` |

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
| 71 | MLA, MoE, limits 10 | 4.13e-3 | 7 | 3.75e-3 | 3.89e-3 | 8.3 GB |

exllamav3's port measured 0.4-1.1e-3 without flips (4 flips each on layers 4 and 7) on the same chain: it keeps fp16
activations, the captured HF chain carries an fp32 residual from layer 3's output on, and this graph rounds the stream
to bf16 as the bf16 model does. On every
real layer a prompt cut at 300 equals the uncut prompt and a 16-row window equals 16 decode rows; for FP8 decode rows
equal the prompt's rows (both run the lane). Loading one MoE layer's 512 experts from the sshfs mount takes ~150 s.

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
`Fp8Experts`/`Nvfp4Experts`). There is no grouped bf16 expert kernel, so bf16 routed experts load as one lane weight
per projection. The router must be bf16 (GLM's router kernel reads bf16); a quantized router is refused.

## What is missing

- Lanes hooks (drafted, concurrent and cut-prompt windows) for Ling: the graph has the pieces (segments, windows,
  `commit(keep)`), nothing schedules them yet. Sampling, the server and a full-checkpoint run are not wired.
- MTP drafting: the MTP layer's tensors load (`weights.Mtp`); the graph leaves it unwired. It would be
  `enorm(embed(next)) | hnorm(hidden)` through `eh_proj`, one MLA + MoE layer on its own cache, the final norm and
  the shared head.
- The SwiGLU limit clamps the gate before SiLU (`silu(min(g, 10))`, the grouped expert kernels' and GLM's rule);
  the Ling reference clamps after (`min(silu(g), 10)`). They differ by at most 4.5e-4 on a clamped element.
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

With the real checkpoint:

```bash
python zig/tests/cuda/oracle/ling_weights.py /path/to/Ling-3.1-flash /tmp/ling-ref   # needs torch, safetensors
zig-out/bin/tf-cuda-test ling-weights-real /path/to/Ling-3.1-flash /tmp/ling-ref
```

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

1. **Lanes hooks.** Drive `Forward` from #804's `cuda_lanes.zig` pattern: drafted windows, concurrent streams as
   segments, cut prompts; `ling-forward-synth` already checks the exactness each needs.
2. **MTP.** Wire `weights.Mtp` as a drafter (see "What is missing").
3. **Fused projections.** Same-format q|k|v (and gate|up) rows as one weight per format, without the column copies.
4. **Bf16 routed experts.** Add a grouped bf16 expert kernel, or quantize at load, and drop the per-expert lanes.
5. **TP.** Cut `weights.load` by heads and widths as `glm/cuda_shard.zig` does; FP8 scale blocks span 128 rows and
   128 inputs, so FP8 cuts fall on multiples of 128.
6. **EXL3.** Wire #832's reader behind the `config.Quant` extension point and lift the refusal.
7. **An NVFP4 checkpoint.** Quantize the release with ModelOpt, keep the routers in bf16, and run `ling-weights` on
   it; the loader expects `weight`, `weight_scale`, `weight_scale_2` and `input_scale` per linear.
8. **GPU packing of FP8 experts.** They pack on the host today (`Fp8Experts.packGateUp`); `modelopt.Packer` has the
   NVFP4 experts' GPU packer to follow.
