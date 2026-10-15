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
| MLA at Ling's shape: 64 roped key columns, 128-wide heads, GPT-J rope, head-wise gate | branch `ling-mla`, f8e594f6 | `ling-mla`, `ling-mla-real`, `glm-mla-same` |
| KDA on Ling's layer 0 (q/k/v convs as one q\|k\|v conv), opt-in chunked prompt path | branch `ling-kda`, b55aa13d | `glm-kda-ling`, `glm-kda-chunk` |
| Config loader and weight map (all 214,025 names of the real index) | `zig/src/families/ling/config.zig`, `names.zig` | `zig build test-ling` |
| Weight loader at TP=1: bf16, FP8 block, ModelOpt NVFP4 | `zig/src/families/ling/weights.zig`, `weights_proj.zig` | `ling-weights`, `ling-weights-real` |
| Synthetic checkpoints in all three formats | `tf-cuda-test ling-synth` | its own name check |

The MLA and KDA commits live on their own branches and are not merged into `ling-family`.

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

- The layer graph: `weights.Layer` holds one `qlinear.Weight` per projection, while #804's GLM blocks read raw bf16
  pointers (KDA's fused `[q | k | v | f_a | g_a | b]`, MLA's latent projections). Ling's blocks have to call
  `qlinear.Linear` per projection, or fuse same-format rows.
- Lanes hooks (drafted, concurrent and cut-prompt windows) for Ling.
- MTP drafting: the MTP layer's tensors load (`weights.Mtp`), nothing drafts with them.
- EXL3, through ashhart's EXL3 module (tracker #832).
- Tensor parallelism: the loader is TP=1; shares and transport would come from #804's `cuda_shard.zig` and #855.

## Testing without the full model

```bash
zig build test-ling                                    # config, name map, tiny configs (host only)
TF_LING_INDEX=/path/to/model.safetensors.index.json zig build test-ling   # adds the real index
zig build -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc
zig-out/bin/tf-cuda-test ling-synth /tmp/ling-fp8 --format fp8      # bf16 | fp8 | nvfp4, --layers N (default 4)
zig-out/bin/tf-cuda-test ling-weights /tmp/ling-fp8
```

`ling-synth` writes `config.json` (the real file's schema) and `model.safetensors` with the real key names: hidden 512,
4 heads, 16 experts in 4 groups keeping 2, `q_lora_rank` 256, vocabulary 1,024, layer groups of 2 (KDA, MLA, KDA,
MLA), layer 0 dense, the last layer clamped, plus the MTP layer. KDA's head dim 128 and MLA's 512 latent, 64 rope and
128-wide heads stay as the kernels take them. The FP8 file carries the real exception list's pattern; the NVFP4 file
uses globs and leaves layer 0's dense MLP in bf16, so one checkpoint mixes formats. It checks its names against the
weight map before it passes. Sizes: bf16 282 tensors, 43 MiB; FP8 517, 23 MiB; NVFP4 978, 15 MiB.

`ling-weights` loads every layer and checks every projection (decode and prompt rows, 3 rows) against float64 from
the file, the grouped experts' gate-up (SwiGLU) and down, and each small tensor (norms, convs as fp32 q|k|v, `A_log`,
`dt_bias`, router, bias) byte for byte. Measured: bf16 85 projections, FP8 and NVFP4 61 each, worst 0.456 of the bound
(2^-7 of |y| plus 2^-12 of the sum of |x w|); 37 small tensors equal.

With the real checkpoint:

```bash
python zig/tests/cuda/oracle/ling_weights.py /path/to/Ling-3.1-flash /tmp/ling-ref   # needs torch, safetensors
zig-out/bin/tf-cuda-test ling-weights-real /path/to/Ling-3.1-flash /tmp/ling-ref
```

It loads layers 0, 4 and 7 with experts 0-3 (769 MiB on the device) and compares 35 products with torch's float64
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

1. **KDA block on `qlinear`.** Write Ling's KDA block over `weights.Kda`: seven `qlinear.Linear` calls (or one fused
   `[q | k | v]` per format) feeding #804's chain with `conv`, `a_log`, `dt_bias`, `o_norm` as loaded. Check it with
   `ling-synth` plus `glm-kda-ling`'s fixture.
2. **MLA block on `qlinear`.** The same for `weights.Mla` on `ling-mla`'s kernels (`q_a`, `q_b`, `kv_a`, `kv_b`,
   head gate, `dense`).
3. **MoE block.** Route with GLM's `moe.Layer` (`ling.route()` fills top-k, groups and limits), run `experts.Grouped`
   on `Moe.routed`, the shared expert as three projections.
4. **Bf16 routed experts.** Add a grouped bf16 expert kernel, or quantize at load, and drop the per-expert lanes.
5. **TP.** Cut `weights.load` by heads and widths as `glm/cuda_shard.zig` does; FP8 scale blocks span 128 rows and
   128 inputs, so FP8 cuts fall on multiples of 128.
6. **EXL3.** Wire #832's reader behind the `config.Quant` extension point and lift the refusal.
7. **An NVFP4 checkpoint.** Quantize the release with ModelOpt, keep the routers in bf16, and run `ling-weights` on
   it; the loader expects `weight`, `weight_scale`, `weight_scale_2` and `input_scale` per linear.
8. **GPU packing of FP8 experts.** They pack on the host today (`Fp8Experts.packGateUp`); `modelopt.Packer` has the
   NVFP4 experts' GPU packer to follow.
