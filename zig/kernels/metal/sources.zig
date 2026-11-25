//! Metal kernel sources the engine compiles at run time, embedded in the executable.

const std = @import("std");

/// DeepSeek-V4.1's sources live beside their kernels: users say sources.dsv41.NAME.
pub const dsv41 = @import("deepseek_v41/sources.zig");

pub const hy4_flash_indexpool = @embedFile("hy4_flash/indexpool.metal");
pub const hy4_flash_index_recall = @embedFile("hy4_flash/index_recall.metal");
pub const hy4_flash_index_build = @embedFile("hy4_flash/index_build.metal");
pub const hy4_flash_attention_sink = @embedFile("hy4_flash/attention_sink.metal");
pub const hy4_flash_mla_project = @embedFile("hy4_flash/mla_project.metal");
pub const hy4_flash_mla_attention = @embedFile("hy4_flash/mla_attention.metal");
pub const hy4_flash_ihc = @embedFile("hy4_flash/ihc.metal");
pub const hy4_flash_rows = @embedFile("hy4_flash/rows.metal");
pub const hy4_flash_mla_nax = @embedFile("hy4_flash/mla_nax.metal");
pub const hy4_flash_route_rows = @embedFile("hy4_flash/route_rows.metal");
pub const hy4_flash_engram = @embedFile("hy4_flash/engram.metal");
pub const hy4_flash_affine_experts = @embedFile("hy4_flash/affine_experts.metal");
pub const hy4_flash_tensor_dispatch = @embedFile("hy4_flash/tensor_dispatch.metal");

/// Nemotron's generated kernels (tools/zig/gen_nemotron_kernels.py).
pub const nemotron = @import("nemotron/kernels.zig");
/// Nemotron's KV in pages of the core pool: twins of its seven K/V kernels, each library beside its original.
pub const nemotron_paged = @import("nemotron/paged.zig");
/// The 27B's KV in pages of the core pool: its attention sources with the K/V address through a stream's table.
pub const qwen27_paged = @import("qwen27/paged.zig");

pub const qwen35 = @import("qwen3_5/kernels.zig");
pub const qwen35_layout = @embedFile("qwen3_5/layout.metal");
/// The core KV pool's page address (core/metal_kv_pages.zig checks its page size).
pub const kv_pages = @embedFile("core/kv_pages.h");
/// Qwen3.5-2B's KV in pages of the core pool: its layout kernels' helpers, the page address, the paged kernels.
pub const qwen35_paged = @embedFile("qwen3_5/layout.metal") ++ kv_pages ++ @embedFile("qwen3_5/paged.metal");
/// Gemma 4 dense decode glue around the core lane projections (families/gemma4/kernels.zig).
pub const gemma4_glue = @embedFile("gemma4/glue.metal");
/// Gemma 4 decode glue as the reg lane kernel's prologues and epilogue (families/gemma4/weights.zig).
pub const gemma4_fused = @embedFile("gemma4/fused.metal");
/// Gemma 4 prompt attention on the tensor units (families/gemma4/kernels.zig).
pub const gemma4_prompt_attention = @embedFile("gemma4/prompt_attention.metal");
/// The u32 copy kernels' macros (core/copy_u32.metal), ahead of the glue that instantiates them.
pub const core_copy_u32 = @embedFile("core/copy_u32.metal");
/// A hyper-connection stream's RMS from its partial sums (core/stream_rinv.metal), ahead of the HC glue that calls it.
pub const core_stream_rinv = @embedFile("core/stream_rinv.metal");
/// The prefill GEMM template and its one kernel macro, ahead of every family's gemm_nax instantiations.
const gemm_nax = @embedFile("prefill/gemm_nax.metal") ++ @embedFile("prefill/gemm_nax_kernel.metal");
/// Shared pointwise activations (core/activations.metal): bf16 SiLU, the tanh GELU's inner term.
const core_activations = @embedFile("core/activations.metal");
/// The image towers' shared glue: bf16 rounding, threadgroup sum and max, row softmax, LayerNorm, bias.
const vision_helpers = core_activations ++ @embedFile("prefill/vision_glue.metal");
/// gemm_nax and the image towers' shared glue.
const vision_glue = gemm_nax ++ vision_helpers;
/// Gemma 4's image tower: gemm_nax and its glue (families/gemma4/vision.zig).
pub const gemma4_vision = vision_glue ++ @embedFile("gemma4/vision.metal");

/// LFM2-MoE: embedding, norms, the short conv, attention, the route and the experts' glue.
pub const lfm2 = @embedFile("lfm2/lfm2.metal") ++ @embedFile("lfm2/lfm2_segments.metal");
/// LFM2's prompt attention on the tensor units (nax.h inlined at load, after lfm2.metal's text).
pub const lfm2_attn_nax = @embedFile("lfm2/lfm2_attn_nax.metal");
pub const nemotron_prompt_masked = @import("nemotron/prompt_masked.zig");

pub const lfm3 = @embedFile("lfm3/ops.metal");
pub const lfm3_attention = @embedFile("lfm3/attention.metal");
pub const lfm3_image_rows = @embedFile("lfm3/image_rows.metal");
pub const lfm3_projection_dual = @embedFile("lfm3/projectiondual.metal");
pub const lfm3_projection = @embedFile("lfm3/projection.metal");

/// Our glue kernels around them: embedding, MLX's RMS arithmetic, argmax, layout moves.
pub const nemotron_glue = core_copy_u32 ++ @embedFile("nemotron_glue.metal");
pub const core_row_projection = @embedFile("core/row_projection.metal");
pub const core_row_projection_legacy = @embedFile("core/row_projection_legacy.metal");
pub const core_binned_projection = @embedFile("core/binned_projection.metal");
pub const core_tile_qmm = @embedFile("core/tile_qmm.metal") ++ @embedFile("core/tile_qmv.metal") ++
    @embedFile("core/tile_qmv_rows.metal");
pub const core_exl3_projection = @embedFile("core/exl3_projection.metal");
pub const core_exlr8_projection = @embedFile("core/exlr8_projection.metal");
pub const core_exlr8_dense = @embedFile("core/exlr8_dense.metal");
pub const core_exlr8_prompt = @embedFile("core/exlr8_prompt.metal");
pub const core_exlr8_dense_prompt = @embedFile("core/exlr8_dense_prompt.metal");
pub const core_pair = @embedFile("core/pair.metal");
/// The row-split seam's kernels (see `pair_rows.zig`): `pair.metal` above stays as it is for glm_dsa.
pub const core_pair_rows = @embedFile("core/pair_rows.metal");

/// GLM-5.3 (glm_moe_dsa): MLA with rope over a bf16 latent cache, the DSA indexer and its top-k, fp32 glue.
pub const glm_dsa_mla = @embedFile("glm_dsa/mla.metal");
pub const glm_dsa_index = @embedFile("glm_dsa/index.metal");
pub const glm_dsa_glue = @embedFile("glm_dsa/glue.metal");
pub const glm_dsa_moe_jobs = @embedFile("glm_dsa/moe_jobs.metal");
pub const core_grammar_mask = @embedFile("core/grammar_mask.metal");
pub const counter_probe = @embedFile("core/counter_probe.metal");

/// Routed experts with an expert's member rows two or four at a time (each row's sums unchanged).
pub const nemotron_experts = @embedFile("nemotron_experts.metal");

/// Tree windows: Mamba by parent, attention by logical key position, KV compaction, row gathers, draft top-k.
pub const nemotron_tree = @embedFile("nemotron_tree.metal");

/// A lone stream's GPU-side round: the verify's arguments, the accept, the confidence stop, a row gather.
pub const nemotron_round = @embedFile("nemotron_round.metal");

/// The MTP head's one-row kernels fused (bit-identical to the kernels they replace).
pub const nemotron_head = @embedFile("nemotron_head.metal");

/// Keyed draws over the whole vocabulary where tf_gpu_sample would keep only its 1,024 candidates.
pub const nemotron_sample = @embedFile("nemotron_sample.metal");

/// Sliding Weights: learned change and bounded step on captured rows; low-rank change per layer, with gradients to it.
pub const train = @embedFile("train.metal");
pub const train_mixers = @embedFile("train_mixers.metal");

/// The NAX helpers the prefill files include (`#include "../nax.h"`, inlined before compiling).
pub const nax = @embedFile("nax.h");

/// Flash Next 6-bit (group 32) prompt projections on the tensor units: dense and sorted-expert gather.
pub const flashnext_qmm6 = @embedFile("prefill/qmm6_nax.metal") ++ @embedFile("prefill/qmm6_nax_b.metal");
/// Flash Next's prompt products at any MLX width (WB and WG defined before it; nax.h inlined at load).
pub const flashnext_qmmw = @embedFile("prefill/qmmw_nax.metal");
/// Flash Next prompt-chunk glue: hyper-connection pieces, router rows, top-k, the expert sort, gathers and scatters.
pub const flashnext_prompt = core_copy_u32 ++ core_stream_rinv ++ @embedFile("prefill/fn_prompt.metal");
/// Flash Next block selection in GPU-side rounds: per-row metadata from the arena and pooling at absolute blocks.
pub const flashnext_select = withRotaryScalar(@embedFile("prefill/fn_select.metal"));
/// Flash Next indexer pooling at an image stream's block positions (the standard frame on the shared sessions).
pub const flashnext_pool_pos = withRotaryScalar(@embedFile("prefill/fn_pool_pos.metal"));
/// Flash Next prompt rows' block scores and sparse attention on the tensor units.
pub const flashnext_attn = @embedFile("prefill/fn_attn.metal");
/// Flash Next's attention prep at three positions a row, for prompt rows up to an image prompt's last image.
pub const flashnext_mrope = withRotaryScalar(@embedFile("prefill/fn_mrope.metal"));
pub const flashnext_rotary_sweep = @embedFile("prefill/fn_rotary_sweep.metal");
pub const flashnext_rotary_scalar = @embedFile("prefill/fn_rotary_scalar.metal");
/// `text` with its fn_rotary_scalar.metal include line replaced by that file: the runtime compiler has no include path.
fn withRotaryScalar(comptime text: []const u8) []const u8 {
    @setEvalBranchQuota(1_000_000);
    const include = "#include \"fn_rotary_scalar.metal\"\n";
    const at = std.mem.indexOf(u8, text, include).?;
    return text[0..at] ++ flashnext_rotary_scalar ++ text[at + include.len ..];
}
pub const flashnext_rotary_lookup = @embedFile("prefill/fn_rotary_lookup.metal");
pub const flashnext_rotary_prepare = @embedFile("prefill/fn_rotary_prepare.metal");

/// A source file of MLX-exact kernels: compiled as one library, its kernels found by name.
pub const File = struct { name: []const u8, text: []const u8 };
pub const elementwise = @embedFile("ops/elementwise.metal");
pub const embed_norm = @embedFile("ops/embed_norm.metal");

/// Prompt-chunk kernels (tools/zig/prefill_*.py), bit-identical to MLX 0.32.3's prefill kernels.
pub const prefill = [_]File{
    .{ .name = "attention_nax", .text = @embedFile("prefill/attention_nax.metal") },
    .{ .name = "conv", .text = @embedFile("prefill/conv.metal") },
    .{ .name = "gemm_nax", .text = @embedFile("prefill/gemm_nax.metal") },
    .{ .name = "gemv", .text = @embedFile("prefill/gemv.metal") },
    .{ .name = "glue", .text = @embedFile("prefill/glue.metal") },
    .{ .name = "qmm_nax", .text = @embedFile("prefill/qmm_nax.metal") },
    .{ .name = "scan", .text = @embedFile("prefill/scan.metal") },
    .{ .name = "sort", .text = @embedFile("prefill/sort.metal") },
    .{ .name = "embed_norm", .text = embed_norm },
    .{ .name = "elementwise", .text = elementwise },
    .{ .name = "route", .text = @embedFile("ops/route.metal") },
    .{ .name = "qmv", .text = @embedFile("ops/qmv.metal") },
    .{ .name = "gemv_ops", .text = @embedFile("ops/gemv.metal") },
};

/// Flash Next decode: the DeltaNet window step with every row's independent work at once.
pub const flashnext_gdn = @embedFile("decode/fn_gdn.metal");
/// Flash Next decode products one input row a simdgroup, for GPUs without tensor units (0.6.6's rows.py).
pub const flashnext_rows = @embedFile("decode/fn_rows.metal");
pub const flashnext_rows_hc = core_stream_rinv ++ @embedFile("decode/fn_rows_hc.metal");
/// Flash Next experts grouped by expert for wide decode windows, the recorded expert kernels' bits (gx.zig).
pub const flashnext_gx = @embedFile("decode/fn_gx.metal");
/// The opt-in grouped gate/up decoding an expert's weights once for two pairs (gx_mb2.zig).
pub const flashnext_gx_mb2 = @embedFile("decode/fn_gx_mb2.metal");

/// GLM-5.3-Flash's decode kernels from our Python family (tools/zig/gen_glm_kernels.py).
pub const glm = @import("glm/kernels.zig");
/// GLM-5.3-Flash's glue: MLX's arithmetic where the Python family calls MLX ops; selection, argmax, small steps.
pub const glm_glue = core_copy_u32 ++ @embedFile("glm_glue.metal");
/// Its products at the widths the generated kernels do not read, after core_row_projection (glm/wide.zig).
pub const glm_rows = @embedFile("glm_rows.metal");
pub const glm_gather = @embedFile("glm_gather.metal");
/// GLM-5.3-Flash's latent attention for rows that read every key: 64 heads as one matrix on the tensor units.
pub const glm_attn = @embedFile("glm_attn.metal");
/// A window's rows' dense-attention softmaxes in one launch, appended to ops/softmax.metal (tf_softmax_row).
pub const glm_softmax_rows = @embedFile("glm_softmax_rows.metal");
/// GLM-5.3-Flash's image tower: the prefill GEMM template's projections and attention products, then its own steps.
pub const glm_vision = vision_glue ++ @embedFile("glm_vision.metal");
/// The Qwen-VL image tower (core/qwen_vl): the prefill GEMM template's products, then its own steps.
pub const core_qwen_vision = vision_glue ++ @embedFile("core/qwen_vision.metal");
/// Qwen-Image-2.1's transformer: the prefill attention template, then its own steps (projections: core_qwen_vision).
pub const qwen_image = @embedFile("prefill/attention_nax.metal") ++ vision_helpers ++ @embedFile("qwen_image.metal");
/// Idol's denoiser: the prefill GEMM template, then its own steps and head-64 attention.
pub const idol = @embedFile("prefill/gemm_nax.metal") ++ vision_helpers ++ @embedFile("idol.metal") ++ @embedFile("idol_attention.metal");
/// MOSS-Image: the prefill attention template, then its transformer's and encoder's steps (gemms: core_qwen_vision).
pub const moss_image = @embedFile("prefill/attention_nax.metal") ++ vision_helpers ++ @embedFile("moss_image.metal") ++ @embedFile("moss_image_encoder.metal");
/// Unlimited-OCR-2: its image encoder's and decoder's own steps (projections: core_qwen_vision).
pub const uocr = vision_helpers ++ @embedFile("uocr/vision.metal");
pub const uocr_decoder = vision_helpers ++ @embedFile("uocr/decoder.metal");
/// MOSS-Image's image decoder: its own float32 steps, without a template.
pub const moss_image_decoder = @embedFile("moss_image_decoder.metal");
/// Idol's decoders' own steps and attention (families/idol/decoder_kernels.zig); projections, norms: the denoiser's.
pub const idol_decoder = @embedFile("idol_decoder.metal");
/// Idol's image encoder (FLUX.2's VAE encoder) in fp32.
pub const idol_vae = @embedFile("idol_vae.metal");
/// Idol's DINOv3 ViT-H+/16 image encoder: the prefill GEMM template, then its own bf16 steps and attention.
pub const idol_dino = @embedFile("prefill/gemm_nax.metal") ++ vision_helpers ++ @embedFile("idol_dino.metal");
/// A prompt chunk's KDA layer in three passes, appended to the generated kda_rows source (its helpers).
pub const glm_kda_prompt = @embedFile("glm_kda_prompt.metal");
/// A decode window's fused KDA step, appended to kda_rows' source too: its arithmetic, rows' independent parts at once.
pub const glm_kda_window = @embedFile("glm_kda_window.metal");
/// And on two threadgroups a head (their value columns), then each head's norm: kda_rows' values (Ultra-class GPUs).
pub const glm_kda_parts = @embedFile("glm_kda_parts.metal");
/// Prompt chunks' expert gathers walked in column groups, appended to core/affine_mm.metal at 4 bits, groups of 64.
pub const glm_mm_groups = @embedFile("glm_mm_groups.metal");
/// The router's logits for a one-row window, appended to core/moe_route.metal at GLM's two shapes.
pub const glm_route_one = @embedFile("glm_route_one.metal");
/// GLM-5.3-Flash speed-up mode's exchange kernels (families/glm/ep.zig).
pub const glm_ep = @embedFile("glm_ep.metal");
/// A prompt chunk's sparse MLA attention on the tensor units (nax.h inlined at load).
pub const glm_sparse_nax = @embedFile("glm_sparse_nax.metal");
/// A prompt chunk's index scores on the tensor units (nax.h inlined at load).
pub const glm_index_nax = @embedFile("glm_index_nax.metal");
/// Decode rows' key lists on many threadgroups: a sampled floor marks candidates, a merge selects among them.
pub const glm_select = @embedFile("glm_select.metal");
/// A prompt chunk's MLA absorb on the tensor units (nax.h inlined at load).
pub const glm_absorb_nax = @embedFile("glm_absorb_nax.metal");
/// A MoE layer's route in two launches (core/moe_route.zig): router logits, then the top-k and the expert groups.
pub const core_moe_route = @embedFile("core/moe_route.metal");
pub const core_affine_mm = @embedFile("core/affine_mm.metal") ++ @embedFile("core/affine_mm_glu.metal");
pub const core_affine_nonfinite = @embedFile("core/affine_mm_nonfinite.h");
/// A hyper-connection block boundary in two launches (core/hc.zig).
pub const core_hc_boundary = @embedFile("core/hc_boundary.metal");
/// MLX's precise row softmax and its embedding and RMS kernels, as the replicas in ops/ write them.
pub const ops_softmax = @embedFile("ops/softmax.metal");
pub const ops_embed_norm = @embedFile("ops/embed_norm.metal");
/// The Flash Next checked-in kernels' embedded texts (flashnext_checked.py).
pub const flashnext_gen = @import("flashnext/sources_gen.zig");
/// Flash Next TFQ3 (2D-AffineCoset/half4) routed MoE decode and prefill kernels.
pub const flashnext_tfq3 = @import("flashnext_tfq3/sources.zig");
/// Nemotron's packed kernels prebuilt as one metallib, for a macOS whose runtime compiler refuses uint4b_format.
pub const packed_metallib = @import("nemotron_packed_metallib").bytes;
/// The core lane projection (tensor units): every window row's sums in one weight read, and its row order.
pub const core_lane_projection = @embedFile("core/lane_projection.metal");
pub const core_lane_order = @embedFile("core/lane_order.metal");
/// Lane rows attending a shared context through an explicit mask (core/shared_attention.zig).
pub const core_shared_attention = @embedFile("core/shared_attention.metal");
/// The core DeltaNet kernels (core/deltanet_kernels.zig).
pub const core_deltanet = @embedFile("core/deltanet.metal") ++ "\n" ++ @embedFile("core/deltanet_slots.metal");
/// SenseNova-Loop two-stream mHC routing, stream enter/leave collapse, and norm kernels.
pub const core_sensenova_mhc = @embedFile("core/sensenova_mhc.metal");
/// SenseNova-Loop BF16 row matvec and nax.h prefill GEMM kernels.
pub const qwen27_bf16_linear = @embedFile("qwen27/bf16_linear.metal");
/// SenseNova-Loop AWQ W4A16 g128 lossless FP32-metadata M5 and M3 kernels.
pub const qwen27_awq_w4a16 = @embedFile("qwen27/awq_w4a16.metal") ++ "\n" ++ @embedFile("qwen27/awq_m5_fast.metal");
/// 4-bit projections on simdgroup-matrix tiles before the M5 (core/simd_qmm.zig).
pub const core_simd_qmm = @embedFile("core/simd_qmm.metal");
/// 2- to 8-bit projections on the same tiles, rows written at a stride (core/simd_qmm_bits.zig).
pub const core_simd_qmm_bits = @embedFile("core/simd_qmm_bits.metal");
/// Step-5's glue at StepFun's rounding points: embed, norms, q/k rotation, gate, SwiGLU, residual adds.
pub const step5_glue = @embedFile("step5/glue.metal");
/// MiniMax-M3's glue at transformers' rounding points: norms, rotation, SwiGLU-OAI, router, indexer, decode attention.
pub const minimax_m3_glue = @embedFile("core/bf16_argmax.metal") ++ @embedFile("minimax_m3/glue.metal");
/// MiniMax-M3's prompt attention on the tensor units over each row's kept index blocks.
pub const minimax_m3_attention_nax = @embedFile("minimax_m3/attention_nax.metal");
/// MiniMax-M3's DSpark drafter (norms, q/k into its context rings, block attention, the Markov chain) and the taps.
pub const minimax_m3_dspark = @embedFile("minimax_m3/dspark.metal");
/// MiniMax-M3's decode-window experts expert by expert (after core_row_projection: its sums, each weight read once).
pub const minimax_m3_experts = @embedFile("minimax_m3/experts.metal");
/// incoai's DFlash2 drafter for MiniMax-M3: Qwen3 norms, full RoPE, the block convolution, top-k and the selector.
pub const minimax_m3_dflash2 = @embedFile("minimax_m3/dflash2.metal");
/// MiniMax-M3's indexer block scores on the tensor units (a prompt chunk's).
pub const minimax_m3_index_nax = @embedFile("minimax_m3/index_nax.metal");
/// Flint's glue at the reference's rounding points: embed, norms, q/k rotation, SwiGLU, residual and MoE adds.
pub const flint_glue = @embedFile("flint/glue.metal");
/// Flint's attention on the tensor units: prompt tiles, decode partials and their merge, with sinks.
pub const flint_attention_nax = @embedFile("flint/attention_nax.metal");
/// Step-5's sparse indexer (summaries, scores, top-k) and its sliding, sparse and dense attention.
pub const step5_sparse = @embedFile("step5/sparse.metal");
/// Step-5's prompt attention on the tensor units (a KV group's 16 heads one operand; nax.h inlined at load).
pub const step5_attend_nax = @embedFile("step5/attend_nax.metal");
/// Step-5's image tower (perception encoder, downsampler convs, projector) on gemm_nax's tensor-unit GEMMs.
pub const step5_vision = vision_glue ++ @embedFile("step5/vision.metal");
/// Spark2.5's glue at its PyTorch rounding points: embed, fp32 residual adds and norms, rope, head gate, exact GELU.
pub const spark25_glue = @embedFile("core/bf16_matvec.metal") ++ @embedFile("core/bf16_argmax.metal") ++ @embedFile("spark25/glue.metal");
/// Spark2.5's and Llama's prompt attention on the tensor units (gemma4's prompt kernel; nax.h inlined at load).
const prompt_attention_nax = @embedFile("core/prompt_attention_nax.metal");
/// Spark2.5's prompt attention (core/prompt_attention_nax.metal as sp_prompt_attention).
pub const spark25_attention_nax = "#define TF_PROMPT_ATTENTION sp_prompt_attention\n" ++ prompt_attention_nax;
/// Spark2.5's bf16 prompt projections on gemm_nax's tensor-unit GEMM (nax.h inlined at load).
pub const spark25_gemm = gemm_nax ++ @embedFile("spark25/gemm.metal");
/// Llama's glue at HF's rounding points: embed, a bf16 residual and its norm, rope with bf16 cos and sin, SiLU.
pub const llama_glue = core_activations ++ @embedFile("core/bf16_matvec.metal") ++ @embedFile("core/bf16_argmax.metal") ++ @embedFile("llama/glue.metal");
/// Llama's prompt attention: core/prompt_attention_nax.metal as ll_prompt_attention (its loader: WINDOW and RING 0).
pub const llama_attention_nax = "#define TF_PROMPT_ATTENTION ll_prompt_attention\n" ++ prompt_attention_nax;
/// Llama's bf16 prompt projections on gemm_nax's tensor-unit GEMM (nax.h inlined at load).
pub const llama_gemm = gemm_nax ++ @embedFile("llama/gemm.metal");
/// Ripple's glue at the release's rounding points: embedding, four norms a layer, q/k norm, RoPE, gate, clamp.
pub const ricursive_glue = @embedFile("ricursive/glue.metal");
/// The NVFP4 release's MLP row kernels and dequantization; its tensor-unit GEMM after gemm_nax.metal.
pub const ricursive_fp4 = @embedFile("ricursive/fp4.metal");
pub const ricursive_fp4_gemm = gemm_nax ++ @embedFile("ricursive/fp4_gemm.metal");
/// Ripple's prompt attention on the tensor units (llama's kernel over a window, keys in a ring; nax.h inlined at load).
pub const ricursive_attention_nax = @embedFile("ricursive/attention_nax.metal");
/// Ripple's DFlash2 drafter (norms, head prep, K-tap block convolution, block attention, selector) and its taps.
pub const ricursive_dflash2 = @embedFile("ricursive/dflash2.metal");
/// The kernel contract's planted controls: one kernel breaking each gate (zig/tests/contract/).
pub const core_contract_controls = @embedFile("core/contract_controls.metal");

/// Gemma-only rounded weights and fused lane arithmetic; shared sources retain the released bytes.
pub const gemma4_affine_mm = @embedFile("gemma4/affine_mm.metal");
pub const gemma4_lane_projection = @embedFile("gemma4/lane_projection.metal");

/// Drama 3's kernels (drama3.metallib is built ahead of time); the kernel contract compiles them from this text.
pub const drama3 = struct {
    pub const common = @embedFile("drama3/d3_common.h");
    pub const dense = @embedFile("drama3/dense.metal");
    pub const gdn = @embedFile("drama3/gdn.metal");
    pub const attn = @embedFile("drama3/attn.metal");
    pub const moe = @embedFile("drama3/moe.metal");
    pub const nvfp4 = @embedFile("drama3/nvfp4.metal");
    pub const guidance = @embedFile("drama3/guidance.metal");
    pub const sample = @embedFile("drama3/sample.metal");
    /// codec.metal with its core/copy_u32.metal include inlined, as the offline compiler reads it.
    pub const codec = blk: {
        const text = @embedFile("drama3/codec.metal");
        const include = "#include \"../core/copy_u32.metal\"\n";
        const at = std.mem.indexOf(u8, text, include).?;
        break :blk text[0..at] ++ core_copy_u32 ++ text[at + include.len ..];
    };
};

pub const core_weight_only_body = @embedFile("core/weight_only.h");
pub const core_weight_only = @embedFile("core/weight_only.metal");
pub const core_hc_activation_body = @embedFile("core/hc_activation.h");
pub const core_hc_activation = @embedFile("core/hc_activation.metal");
pub const core_hc_pointwise = @embedFile("core/hc_pointwise.metal");

pub const lfm2_prompt_masked = @import("lfm2/prompt_masked.zig");

/// Qwen3 embeddings' projected-head RMS norm (128-wide Q and K, V unchanged).
pub const qwen3_heads = @embedFile("qwen3/heads.metal");
/// GLM-5.3-Flash's DFlash2 drafter attention: Q/K rotary + normalized GQA partials and their merge.
pub const glm_draft_attention = @embedFile("glm_draft_attention.metal");
/// GLM-5.3-Flash's DFlash2 drafter top-k: row-wise top 16 over vocabulary logits into packed candidates.
pub const glm_draft_topk = @embedFile("glm_draft_topk.metal");
/// GLM-5.3-Flash's DFlash2 inline chain accept: compares verified picks against drafted tokens on the GPU.
pub const glm_draft_accept = @embedFile("glm_draft_accept.metal");
/// GLM-5.3-Flash's DFlash2 inline follow: copies accepted tap rows into the drafter ring ahead of its block.
pub const glm_draft_follow = @embedFile("glm_draft_follow.metal");
/// GLM-5.3-Flash's TFQ3 (2D-AffineCoset/half4) decode and prefill kernels.
pub const glm_tfq3 = @import("glm_tfq3/sources.zig");

/// Inkling 2 Mini's runtime helpers, the exact source compiled by its model and kernel contract.
pub const inkling2_runtime = @embedFile("inkling2/runtime.metal");

/// Fused DFlash2 decode kernels: K-norm+RoPE+append, QKV prep, and dynamic-conv2+residual+RMSNorm.
pub const glm_dflash_opt = @embedFile("glm_dflash_opt.metal");
