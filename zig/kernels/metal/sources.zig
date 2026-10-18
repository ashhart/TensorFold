//! Metal kernel sources the engine compiles at run time, embedded in the executable.

/// Nemotron's generated kernels (tools/zig/gen_nemotron_kernels.py).
pub const nemotron = @import("nemotron/kernels.zig");

pub const qwen35 = @import("qwen3_5/kernels.zig");
pub const qwen35_layout = @embedFile("qwen3_5/layout.metal");
/// Gemma 4 dense decode glue around the core lane projections (families/gemma4/kernels.zig).
pub const gemma4_glue = @embedFile("gemma4/glue.metal");
/// Gemma 4 decode glue as the reg lane kernel's prologues and epilogue (families/gemma4/weights.zig).
pub const gemma4_fused = @embedFile("gemma4/fused.metal");
/// Gemma 4 prompt attention on the tensor units (families/gemma4/kernels.zig).
pub const gemma4_prompt_attention = @embedFile("gemma4/prompt_attention.metal");
/// Gemma 4's image tower: gemm_nax and its glue (families/gemma4/vision.zig).
pub const gemma4_vision = @embedFile("prefill/gemm_nax.metal") ++ @embedFile("gemma4/vision.metal");

/// LFM2-MoE: embedding, norms, the short conv, attention, the route and the experts' glue.
pub const lfm2 = @embedFile("lfm2/lfm2.metal");
/// LFM2's prompt attention on the tensor units (nax.h inlined at load, after lfm2.metal's text).
pub const lfm2_attn_nax = @embedFile("lfm2/lfm2_attn_nax.metal");

/// Our glue kernels around them: embedding, MLX's RMS arithmetic, argmax, layout moves.
pub const nemotron_glue = @embedFile("nemotron_glue.metal");
pub const core_row_projection = @embedFile("core/row_projection.metal");
pub const core_tile_qmm = @embedFile("core/tile_qmm.metal") ++ @embedFile("core/tile_qmv.metal") ++
    @embedFile("core/tile_qmv_rows.metal");
pub const core_exl3_projection = @embedFile("core/exl3_projection.metal");
pub const core_exlr8_projection = @embedFile("core/exlr8_projection.metal");
pub const core_exlr8_dense = @embedFile("core/exlr8_dense.metal");
pub const core_exlr8_prompt = @embedFile("core/exlr8_prompt.metal");
pub const core_exlr8_dense_prompt = @embedFile("core/exlr8_dense_prompt.metal");
pub const core_pair = @embedFile("core/pair.metal");

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

/// Sliding Weights: the learned change in the forward and the bounded step on captured rows.
/// Learning in the weights: a low-rank change at every layer, and the gradients through each kind of layer to it.
pub const train = @embedFile("train.metal");
pub const train_mixers = @embedFile("train_mixers.metal");

/// The NAX helpers the prefill files include (`#include "../nax.h"`, inlined before compiling).
pub const nax = @embedFile("nax.h");

/// Flash Next 6-bit (group 32) prompt projections on the tensor units: dense and sorted-expert gather.
pub const flashnext_qmm6 = @embedFile("prefill/qmm6_nax.metal") ++ @embedFile("prefill/qmm6_nax_b.metal");
/// Flash Next's prompt products at any MLX width (WB and WG defined before it; nax.h inlined at load).
pub const flashnext_qmmw = @embedFile("prefill/qmmw_nax.metal");
/// Flash Next prompt-chunk glue: hyper-connection pieces, router rows, top-k, the expert sort, gathers and scatters.
pub const flashnext_prompt = @embedFile("prefill/fn_prompt.metal");
/// Flash Next block selection in GPU-side rounds: per-row metadata from the arena and pooling at absolute blocks.
pub const flashnext_select = @embedFile("prefill/fn_select.metal");
/// Flash Next prompt rows' block scores and sparse attention on the tensor units.
pub const flashnext_attn = @embedFile("prefill/fn_attn.metal");

/// A source file of MLX-exact kernels: compiled as one library, its kernels found by name.
pub const File = struct { name: []const u8, text: []const u8 };

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
    .{ .name = "embed_norm", .text = @embedFile("ops/embed_norm.metal") },
    .{ .name = "elementwise", .text = @embedFile("ops/elementwise.metal") },
    .{ .name = "route", .text = @embedFile("ops/route.metal") },
    .{ .name = "qmv", .text = @embedFile("ops/qmv.metal") },
};

/// Flash Next decode: the DeltaNet window step with every row's independent work at once.
pub const flashnext_gdn = @embedFile("decode/fn_gdn.metal");
/// Flash Next decode products one input row a simdgroup, for GPUs without tensor units (0.6.6's rows.py).
pub const flashnext_rows = @embedFile("decode/fn_rows.metal");
pub const flashnext_rows_hc = @embedFile("decode/fn_rows_hc.metal");

/// GLM-5.3-Flash's decode kernels from our Python family (tools/zig/gen_glm_kernels.py).
pub const glm = @import("glm/kernels.zig");
/// GLM-5.3-Flash's glue: MLX's arithmetic where the Python family calls MLX ops; selection, argmax, small steps.
pub const glm_glue = @embedFile("glm_glue.metal");
/// Its products at the widths the generated kernels do not read, after core_row_projection (glm/wide.zig).
pub const glm_rows = @embedFile("glm_rows.metal");
pub const glm_gather = @embedFile("glm_gather.metal");
/// GLM-5.3-Flash's latent attention for rows that read every key: 64 heads as one matrix on the tensor units.
pub const glm_attn = @embedFile("glm_attn.metal");
/// GLM-5.3-Flash's image tower: the prefill GEMM template's projections and attention products, then its own steps.
pub const glm_vision = @embedFile("prefill/gemm_nax.metal") ++ @embedFile("glm_vision.metal");
/// The Qwen-VL image tower (core/qwen_vl): the prefill GEMM template's products, then its own steps.
pub const core_qwen_vision = @embedFile("prefill/gemm_nax.metal") ++ @embedFile("core/qwen_vision.metal");
/// Qwen-Image-2.1's transformer: the prefill attention template, then its own steps (projections: core_qwen_vision).
pub const qwen_image = @embedFile("prefill/attention_nax.metal") ++ @embedFile("qwen_image.metal");
/// A prompt chunk's KDA layer in three passes, appended to the generated kda_rows source (its helpers).
pub const glm_kda_prompt = @embedFile("glm_kda_prompt.metal");
/// GLM-5.3-Flash speed-up mode's exchange kernels (families/glm/ep.zig).
pub const glm_ep = @embedFile("glm_ep.metal");
/// A prompt chunk's sparse MLA attention on the tensor units (nax.h inlined at load).
pub const glm_sparse_nax = @embedFile("glm_sparse_nax.metal");
/// A prompt chunk's index scores on the tensor units (nax.h inlined at load).
pub const glm_index_nax = @embedFile("glm_index_nax.metal");
/// A prompt chunk's MLA absorb on the tensor units (nax.h inlined at load).
pub const glm_absorb_nax = @embedFile("glm_absorb_nax.metal");
/// A MoE layer's route in two launches (core/moe_route.zig): router logits, then the top-k and the expert groups.
pub const core_moe_route = @embedFile("core/moe_route.metal");
pub const core_affine_mm = @embedFile("core/affine_mm.metal");
/// A hyper-connection block boundary in two launches (core/hc.zig).
pub const core_hc_boundary = @embedFile("core/hc_boundary.metal");
/// MLX's precise row softmax and its embedding and RMS kernels, as the replicas in ops/ write them.
pub const ops_softmax = @embedFile("ops/softmax.metal");
pub const ops_embed_norm = @embedFile("ops/embed_norm.metal");
/// The Flash Next checked-in kernels' embedded texts (flashnext_checked.py).
pub const flashnext_gen = @import("flashnext/sources_gen.zig");
/// Nemotron's packed kernels prebuilt as one metallib, for a macOS whose runtime compiler refuses uint4b_format.
pub const packed_metallib = @import("nemotron_packed_metallib").bytes;
/// The core lane projection (tensor units): every window row's sums in one weight read, and its row order.
pub const core_lane_projection = @embedFile("core/lane_projection.metal");
pub const core_lane_order = @embedFile("core/lane_order.metal");
/// Lane rows attending a shared context through an explicit mask (core/shared_attention.zig).
pub const core_shared_attention = @embedFile("core/shared_attention.metal");
/// The core DeltaNet kernels (core/deltanet_kernels.zig).
pub const core_deltanet = @embedFile("core/deltanet.metal") ++ "\n" ++ @embedFile("core/deltanet_slots.metal");
/// 4-bit projections on simdgroup-matrix tiles before the M5 (core/simd_qmm.zig).
pub const core_simd_qmm = @embedFile("core/simd_qmm.metal");
/// 2- to 8-bit projections on the same tiles, rows written at a stride (core/simd_qmm_bits.zig).
pub const core_simd_qmm_bits = @embedFile("core/simd_qmm_bits.metal");
/// Step-5's glue at StepFun's rounding points: embed, norms, q/k rotation, gate, SwiGLU, residual adds.
pub const step5_glue = @embedFile("step5/glue.metal");
/// Step-5's sparse indexer (summaries, scores, top-k) and its sliding, sparse and dense attention.
pub const step5_sparse = @embedFile("step5/sparse.metal");
/// Step-5's prompt attention on the tensor units (a KV group's 16 heads one operand; nax.h inlined at load).
pub const step5_attend_nax = @embedFile("step5/attend_nax.metal");
/// Step-5's image tower (perception encoder, downsampler convs, projector) on gemm_nax's tensor-unit GEMMs.
pub const step5_vision = @embedFile("prefill/gemm_nax.metal") ++ @embedFile("step5/vision.metal");
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
    pub const sample = @embedFile("drama3/sample.metal");
    pub const codec = @embedFile("drama3/codec.metal");
};
