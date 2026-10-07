//! Metal kernel sources the engine compiles at run time, embedded in the executable.

/// Nemotron's generated kernels (tools/zig/gen_nemotron_kernels.py).
pub const nemotron = @import("nemotron/kernels.zig");

pub const qwen35 = @import("qwen3_5/kernels.zig");
pub const qwen35_layout = @embedFile("qwen3_5/layout.metal");

/// Our glue kernels around them: embedding, MLX's RMS arithmetic, argmax, layout moves.
pub const nemotron_glue = @embedFile("nemotron_glue.metal");
pub const core_row_projection = @embedFile("core/row_projection.metal");

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

/// The NAX helpers the prefill files include (`#include "../nax.h"`, inlined before compiling).
pub const nax = @embedFile("nax.h");

/// Flash Next 6-bit (group 32) prompt projections on the tensor units: dense and sorted-expert gather.
pub const flashnext_qmm6 = @embedFile("prefill/qmm6_nax.metal") ++ @embedFile("prefill/qmm6_nax_b.metal");
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

/// Flash Next decode: the lane projection (lane_qmm's sums, the next group read ahead) for the target's dense rows.
pub const flashnext_lane = @embedFile("decode/fn_lane.metal");
/// Flash Next decode: the DeltaNet window step with every row's independent work at once.
pub const flashnext_gdn = @embedFile("decode/fn_gdn.metal");

/// GLM-5.3-Flash's decode kernels from our Python family (tools/zig/gen_glm_kernels.py).
pub const glm = @import("glm/kernels.zig");
/// GLM-5.3-Flash's glue: MLX's arithmetic where the Python family calls MLX ops; selection, argmax, small steps.
pub const glm_glue = @embedFile("glm_glue.metal");
/// GLM-5.3-Flash's latent attention for rows that read every key: 64 heads as one matrix on the tensor units.
pub const glm_attn = @embedFile("glm_attn.metal");
/// A prompt chunk's KDA layer in three passes, appended to the generated kda_rows source (its helpers).
pub const glm_kda_prompt = @embedFile("glm_kda_prompt.metal");
/// A prompt chunk's sparse MLA attention on the tensor units (nax.h inlined at load).
pub const glm_sparse_nax = @embedFile("glm_sparse_nax.metal");
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
