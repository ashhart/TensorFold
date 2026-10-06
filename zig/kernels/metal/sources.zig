//! Metal kernel sources the engine compiles at run time, embedded in the executable.

/// Nemotron's generated kernels (tools/zig/gen_nemotron_kernels.py).
pub const nemotron = @import("nemotron/kernels.zig");

/// Our glue kernels around them: embedding, MLX's RMS arithmetic, argmax, layout moves.
pub const nemotron_glue = @embedFile("nemotron_glue.metal");

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
pub const flashnext_qmm6 = @embedFile("prefill/qmm6_nax.metal");
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
