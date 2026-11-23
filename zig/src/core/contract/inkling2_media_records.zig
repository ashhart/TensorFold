//! Inkling 2's image tower and DSpark draft launches (inkling2/{gemm,vision,draft}.metal).
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

const reduce = std.mem.alignForward(u32, 4 * 33, 16);
const gemm_kinds = [_][]const u8{ "bf16_nt", "bf16_nn", "bf16f_nt", "score_nt", "prob_nn" };
/// The fp32 dots (scores and p v) stage two 16 x 68 float chunks (gemm.metal's IK2_DOT_KC and IK2_DOT_LD).
const dot_tg_memory = 2 * 16 * 68 * 4;

const kernels = [_]M{
    .{ .name = "inkling2 vision ik2v_pack", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 vision ik2v_embed", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 vision ik2v_layer_norm", .threads = 256, .tg_memory = reduce, .reserved = 256 },
    .{ .name = "inkling2 vision ik2v_qkv", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 vision ik2v_softmax", .threads = 256, .tg_memory = reduce, .reserved = 256 },
    .{ .name = "inkling2 vision ik2v_heads", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 vision ik2v_bias_gelu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 vision ik2v_gelu_rows", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 vision ik2v_residual", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 draft ik2d_embed_block", .threads = 256, .tg_memory = reduce, .reserved = 256 },
    .{ .name = "inkling2 draft ik2d_rms", .threads = 256, .tg_memory = reduce, .reserved = 256 },
    .{ .name = "inkling2 draft ik2d_add_rms", .threads = 256, .tg_memory = reduce, .reserved = 256 },
    .{ .name = "inkling2 draft ik2d_mix_rms", .threads = 256, .tg_memory = reduce, .reserved = 256 },
    .{ .name = "inkling2 draft ik2d_qk_prep", .threads = 128, .tg_memory = reduce + 4 * 128, .reserved = 128 },
    .{ .name = "inkling2 draft ik2d_ctx_store", .threads = 128, .tg_memory = reduce + 4 * 128, .reserved = 128 },
    .{ .name = "inkling2 draft ik2d_attn_split", .threads = 96, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 draft ik2d_attn_merge", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "inkling2 draft ik2d_swiglu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 draft ik2d_rnn", .threads = 256, .tg_memory = reduce, .reserved = 256 },
    .{ .name = "inkling2 draft ik2d_cell", .threads = 64, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 draft ik2d_step_logits", .threads = 256, .tg_memory = 8 * 8 * 8, .reserved = 256 },
    .{ .name = "inkling2 draft ik2d_pick", .threads = 1024, .tg_memory = 4 * 2048, .reserved = 1024 },
    .{ .name = "inkling2 draft ik2d_fold_b", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 draft ik2d_alpha", .threads = 64, .tg_memory = 0, .reserved = 64 },
};

pub const records = blk: {
    var out: [kernels.len + gemm_kinds.len * 4 + 1]M = undefined;
    for (kernels, 0..) |k, i| out[i] = k;
    var at: usize = kernels.len;
    for (gemm_kinds) |kind| for (0..2) |m| for (0..2) |n| {
        out[at] = .{
            .name = std.fmt.comptimePrint("inkling2 media ik2_gemm_{s}_{d}{d}", .{ kind, m, n }),
            .threads = 256,
            .tg_memory = if (std.mem.eql(u8, kind, "score_nt") or std.mem.eql(u8, kind, "prob_nn")) dot_tg_memory else 0,
            .reserved = 256,
            .offline = false,
        };
        at += 1;
    };
    // Without tensor units the draft's small-row GEMM runs on 8x8 matrices, one simdgroup a threadgroup.
    out[at] = .{ .name = "inkling2 media ik2_gemm_bf16_nt_rows", .threads = 32, .tg_memory = 0, .reserved = 32, .offline = false };
    break :blk out;
};

test "Inkling 2 Mini media launch declarations fit all Apple families" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
