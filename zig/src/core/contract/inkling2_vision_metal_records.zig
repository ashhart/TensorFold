//! Inkling 2 Mini image-tower launches (zig/kernels/metal/inkling2/vision*.metal).
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

const reduce = std.mem.alignForward(u32, 4 * 33, 16);
const gemm_kinds = [_][]const u8{ "bf16_nt", "bf16f_nt", "f32_nt", "f32_nn" };

const kernels = [_]M{
    .{ .name = "metal inkling2 vision ik2v_pack", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 vision ik2v_embed", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 vision ik2v_layer_norm", .threads = 256, .tg_memory = reduce, .reserved = 256 },
    .{ .name = "metal inkling2 vision ik2v_qkv", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 vision ik2v_softmax", .threads = 256, .tg_memory = reduce, .reserved = 256 },
    .{ .name = "metal inkling2 vision ik2v_heads", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 vision ik2v_bias_gelu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 vision ik2v_gelu_rows", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 vision ik2v_residual", .threads = 256, .tg_memory = 0, .reserved = 256 },
};

pub const records = blk: {
    var out: [kernels.len + gemm_kinds.len * 4 + 1]M = undefined;
    for (kernels, 0..) |k, i| out[i] = k;
    var at: usize = kernels.len;
    for (gemm_kinds) |kind| for (0..2) |m| for (0..2) |n| {
        out[at] = .{
            .name = std.fmt.comptimePrint("metal inkling2 vision ik2_gemm_{s}_{d}{d}", .{ kind, m, n }),
            .threads = 256,
            .tg_memory = 0,
            .reserved = 256,
            .offline = false,
        };
        at += 1;
    };
    // Without tensor units the small-row GEMM runs on 8x8 matrices, one simdgroup a threadgroup.
    out[at] = .{ .name = "metal inkling2 vision ik2_gemm_bf16_nt_rows", .threads = 32, .tg_memory = 0, .reserved = 32, .offline = false };
    break :blk out;
};

test "Inkling 2 Mini image-tower launch declarations fit all Apple families" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
