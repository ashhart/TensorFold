//! Inkling 2's backbone launches: each kernel's max threads matches its launch, threadgroup bytes are static arrays.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

const attn_groups = 8;

/// Threadgroup floats of the attention chunk kernel: q G*128, rel G*16, merge m/l 2*4*G, merge acc 4*G*128.
fn attentionBytes(g: u32) u32 {
    return 4 * (g * 128 + g * 16 + 2 * 4 * g + 4 * g * 128);
}

pub const records = blk: {
    var out: [14 + 2 * attn_groups]M = undefined;
    const singles = [_]M{
        .{ .name = "inkling2 ops inkling2_embed", .threads = 256, .tg_memory = 4 * 8, .reserved = 256 },
        .{ .name = "inkling2 ops inkling2_rms_norm", .threads = 256, .tg_memory = 4 * 8, .reserved = 256 },
        .{ .name = "inkling2 ops inkling2_copy_rows", .threads = 256, .tg_memory = 0, .reserved = 256 },
        // part[8][51] + inv[17] + scratch[32] floats; Metal rounds static storage to 16 bytes.
        .{ .name = "inkling2 ops inkling2_attn_res", .threads = 256, .tg_memory = std.mem.alignForward(u32, 4 * (8 * 51 + 17 + 32), 16), .reserved = 256 },
        .{ .name = "inkling2 ops inkling2_conv_static", .threads = 256, .tg_memory = 0, .reserved = 256 },
        .{ .name = "inkling2 ops inkling2_conv_dynamic", .threads = 256, .tg_memory = 0, .reserved = 256 },
        .{ .name = "inkling2 ops inkling2_conv_store", .threads = 256, .tg_memory = 0, .reserved = 256 },
        .{ .name = "inkling2 ops inkling2_scaled_l2_clip", .threads = 32, .tg_memory = 0, .reserved = 32 },
        .{ .name = "inkling2 ops inkling2_kv_prep", .threads = 32, .tg_memory = 0, .reserved = 32 },
        .{ .name = "inkling2 ops inkling2_attn_merge", .threads = 128, .tg_memory = 0, .reserved = 128 },
        .{ .name = "inkling2 ops inkling2_route_select", .threads = 32, .tg_memory = 0, .reserved = 32 },
        .{ .name = "inkling2 ops inkling2_softcap", .threads = 256, .tg_memory = 0, .reserved = 256 },
        .{ .name = "inkling2 ops inkling2_residual_attn", .threads = 256, .tg_memory = 0, .reserved = 256 },
        .{ .name = "inkling2 ops inkling2_residual_mlp", .threads = 256, .tg_memory = 0, .reserved = 256 },
    };
    for (singles, 0..) |s, i| out[i] = s;
    for (0..attn_groups) |gi| {
        const g: u32 = @intCast(gi + 1);
        const bytes = attentionBytes(g);
        out[singles.len + gi] = .{
            .name = std.fmt.comptimePrint("inkling2 ops inkling2_attn_partial_g{d}", .{g}),
            .threads = 128,
            .tg_memory = bytes,
            .reserved = 128,
        };
        out[singles.len + attn_groups + gi] = .{
            .name = std.fmt.comptimePrint("inkling2 ops inkling2_attn_final_g{d}", .{g}),
            .threads = 128,
            .tg_memory = bytes,
            .reserved = 128,
        };
    }
    break :blk out;
};

test "the Inkling 2 operator launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
