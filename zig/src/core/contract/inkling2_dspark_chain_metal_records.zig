//! Mini DSpark's greedy chain launches (zig/tests/contract/metal_inkling2_dspark_chain.zig).
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "metal inkling2 dspark chain_step tf_ink2_dspark_chain_step", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dspark chain_pick tf_ink2_dspark_chain_pick", .threads = 1024, .tg_memory = 256, .reserved = 1024 },
    .{ .name = "metal inkling2 dspark chain_q2_pack tf_ink2_dspark_q2_pack", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dspark chain_q4_pack tf_ink2_dspark_q4_pack", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dspark chain_q2_proj tf_ink2_dspark_q2_proj", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dspark chain_q2_screen tf_ink2_dspark_q2_screen", .threads = 256, .tg_memory = 768, .reserved = 256 },
    .{ .name = "metal inkling2 dspark chain_q2_refine tf_ink2_dspark_q2_refine", .threads = 256, .tg_memory = 1024, .reserved = 256 },
    .{ .name = "metal inkling2 dspark chain_q2_pick tf_ink2_dspark_q2_pick", .threads = 256, .tg_memory = 64, .reserved = 256 },
    .{ .name = "metal inkling2 dspark chain_q2_sample tf_ink2_dspark_q2_sample", .threads = 256, .tg_memory = 3088, .reserved = 256 },
};

test "the chain's launches reserve their threads" {
    for (records) |record| {
        try std.testing.expectEqual(record.threads, record.reserved);
        try std.testing.expect(std.mem.startsWith(u8, record.name, "metal inkling2 dspark chain_"));
    }
}
