//! Mini DSpark's greedy chain launches (zig/tests/contract/metal_inkling2_dspark_chain.zig).
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "metal inkling2 dspark chain_step tf_ink2_dspark_chain_step", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dspark chain_pick tf_ink2_dspark_chain_pick", .threads = 1024, .tg_memory = 256, .reserved = 1024 },
};

test "the chain's launches reserve their threads" {
    for (records) |record| {
        try std.testing.expectEqual(record.threads, record.reserved);
        try std.testing.expect(std.mem.startsWith(u8, record.name, "metal inkling2 dspark chain_"));
    }
}
