//! Hy4-flash's nax launches on Metal (tests/contract/metal_hy4_nax*.zig), as mla_metal.zig's Nax encodes them.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
// nax.h's tensor ops keep every pipeline but the combine from an M1/M2 build ahead of time: 128 threads each.
pub const records = [_]M{
    .{ .name = "hy4 units mla_nax tf_hy4_absorb_nax", .threads = 128, .tg_memory = 0, .offline = false },
    .{ .name = "hy4 units mla_nax tf_hy4_expand_gated_nax", .threads = 128, .tg_memory = 0, .offline = false },
    // part[4][16][32] floats: each simdgroup's slice of a tile's scores
    .{ .name = "hy4 units mla_nax tf_hy4_attend_nax", .threads = 128, .tg_memory = 4 * 16 * 32 * 4, .offline = false },
    .{ .name = "hy4 units mla_nax tf_hy4_attend_nax_split", .threads = 128, .tg_memory = 4 * 16 * 32 * 4, .offline = false },
    .{ .name = "hy4 units mla_nax tf_hy4_attend_combine", .threads = 128, .tg_memory = 0, .m1 = 928, .m2 = 928 },
};

test "Hy4-flash's nax launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
