//! Hyper-connection launches: reserved legacy groups and the native boundary's measured M1/M2 compiled limits.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "dsv41 hc mix tf_dsv41_hc_mix", .threads = 256, .tg_memory = 800, .reserved = 256 },
    .{ .name = "dsv41 hc pre tf_dsv41_hc_pre", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "dsv41 hc post tf_dsv41_hc_post", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "dsv41-native-boundary coefficients tf_hc_boundary_mix", .threads = 1024, .tg_memory = 256, .m1 = 1024, .m2 = 1024 },
};

test "the DeepSeek V4.1 hyper-connection launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
