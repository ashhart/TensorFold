//! DeepSeek V4.1's routed-expert prompt launches: simdgroup MMA and M5 tensor units, every thread count reserved.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "dsv41 xmma tiles tf_dsv41_xm_tiles", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 xmma quant tf_dsv41_xm_quant", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 xmma up tf_dsv41_xm_up", .threads = 64, .tg_memory = 4096, .reserved = 64 },
    .{ .name = "dsv41 xmma down tf_dsv41_xm_down", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 xmma units quant tf_dsv41_xu_quant", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 xmma units up tf_dsv41_xu_up", .threads = 512, .tg_memory = 16384, .reserved = 512, .offline = false },
    .{ .name = "dsv41 xmma units dprep tf_dsv41_xu_dprep", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 xmma units down tf_dsv41_xu_down", .threads = 512, .tg_memory = 16384, .reserved = 512, .offline = false },
};

test "the DeepSeek V4.1 expert MMA launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
