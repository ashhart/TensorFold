//! Experimental K3.1 launches: non-adaptive widths are reserved in MSL; attention adapts to pipeline reflection.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "kimi index k3_index_key", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "kimi index k3_index_scores", .threads = 128, .tg_memory = 16640, .reserved = 128 },
    .{ .name = "kimi index k3_index_topk", .threads = 1024, .tg_memory = 1312, .reserved = 1024 },
    .{ .name = "kimi index k3_index_attend", .threads = 1024, .tg_memory = 18432, .adapts = true },
    .{ .name = "kimi index k3_index_merge", .threads = 256, .tg_memory = 0, .reserved = 256 },
};
test "experimental K3.1 launch declarations fit M1 through M5" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
