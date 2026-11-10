//! MiniMax-H3-Image's launches: its edit encoder's, prompt path's and image decoder's kernels, each source-reserved.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const encoder = [_]M{
    .{ .name = "minimax image encoder hie_conv", .threads = 256, .tg_memory = 16384, .reserved = 256 },
    .{ .name = "minimax image encoder hie_group_part", .threads = 1024, .tg_memory = 0, .reserved = 1024 },
    .{ .name = "minimax image encoder hie_group_finish", .threads = 1024, .tg_memory = 0, .reserved = 1024 },
    .{ .name = "minimax image encoder hie_norm_silu", .threads = 256, .tg_memory = 0, .reserved = 256 },
};
pub const prompt = [_]M{};
pub const decoder = [_]M{};
pub const records = encoder ++ prompt ++ decoder;

test "the MiniMax-H3-Image launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
