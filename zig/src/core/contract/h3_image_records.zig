//! MiniMax-H3-Image's launches: its edit encoder's, prompt path's and image decoder's kernels, each source-reserved.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const encoder = [_]M{};
pub const prompt = [_]M{};
pub const decoder = [_]M{};
pub const records = encoder ++ prompt ++ decoder;

test "the MiniMax-H3-Image launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
