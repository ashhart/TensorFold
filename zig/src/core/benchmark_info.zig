//! An HTTP request owns its snapshot even if the loaded engine parks immediately after the copy.
const std = @import("std");
pub const Snapshot = struct {
    ranks: ?u32 = null,
    group: ?[]const u8 = null,

    pub fn copy(a: std.mem.Allocator, info: anytype) !Snapshot {
        return .{ .ranks = info.benchmark_ranks, .group = if (info.benchmark_group) |g| try a.dupe(u8, g) else null };
    }
};
