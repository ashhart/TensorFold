//! Optional per-level decode topology, independent of the device decoder.
const std = @import("std");
const dc = @import("decoder_config.zig");
const octree = @import("octree.zig");

/// Each level's parents and their counts, kept when a caller asks (for comparing a run's topology).
pub const Trace = struct {
    parents: [dc.max_level][]const [3]u32 = @splat(&.{}),
    counts: [dc.max_level][]const u32 = @splat(&.{}),
    probs: [dc.max_level][]const f32 = @splat(&.{}),

    pub fn deinit(t: *Trace, gpa: std.mem.Allocator) void {
        for (t.parents, t.counts, t.probs) |p, c, q| {
            gpa.free(p);
            gpa.free(c);
            gpa.free(q);
        }
        t.* = .{};
    }
};

pub fn capture(t: *Trace, gpa: std.mem.Allocator, index: usize, level: octree.Level, probs: []const f32) !void {
    const parents = try gpa.dupe([3]u32, level.coords);
    errdefer gpa.free(parents);
    const counts = try gpa.dupe(u32, level.counts);
    errdefer gpa.free(counts);
    const probs_copy = try gpa.dupe(f32, probs);
    t.parents[index] = parents;
    t.counts[index] = counts;
    t.probs[index] = probs_copy;
}

test "trace capture leaves the failed level empty and balances allocations" {
    const level: octree.Level = .{ .coords = &.{.{ 0, 0, 0 }}, .counts = &.{1} };
    const probs = [_]f32{1};
    for (0..3) |fail_at| {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_at });
        var trace: Trace = .{};
        defer trace.deinit(std.testing.allocator);
        try std.testing.expectError(error.OutOfMemory, capture(&trace, failing.allocator(), 0, level, &probs));
        try std.testing.expectEqual(@as(usize, 0), trace.parents[0].len);
        try std.testing.expectEqual(@as(usize, 0), trace.counts[0].len);
        try std.testing.expectEqual(@as(usize, 0), trace.probs[0].len);
        try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
    }
}
