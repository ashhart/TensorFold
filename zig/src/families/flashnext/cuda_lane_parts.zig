//! A contiguous lane's fixed-size buffers reused across streams (TF_FLASHNEXT_LANE_PARTS=0: allocated each stream).
const std = @import("std");
const cuda = @import("cuda");

/// Fixed-size lane buffers reused across streams: at most `kept` idle blocks of sequence and draft parts each.
pub const Parts = struct {
    on: bool = false,
    seqs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    drafts: std.ArrayList(cuda.DeviceBuffer) = .empty,
    const kept = 4;

    /// A graph-recycled lane keeps its whole sequence and draft, not borrowed parts of them.
    pub fn forLane(p: *const Parts, ranked: bool, pooled: bool) bool {
        return p.on and !ranked and !pooled and !@import("exl3/offload/cuda_exl3_recycle.zig").enabled();
    }

    pub fn take(list: *std.ArrayList(cuda.DeviceBuffer), d: *cuda.Driver, len: usize) !cuda.DeviceBuffer {
        while (list.pop()) |b| {
            if (b.len >= len) return b;
            var x = b;
            x.free();
        }
        return cuda.DeviceBuffer.alloc(d, len);
    }

    pub fn give(list: *std.ArrayList(cuda.DeviceBuffer), gpa: std.mem.Allocator, b: cuda.DeviceBuffer) void {
        var x = b;
        if (list.items.len >= kept) return x.free();
        list.append(gpa, b) catch x.free();
    }

    pub fn deinit(p: *Parts, gpa: std.mem.Allocator) void {
        for (p.seqs.items) |*b| b.free();
        for (p.drafts.items) |*b| b.free();
        p.seqs.deinit(gpa);
        p.drafts.deinit(gpa);
    }
};

test "exl3: graph recycling and lane parts never own the same buffers" {
    const graphs = @import("exl3/graph/cuda_exl3_graph.zig");
    const previous = graphs.force;
    defer graphs.force = previous;
    const parts: Parts = .{ .on = true };
    graphs.force = .off;
    try std.testing.expect(parts.forLane(false, false));
    try std.testing.expect(!parts.forLane(true, false));
    try std.testing.expect(!parts.forLane(false, true));
    graphs.force = .graph;
    try std.testing.expect(!parts.forLane(false, false));
}
