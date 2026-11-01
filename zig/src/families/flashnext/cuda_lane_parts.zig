//! A contiguous lane's fixed-size buffers reused across streams (TF_FLASHNEXT_LANE_PARTS=0: allocated each stream).
const std = @import("std");
const cuda = @import("cuda");

/// Fixed-size lane buffers reused across streams: at most `kept` idle blocks of sequence and draft parts each.
pub const Parts = struct {
    on: bool = false,
    seqs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    drafts: std.ArrayList(cuda.DeviceBuffer) = .empty,
    const kept = 4;

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
