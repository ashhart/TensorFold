//! Byte ranges that dispatches in the open command buffer have read or written since the last barrier.
const std = @import("std");

/// A byte range of one buffer, by handle.
pub const Range = struct { buffer: usize, start: u64, end: u64 };

const max_written = 256;
const max_read = 1024;

pub const Hazards = struct {
    written: [max_written]Range = undefined,
    read: [max_read]Range = undefined,
    n_written: usize = 0,
    n_read: usize = 0,
    /// A list ran out of room: the next dispatch waits, which covers everything not tracked.
    full: bool = false,

    pub fn clear(h: *Hazards) void {
        h.n_written = 0;
        h.n_read = 0;
        h.full = false;
    }

    pub fn empty(h: *const Hazards) bool {
        return h.n_written == 0 and h.n_read == 0 and !h.full;
    }

    /// True when a dispatch that reads `reads` and writes `writes` must wait for the dispatches tracked so far.
    pub fn conflicts(h: *const Hazards, reads: []const Range, writes: []const Range) bool {
        if (h.full) return true;
        for (reads) |r| if (hits(h.written[0..h.n_written], r)) return true;
        for (writes) |w| if (hits(h.written[0..h.n_written], w) or hits(h.read[0..h.n_read], w)) return true;
        return false;
    }

    /// Tracks a dispatch that has been recorded without a barrier after it.
    pub fn add(h: *Hazards, reads: []const Range, writes: []const Range) void {
        if (h.n_read + reads.len > max_read or h.n_written + writes.len > max_written) {
            h.full = true;
            return;
        }
        @memcpy(h.read[h.n_read..][0..reads.len], reads);
        h.n_read += reads.len;
        @memcpy(h.written[h.n_written..][0..writes.len], writes);
        h.n_written += writes.len;
    }
};

fn hits(tracked: []const Range, r: Range) bool {
    for (tracked) |t| if (t.buffer == r.buffer and t.start < r.end and r.start < t.end) return true;
    return false;
}

const testing = std.testing;

fn span(buffer: usize, start: u64, end: u64) Range {
    return .{ .buffer = buffer, .start = start, .end = end };
}

test "a read of earlier writes, or a write over earlier reads or writes, waits; reads beside reads and other ranges do not" {
    var h: Hazards = .{};
    h.add(&.{span(1, 0, 64)}, &.{span(2, 0, 32)});
    try testing.expect(h.conflicts(&.{span(2, 16, 48)}, &.{}));
    try testing.expect(h.conflicts(&.{}, &.{span(2, 31, 40)}));
    try testing.expect(h.conflicts(&.{}, &.{span(1, 63, 80)}));
    try testing.expect(!h.conflicts(&.{span(1, 0, 64)}, &.{span(3, 0, 8)}));
    try testing.expect(!h.conflicts(&.{span(2, 32, 64)}, &.{span(1, 64, 128)}));
    try testing.expect(!h.conflicts(&.{span(4, 0, 64)}, &.{span(5, 0, 32)}));
}

test "a barrier clears the tracked ranges, and a full list makes the next dispatch wait" {
    var h: Hazards = .{};
    h.add(&.{}, &.{span(1, 0, 8)});
    try testing.expect(h.conflicts(&.{span(1, 0, 8)}, &.{}));
    h.clear();
    try testing.expect(!h.conflicts(&.{span(1, 0, 8)}, &.{}));
    var i: u64 = 0;
    while (i < 300) : (i += 1) h.add(&.{}, &.{span(1, i * 8, i * 8 + 8)});
    try testing.expect(h.conflicts(&.{span(9, 0, 8)}, &.{}));
    h.clear();
    try testing.expect(!h.conflicts(&.{span(9, 0, 8)}, &.{}));
}
