//! LFM2.5-24B-A2B on CUDA: message starts keep a resumable state each, exact on resume, without a GPU.
const std = @import("std");
const be = @import("cuda_backend.zig");
const cut = be.Lfm.cut;
const t = std.testing;

/// A pass's chunk ends and the points it keeps a state at (taps and cut-point ends).
const Walk = struct { ends: []usize, kept: []usize };
fn walk(from: usize, len: usize, marks: []const u32, chunks: []const u32) !Walk {
    var ends: std.ArrayList(usize) = .empty;
    errdefer ends.deinit(t.allocator);
    var kept: std.ArrayList(usize) = .empty;
    errdefer kept.deinit(t.allocator);
    var at = from;
    while (at < len) {
        const c = cut(at, len, marks, chunks);
        try ends.append(t.allocator, c.end);
        for (c.taps()) |tap| try kept.append(t.allocator, tap);
        for ([_][]const u32{ marks, chunks }) |points| {
            if (std.mem.indexOfScalar(u32, points, @intCast(c.end)) != null) {
                try kept.append(t.allocator, c.end);
                break;
            }
        }
        at = c.end;
    }
    return .{ .ends = try ends.toOwnedSlice(t.allocator), .kept = try kept.toOwnedSlice(t.allocator) };
}

test "message starts keep a state at each, grid-aligned or not (the resume point main misses)" {
    const len = 9000;
    const starts = [_]u32{ 700, 1024, 2100, 4096, 5700, 8192 };
    const w = try walk(0, len, &.{}, &starts);
    defer t.allocator.free(w.ends);
    defer t.allocator.free(w.kept);
    for (starts) |s| try t.expect(std.mem.indexOfScalar(usize, w.kept, s) != null);
    try t.expectEqual(starts.len, w.kept.len); // the grid alone keeps nothing
}

test "message starts past the tap limit end their chunk so each still keeps a state" {
    const len = 3000;
    const starts = [_]u32{ 100, 200, 300, 400, 500, 600 };
    const w = try walk(0, len, &.{}, &starts);
    defer t.allocator.free(w.ends);
    defer t.allocator.free(w.kept);
    for (starts) |s| try t.expect(std.mem.indexOfScalar(usize, w.kept, s) != null);
    try t.expectEqual(starts.len, w.kept.len);
}

test "a resume at a kept message start recomputes the chunk ends a fresh pass did past it" {
    const len = 6000;
    const marks = [_]u32{1500};
    const starts = [_]u32{ 700, 2100, 3300, 4500 };
    const fresh = try walk(0, len, &marks, &starts);
    defer t.allocator.free(fresh.ends);
    defer t.allocator.free(fresh.kept);
    const resumed = try walk(2100, len, &marks, &starts);
    defer t.allocator.free(resumed.ends);
    defer t.allocator.free(resumed.kept);
    for (resumed.ends) |e| {
        try t.expect(e > 2100);
        try t.expect(std.mem.indexOfScalar(usize, fresh.ends, e) != null);
    }
    for (fresh.ends) |e| {
        if (e > 2100) try t.expect(std.mem.indexOfScalar(usize, resumed.ends, e) != null);
    }
}
