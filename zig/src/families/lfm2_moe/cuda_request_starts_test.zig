//! LFM2.5-24B-A2B on CUDA: message starts keep a resumable state each, exact on resume, without a GPU.
const std = @import("std");
const be = @import("cuda_backend.zig");
const lanes = @import("lanes");
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

test "lfm: cold and reused passes both bound chunks by the shared cut points" {
    var prompt: [2000]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(i);
    const starts = [_]u32{ 1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129 };
    var s = try lanes.Stream.init(t.allocator, .{
        .id = "cold-and-reused", .prompt = &prompt, .max_new = 1, .drafts = false, .chunks = &starts,
    });
    defer s.deinit(t.allocator);
    const points = lanes.cutPoints(&s);
    try t.expectEqualSlices(u32, &starts, points);
    const cold = try walk(0, prompt.len, &.{}, points);
    defer t.allocator.free(cold.ends);
    defer t.allocator.free(cold.kept);
    for (starts) |at| try t.expect(std.mem.indexOfScalar(usize, cold.kept, at) != null);
    // A state kept at 128 resumes there; with no store marks the starts after it still end chunks.
    const reused = try walk(128, prompt.len, &.{}, points);
    defer t.allocator.free(reused.ends);
    defer t.allocator.free(reused.kept);
    for (starts) |at| if (at > 128) try t.expect(std.mem.indexOfScalar(usize, reused.kept, at) != null);
}
