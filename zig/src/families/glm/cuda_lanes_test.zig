//! The lane backend's slots: chosen before a stream's map entry exists, and none left behind by a failed insertion.
const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const Lanes = @import("cuda_lanes.zig").Lanes;
const fw = @import("cuda_forward.zig");
const solo = @import("cuda_prompt_solo.zig");
const prompt_cuts = @import("cuda_prompt_cuts.zig");

test "a lane's slot is chosen before its map entry exists" {
    var storage: [8192]u8 = @splat(0);
    var fixed = std.heap.FixedBufferAllocator.init(&storage);
    var e: engine.Engine = undefined;
    e.gpa = fixed.allocator();
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var streams: [3]lanes.Stream = undefined;
    const Probe = struct {
        var bad = false;
        fn choose(l: *const Lanes) u32 {
            bad = l.lanes.count() != 0;
            return 0;
        }
    };
    _ = try self.claimUsing(&streams[0], Probe.choose);
    try std.testing.expect(!Probe.bad);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&streams[0])).slot);
    try std.testing.expectEqual(@as(u32, 1), (try self.claim(&streams[1])).slot);
    _ = self.lanes.remove(&streams[0]);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&streams[2])).slot);
    try std.testing.expectEqual(@as(u32, 1), (try self.claim(&streams[1])).slot);
}

test "a failed lane insertion leaves no entry, and a retry takes slot zero" {
    var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var e: engine.Engine = undefined;
    e.gpa = fail.allocator();
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var stream: lanes.Stream = undefined;
    try std.testing.expectError(error.OutOfMemory, self.claim(&stream));
    try std.testing.expectEqual(@as(usize, 0), self.lanes.count());
    fail.fail_index = std.math.maxInt(usize);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&stream)).slot);
}

test "a prompt pass goes solo only while no other stream is past its prompt pass" {
    var e: engine.Engine = undefined;
    e.gpa = std.testing.allocator;
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var streams: [3]lanes.Stream = undefined;
    const first = try self.claim(&streams[0]); // a stream's own lane, fresh as its first chunk starts
    try std.testing.expect(!solo.othersDecode(&self.lanes, &streams[0]));
    first.fill = fw.prefill_rows;
    (try self.claim(&streams[1])).fill = 2 * fw.prefill_rows; // another prompt pass, partway
    try std.testing.expect(!solo.othersDecode(&self.lanes, &streams[0]));
    _ = try self.claim(&streams[2]); // past its prompt pass: decoding
    try std.testing.expect(solo.othersDecode(&self.lanes, &streams[0]) and solo.othersDecode(&self.lanes, &streams[1]));
    try std.testing.expect(!solo.othersDecode(&self.lanes, &streams[2])); // the other two still fill
    _ = self.lanes.remove(&streams[2]); // released
    try std.testing.expect(!solo.othersDecode(&self.lanes, &streams[0]));
}

test "glm request starts: a kept state off the new request's cold cuts, or over other chunks below, passes from 0" {
    const m = fw.prefill_rows;
    var e: engine.Engine = undefined;
    e.gpa = std.testing.allocator;
    e.snaps = .empty;
    defer e.snaps.deinit(e.gpa);
    try e.snaps.put(e.gpa, 7, undefined); // every rank holds state 7; resumeFrom reads no more of it
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    const Case = struct { kept: []const u32, cuts: []const u32, why: ?anyerror, id: u64 = 7, at: u32 = 2 * m };
    const cases = [_]Case{
        .{ .kept = &.{}, .cuts = &.{m + 300}, .why = error.SnapshotOutOfStep }, // 2m lies inside a chunk now
        .{ .kept = &.{300}, .cuts = &.{}, .why = error.SnapshotOutOfStep, .at = 300 }, // a chat's start, a grid pass
        .{ .kept = &.{300}, .cuts = &.{ 300, 2 * m }, .why = null, .at = 300 }, // the same start: it resumes
        .{ .kept = &.{}, .cuts = &.{ 300, 2 * m }, .why = error.SnapshotOutOfStep }, // a grid state, a start below
        .{ .kept = &.{ 300, 2 * m }, .cuts = &.{ 700, 2 * m }, .why = error.SnapshotOutOfStep }, // other starts below
        .{ .kept = &.{ 300, 2 * m }, .cuts = &.{}, .why = error.SnapshotOutOfStep }, // a chat's state, a grid pass
        .{ .kept = &.{}, .cuts = &.{ 2 * m, 2 * m + 500 }, .why = null }, // the starts from 2m on chunk below it alike
        .{ .kept = &.{m}, .cuts = &.{}, .why = null }, // a start on the grid chunks as the grid does
        .{ .kept = &.{ 300, 2 * m }, .cuts = &.{ 300, 2 * m, 3 * m }, .why = null },
        .{ .kept = &.{ 300, 2 * m }, .cuts = &.{ 300, 2 * m }, .why = error.SnapshotStale, .id = 8 }, // no rank holds 8
    };
    for (cases) |c| {
        var k: engine.Kept = .{ .id = c.id, .at = c.at, .below = prompt_cuts.below(c.kept, c.at) };
        var s: lanes.Stream = undefined;
        s.chunks = c.cuts;
        s.reuse = .{ .saved = &k, .at = c.at };
        s.reuse_failed = false;
        s.reuse_error = null;
        s.cached = 0;
        const from = self.resumeFrom(&s, 4 * m);
        try std.testing.expectEqual(c.why, s.reuse_error);
        try std.testing.expectEqual(c.why != null, s.reuse_failed);
        try std.testing.expectEqual(@as(usize, if (c.why == null) c.at else 0), if (from) |r| r.at else 0);
        try std.testing.expectEqual(if (c.why == null) c.at else 0, s.cached); // refused: the pass starts at 0
    }
}
