//! Failure injection for the production staging fence.
const std = @import("std");
const Staging = @import("cuda_batch_state.zig").Staging;
const FakeQueue = struct {
    queued: usize = 0,
    covered: usize = 0,
    copies: usize = 0,
    fail_copy: ?usize = null,
    fail_record: bool = false,
    fail_drain: bool = false,
    drains: usize = 0,

    pub fn copy(q: *FakeQueue, _: u64, _: []const u8) !void {
        q.copies += 1;
        q.queued += 1;
        if (q.fail_copy) |k| if (k == q.copies) return error.CopyFailed;
    }
    pub fn record(q: *FakeQueue) !void {
        if (q.fail_record) return error.RecordFailed;
        q.covered = q.queued;
    }
    pub fn wait(q: *FakeQueue) !void {
        q.queued -= q.covered; // the event's copies have landed; any queued after it may still be reading
        q.covered = 0;
    }
    pub fn drain(q: *FakeQueue) !void {
        q.drains += 1;
        if (q.fail_drain) return error.DrainFailed;
        q.queued = 0;
        q.covered = 0;
    }
};

test "a pass whose second upload or whose record fails leaves no copy reading the staging when it is reused" {
    const Case = struct { fail_copy: ?usize, fail_record: bool };
    const cases = [_]Case{
        .{ .fail_copy = 1, .fail_record = false },
        .{ .fail_copy = 2, .fail_record = false },
        .{ .fail_copy = null, .fail_record = true },
        .{ .fail_copy = null, .fail_record = false },
    };
    for (cases) |c| {
        var fq: FakeQueue = .{};
        var g: Staging(*FakeQueue) = .{ .q = &fq };
        // a pass that stages and fences cleanly, then one that fails part way, as `stage()` uploads ids and positions
        try g.ready();
        try g.upload(0, "ids");
        try g.upload(0, "pos");
        try g.fence();
        fq.fail_copy = if (c.fail_copy) |k| fq.copies + k else null;
        fq.fail_record = c.fail_record;
        try g.ready();
        const failed = blk: {
            g.upload(0, "ids") catch break :blk true;
            g.upload(0, "pos") catch break :blk true;
            g.fence() catch break :blk true;
            break :blk false;
        };
        try std.testing.expectEqual(c.fail_copy != null or c.fail_record, failed);
        // the next pass (or `free`) waits before it writes the bytes: nothing may still be queued
        try g.ready();
        try std.testing.expectEqual(@as(usize, 0), fq.queued);
        try std.testing.expectEqual(@as(usize, if (failed) 1 else 0), fq.drains);
    }
}

test "a failed staging drain keeps the upload guard set until a successful retry" {
    var fq: FakeQueue = .{ .fail_record = true, .fail_drain = true };
    var g: Staging(*FakeQueue) = .{ .q = &fq };
    try g.upload(0, "ids");
    try std.testing.expectError(error.RecordFailed, g.fence());
    try std.testing.expectError(error.DrainFailed, g.ready());
    try std.testing.expect(g.open);
    try std.testing.expectEqual(@as(usize, 1), fq.queued);
    fq.fail_drain = false;
    try g.ready();
    try std.testing.expect(!g.open);
    try std.testing.expectEqual(@as(usize, 0), fq.queued);
    try std.testing.expectEqual(@as(usize, 2), fq.drains);
}
