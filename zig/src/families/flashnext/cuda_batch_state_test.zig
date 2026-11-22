//! Failure injection for the production staging fence.
const std = @import("std");
const Staging = @import("cuda_batch_state.zig").Staging;
const prompt_async = @import("cuda_prompt_async.zig");
const FakeQueue = struct {
    queued: usize = 0,
    covered: usize = 0,
    copies: usize = 0,
    fail_copy: ?usize = null,
    fail_record: bool = false,
    fail_drain: bool = false,
    waits: usize = 0,
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
        q.waits += 1;
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

const FakeHost = struct { bytes: []u8 };

test "Flash Next completion: async staging overlaps the next bank and reuses an in-flight bank only after its own fence" {
    var bytes0: [16]u8 = @splat(0);
    var bytes1: [16]u8 = @splat(0);
    var q0: FakeQueue = .{};
    var q1: FakeQueue = .{};
    var bank: prompt_async.Bank(*FakeQueue, FakeHost) = .{
        .enabled = true,
        .hosts = .{ FakeHost{ .bytes = &bytes0 }, FakeHost{ .bytes = &bytes1 } },
        .queues = .{ .{ .q = &q0 }, .{ .q = &q1 } },
    };
    try bank.upload(10, "first ids");
    try bank.upload(20, "first rows");
    try bank.fence();
    try std.testing.expect(bank.select());
    try bank.ready();
    try std.testing.expectEqual(@as(usize, 1), bank.active);
    try std.testing.expectEqual(@as(usize, 2), q0.queued);
    bank.bytes()[0] = 0x42;
    try std.testing.expectEqual(@as(u8, 0), bytes0[0]);
    try std.testing.expectEqual(@as(u8, 0x42), bytes1[0]);
    try bank.upload(10, "second ids");
    try bank.upload(20, "second rows");
    try bank.fence();
    try std.testing.expect(bank.select());
    try bank.ready();
    try std.testing.expectEqual(@as(usize, 0), bank.active);
    try std.testing.expectEqual(@as(usize, 0), q0.queued);
    try std.testing.expectEqual(@as(usize, 2), q1.queued);
    bank.bytes()[0] = 0x84;
    try std.testing.expectEqual(@as(u8, 0x84), bytes0[0]);
    try std.testing.expectEqual(@as(u8, 0x42), bytes1[0]);
    try bank.upload(10, "third ids");
    try bank.upload(20, "third rows");
    try bank.fence();
    bank.drain();
    try std.testing.expectEqual(@as(usize, 0), q0.queued);
    try std.testing.expectEqual(@as(usize, 0), q1.queued);
    try std.testing.expectEqual(@as(usize, 2), q0.waits);
}

test "Flash Next completion: prompt async skips only no-logits drains, final completion precedes logits" {
    const Event = enum { complete, logits_read };
    const Owner = struct {
        events: [3]Event = undefined,
        len: usize = 0,
        pub fn complete(o: *@This()) !void {
            o.events[o.len] = .complete;
            o.len += 1;
        }
        pub fn read(o: *@This()) void {
            o.events[o.len] = .logits_read;
            o.len += 1;
        }
    };
    var owner: Owner = .{};
    try prompt_async.afterChunk(&owner, true, false);
    try std.testing.expectEqual(@as(usize, 1), owner.len);
    try prompt_async.afterChunk(&owner, true, true);
    try prompt_async.afterChunk(&owner, false, true);
    try std.testing.expectEqual(@as(usize, 1), owner.len);
    try prompt_async.beforeLogits(&owner);
    owner.read();
    try std.testing.expectEqual([3]Event{ .complete, .complete, .logits_read }, owner.events);
}

const Copy = struct { dst: u64, bytes: [16]u8, len: usize };
const Pending = struct { dst: u64, src: []const u8, owner: usize, covered: bool = false };
const Trace = struct {
    pending: [16]Pending = undefined,
    pending_len: usize = 0,
    landed: usize = 0,
    ops: [16]Copy = undefined,
    len: usize = 0,
    fn finish(t: *Trace) void {
        while (t.landed < t.pending_len and t.pending[t.landed].covered) {
            const p = t.pending[t.landed];
            var op: Copy = .{ .dst = p.dst, .bytes = @splat(0), .len = p.src.len };
            @memcpy(op.bytes[0..p.src.len], p.src);
            t.ops[t.len] = op;
            t.len += 1;
            t.landed += 1;
        }
    }
};
const TraceQueue = struct {
    queued: usize = 0,
    covered: usize = 0,
    drains: usize = 0,
    id: usize,
    trace: *Trace,
    pub fn copy(q: *TraceQueue, dst: u64, bytes: []const u8) !void {
        q.queued += 1;
        q.trace.pending[q.trace.pending_len] = .{ .dst = dst, .src = bytes, .owner = q.id };
        q.trace.pending_len += 1;
    }
    pub fn record(q: *TraceQueue) !void {
        q.covered = q.queued;
        for (q.trace.pending[0..q.trace.pending_len]) |*p| {
            if (p.owner == q.id) p.covered = true;
        }
    }
    pub fn wait(q: *TraceQueue) !void {
        if (q.covered != 0) q.trace.finish();
        q.queued -= q.covered;
        q.covered = 0;
    }
    pub fn drain(q: *TraceQueue) !void {
        q.drains += 1;
        q.queued = 0;
        q.covered = 0;
        for (q.trace.pending[0..q.trace.pending_len]) |*p| p.covered = true;
        q.trace.finish();
    }
};

test "Flash Next completion: async preserves three chunks of device copies and bytes while removing only full drains" {
    const Runner = struct {
        fn run(async_enabled: bool, trace: *Trace) !usize {
            var bytes0: [16]u8 = @splat(0);
            var bytes1: [16]u8 = @splat(0);
            var q0: TraceQueue = .{ .id = 0, .trace = trace };
            var q1: TraceQueue = .{ .id = 1, .trace = trace };
            var bank: prompt_async.Bank(*TraceQueue, FakeHost) = .{
                .enabled = async_enabled,
                .hosts = .{ FakeHost{ .bytes = &bytes0 }, FakeHost{ .bytes = &bytes1 } },
                .queues = .{ .{ .q = &q0 }, .{ .q = &q1 } },
            };
            for (0..3) |chunk| {
                if (chunk != 0) {
                    try std.testing.expectEqual(async_enabled, bank.select());
                    try bank.ready();
                }
                try std.testing.expectEqual(if (async_enabled) chunk % 2 else 0, bank.active);
                const copied = try std.fmt.bufPrint(bank.bytes(), "chunk {d}", .{chunk});
                try bank.upload(10, copied);
                try bank.upload(20, copied);
                try bank.fence();
                if (!async_enabled) try q0.drain();
            }
            bank.drain();
            return q0.drains + q1.drains;
        }
    };
    var plain: Trace = .{};
    var async_trace: Trace = .{};
    try std.testing.expectEqual(@as(usize, 3), try Runner.run(false, &plain));
    try std.testing.expectEqual(@as(usize, 0), try Runner.run(true, &async_trace));
    try std.testing.expectEqual(plain.len, async_trace.len);
    for (plain.ops[0..plain.len], async_trace.ops[0..async_trace.len]) |a, b| {
        try std.testing.expectEqual(a.dst, b.dst);
        try std.testing.expectEqual(a.len, b.len);
        try std.testing.expectEqualSlices(u8, a.bytes[0..a.len], b.bytes[0..b.len]);
    }
}
