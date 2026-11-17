//! The paired request probe commits on its own queue under the owner's lock, and only while a request holds the gate.
const std = @import("std");
const mtl = @import("metal");
const PairWarm = @import("src/native/flashnext_pair_warm.zig").PairWarm;
const Gate = @import("src/families/flashnext/tp_request_gate.zig").Gate;
const a = std.testing.allocator;

/// The owner's flush and lock, noting the commits made while the lock is held.
const Owner = struct {
    record: *mtl.Record,
    flushes: u32 = 0,
    at_lock: usize = 0,
    inside: usize = 0,
    locks: std.atomic.Value(u32) = .init(0),
    unlocks: std.atomic.Value(u32) = .init(0),

    fn flush(ctx: *anyopaque) void {
        const o: *Owner = @ptrCast(@alignCast(ctx));
        o.flushes += 1;
    }

    fn lock(ctx: *anyopaque) void {
        const o: *Owner = @ptrCast(@alignCast(ctx));
        o.at_lock = o.record.commits;
        _ = o.locks.fetchAdd(1, .acq_rel);
    }

    fn unlock(ctx: *anyopaque) void {
        const o: *Owner = @ptrCast(@alignCast(ctx));
        o.inside += o.record.commits - o.at_lock;
        _ = o.unlocks.fetchAdd(1, .acq_rel);
    }

    fn target(o: *Owner, sets: []const mtl.ResidencySet, scratch: mtl.Buffer) mtl.keepalive.Target {
        return .{ .queue = o.record.queue(), .sets = sets, .scratch = scratch, .flush = .{ .ctx = o, .run = flush }, .hold = .{ .ctx = o, .lock = lock, .unlock = unlock } };
    }
};

/// A recorded set and owner scratch beside one pair probe on the same device.
const Fixture = struct {
    record: mtl.Record = .{ .gpa = a },
    gate: Gate = .{},
    sets: [1]mtl.ResidencySet = undefined,
    scratch: mtl.Buffer = undefined,
    owner: Owner = undefined,
    warm: PairWarm = undefined,

    fn init(f: *Fixture) !void {
        f.sets[0] = try f.record.device().residencySet(1);
        f.scratch = try f.record.device().buffer(64, 0);
        f.owner = .{ .record = &f.record };
        f.warm = try PairWarm.init(f.record.device(), std.testing.io, &f.gate.active);
        f.warm.track(f.owner.target(&f.sets, f.scratch));
    }

    fn deinit(f: *Fixture) !void {
        f.warm.deinit();
        f.sets[0].deinit();
        f.scratch.deinit();
        try f.record.allFreed();
        try std.testing.expectEqual(@as(usize, 1), f.record.queue_releases);
        f.record.deinit();
    }
};

fn until(value: *const std.atomic.Value(u32), want: u32) !void {
    for (0..5000) |_| {
        if (value.load(.acquire) >= want) return;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.Timeout;
}

test "residency owner Flash Next pair warm: while a paired request holds the gate each step commits one probe on the second queue under the owner's lock" {
    var f: Fixture = .{};
    try f.init();
    f.gate.begin();
    for (1..4) |n| {
        try std.testing.expect(f.warm.step());
        try std.testing.expectEqual(n, f.record.side_commits);
    }
    f.gate.end();
    try std.testing.expectEqual(@as(usize, 1), f.record.side_queues);
    try std.testing.expectEqual(f.record.side_commits, f.record.commits);
    try std.testing.expectEqual(@as(usize, 3), f.record.uses);
    try std.testing.expectEqual(@as(usize, 3), f.record.copies);
    try std.testing.expectEqual(@as(usize, 3), f.owner.inside);
    try std.testing.expectEqual(@as(u32, 3), f.owner.unlocks.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), f.owner.flushes);
    try std.testing.expectEqual(@as(usize, 0), f.record.waits);
    try std.testing.expect(f.warm.target.flush == null);
    try std.testing.expect(f.warm.target.scratch.?.id != f.scratch.id);
    try f.deinit();
}

test "residency owner Flash Next pair warm: with no request holding the gate a step commits nothing" {
    var f: Fixture = .{};
    try f.init();
    try std.testing.expect(!f.warm.step());
    f.gate.begin();
    f.gate.end();
    try std.testing.expect(!f.warm.step());
    var idle = f.owner.target(&f.sets, f.scratch);
    try std.testing.expect(try f.gate.keepWarm(&idle));
    try std.testing.expect(!f.warm.step());
    try std.testing.expectEqual(@as(usize, 0), f.record.side_commits);
    try std.testing.expectEqual(@as(usize, 1), f.record.commits);
    try std.testing.expectEqual(@as(u32, 0), f.owner.locks.load(.acquire));
    try f.deinit();
}

test "Flash Next cleanup: stopping the pair warm thread joins it and no probe commits afterwards" {
    var f: Fixture = .{};
    try f.init();
    f.warm.interval_ns = std.time.ns_per_ms;
    f.gate.begin();
    try f.warm.start();
    try until(&f.owner.unlocks, 3);
    f.warm.stop();
    try std.testing.expect(f.warm.thread == null);
    const probes = f.owner.unlocks.load(.acquire);
    try std.testing.expectEqual(@as(usize, probes), f.record.side_commits);
    std.Io.sleep(std.testing.io, .fromMilliseconds(20), .awake) catch {};
    try std.testing.expectEqual(probes, f.owner.unlocks.load(.acquire));
    try std.testing.expectEqual(@as(usize, probes), f.record.side_commits);
    f.gate.end();
    try f.warm.start();
    std.Io.sleep(std.testing.io, .fromMilliseconds(20), .awake) catch {};
    f.warm.stop();
    try std.testing.expectEqual(@as(usize, probes), f.record.side_commits);
    try f.deinit();
}
