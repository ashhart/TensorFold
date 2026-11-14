//! Pair GPU operations exclude idle residency changes and finish any preceding keepalive probe first.
const std = @import("std");

pub const Gate = struct {
    mutex: std.atomic.Mutex = .unlocked,

    pub fn begin(g: *Gate) void {
        while (!g.mutex.tryLock()) std.atomic.spinLoopHint();
    }

    pub fn end(g: *Gate) void {
        g.mutex.unlock();
    }

    pub fn keepWarm(g: *Gate, warm: anytype) !bool {
        if (!g.mutex.tryLock()) return false;
        defer g.mutex.unlock();
        if (warm.flush) |flush| flush.run(flush.ctx);
        const cb = warm.probe();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailure;
        return true;
    }
};

const Recorded = struct {
    gate: Gate = .{},
    flushes: u32 = 0,
    probes: u32 = 0,
    waiting: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    retired: std.atomic.Value(bool) = .init(false),
    attempted: std.atomic.Value(bool) = .init(false),
    entered: std.atomic.Value(bool) = .init(false),
    ordered: bool = false,
    ticked: bool = false,
    failed: bool = false,

    const Warm = struct {
        owner: *Recorded,
        flush: ?struct { ctx: *anyopaque, run: *const fn (*anyopaque) void },

        pub fn probe(w: Warm) Command {
            w.owner.probes += 1;
            return .{ .owner = w.owner };
        }
    };
    const Command = struct {
        owner: *Recorded,

        pub fn wait(c: Command) void {
            c.owner.waiting.store(true, .release);
            while (!c.owner.release.load(.acquire)) std.atomic.spinLoopHint();
            c.owner.retired.store(true, .release);
        }

        pub fn failure(c: Command) ?[]const u8 {
            return if (c.owner.failed) "planted probe failure" else null;
        }
    };

    fn flush(ctx: *anyopaque) void {
        const r: *Recorded = @ptrCast(@alignCast(ctx));
        r.flushes += 1;
    }

    fn warm(r: *Recorded) Warm {
        return .{ .owner = r, .flush = .{ .ctx = r, .run = flush } };
    }

    fn tick(r: *Recorded) void {
        var w = r.warm();
        r.ticked = r.gate.keepWarm(&w) catch false;
    }

    fn operation(r: *Recorded) void {
        r.attempted.store(true, .release);
        r.gate.begin();
        defer r.gate.end();
        r.ordered = r.retired.load(.acquire);
        r.entered.store(true, .release);
    }
};

fn until(flag: *const std.atomic.Value(bool)) !void {
    for (0..1000) |_| {
        if (flag.load(.acquire)) return;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.TestTimeout;
}

test "GLM DSA request gate: an operation excludes idle residency flushes and probes" {
    var r: Recorded = .{};
    var warm = r.warm();
    r.gate.begin();
    try std.testing.expect(!try r.gate.keepWarm(&warm));
    try std.testing.expectEqual(@as(u32, 0), r.flushes);
    try std.testing.expectEqual(@as(u32, 0), r.probes);
    r.gate.end();
    r.release.store(true, .release);
    try std.testing.expect(try r.gate.keepWarm(&warm));
    try std.testing.expectEqual(@as(u32, 1), r.flushes);
    try std.testing.expectEqual(@as(u32, 1), r.probes);
    try std.testing.expect(r.retired.load(.acquire));
}

test "GLM DSA request gate: an earlier idle probe retires before operation encoding can enter" {
    var r: Recorded = .{};
    const tick = try std.Thread.spawn(.{}, Recorded.tick, .{&r});
    var operation: ?std.Thread = null;
    defer {
        r.release.store(true, .release);
        tick.join();
        if (operation) |thread| thread.join();
    }
    try until(&r.waiting);
    operation = try std.Thread.spawn(.{}, Recorded.operation, .{&r});
    try until(&r.attempted);
    std.Io.sleep(std.testing.io, .fromMilliseconds(20), .awake) catch {};
    try std.testing.expect(!r.entered.load(.acquire));
    r.release.store(true, .release);
    try until(&r.entered);
    try std.testing.expect(r.ordered);
}

test "GLM DSA request gate: a failed idle probe reports failure and releases ownership" {
    var r: Recorded = .{ .failed = true };
    var warm = r.warm();
    r.release.store(true, .release);
    try std.testing.expectError(error.GpuFailure, r.gate.keepWarm(&warm));
    try std.testing.expect(r.retired.load(.acquire));
    try std.testing.expect(r.gate.mutex.tryLock());
    r.gate.end();
}
