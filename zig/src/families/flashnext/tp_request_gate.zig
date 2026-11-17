//! Paired requests exclude idle residency changes and finish an earlier idle probe before encoding.
const std = @import("std");

pub const Gate = struct {
    mutex: std.atomic.Mutex = .unlocked,
    active: std.atomic.Value(bool) = .init(false), // a request holds the gate, for readers that never take it

    pub fn begin(g: *Gate) void {
        while (!g.mutex.tryLock()) std.atomic.spinLoopHint();
        g.active.store(true, .release);
    }

    pub fn end(g: *Gate) void {
        g.active.store(false, .release);
        g.mutex.unlock();
    }

    pub fn held(g: *const Gate) bool {
        return g.active.load(.acquire);
    }

    pub fn keepWarm(g: *Gate, warm: anytype) !bool {
        if (!g.mutex.tryLock()) return false;
        defer g.mutex.unlock();
        if (warm.flush) |f| f.run(f.ctx);
        const cb = warm.probe();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
        return true;
    }
};

const Probe = struct {
    gate: Gate = .{},
    flushes: std.atomic.Value(u32) = .init(0),
    commits: std.atomic.Value(u32) = .init(0),
    waits: std.atomic.Value(u32) = .init(0),
    release: std.atomic.Value(bool) = .init(false),
    completed: std.atomic.Value(bool) = .init(false),
    attempted: std.atomic.Value(bool) = .init(false),
    entered: std.atomic.Value(bool) = .init(false),
    early: std.atomic.Value(bool) = .init(false),
    ticked: bool = false,
    gpu_error: bool = false,

    const Flush = struct { ctx: *anyopaque, run: *const fn (*anyopaque) void };
    const Warm = struct {
        p: *Probe,
        flush: ?Flush,

        pub fn probe(w: *Warm) Buffer {
            _ = w.p.commits.fetchAdd(1, .acq_rel);
            return .{ .p = w.p };
        }
    };
    const Buffer = struct {
        p: *Probe,

        pub fn wait(cb: Buffer) void {
            _ = cb.p.waits.fetchAdd(1, .acq_rel);
            while (!cb.p.release.load(.acquire)) std.atomic.spinLoopHint();
            cb.p.completed.store(true, .release);
        }

        pub fn failure(cb: Buffer) ?[]const u8 {
            return if (cb.p.gpu_error) "planted failure" else null;
        }
    };

    fn flush(ctx: *anyopaque) void {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        _ = p.flushes.fetchAdd(1, .acq_rel);
    }

    fn warm(p: *Probe) Warm {
        return .{ .p = p, .flush = .{ .ctx = p, .run = flush } };
    }

    fn tick(p: *Probe) void {
        var target = p.warm();
        p.ticked = p.gate.keepWarm(&target) catch false;
    }

    fn request(p: *Probe) void {
        p.attempted.store(true, .release);
        p.gate.begin();
        defer p.gate.end();
        p.early.store(!p.completed.load(.acquire), .release);
        p.entered.store(true, .release);
    }
};

fn until(value: anytype, want: @TypeOf(value.load(.acquire))) !void {
    for (0..1000) |_| {
        if (value.load(.acquire) == want) return;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.Timeout;
}

test "an idle residency flush and submitted probe finish before a paired request can enter" {
    var p: Probe = .{};
    const ticker = try std.Thread.spawn(.{}, Probe.tick, .{&p});
    var request: ?std.Thread = null;
    defer {
        p.release.store(true, .release);
        ticker.join();
        if (request) |th| th.join();
    }
    try until(&p.waits, 1);
    request = try std.Thread.spawn(.{}, Probe.request, .{&p});
    try until(&p.attempted, true);
    std.Io.sleep(std.testing.io, .fromMilliseconds(20), .awake) catch {};
    try std.testing.expect(!p.entered.load(.acquire));
    p.release.store(true, .release);
    try until(&p.entered, true);
    try std.testing.expect(!p.early.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), p.flushes.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), p.commits.load(.acquire));
}

test "a paired request prevents every idle flush and commit and idle ticks resume afterwards" {
    var p: Probe = .{};
    p.gate.begin();
    const ticker = try std.Thread.spawn(.{}, Probe.tick, .{&p});
    ticker.join();
    p.gate.end();
    try std.testing.expect(!p.ticked);
    try std.testing.expectEqual(@as(u32, 0), p.flushes.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), p.commits.load(.acquire));
    p.release.store(true, .release);
    Probe.tick(&p);
    try std.testing.expect(p.ticked);
    try std.testing.expectEqual(@as(u32, 1), p.flushes.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), p.commits.load(.acquire));
    try std.testing.expect(p.completed.load(.acquire));
}

test "the request gate reports a request active only between begin and end" {
    var p: Probe = .{};
    try std.testing.expect(!p.gate.held());
    p.release.store(true, .release);
    var target = p.warm();
    try std.testing.expect(try p.gate.keepWarm(&target));
    try std.testing.expect(!p.gate.held());
    p.gate.begin();
    try std.testing.expect(p.gate.held());
    p.gate.end();
    try std.testing.expect(!p.gate.held());
}

test "a failed idle probe releases the request gate and reports its GPU error" {
    var p: Probe = .{ .gpu_error = true };
    p.release.store(true, .release);
    var target = p.warm();
    try std.testing.expectError(error.GpuFailed, p.gate.keepWarm(&target));
    try std.testing.expect(p.gate.mutex.tryLock());
    p.gate.mutex.unlock();
}
