//! An exit notification is not a hook join: shutdown must retain the fate and its callback owners until both finish.
const std = @import("std");
const Fate = @import("fate.zig").Fate;
const fate = @import("fate.zig");
const sock = @import("sock.zig");
const t = std.testing;
const Flag = std.atomic.Value(bool);

var exits: std.atomic.Value(u32) = .init(0);

fn recordExit(code: u8) void {
    std.debug.assert(code == fate.exit_code);
    _ = exits.fetchAdd(1, .acq_rel);
}

fn wait(flag: *const Flag) !void {
    const until = sock.nowNs() + 10 * std.time.ns_per_s;
    while (!flag.load(.acquire)) {
        if (sock.nowNs() > until) return error.StalledFixture;
        sock.sleepNs(100 * std.time.ns_per_us);
    }
}

const Case = struct {
    f: Fate,
    entered: Flag = .init(false),
    release: Flag = .init(false),
    close_started: Flag = .init(false),
    closed: Flag = .init(false),
    early: Flag = .init(false),
    calls: std.atomic.Value(u32) = .init(0),
    want_calls: u32 = 2,
    exit_done: ?*const Flag = null,

    fn init() Case {
        var c: Case = .{ .f = Fate.init(.{ .world = 1, .grace_ns = 0 }, @splat(-1)) };
        c.f.exit_fn = recordExit;
        return c;
    }

    fn first(ctx: *anyopaque, reason: []const u8) void {
        const c: *Case = @ptrCast(@alignCast(ctx));
        c.entered.store(true, .release);
        while (!c.release.load(.acquire)) sock.sleepNs(100 * std.time.ns_per_us);
        std.debug.assert(std.mem.eql(u8, reason, "rank 0 failed in held hook: Unexpected"));
        _ = c.calls.fetchAdd(1, .acq_rel);
    }

    fn last(ctx: *anyopaque, reason: []const u8) void {
        const c: *Case = @ptrCast(@alignCast(ctx));
        std.debug.assert(std.mem.eql(u8, reason, "rank 0 failed in held hook: Unexpected"));
        _ = c.calls.fetchAdd(1, .acq_rel);
    }

    fn close(c: *Case) void {
        c.close_started.store(true, .release);
        c.f.close();
        const owner_done = if (c.exit_done) |done| done.load(.acquire) else true;
        c.early.store(c.calls.load(.acquire) != c.want_calls or !owner_done, .release);
        c.closed.store(true, .release);
    }

    fn fatal(c: *Case) void {
        c.f.fatal("held hook", error.Unexpected);
    }

    fn releaseAfterCloseStarts(c: *Case) !void {
        try wait(&c.close_started);
        const until = sock.nowNs() + 5 * std.time.ns_per_ms;
        // Observe an early return if one occurs; the hold merely creates an overlap, never a passing time bound.
        while (!c.closed.load(.acquire) and sock.nowNs() < until) sock.sleepNs(100 * std.time.ns_per_us);
        c.release.store(true, .release);
    }

    fn run(c: *Case) !void {
        c.f.onFail(.{ .ctx = c, .call = first });
        c.f.onFail(.{ .ctx = c, .call = last });
        try c.f.start();
        defer c.f.close();
        defer c.release.store(true, .release);
        const other = try std.Thread.spawn(.{}, fatal, .{c});
        c.fatal();
        other.join();
        try wait(&c.entered);
        // The exit callback already returned while the first abort is held, so aborts cannot hold the exit deadline.
        try t.expect(c.f.exited.load(.acquire));
        const closer = try std.Thread.spawn(.{}, close, .{c});
        defer closer.join();
        defer c.release.store(true, .release);
        try c.releaseAfterCloseStarts();
        try wait(&c.closed);
        const until = sock.nowNs() + 10 * std.time.ns_per_s;
        while (c.calls.load(.acquire) != 2) {
            if (sock.nowNs() > until) return error.StalledFixture;
            sock.sleepNs(100 * std.time.ns_per_us);
        }
    }
};

const Worker = struct {
    cases: []Case,
    failed: bool = false,
    fn run(w: *Worker) void {
        for (w.cases) |*c| c.run() catch {
            w.failed = true;
            return;
        };
    }
};

test "fate lifetime: four threads close 256 failed ranks only after both held hooks finish" {
    exits.store(0, .release);
    const cases = try t.allocator.alloc(Case, 256);
    defer t.allocator.free(cases);
    for (cases) |*c| c.* = .init();
    var workers: [4]Worker = undefined;
    var threads: [4]std.Thread = undefined;
    var started: usize = 0;
    defer for (threads[0..started]) |thread| thread.join();
    for (&workers, &threads, 0..) |*w, *thread, i| {
        w.* = .{ .cases = cases[i * 64 ..][0..64] };
        thread.* = try std.Thread.spawn(.{}, Worker.run, .{w});
        started += 1;
    }
    for (threads[0..started]) |thread| thread.join();
    started = 0;
    for (workers) |w| try t.expect(!w.failed);
    for (cases) |*c| try t.expect(!c.early.load(.acquire));
    try t.expectEqual(@as(u32, 256), exits.load(.acquire));
}

const ExitGate = struct {
    entered: Flag = .init(false),
    release: Flag = .init(false),
    done: Flag = .init(false),
    fn exit(code: u8) void {
        std.debug.assert(code == fate.exit_code);
        gate.entered.store(true, .release);
        while (!gate.release.load(.acquire)) sock.sleepNs(100 * std.time.ns_per_us);
        gate.done.store(true, .release);
    }
};
var gate: ExitGate = .{};

test "fate lifetime: close waits for an external fatal caller to finish its returning exit callback" {
    gate = .{};
    var c: Case = .init();
    c.want_calls = 0;
    c.exit_done = &gate.done;
    c.f.exit_fn = ExitGate.exit;
    const owner = try std.Thread.spawn(.{}, Case.fatal, .{&c});
    defer owner.join();
    defer gate.release.store(true, .release);
    try wait(&gate.entered);
    const closer = try std.Thread.spawn(.{}, Case.close, .{&c});
    defer closer.join();
    defer gate.release.store(true, .release);
    try c.releaseAfterCloseStarts();
    gate.release.store(true, .release);
    try wait(&c.closed);
    try t.expect(!c.early.load(.acquire));
    try t.expect(gate.done.load(.acquire));
}
