//! A held fake phase lock cannot stop the observer, and a missing observer misses its warning.
const std = @import("std");
const watch = @import("startup_watch.zig");
const t = std.testing;
const Seen = struct {
    var warnings: std.atomic.Value(u32) = .init(0);
    var entries: usize = 0;
    var last: watch.Event = undefined;
    fn sink(waiting: bool, e: watch.Event) void {
        if (!waiting) {
            entries += 1;
            return;
        }
        last = e;
        _ = warnings.fetchAdd(1, .release);
    }
    fn clear() void {
        warnings.store(0, .release);
        entries = 0;
    }
};
test "startup watcher: minute cadence resets per peer phase, never warns for weights or one rank" {
    Seen.clear();
    var m: watch.Monitor = .{ .rank = 1, .world = 2, .sink = Seen.sink };
    m.enter(.nccl_init, 0);
    m.poll(59_999);
    try t.expectEqual(@as(u32, 0), Seen.warnings.load(.acquire));
    m.poll(60_000);
    m.poll(60_001);
    m.poll(119_999);
    try t.expectEqual(@as(u32, 1), Seen.warnings.load(.acquire));
    m.poll(120_000);
    try t.expectEqual(@as(u32, 2), Seen.warnings.load(.acquire));
    try t.expectEqual(@as(u64, 120), Seen.last.elapsed_s);
    try t.expectEqual(@as(usize, 2), Seen.last.world);
    m.enter(.peer_warmup, 120_001);
    m.poll(180_000);
    try t.expectEqual(@as(u32, 2), Seen.warnings.load(.acquire));
    m.poll(180_001);
    try t.expectEqual(@as(u32, 3), Seen.warnings.load(.acquire));
    for ([_]watch.Phase{ .weights, .calibration, .ready, .following, .failed }) |phase| {
        m.enter(phase, 200_000);
        m.poll(1_000_000);
    }
    try t.expectEqual(@as(u32, 3), Seen.warnings.load(.acquire));
    try t.expectEqual(@as(usize, 7), Seen.entries);
    m.world = 1;
    m.enter(.nccl_init, 0);
    m.poll(1_000_000);
    try t.expectEqual(@as(u32, 3), Seen.warnings.load(.acquire));
    try t.expectEqual(@as(usize, 7), Seen.entries);
}
const FakeClock = struct {
    now_ms: *std.atomic.Value(u64),
    pub fn now(c: FakeClock) u64 {
        return c.now_ms.load(.acquire);
    }
    pub fn pause(_: FakeClock) void {
        std.Thread.yield() catch {};
    }
};
fn awaitWarning(n: u32) !void {
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 2 * std.time.ns_per_s;
    while (Seen.warnings.load(.acquire) < n and std.Io.Clock.awake.now(t.io).toNanoseconds() < until) {
        std.Thread.yield() catch {};
    }
    try t.expectEqual(n, Seen.warnings.load(.acquire));
}
test "startup watcher: a blocked fake phase still emits, and the no-watcher control misses the line" {
    Seen.clear();
    var now: std.atomic.Value(u64) = .init(0);
    var m: watch.Monitor = .{ .rank = 0, .world = 2, .sink = Seen.sink };
    var held: std.Io.Mutex = .init;
    held.lockUncancelable(t.io);
    defer held.unlock(t.io);
    m.enter(.nccl_init, 0);
    now.store(60_000, .release);
    try t.expectEqual(@as(u32, 0), Seen.warnings.load(.acquire));
    const thread = try std.Thread.spawn(.{}, watch.drive, .{ &m, FakeClock{ .now_ms = &now } });
    defer {
        m.done.store(true, .release);
        thread.join();
    }
    try awaitWarning(1);
    try t.expectEqual(watch.Phase.nccl_init, Seen.last.phase);
    try t.expectEqual(@as(u64, 60), Seen.last.elapsed_s);
    now.store(120_000, .release);
    try awaitWarning(2);
}
test "startup watcher: transport reports only this rank's observed classes, with unknown when absent" {
    const text = "NCCL INFO Channel 00 : 0[0] -> 1[1] via P2P/IPC/read\n" ++
        "NCCL INFO Channel 01 : 1[1] -> 0[0] via NET/Socket/0\n" ++
        "NCCL INFO Channel 02 : 2[2] -> 3[3] via SHM/direct/direct\n";
    try t.expectEqual(@as(u3, 5), watch.transport(text, 0));
    try t.expectEqual(@as(u3, 2), watch.transport(text, 2));
    try t.expectEqual(@as(u3, 0), watch.transport(text, 4));
    try t.expectEqual(@as(u3, 0), watch.transport("", 0));
}

test "startup watcher: P2P advice requires two ranks and an observed P2P transport" {
    for (1..5) |world| for (0..8) |bits| {
        const e: watch.Event = .{ .world = world, .rank = 0, .phase = .peer_warmup, .transport = @intCast(bits) };
        const hint = watch.advice(e);
        try t.expectEqual(world == 2 and bits & 1 != 0, std.mem.indexOf(u8, hint, "NCCL_P2P_DISABLE=1") != null);
    };
    try t.expectEqual(@as(usize, 0), watch.advice(.{ .rank = 0, .phase = .nccl_init }).len);
}

test "startup log: minute warnings retain the active capture pathname" {
    Seen.clear();
    var m: watch.Monitor = .{ .rank = 1, .world = 2, .sink = Seen.sink };
    m.capture.state = .on;
    m.capture.dir[0] = '.';
    m.capture.dir_len = 1;
    m.enter(.peer_warmup, 0);
    m.poll(60_000);
    try t.expectEqual(@as(u32, 1), Seen.warnings.load(.acquire));
    try t.expectEqualStrings("./nccl.log", std.mem.sliceTo(&Seen.last.log_path, 0));
    m.poll(120_000);
    try t.expectEqual(@as(u32, 2), Seen.warnings.load(.acquire));
    try t.expectEqualStrings("./nccl.log", std.mem.sliceTo(&Seen.last.log_path, 0));
}
