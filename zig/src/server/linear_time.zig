//! Timing for linear-time tests: two sizes run in turn on the thread's CPU clock, each keeping its fastest run.
const std = @import("std");
const builtin = @import("builtin");
const windows = std.os.windows;

extern "kernel32" fn QueryThreadCycleTime(thread: windows.HANDLE, cycles: *u64) callconv(.winapi) windows.BOOL;

/// Each size's fastest run: `ns` for limits in time, `work` to compare the two, in cycles on Windows.
pub const Fastest = struct { ns: [2]u64, work: [2]u64 };

/// The thread's CPU clock where it counts to the microsecond, so other work on the machine adds nothing; else awake.
pub fn clock(io: std.Io) std.Io.Clock {
    const r = std.Io.Clock.cpu_thread.resolution(io) catch return .awake;
    return if (r.nanoseconds > 0 and r.nanoseconds <= std.time.ns_per_us) .cpu_thread else .awake;
}

/// This thread's work so far: its cycles on Windows, where `clock` can only give awake, else ns on `c`.
fn work(c: std.Io.Clock) !u64 {
    if (builtin.os.tag != .windows) return @intCast(c.now(std.testing.io).nanoseconds);
    var cycles: u64 = undefined;
    if (!QueryThreadCycleTime(windows.GetCurrentThread(), &cycles).toBool()) return error.NoCycleTime;
    return cycles;
}

/// The fastest of `rounds` runs each of `run(ctx, 0)` and `run(ctx, 1)`, taken in turn so load slows both alike.
pub fn fastest(rounds: usize, ctx: anytype, comptime run: fn (@TypeOf(ctx), usize) anyerror!void) !Fastest {
    const io = std.testing.io;
    const c = clock(io);
    var best: Fastest = .{ .ns = @splat(std.math.maxInt(u64)), .work = @splat(std.math.maxInt(u64)) };
    for (0..rounds) |_| for (0..2) |i| {
        const start = c.now(io);
        const done = try work(c);
        try run(ctx, i);
        best.work[i] = @min(best.work[i], try work(c) -| done);
        best.ns[i] = @min(best.ns[i], @as(u64, @intCast(@max(0, start.durationTo(c.now(io)).nanoseconds))));
    };
    return best;
}

test "fastest's work leaves out a sleep in a run" {
    if (builtin.os.tag != .windows and clock(std.testing.io) != .cpu_thread) return error.SkipZigTest;
    const Nap = struct {
        fn run(_: void, i: usize) anyerror!void {
            for (0..1 << 20) |n| std.mem.doNotOptimizeAway(n);
            if (i == 1) try std.Io.sleep(std.testing.io, .fromMilliseconds(50), .awake);
        }
    };
    const f = try fastest(3, {}, Nap.run);
    try std.testing.expect(f.work[1] < 2 * f.work[0]);
}

test "the CPU clock leaves out a sleep that the awake clock counts" {
    const io = std.testing.io;
    if (clock(io) != .cpu_thread) return error.SkipZigTest;
    const cpu = std.Io.Clock.cpu_thread.now(io);
    const awake = std.Io.Clock.awake.now(io);
    try std.Io.sleep(io, .fromMilliseconds(50), .awake);
    try std.testing.expect(awake.durationTo(std.Io.Clock.awake.now(io)).nanoseconds >= 50 * std.time.ns_per_ms);
    try std.testing.expect(cpu.durationTo(std.Io.Clock.cpu_thread.now(io)).nanoseconds < 25 * std.time.ns_per_ms);
}
