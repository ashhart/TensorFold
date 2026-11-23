//! A score asker's wake is never taken by the lane thread's timed idle wait, on a schedule made certain by a held Io.
const std = @import("std");
const lanes = @import("lanes");
const LaneHost = @import("lane_host.zig").LaneHost;
const api = @import("engine_api.zig");
const t = std.testing;

/// The schedule: the asker, once woken, is held before it takes its wake until the lane thread has waited twice.
const Hold = struct {
    wake: ?*const u32 = null, // the lane thread's idle condition (the host's `wake`)
    mutex: ?*const u32 = null, // the host's lock, whose waits are never held
    arrived: std.atomic.Value(bool) = .init(false), // the asker woke and is held
    waits: std.atomic.Value(u32) = .init(0), // the lane thread's timed waits on `wake` begun since the asker arrived
    held: std.atomic.Value(bool) = .init(false), // the asker was held (the schedule ran)
};

var hold: Hold = .{};
var inner: *const std.Io.VTable = undefined;
var vtable: std.Io.VTable = undefined;
threadlocal var asker = false;

fn timed(ud: ?*anyopaque, ptr: *const u32, expected: u32, timeout: std.Io.Timeout) std.Io.Cancelable!void {
    if (ptr == hold.wake and timeout != .none and hold.arrived.load(.acquire)) _ = hold.waits.fetchAdd(1, .acq_rel);
    return inner.futexWait(ud, ptr, expected, timeout);
}

fn untimed(ud: ?*anyopaque, ptr: *const u32, expected: u32) void {
    inner.futexWaitUncancelable(ud, ptr, expected);
    if (!asker or ptr == hold.mutex) return;
    asker = false; // held once: its later waits are the real ones
    hold.arrived.store(true, .release);
    hold.held.store(true, .release);
    for (0..3000) |_| { // bounded: three seconds at most
        if (hold.waits.load(.acquire) >= 2) break;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
}

/// `std.testing.io` with its futex waits observed: the host's own code runs unchanged on it.
fn heldIo() std.Io {
    inner = t.io.vtable;
    vtable = inner.*;
    vtable.futexWait = timed;
    vtable.futexWaitUncancelable = untimed;
    return .{ .userdata = t.io.userdata, .vtable = &vtable };
}

fn score(_: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64, _: bool) anyerror!f64 {
    for (logits, 0..) |*v, i| v.* = @floatFromInt(prompt.len + labels[i]);
    return 1.5;
}

const Asked = struct { lse: f64 = 0, failed: bool = false, done: std.atomic.Value(bool) = .init(false) };

fn ask(engine: api.Engine, out: *Asked) void {
    asker = true;
    var logits: [2]f64 = undefined;
    const prompt = [_]u32{ 1, 2, 3 };
    const labels = [_]u32{ 7, 9 };
    out.lse = engine.score(&prompt, &labels, &logits, true) catch blk: {
        out.failed = true;
        break :blk 0;
    };
    out.done.store(true, .release);
}

test "lane condition stress: an answered asker held past the lane thread's timed waits still gets its answer" {
    const gpa = t.allocator;
    hold = .{};
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    const plain = target.backend();
    var vt = plain.vtable.*;
    vt.score = score;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, .{ .ptr = plain.ptr, .vtable = &vt }, clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, heldIo(), &core, .{ .lanes = 1 });
    hold.wake = &host.wake.epoch.raw;
    hold.mutex = @ptrCast(&host.mutex.state.raw);
    try host.start();
    var out: Asked = .{};
    const th = std.Thread.spawn(.{}, ask, .{ host.engine(), &out }) catch |err| {
        host.stop();
        return err;
    };
    var answered = false;
    for (0..3000) |_| { // bounded: a stranded asker is reported, not waited for
        if (out.done.load(.acquire)) {
            answered = true;
            break;
        }
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    host.stop(); // its close broadcasts: a stranded asker returns, so the thread joins either way
    th.join();
    try t.expect(hold.held.load(.acquire)); // the schedule ran: the asker was woken and held
    try t.expect(answered); // stranded when its wake was taken by the lane thread's timed wait
    try t.expect(!out.failed and out.lse == 1.5);
}
