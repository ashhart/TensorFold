//! The chunk gate every rank passes before a prompt chunk (kept state, heat wait, cancel) and the ranks' maximum.
const std = @import("std");
const heat = @import("heat");
const snapshots = @import("cuda_snapshots.zig");
const prof_mod = @import("cuda_prof.zig");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;
const Cancel = engine.Cancel;

/// The chunk gate on every rank: a kept state at the mark, the heat wait, rank 0's cancel; the id exchanged first.
pub fn gate(e: *Engine, cancel: ?Cancel) !void {
    try snapshots.gate(e);
    if (e.heat_gate.bands != null) {
        var w: Wait = .{ .e = e, .cancel = cancel };
        const before = e.heat_gate.waited_s;
        e.heat_gate.waitUntil(Wait.read, Wait.sleep, &w, .{ .ptr = &w, .check = Wait.stop }) catch |err| return w.failed orelse err;
        if (e.heat_gate.waited_s > before) heat.note(e.heat_gate.waited_s - before);
    }
    if (try e.shared(if (cancel) |c| @floatFromInt(@intFromBool(c.check(c.ptr))) else 0) > 0) return error.Cancelled;
}

/// The ranks' maximum of `v` (an all-reduce on the engine's stream).
pub fn shared(e: *Engine, v: f64) !f64 {
    var x = v;
    try e.shared_buf.uploadAsync(0, std.mem.asBytes(&x), e.stream.handle);
    const stamp = prof_mod.span(e.profiling_prompt, e.stream, .control_reduce);
    try e.lib.check(e.lib.api.ncclAllReduce(e.shared_buf.ptr, e.shared_buf.ptr, 1, .f64, .max, e.comm, e.stream.handle), "ncclAllReduce");
    stamp.end();
    try e.stream.synchronize();
    try e.shared_buf.download(0, std.mem.asBytes(&x));
    return x;
}

/// heat.Gate.waitUntil's hooks over the ranks: gathered max reading, rank 0's stop: all pause together.
const Wait = struct {
    e: *Engine,
    cancel: ?Cancel,
    failed: ?anyerror = null,

    fn read(ptr: *anyopaque) anyerror!f64 {
        const w: *Wait = @ptrCast(@alignCast(ptr));
        const local: f64 = blk: {
            var dir = std.Io.Dir.cwd().openDir(w.e.io, w.e.heat_gate.root, .{ .iterate = true }) catch break :blk -std.math.inf(f64);
            defer dir.close(w.e.io);
            break :blk heat.hottest(w.e.io, dir) catch -std.math.inf(f64);
        };
        return w.e.shared(local);
    }

    fn sleep(ptr: *anyopaque) void {
        const w: *Wait = @ptrCast(@alignCast(ptr));
        std.Io.sleep(w.e.io, .fromMilliseconds(2000), .awake) catch {};
    }

    fn stop(ptr: *anyopaque) bool {
        const w: *Wait = @ptrCast(@alignCast(ptr));
        const mine: f64 = if (w.cancel) |c| @floatFromInt(@intFromBool(c.check(c.ptr))) else 0;
        const any = w.e.shared(mine) catch |err| {
            w.failed = err;
            return true;
        };
        return any > 0;
    }
};
