//! Engine.memoryOp for LaneHost: HTTP/background threads post Folded Memory Weights ops, and the
//! engine thread executes them either idle OR interleaved mid-flight between decode rounds!
const std = @import("std");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;

pub const Ask = struct {
    op: *const api.MemoryOp,
    result: api.MemoryResult = .{},
    done: bool = false,
};

pub const Queue = struct {
    asks: std.ArrayList(*Ask) = .empty,
    answered: std.Io.Condition = .init,
};

fn isCpuOnly(op: api.MemoryOp) bool {
    return switch (op) {
        .save, .forget_parent, .supersede_parent, .consolidate, .clear_all, .stats => true,
        .init, .teach_chunk, .find_conflicts => false,
    };
}

pub fn call(ctx: *anyopaque, op: *const api.MemoryOp) api.MemoryResult {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    var ask: Ask = .{ .op = op };
    h.lock();
    defer h.unlock();
    if (h.closing.load(.acquire)) return .{ .err = "the engine closed" };
    if (h.folded_memory == null) return .{ .err = "folded memory weights are not attached" };
    h.mem_ops.asks.append(h.gpa, &ask) catch return .{ .err = "out of memory" };
    h.wake.broadcast(h.io);
    while (!ask.done) h.mem_ops.answered.waitUncancelable(h.io, &h.mutex);
    return ask.result;
}

pub fn ready(h: *const LaneHost) bool {
    return h.folded_memory != null and h.mem_ops.asks.items.len > 0 and h.admitted.items.len == 0;
}

/// Runs AT MOST ONE queued memory op mid-flight between decode rounds whenever safe:
/// - CPU-only ops (`save`, `forget_parent`, `stats`, `clear_all`) run immediately between rounds.
/// - GPU compile ops (`init`, `teach_chunk`) run whenever the engine is idle OR at least one spare
///   lane is free (`admitted < lanes`), letting live streaming traffic keep flowing with minimal pause!
pub fn answerOneInFlight(h: *LaneHost) void {
    if (h.folded_memory == null) return; // set before the host starts: without a bank, no lock between rounds
    h.lock();
    if (h.folded_memory == null or h.mem_ops.asks.items.len == 0) {
        h.unlock();
        return;
    }
    const spare_lane = h.admitted.items.len == 0 or h.admitted.items.len < h.info_.lanes;
    var pick: ?usize = null;
    for (h.mem_ops.asks.items, 0..) |ask, i| {
        if (spare_lane or isCpuOnly(ask.op.*)) {
            pick = i;
            break;
        }
    }
    const idx = pick orelse {
        h.unlock();
        return;
    };
    const ask = h.mem_ops.asks.orderedRemove(idx);
    const ctrl = h.folded_memory;
    h.unlock();

    if (h.closing.load(.acquire)) {
        ask.result = .{ .err = "the engine closed" };
    } else if (ctrl) |c| {
        ask.result = c.exec(c.ctx, ask.op);
    } else {
        ask.result = .{ .err = "folded memory weights are not attached" };
    }

    h.lock();
    ask.done = true;
    h.mem_ops.answered.broadcast(h.io);
    h.unlock();
}

/// Runs all queued memory ops on the engine thread while idle (zero active streams).
pub fn answer(h: *LaneHost) void {
    h.lock();
    if (h.mem_ops.asks.items.len == 0) {
        h.unlock();
        return;
    }
    const list = h.mem_ops.asks.toOwnedSlice(h.gpa) catch {
        h.unlock();
        return;
    };
    h.unlock();
    defer h.gpa.free(list);

    const ctrl = h.folded_memory;
    for (list) |ask| {
        if (h.closing.load(.acquire)) {
            ask.result = .{ .err = "the engine closed" };
        } else if (ctrl) |c| {
            ask.result = c.exec(c.ctx, ask.op);
        } else {
            ask.result = .{ .err = "folded memory weights are not attached" };
        }
    }
    h.lock();
    for (list) |ask| ask.done = true;
    h.mem_ops.answered.broadcast(h.io);
    h.unlock();
}

pub fn close(h: *LaneHost) void {
    h.lock();
    const list = h.mem_ops.asks.toOwnedSlice(h.gpa) catch &.{};
    for (list) |ask| {
        ask.result = .{ .err = "the engine closed" };
        ask.done = true;
    }
    h.mem_ops.answered.broadcast(h.io);
    h.unlock();
    if (list.len > 0) h.gpa.free(list);
    h.mem_ops.asks.deinit(h.gpa);
}
