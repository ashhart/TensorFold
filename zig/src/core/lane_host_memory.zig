//! Engine.memoryOp for LaneHost: threads post memory ops; the engine thread runs them idle or between rounds.
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

/// Whether a server builds a memory holder at all: only `--slide` with a checkpoint folder (a saved bank alone never).
pub fn wanted(dir: []const u8, slide: bool) bool {
    return slide and dir.len > 0;
}

fn isCpuOnly(op: api.MemoryOp) bool {
    return switch (op) {
        .save, .forget_parent, .supersede_parent, .consolidate, .clear_all, .stats, .publish_parent => true,
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

/// Runs at most one queued memory op between rounds; GPU ops (`init`, `teach_chunk`) need idle or a free lane.
pub fn answerOneInFlight(h: *LaneHost) void {
    if (h.folded_memory == null) return; // set before the host starts: without a bank, no lock between rounds
    h.lock();
    if (h.folded_memory == null or h.mem_ops.asks.items.len == 0) {
        h.unlock();
        return;
    }
    const spare_lane = h.admitted.items.len == 0 or h.admitted.items.len < h.info_.lanes;
    // arrival order: only the oldest op may run, so no later change is published ahead of an earlier one
    const head = h.mem_ops.asks.items[0];
    const idx: usize = if (spare_lane or isCpuOnly(head.op.*)) 0 else {
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
