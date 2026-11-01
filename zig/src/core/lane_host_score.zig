//! Engine.score and Engine.decide for a lanes host: a request thread asks, the engine thread answers between rounds.
const std = @import("std");
const LaneHost = @import("lane_host.zig").LaneHost;
const Decision = @import("engine_api.zig").Decision;

/// A family's whole-decision runner (opt-in): the trunk once and each branch after it, on the engine thread.
pub const Decider = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque, d: *Decision) anyerror!void };

/// One scoring question, on the asking thread's stack until `done`.
pub const Ask = struct {
    prompt: []const u32,
    labels: []const u32,
    logits: []f64,
    mass: bool,
    logsumexp: f64 = 0,
    failed: bool = false,
    done: bool = false,
    /// A whole decision for the host's decider instead of one score; `refused` keeps its problem words.
    decision: ?*Decision = null,
    refused: bool = false,
};

/// Questions waiting for the engine thread (under the host's lock).
pub const Queue = struct {
    asks: std.ArrayList(*Ask) = .empty,
};

/// Engine.score: queued for the engine thread, which runs it between two rounds; Failed when the host is closing.
pub fn score(ctx: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) error{Failed}!f64 {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    var ask: Ask = .{ .prompt = prompt, .labels = labels, .logits = logits, .mass = mass };
    h.lock();
    defer h.unlock();
    if (h.closing) return error.Failed;
    h.scores.asks.append(h.gpa, &ask) catch return error.Failed;
    h.wake.broadcast(h.io);
    while (!ask.done) h.wake.waitUncancelable(h.io, &h.mutex);
    if (ask.failed) return error.Failed;
    return ask.logsumexp;
}

/// Engine.decide: one queued decision answered whole on the engine thread, so concurrent ones take turns.
pub fn decide(ctx: *anyopaque, d: *Decision) error{ Failed, Refused }!void {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    var ask: Ask = .{ .prompt = &.{}, .labels = &.{}, .logits = &.{}, .mass = false, .decision = d };
    h.lock();
    defer h.unlock();
    if (h.closing) return error.Failed;
    h.scores.asks.append(h.gpa, &ask) catch return error.Failed;
    h.wake.broadcast(h.io);
    while (!ask.done) h.wake.waitUncancelable(h.io, &h.mutex);
    if (ask.refused) return error.Refused;
    if (ask.failed) return error.Failed;
}

/// Every queued question answered by the backend (engine thread, between rounds).
pub fn answer(h: *LaneHost) void {
    h.lock();
    if (h.scores.asks.items.len == 0) {
        h.unlock();
        return;
    }
    const list = h.scores.asks.toOwnedSlice(h.gpa) catch {
        h.unlock();
        return;
    };
    h.unlock();
    defer h.gpa.free(list);
    const backend = h.core.backend;
    for (list) |a| {
        if (a.decision) |d| {
            const decider = h.decider orelse {
                a.failed = true;
                continue;
            };
            decider.run(decider.ctx, d) catch |err| switch (err) {
                error.Refused => a.refused = true,
                else => {
                    std.log.warn("lane host: a decision failed ({s})", .{@errorName(err)});
                    a.failed = true;
                },
            };
            continue;
        }
        const run = backend.vtable.score orelse {
            a.failed = true;
            continue;
        };
        if (run(backend.ptr, a.prompt, a.labels, a.logits, a.mass)) |lse| a.logsumexp = lse else |err| {
            std.log.warn("lane host: a decision score failed ({s})", .{@errorName(err)});
            a.failed = true;
        }
    }
    h.lock();
    for (list) |a| a.done = true;
    h.wake.broadcast(h.io);
    h.unlock();
}

/// Questions left when the engine stops: each fails.
pub fn close(h: *LaneHost) void {
    h.lock();
    h.closing = true;
    for (h.scores.asks.items) |a| {
        a.failed = true;
        a.done = true;
    }
    h.scores.asks.clearAndFree(h.gpa);
    h.wake.broadcast(h.io);
    h.unlock();
}
