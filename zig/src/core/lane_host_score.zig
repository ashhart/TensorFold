//! Engine.score and Engine.decide for a lanes host: a request thread asks, the engine thread answers between rounds.
const std = @import("std");
const LaneHost = @import("lane_host.zig").LaneHost;
const Decision = @import("engine_api.zig").Decision;

/// A family's whole-decision runner (opt-in): the trunk once and each branch after it, on the engine thread.
pub const Decider = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, d: *Decision) anyerror!void,
    /// Several queued decisions at once (one batched pass where they fit); errs[i]: decision i's error, null when done.
    run_many: ?*const fn (ctx: *anyopaque, ds: []const *Decision, errs: []?anyerror) void = null,
};

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
    const handled: ?[]bool = h.gpa.alloc(bool, list.len) catch null; // out of memory: one at a time, as before
    defer if (handled) |x| h.gpa.free(x);
    if (handled) |x| {
        @memset(x, false);
        if (h.decider) |decider| if (decider.run_many) |many| batched(h, decider, many, list, x);
    }
    for (list, 0..) |a, at| {
        if (handled) |x| if (x[at]) continue; // answered in a batch
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

/// Queued decisions (two or more) through `many` at once, marked `handled`; `answer` sets done with the rest.
fn batched(h: *LaneHost, decider: Decider, many: *const fn (ctx: *anyopaque, ds: []const *Decision, errs: []?anyerror) void, list: []*Ask, handled: []bool) void {
    var count: usize = 0;
    for (list) |a| count += @intFromBool(a.decision != null);
    if (count < 2) return;
    const ds = h.gpa.alloc(*Decision, count) catch return;
    defer h.gpa.free(ds);
    const errs = h.gpa.alloc(?anyerror, count) catch return;
    defer h.gpa.free(errs);
    var i: usize = 0;
    for (list) |a| if (a.decision) |d| {
        ds[i] = d;
        i += 1;
    };
    @memset(errs, null);
    many(decider.ctx, ds, errs);
    i = 0;
    for (list, handled) |a, *done| if (a.decision != null) {
        if (errs[i]) |err| switch (err) {
            error.Refused => a.refused = true,
            else => {
                std.log.warn("lane host: a decision failed ({s})", .{@errorName(err)});
                a.failed = true;
            },
        };
        done.* = true;
        i += 1;
    };
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
