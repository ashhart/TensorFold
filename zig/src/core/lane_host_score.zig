//! Engine.score and Engine.decide for a lanes backend that scores: a request thread asks, the engine thread answers.
const std = @import("std");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;

/// One scoring question, on the asking thread's stack until `done`: a prompt's labels, or a decision's branches.
pub const Ask = struct {
    kind: enum { score, decide } = .score,
    /// The prompt scored, or the decision's trunk.
    prompt: []const u32,
    labels: []const u32,
    logits: []f64,
    mass: bool = false,
    branches: []const []const u32 = &.{},
    counts: []const usize = &.{},
    logsumexp: f64 = 0,
    /// A decision the backend refused, and its reason (a client's error, not the engine's).
    refused: bool = false,
    problem: []const u8 = "",
    failed: bool = false,
    done: bool = false,
};

/// Questions waiting for the engine thread (under the host's lock).
pub const Queue = struct {
    asks: std.ArrayList(*Ask) = .empty,
};

/// Engine.score: queued for the engine thread, which runs it between two rounds; Failed when the host is closing.
pub fn score(ctx: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) error{Failed}!f64 {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    var ask: Ask = .{ .prompt = prompt, .labels = labels, .logits = logits, .mass = mass };
    try put(h, &ask);
    return ask.logsumexp;
}

/// Engine.decide: one queued question for every branch, answered as `score` is; text only (no pictures).
pub fn decide(ctx: *anyopaque, d: *api.Decision) error{ Failed, Refused }!void {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    if (d.pictures.len > 0) {
        d.problem = "this engine reads text only: a decision with pictures is not supported";
        return error.Refused;
    }
    var ask: Ask = .{ .kind = .decide, .prompt = d.trunk, .labels = d.labels, .logits = d.logits, .branches = d.branches, .counts = d.counts };
    try put(h, &ask);
    if (ask.refused) {
        d.problem = ask.problem;
        return error.Refused;
    }
    var read = d.trunk.len;
    for (d.branches) |b| read += b.len;
    d.read = read;
}

/// Queues `ask` for the engine thread and waits for its answer; Failed when the host is closing or the answer failed.
fn put(h: *LaneHost, ask: *Ask) error{Failed}!void {
    h.lock();
    defer h.unlock();
    if (h.closing) return error.Failed;
    h.scores.asks.append(h.gpa, ask) catch return error.Failed;
    h.wake.broadcast(h.io);
    while (!ask.done) h.wake.waitUncancelable(h.io, &h.mutex);
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
        // a drain's deadline stops the queue: the started score finishes, the rest fail without running
        h.lock();
        const closing = h.closing;
        h.unlock();
        if (closing) {
            a.failed = true;
            continue;
        }
        switch (a.kind) {
            .score => {
                const run = backend.vtable.score orelse {
                    a.failed = true;
                    continue;
                };
                if (run(backend.ptr, a.prompt, a.labels, a.logits, a.mass)) |lse| a.logsumexp = lse else |err| {
                    std.log.warn("lane host: a decision score failed ({s})", .{@errorName(err)});
                    a.failed = true;
                }
            },
            .decide => {
                const run = backend.vtable.decide orelse {
                    a.failed = true;
                    continue;
                };
                run(backend.ptr, a.prompt, a.branches, a.counts, a.labels, a.logits, &a.problem) catch |err| {
                    if (err == error.Refused) {
                        a.refused = true;
                        continue;
                    }
                    std.log.warn("lane host: a packed decision failed ({s})", .{@errorName(err)});
                    a.failed = true;
                };
            },
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
