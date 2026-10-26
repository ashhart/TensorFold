//! Engine.score for a lanes backend that scores labels: a request thread asks, the engine thread answers.
const std = @import("std");
const LaneHost = @import("lane_host.zig").LaneHost;

/// One scoring question, on the asking thread's stack until `done`.
pub const Ask = struct {
    prompt: []const u32,
    labels: []const u32,
    logits: []f64,
    mass: bool,
    hidden: ?[]u16 = null, // a hidden-rows ask: the final normed rows go here instead of label logits
    logsumexp: f64 = 0,
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
    h.lock();
    defer h.unlock();
    if (h.closing) return error.Failed;
    h.scores.asks.append(h.gpa, &ask) catch return error.Failed;
    h.wake.broadcast(h.io);
    while (!ask.done) h.wake.waitUncancelable(h.io, &h.mutex);
    if (ask.failed) return error.Failed;
    return ask.logsumexp;
}

/// Engine.hidden: the prompt's final normed rows, queued for the engine thread as scores are.
pub fn hidden(ctx: *anyopaque, prompt: []const u32, out: []u16) error{Failed}!void {
    var logits: [0]f64 = .{};
    var ask: Ask = .{ .prompt = prompt, .labels = &.{}, .logits = &logits, .mass = false, .hidden = out };
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    h.lock();
    defer h.unlock();
    if (h.closing) return error.Failed;
    h.scores.asks.append(h.gpa, &ask) catch return error.Failed;
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
        if (a.hidden) |out| {
            const rows = backend.vtable.hidden orelse {
                a.failed = true;
                continue;
            };
            rows(backend.ptr, a.prompt, out) catch |err| {
                std.log.warn("lane host: a decision's hidden rows failed ({s})", .{@errorName(err)});
                a.failed = true;
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
