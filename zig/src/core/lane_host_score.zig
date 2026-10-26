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
    at: usize = 0, // a hidden-rows ask's next prompt row: it runs one backend piece per engine-thread pass
    logsumexp: f64 = 0,
    failed: bool = false,
    done: bool = false,
};

/// Questions waiting for the engine thread (under the host's lock).
pub const Queue = struct {
    asks: std.ArrayList(*Ask) = .empty,
    stepping: std.ArrayList(*Ask) = .empty, // hidden-rows asks part way through their prompt, oldest first
};

/// A decision's prompt is part way through (under the host's lock): the engine thread should not idle.
pub fn busy(h: *LaneHost) bool {
    return h.scores.stepping.items.len > 0;
}

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

/// Queued scores answered, and one prompt piece of the oldest decision (engine thread, between rounds).
pub fn answer(h: *LaneHost) void {
    h.lock();
    const list = h.scores.asks.toOwnedSlice(h.gpa) catch &.{};
    h.unlock();
    defer h.gpa.free(list);
    const backend = h.core.backend;
    for (list) |a| {
        if (a.hidden != null) {
            if (backend.vtable.hidden == null) {
                a.failed = true;
                continue;
            }
            h.lock();
            defer h.unlock();
            h.scores.stepping.append(h.gpa, a) catch {
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
    step(h);
    h.lock();
    for (list) |a| if (a.hidden == null or a.failed) {
        a.done = true;
    };
    h.wake.broadcast(h.io);
    h.unlock();
}

/// One backend piece of the oldest decision's prompt; done (and dropped) at its last row or on a failure.
fn step(h: *LaneHost) void {
    h.lock();
    const a = if (h.scores.stepping.items.len > 0) h.scores.stepping.items[0] else null;
    h.unlock();
    const ask = a orelse return;
    const backend = h.core.backend;
    var finished = false;
    if (backend.vtable.hidden.?(backend.ptr, ask.prompt, ask.hidden.?, ask.at)) |next| {
        ask.at = next;
        finished = next >= ask.prompt.len;
    } else |err| {
        std.log.warn("lane host: a decision's hidden rows failed ({s})", .{@errorName(err)});
        ask.failed = true;
        finished = true;
    }
    if (!finished) return;
    h.lock();
    _ = h.scores.stepping.orderedRemove(0);
    ask.done = true;
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
    for (h.scores.stepping.items) |a| {
        a.failed = true;
        a.done = true;
    }
    h.scores.stepping.clearAndFree(h.gpa);
    h.wake.broadcast(h.io);
    h.unlock();
}
