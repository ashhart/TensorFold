//! Prompt-forget requests: callers wait for the engine thread that owns the cache.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Allocator = std.mem.Allocator;

/// One caller's span and, once the engine thread ran it, what it forgot.
pub const Ask = struct { span: []const u32, done: bool = false, closed: bool = false, out: pc.Forgotten = .{} };

/// Forgets asked for and not yet run; every call is made under the host's mutex.
pub const Queue = struct {
    asks: std.ArrayList(*Ask) = .empty,
    answered: std.Io.Condition = .init,

    /// Queues `span` for the engine thread, wakes it with `wake` and waits (unlocking `mutex`) for its answer.
    pub fn ask(q: *Queue, gpa: Allocator, io: std.Io, mutex: *std.Io.Mutex, wake: *std.Io.Condition, span: []const u32) error{ Closed, Busy }!pc.Forgotten {
        var a: Ask = .{ .span = span };
        q.asks.append(gpa, &a) catch return error.Busy;
        wake.signal(io);
        while (!a.done) q.answered.waitUncancelable(io, mutex);
        if (a.closed) return error.Closed;
        return a.out;
    }

    pub fn pending(q: *const Queue) bool {
        return q.asks.items.len > 0;
    }

    /// Take all waiting requests without allocating; the engine thread answers them and frees the list.
    pub fn take(q: *Queue) std.ArrayList(*Ask) {
        const out = q.asks;
        q.asks = .empty;
        return out;
    }

    /// The taken asks answered: their callers return.
    pub fn answer(q: *Queue, io: std.Io, asks: []const *Ask) void {
        for (asks) |a| a.done = true;
        q.answered.broadcast(io);
    }

    /// Every ask still waiting refused: the engine thread has stopped.
    pub fn close(q: *Queue, gpa: Allocator, io: std.Io) void {
        for (q.asks.items) |a| {
            a.closed = true;
            a.done = true;
        }
        q.asks.deinit(gpa);
        q.asks = .empty;
        q.answered.broadcast(io);
    }
};

/// Whether a request's prompt carries `span` (an empty span is in none).
pub fn carries(prompt: []const u32, span: []const u32) bool {
    return span.len > 0 and std.mem.indexOf(u32, prompt, span) != null;
}
