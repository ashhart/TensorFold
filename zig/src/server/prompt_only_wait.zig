//! A prompt-only request (an embedding or a readout) submitted and waited on until the engine's terminal event.
const std = @import("std");
const api = @import("engine_api");

/// The terminal events' sink: the finish reason, and whether any token came (a prompt-only reply has none).
pub const Completion = struct {
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    reason: ?api.Reason = null,
    unexpected_tokens: bool = false,

    pub fn event(ptr: *anyopaque, _: api.Id, value: *const api.Event) void {
        const c: *Completion = @ptrCast(@alignCast(ptr));
        c.mutex.lockUncancelable(c.io);
        defer c.mutex.unlock(c.io);
        switch (value.*) {
            .tokens => |tokens| c.unexpected_tokens = c.unexpected_tokens or tokens.len != 0,
            .finished => |f| c.reason = f.reason,
            else => {},
        }
    }
};

/// How the request ended: its answer is in its output (`length` and no token), it failed, or it was cancelled.
pub const Ended = enum { done, failed, cancelled };

/// Submits `ask` and polls for its finish; a gone client cancels it once, and the wait outlasts the engine.
pub fn run(engine: api.Engine, io: std.Io, id: api.Id, ask: *const api.Request, gone: anytype) api.SubmitError!Ended {
    var done: Completion = .{ .io = io };
    try engine.submit(id, ask, .{ .ctx = &done, .event = Completion.event });
    var cancelling = false;
    while (true) {
        done.mutex.lockUncancelable(io);
        const reason = done.reason;
        const bad = done.unexpected_tokens;
        done.mutex.unlock(io);
        if (reason) |why| {
            if (cancelling or why == .cancelled) return .cancelled;
            return if (why != .length or bad) .failed else .done;
        }
        if (!cancelling and gone.check()) {
            engine.cancel(id);
            cancelling = true;
        }
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
}
