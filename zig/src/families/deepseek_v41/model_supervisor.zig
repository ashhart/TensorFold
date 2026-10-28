//! An owned host thread turns a published context fault into fail-fast without waiting for an HTTP health poll.
const std = @import("std");

pub const Supervisor = struct {
    io: std.Io,
    ctx: *anyopaque,
    healthy: *const fn (*anyopaque) bool,
    fail: *const fn (*anyopaque, anyerror) void,
    stopping: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    pub fn start(s: *Supervisor) !void {
        std.debug.assert(s.thread == null);
        s.thread = try std.Thread.spawn(.{}, run, .{s});
    }

    /// The model joins this before its TP session, context or callback owner can be destroyed.
    pub fn stop(s: *Supervisor) void {
        s.stopping.store(true, .release);
        if (s.thread) |thread| thread.join();
        s.thread = null;
    }

    /// Readiness was withdrawn by the context's sticky-fault publisher before this notification.
    pub fn check(s: *Supervisor) bool {
        if (s.healthy(s.ctx)) return false;
        if (!s.failed.swap(true, .acq_rel)) s.fail(s.ctx, error.CudaContextLost);
        return true;
    }

    fn run(s: *Supervisor) void {
        while (!s.stopping.load(.acquire)) {
            if (s.check()) return;
            std.Io.sleep(s.io, .fromMilliseconds(100), .awake) catch {};
        }
    }
};
