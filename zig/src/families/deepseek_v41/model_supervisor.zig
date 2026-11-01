//! An owned host thread turns a published context fault into fail-fast without waiting for an HTTP health poll.
const std = @import("std");

pub const Supervisor = struct {
    io: std.Io,
    ctx: *anyopaque,
    healthy: *const fn (*anyopaque) bool,
    fail: *const fn (*anyopaque, anyerror) void,
    progress: ?*@import("round_progress.zig").Progress = null,
    stall_ns: u64 = 0,
    stopping: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    notified: std.atomic.Value(bool) = .init(false),
    pending: std.atomic.Value(u16) = .init(0),
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

    /// Fatal backend errors withdraw readiness immediately; the owned thread performs the potentially blocking exit.
    pub fn report(s: *Supervisor, err: anyerror) void {
        _ = s.pending.cmpxchgStrong(0, @intFromError(err), .acq_rel, .acquire);
        s.failed.store(true, .release);
    }

    /// Publish failure before notification; readiness refuses even while the exit callback is blocked.
    pub fn check(s: *Supervisor) bool {
        const code = s.pending.load(.acquire);
        const err: anyerror = if (code != 0) @errorFromInt(code) else if (!s.healthy(s.ctx)) error.CudaContextLost else stalled: {
            const progress = s.progress orelse return false;
            const phase = progress.stalled(s.stall_ns) orelse return false;
            if (!@import("builtin").is_test) std.log.warn("DeepSeek: no progress in {t}; withdrawing readiness and exiting", .{phase});
            break :stalled error.StalledRound;
        };
        s.failed.store(true, .release);
        if (!s.notified.swap(true, .acq_rel)) s.fail(s.ctx, err);
        return true;
    }

    fn run(s: *Supervisor) void {
        while (!s.stopping.load(.acquire)) {
            if (s.check()) return;
            std.Io.sleep(s.io, .fromMilliseconds(100), .awake) catch {};
        }
    }
};
