//! An owned host thread turns a published context fault into fail-fast without waiting for an HTTP health poll.
const std = @import("std");
const sys = @import("sys");
const posix = sys.posix;

/// Set by `installStopSignals`' handler (tf-dsv41 follow / generate); every supervisor reads it, nothing waits on it.
pub var stop_signal: std.atomic.Value(bool) = .init(false);

fn onStop(sig: posix.SIG) callconv(.c) void {
    if (!stop_signal.swap(true, .acq_rel)) return;
    const default: posix.Sigaction = .{ .handler = .{ .handler = posix.SIG.DFL }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(sig, &default, null); // a second stop signal: the default action, as with no handler
    _ = sys.c.raise(sig);
}

/// SIGTERM and SIGINT after the load: the supervisor takes fail-fast (collectives aborted, every rank down).
pub fn installStopSignals() void {
    const act: posix.Sigaction = .{ .handler = .{ .handler = onStop }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(.TERM, &act, null);
    posix.sigaction(.INT, &act, null);
}

pub const Supervisor = struct {
    io: std.Io,
    ctx: *anyopaque,
    healthy: *const fn (*anyopaque) bool,
    fail: *const fn (*anyopaque, anyerror) void,
    progress: ?*@import("round_progress.zig").Progress = null,
    stall_ns: u64 = 0,
    /// The stop flag the poll reads (tests pass their own).
    signal: *const std.atomic.Value(bool) = &stop_signal,
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

    /// The failure to report now: a transport report, a stop signal, a lost context, or stalled owned work.
    fn failure(s: *Supervisor) ?anyerror {
        const code = s.pending.load(.acquire);
        if (code != 0) return @errorFromInt(code);
        if (s.signal.load(.acquire)) return error.StopRequested;
        if (!s.healthy(s.ctx)) return error.CudaContextLost;
        const progress = s.progress orelse return null;
        const phase = progress.stalled(s.stall_ns) orelse return null;
        if (!@import("builtin").is_test) std.log.warn("DeepSeek: no progress in {t}; withdrawing readiness and exiting", .{phase});
        return error.StalledRound;
    }

    /// Publish failure before notification; readiness refuses even while the exit callback is blocked.
    pub fn check(s: *Supervisor) bool {
        const err = s.failure() orelse return false;
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
