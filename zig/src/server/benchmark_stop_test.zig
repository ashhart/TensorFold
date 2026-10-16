//! The stop flag through its real handler, in forked copies of the test process, with children the tests reap.
const std = @import("std");
const platform = @import("sys");
const builtin = @import("builtin");
const fake = @import("benchmark_fake.zig");
const posix = std.posix;
const stop = @import("benchmark_stop.zig");

test "the production modules import no test fake" {
    const modules = [_][]const u8{ @embedFile("benchmark_stop.zig"), @embedFile("benchmark_stream.zig"), @embedFile("benchmark_sse.zig"), @embedFile("benchmark_receipt.zig"), @embedFile("benchmark_check.zig"), @embedFile("benchmark_suite.zig") };
    for (modules) |source| try std.testing.expect(std.mem.indexOf(u8, source, "benchmark_fake") == null);
}

test "the first signal wins and maps to the shell's exit codes" {
    stop.reset();
    defer stop.reset();
    try std.testing.expect(stop.requested() == null);
    stop.request(.TERM);
    stop.request(.INT);
    try std.testing.expectEqual(posix.SIG.TERM, stop.requested().?);
    try std.testing.expectEqual(@as(u8, 143), stop.exitCode(.TERM));
    try std.testing.expectEqual(@as(u8, 130), stop.exitCode(.INT));
    try std.testing.expectEqual(@as(u8, 129), stop.exitCode(.HUP));
}

/// A ``sleep 30`` child the test reaps itself; ``end`` kills and reaps it unless it has already ended.
const Sleeper = struct {
    pid: posix.pid_t,
    status: ?u32 = null,

    fn start() !Sleeper {
        const c = try std.process.spawn(std.testing.io, .{ .argv = &.{ "/bin/sh", "-c", "exec sleep 30" }, .stdin = .ignore, .stdout = .ignore, .stderr = .ignore });
        return .{ .pid = c.id.? };
    }

    /// Its wait status once it ends within ``ms``, else null; a zombie counts as ended, a live child does not.
    fn ended(s: *Sleeper, ms: u32) ?u32 {
        var waited: u32 = 0;
        while (s.status == null and waited <= ms) : (waited += 10) {
            var status: c_int = 0;
            if (platform.c.waitpid(s.pid, &status, platform.c.W.NOHANG) == s.pid) s.status = @bitCast(status) else if (waited < ms) fake.nap(10);
        }
        return s.status;
    }

    fn end(s: *Sleeper) void {
        if (s.ended(0) != null) return;
        _ = platform.c.kill(s.pid, .KILL);
        var status: c_int = 0;
        _ = platform.c.waitpid(s.pid, &status, 0);
    }
};

fn killedBy(status: ?u32, sig: posix.SIG) bool {
    const s = status orelse return false;
    return platform.c.W.IFSIGNALED(s) and platform.c.W.TERMSIG(s) == sig;
}

test "a second Ctrl-C kills the guarded child, and a released child is left alone" {
    var kept = try Sleeper.start();
    defer kept.end();
    stop.guardChild(kept.pid);
    stop.releaseChild(kept.pid);
    stop.abandon();
    try std.testing.expect(kept.ended(100) == null);
    var doomed = try Sleeper.start();
    defer doomed.end();
    stop.guardChild(doomed.pid);
    stop.abandon();
    try std.testing.expect(killedBy(doomed.ended(2000), .KILL));
    try std.testing.expectEqual(@as(posix.pid_t, 0), stop.guardedChild());
}

/// Runs ``body`` in a forked copy of this process; its wait status, or null when it ran past 5 s and was killed.
fn forked(body: *const fn () u8) !?u32 {
    const pid = platform.c.fork();
    if (pid < 0) return error.SkipZigTest;
    if (pid == 0) platform.c._exit(body());
    var status: c_int = 0;
    var waited: u32 = 0;
    while (waited < 5000) : (waited += 10) {
        if (platform.c.waitpid(pid, &status, platform.c.W.NOHANG) == pid) return @bitCast(status);
        fake.nap(10);
    }
    _ = platform.c.kill(pid, .KILL);
    _ = platform.c.waitpid(pid, &status, 0);
    return null;
}

const InFork = struct {
    var target: posix.pid_t = 0;

    fn doubleCtrlC() u8 {
        stop.reset();
        stop.releaseTerminal();
        stop.install();
        stop.guardChild(target);
        _ = platform.c.raise(.INT);
        if (stop.requested() != .INT) return 2;
        _ = platform.c.raise(.INT);
        return 3;
    }

    fn firstStopWins() u8 {
        stop.reset();
        stop.install();
        _ = platform.c.raise(.HUP);
        _ = platform.c.raise(.TERM);
        if (stop.requested() != .HUP) return 4;
        stop.reset();
        _ = platform.c.raise(.TERM);
        _ = platform.c.raise(.HUP);
        if (stop.requested() != .TERM) return 5;
        stop.reset();
        _ = platform.c.raise(.INT);
        if (stop.requested() != .INT) return 6;
        stop.restore();
        return 0;
    }
};

test "through the real handler, a second Ctrl-C kills the guarded child and ends the process by SIGINT" {
    if (builtin.os.tag != .macos and builtin.os.tag != .linux) return error.SkipZigTest;
    var sleeper = try Sleeper.start();
    defer sleeper.end();
    InFork.target = sleeper.pid;
    try std.testing.expect(killedBy(try forked(InFork.doubleCtrlC), .INT));
    try std.testing.expect(killedBy(sleeper.ended(2000), .KILL));
}

test "through the real handler, HUP, TERM and a first INT are recorded and the first one wins" {
    if (builtin.os.tag != .macos and builtin.os.tag != .linux) return error.SkipZigTest;
    const status = (try forked(InFork.firstStopWins)) orelse return error.TestUnexpectedResult;
    try std.testing.expect(platform.c.W.IFEXITED(status));
    try std.testing.expectEqual(@as(u8, 0), platform.c.W.EXITSTATUS(status));
}

test "a second Ctrl-C at the hidden prompt turns echo back on" {
    if (builtin.os.tag != .macos and builtin.os.tag != .linux) return error.SkipZigTest;
    const pty = fake.Pty.open() orelse return error.SkipZigTest;
    defer pty.close();
    const term = pty.term;
    const before = try posix.tcgetattr(term);
    try std.testing.expect(before.lflag.ECHO);
    var quiet = before;
    quiet.lflag.ECHO = false;
    stop.guardTerminal(term, before);
    try posix.tcsetattr(term, .FLUSH, quiet);
    try std.testing.expect(!(try posix.tcgetattr(term)).lflag.ECHO);
    stop.abandon();
    try std.testing.expect((try posix.tcgetattr(term)).lflag.ECHO);
    try std.testing.expectEqual(@as(posix.fd_t, -1), stop.guardedTerminal());
}
