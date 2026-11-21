//! The shards a `kernel-contract --jobs` process starts and the process they must not outlive.
const std = @import("std");
const builtin = @import("builtin");

/// Most shards at once; `contract_shard.max_jobs` may not exceed it.
pub const capacity = 16;

var pids: [capacity]std.atomic.Value(i32) = @splat(.init(0));
var stopped: [capacity]std.atomic.Value(bool) = @splat(.init(false));
var stopping = std.atomic.Value(bool).init(false);

/// Records shard `slot`'s process so a stop reaches it.
pub fn register(slot: usize, pid: std.posix.pid_t) void {
    pids[slot].store(pid, .seq_cst);
}

pub fn forget(slot: usize) void {
    pids[slot].store(0, .seq_cst);
}

/// Whether a stop signalled shard `slot`, as against the shard ending by itself.
pub fn wasStopped(slot: usize) bool {
    return stopped[slot].load(.seq_cst);
}

pub fn isStopping() bool {
    return stopping.load(.seq_cst);
}

/// Signals every registered shard but `except`; true for the first stop of the run, which should say why.
pub fn stopAll(except: ?usize, sig: std.posix.SIG) bool {
    const first = !stopping.swap(true, .seq_cst);
    for (&pids, 0..) |*slot, i| {
        if (except == i) continue;
        const pid = slot.load(.seq_cst);
        if (pid <= 0) continue;
        stopped[i].store(true, .seq_cst);
        std.posix.kill(pid, sig) catch {};
    }
    return first;
}

fn onSignal(sig: std.posix.SIG) callconv(.c) void {
    _ = stopAll(null, sig);
}

/// SIGINT and SIGTERM sent to this process stop its shards too.
pub fn forwardSignals() void {
    const act: std.posix.Sigaction = .{ .handler = .{ .handler = onSignal }, .mask = std.posix.sigemptyset(), .flags = 0 };
    std.posix.sigaction(.INT, &act, null);
    std.posix.sigaction(.TERM, &act, null);
}

/// This process's id, for its shards to hold it to.
pub fn selfPid() std.posix.pid_t {
    return if (builtin.os.tag == .linux) std.os.linux.getpid() else 0;
}

pub const ParentError = error{ ParentGone, ParentWatchFailed, ParentWatchNeedsLinux };

/// A shard's side: die with SIGKILL when the process that started it does, and refuse if that is already not `parent`.
pub fn followParent(parent: std.posix.pid_t) ParentError!void {
    if (builtin.os.tag != .linux) return error.ParentWatchNeedsLinux;
    const linux = std.os.linux;
    const set = linux.prctl(@backingInt(linux.PR.SET_PDEATHSIG), @backingInt(linux.SIG.KILL), 0, 0, 0);
    if (linux.errno(set) != .SUCCESS) return error.ParentWatchFailed;
    if (linux.getppid() != parent) return error.ParentGone;
}
