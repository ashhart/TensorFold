//! Fail fast on every rank: a failure or a vanished rank ends all ranks with exit 70 about a second later.
const std = @import("std");
const log = @import("log.zig");
const sock = @import("sock.zig");
const Config = @import("config.zig").Config;
const posix = std.posix;
/// Linux POLLRDHUP: the peer shut its side down (std has no name for it).
const pollrdhup: i16 = 0x2000;

pub const exit_code: u8 = 70;
const magic: u32 = 0x46463431;
const kind_mem: u32 = 1;
const kind_fail: u32 = 2;
/// rank 0's clean stop: the hang-up that follows is no failure (sent before rank 0 closes its sockets)
const kind_stop: u32 = 3;
const Frame = extern struct { magic: u32, kind: u32, value: i64 };

pub const max_ranks = 64;
pub const max_hooks = 8;

/// A callback with its context (an abort of a collective, rank 0's request failing); `reason` lives until exit.
pub const Hook = struct { ctx: *anyopaque, call: *const fn (ctx: *anyopaque, reason: []const u8) void };

pub const Fate = struct {
    rank: u32,
    world: u32,
    fds: [max_ranks]sock.Fd,
    grace_ns: u64,
    report_ns: u64,
    /// This rank's usable memory in bytes for the MEM report (default: MemAvailable).
    usable: *const fn () u64 = memAvailable,
    /// The process exit; tests replace it (it may return there).
    exit_fn: *const fn (u8) void = exitNow,
    hooks: [max_hooks]Hook = undefined,
    n_hooks: std.atomic.Value(u32) = .init(0),
    dead: std.atomic.Value(bool) = .init(false),
    closing: std.atomic.Value(bool) = .init(false),
    /// Set once the exit was asked for (tests read it; the process is gone in serving).
    exited: std.atomic.Value(bool) = .init(false),
    reason_buf: [512]u8 = undefined,
    reason_len: usize = 0,
    send_lock: std.atomic.Value(bool) = .init(false),
    /// Rank 0: each rank's last report (bytes) and its monotonic time.
    peer_mem: [max_ranks]std.atomic.Value(u64) = @splat(.init(0)),
    peer_mem_at: [max_ranks]std.atomic.Value(u64) = @splat(.init(0)),
    wake: [2]sock.Fd = .{ -1, -1 },
    thread: ?std.Thread = null,

    /// The channel over the bootstrap's sockets (`Bootstrap.release`); `start` runs its thread.
    pub fn init(cfg: Config, fds: [max_ranks]sock.Fd) Fate {
        return .{ .rank = cfg.rank, .world = cfg.world, .fds = fds, .grace_ns = cfg.grace_ns, .report_ns = cfg.report_ns };
    }

    pub fn start(f: *Fate) !void {
        var p: [2]c_int = undefined;
        if (std.c.pipe2(&p, .{ .CLOEXEC = true }) != 0) return error.SocketFailed;
        f.wake = p;
        f.thread = if (f.rank == 0) try std.Thread.spawn(.{}, watch0, .{f}) else try std.Thread.spawn(.{}, watchK, .{f});
    }

    /// Runs `hook` when the ranks go down; collectives register their abort here.
    pub fn onFail(f: *Fate, hook: Hook) void {
        const i = f.n_hooks.load(.acquire);
        std.debug.assert(i < max_hooks);
        f.hooks[i] = hook;
        f.n_hooks.store(i + 1, .release);
    }

    /// This rank failed (`where`: what it was doing): every rank goes down. Returns only in tests.
    pub fn fatal(f: *Fate, where: []const u8, err: anyerror) void {
        var buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, "rank {d} failed in {s}: {t}", .{ f.rank, where, err }) catch "rank failed";
        f.die(msg, true);
    }

    /// This rank failed while booting: every rank goes down, and the thread is joined before the caller frees this.
    pub fn failBoot(f: *Fate, where: []const u8, err: anyerror) void {
        f.fatal(where, err);
        f.close();
    }

    /// A normal stop is coming (rank 0 before it tells the others; the others when told): later hang-ups are fine.
    pub fn expectStop(f: *Fate) void {
        f.closing.store(true, .release);
    }

    /// Rank 0's clean stop of every rank: a STOP frame to each ahead of its close, so followers end quietly.
    pub fn announceStop(f: *Fate) void {
        if (f.closing.swap(true, .acq_rel)) return;
        if (f.rank != 0) return;
        const h: Frame = .{ .magic = magic, .kind = kind_stop, .value = 0 };
        f.lock();
        defer f.unlock();
        for (f.fds[1..f.world]) |fd| if (fd >= 0) {
            _ = sock.trySend(fd, std.mem.asBytes(&h));
        };
    }

    /// A normal stop: the thread ends and the sockets close.
    pub fn close(f: *Fate) void {
        f.closing.store(true, .release);
        if (f.wake[1] >= 0) _ = std.c.write(f.wake[1], "x", 1);
        if (f.thread) |t| t.join();
        f.thread = null;
        for (&f.fds) |*fd| if (fd.* >= 0) {
            sock.close(fd.*);
            fd.* = -1;
        };
        for (f.wake) |fd| if (fd >= 0) sock.close(fd);
        f.wake = .{ -1, -1 };
    }

    /// Rank 0: rank `r`'s last reported usable bytes and the age of the report, or null before its first.
    pub fn peerMem(f: *const Fate, r: u32) ?struct { bytes: u64, age_ns: u64 } {
        const at = f.peer_mem_at[r].load(.acquire);
        if (at == 0) return null;
        return .{ .bytes = f.peer_mem[r].load(.acquire), .age_ns = sock.nowNs() -| at };
    }

    pub fn reason(f: *const Fate) []const u8 {
        return f.reason_buf[0..f.reason_len];
    }

    fn lock(f: *Fate) void {
        while (f.send_lock.swap(true, .acquire)) std.atomic.spinLoopHint();
    }

    fn unlock(f: *Fate) void {
        f.send_lock.store(false, .release);
    }

    fn die(f: *Fate, why: []const u8, own: bool) void {
        if (f.dead.swap(true, .acq_rel)) {
            // the first caller exits the process; a second (the fate thread or a round thread) waits for it
            while (!f.exited.load(.acquire)) sock.sleepNs(10 * std.time.ns_per_ms);
            return;
        }
        const n = @min(why.len, f.reason_buf.len);
        @memcpy(f.reason_buf[0..n], why[0..n]);
        f.reason_len = n;
        const r = f.reason();
        log.warn("[tensorfold] tp fail-fast: rank {d}: {s}; exiting ({d}) in {d} ms so every rank restarts", .{ f.rank, r, exit_code, f.grace_ns / std.time.ns_per_ms });
        // aborts first: kernels waiting on the dead peer end now; a hook that blocks cannot hold the exit
        if (std.Thread.spawn(.{}, runHooks, .{ f, r })) |t| t.detach() else |_| runHooks(f, r);
        if (!f.closing.load(.acquire)) {
            const head: Frame = .{ .magic = magic, .kind = kind_fail, .value = @intCast(r.len) };
            f.lock();
            for (f.fds[0..f.world]) |fd| if (fd >= 0) {
                if (own or f.rank == 0) {
                    _ = sock.trySend(fd, std.mem.asBytes(&head));
                    _ = sock.trySend(fd, r);
                }
                sock.hangup(fd);
            };
            f.unlock();
        }
        sock.sleepNs(f.grace_ns);
        f.exited.store(true, .release);
        f.exit_fn(exit_code);
    }

    fn runHooks(f: *Fate, r: []const u8) void {
        const n = f.n_hooks.load(.acquire);
        for (f.hooks[0..n]) |h| h.call(h.ctx, r);
    }

    /// Reads one frame from `fd`: a MEM report, a FAIL (the ranks go down), or the end (the peer is gone).
    fn readFrame(f: *Fate, peer: u32, fd: sock.Fd) bool {
        var h: Frame = undefined;
        const deadline = sock.nowNs() + std.time.ns_per_s;
        sock.recvExact(fd, std.mem.asBytes(&h), deadline) catch {
            if (f.closing.load(.acquire)) return false;
            var buf: [128]u8 = undefined;
            f.die(std.fmt.bufPrint(&buf, "rank {d} closed the fate channel (it failed, exited or was killed)", .{peer}) catch "peer gone", false);
            return false;
        };
        if (h.magic != magic) {
            f.die("a frame that is not the fate channel's", false);
            return false;
        }
        if (h.kind == kind_stop) {
            // rank 0 stops every rank: its hang-up is next, and no failure
            f.closing.store(true, .release);
            return false;
        }
        if (h.kind == kind_mem) {
            f.peer_mem[peer].store(@intCast(@max(h.value, 0)), .release);
            f.peer_mem_at[peer].store(sock.nowNs(), .release);
            return true;
        }
        var text: [512]u8 = undefined;
        const len: usize = @intCast(std.math.clamp(h.value, 0, text.len));
        sock.recvExact(fd, text[0..len], deadline) catch {};
        f.die(if (len > 0) text[0..len] else "a peer failed", false);
        return false;
    }

    fn watch0(f: *Fate) void {
        var p: [max_ranks + 1]std.c.pollfd = undefined;
        var who: [max_ranks + 1]u32 = undefined;
        var n: usize = 0;
        for (f.fds[1..f.world], 1..) |fd, r| if (fd >= 0) {
            p[n] = .{ .fd = fd, .events = posix.POLL.IN | pollrdhup, .revents = 0 };
            who[n] = @intCast(r);
            n += 1;
        };
        p[n] = .{ .fd = f.wake[0], .events = posix.POLL.IN, .revents = 0 };
        while (!f.dead.load(.acquire) and !f.closing.load(.acquire)) {
            if (std.c.poll(&p, @intCast(n + 1), -1) <= 0) continue;
            for (p[0..n], who[0..n]) |*q, r| if (q.revents != 0) {
                if (!f.readFrame(r, q.fd)) return;
            };
        }
    }

    fn watchK(f: *Fate) void {
        const fd = f.fds[0];
        var p = [_]std.c.pollfd{
            .{ .fd = fd, .events = posix.POLL.IN | pollrdhup, .revents = 0 },
            .{ .fd = f.wake[0], .events = posix.POLL.IN, .revents = 0 },
        };
        const ms: c_int = @intCast(@max(f.report_ns / std.time.ns_per_ms, 1));
        f.report(fd);
        while (!f.dead.load(.acquire) and !f.closing.load(.acquire)) {
            const ready = std.c.poll(&p, 2, ms);
            if (ready == 0) {
                f.report(fd);
                continue;
            }
            if (ready > 0 and p[0].revents != 0 and !f.readFrame(0, fd)) return;
        }
    }

    /// One MEM frame, never blocking (a full buffer drops it: the next comes one period later).
    fn report(f: *Fate, fd: sock.Fd) void {
        const h: Frame = .{ .magic = magic, .kind = kind_mem, .value = @intCast(f.usable()) };
        f.lock();
        defer f.unlock();
        _ = sock.trySend(fd, std.mem.asBytes(&h));
    }
};

fn exitNow(code: u8) void {
    std.c._exit(code);
}

/// MemAvailable from /proc/meminfo in bytes (0 when unreadable).
pub fn memAvailable() u64 {
    const fd = std.c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY });
    if (fd < 0) return 0;
    defer _ = std.c.close(fd);
    var buf: [4096]u8 = undefined;
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return 0;
    const text = buf[0..@intCast(n)];
    const key = "MemAvailable:";
    const at = std.mem.indexOf(u8, text, key) orelse return 0;
    var it = std.mem.tokenizeAny(u8, text[at + key.len ..], " \n");
    const kb = std.fmt.parseInt(u64, it.next() orelse return 0, 10) catch return 0;
    return kb * 1024;
}
