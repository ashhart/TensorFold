//! Owned work scopes and completed operations distinguish a blocked round from an idle or advancing engine.
const std = @import("std");

pub const Phase = enum { round, snapshot, kv_union, follower };
pub const Progress = struct {
    io: std.Io,
    mutex: std.atomic.Mutex = .unlocked,
    active: u32 = 0,
    last_ns: u64 = 0,
    phase: Phase = .round,

    pub const Scope = struct {
        owner: ?*Progress,
        previous: Phase = .round,
        pub fn end(s: *Scope) void {
            const p = s.owner orelse return;
            p.lock();
            defer p.mutex.unlock();
            std.debug.assert(p.active > 0);
            p.active -= 1;
            p.phase = s.previous;
            s.owner = null;
        }
    };

    pub fn begin(p: *Progress, phase: Phase) Scope {
        return p.beginAt(phase, p.now());
    }
    pub fn beginAt(p: *Progress, phase: Phase, now_ns: u64) Scope {
        p.lock();
        defer p.mutex.unlock();
        const previous = p.phase;
        if (p.active == 0) p.last_ns = now_ns;
        p.active += 1;
        p.phase = phase;
        return .{ .owner = p, .previous = previous };
    }
    pub fn touch(p: *Progress) void {
        p.touchAt(p.now());
    }
    pub fn touchAt(p: *Progress, now_ns: u64) void {
        p.lock();
        defer p.mutex.unlock();
        if (p.active > 0) p.last_ns = @max(p.last_ns, now_ns);
    }
    pub fn stalled(p: *Progress, limit_ns: u64) ?Phase {
        return p.stalledAt(limit_ns, p.now());
    }
    pub fn stalledAt(p: *Progress, limit_ns: u64, now_ns: u64) ?Phase {
        p.lock();
        defer p.mutex.unlock();
        if (limit_ns == 0 or p.active == 0 or now_ns -| p.last_ns < limit_ns) return null;
        return p.phase;
    }
    fn lock(p: *Progress) void {
        while (!p.mutex.tryLock()) std.atomic.spinLoopHint();
    }
    fn now(p: *const Progress) u64 {
        return @intCast(@max(0, std.Io.Clock.awake.now(p.io).toNanoseconds()));
    }
};

/// Seconds, default off. Overflow and malformed operator settings refuse before boot.
pub fn timeoutNs(value: ?[]const u8) !u64 {
    const text = value orelse return 0;
    const seconds = std.fmt.parseInt(u64, text, 10) catch return error.BadStallTimeout;
    return std.math.mul(u64, seconds, std.time.ns_per_s) catch error.BadStallTimeout;
}
