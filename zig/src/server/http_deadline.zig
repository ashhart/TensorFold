//! A read budget belongs to a whole head or body, never to an individual socket fill.
const std = @import("std");
pub const Error = error{Timeout};
pub const Policy = struct {
    head_ms: u32 = 30_000,
    body_ms: u32 = 30_000,
    body_bytes_per_s: u32 = 65_536,
};

pub const Budget = struct {
    phase: enum { none, idle, head, body } = .none,
    start: i64 = 0,
    end: i64 = 0,
    bytes: u64 = 0,

    pub fn head(b: *Budget, now: i64, idle: bool, policy: Policy) void {
        b.* = .{ .phase = if (idle) .idle else .head, .start = now, .end = now +| policy.head_ms };
    }

    pub fn arrived(b: *Budget, now: i64, policy: Policy) void {
        if (b.phase == .idle) b.head(now, false, policy);
    }

    pub fn body(b: *Budget, now: i64, policy: Policy) void {
        b.* = .{ .phase = .body, .start = now, .end = now +| policy.body_ms };
    }

    /// Chunk declarations extend the original body's deadline, not the time of the last chunk.
    pub fn declared(b: *Budget, count: u64, policy: Policy) void {
        std.debug.assert(b.phase == .body);
        b.bytes +|= count;
        const rate = @max(1, policy.body_bytes_per_s);
        const ms = (b.bytes *| 1000) / rate + @intFromBool((b.bytes *| 1000) % rate != 0);
        b.end = b.start +| policy.body_ms +| @as(i64, @intCast(@min(ms, std.math.maxInt(i64))));
    }

    pub fn wait(b: Budget, now: i64, idle_ms: i32) Error!i32 {
        if (b.phase == .none or b.phase == .idle) return idle_ms;
        if (now >= b.end) return error.Timeout;
        return @intCast(@min(@as(i64, idle_ms), b.end - now));
    }
};
