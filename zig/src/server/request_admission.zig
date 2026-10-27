//! Opt-in admission for model-using HTTP requests: in-flight caps and per-key rates, held until the reply ends.
const std = @import("std");

pub const max_limit = 128;
pub const principal_capacity = 128;
pub const max_rate = 100_000;
const token = std.time.ns_per_s;
/// A verified key's digest; null is every request of a server without keys.
pub const Principal = ?[32]u8;

pub const Limits = struct {
    max_inflight: u32 = 0,
    max_inflight_per_key: u32 = 0,
    requests_per_second: u32 = 0,
    request_burst: u32 = 0,

    /// All zero, the default: no request is counted, reserved or refused, and /metrics gains nothing.
    pub fn enabled(l: Limits) bool {
        return l.max_inflight > 0 or l.max_inflight_per_key > 0 or l.requests_per_second > 0;
    }
};

pub const Reason = enum { global_limit, key_limit, principal_capacity, rate_limit };
pub const Refusal = struct { reason: Reason, retry_after_s: u32 = 1 };
pub const Snapshot = struct {
    inflight: u32,
    peak: u32,
    principals: u32,
    admitted: u64,
    rejected_global: u64,
    rejected_key: u64,
    rejected_capacity: u64,
    rejected_rate: u64 = 0,
    rate_buckets: u32 = 0,
};

const Slot = struct {
    principal: Principal = null,
    count: u32 = 0,
    occupied: bool = false,
    rate: u32 = 0,
    capacity: u64 = 0,
    credit: u64 = 0,
    updated_ns: i96 = 0,

    fn available(s: *const Slot, now_ns: i96) u64 {
        if (s.rate == 0 or now_ns <= s.updated_ns) return s.credit;
        const missing = s.capacity - s.credit;
        const fill_ns = std.math.divCeil(u64, missing, s.rate) catch unreachable;
        const elapsed: u64 = @intCast(@min(now_ns -| s.updated_ns, fill_ns));
        return @min(s.capacity, s.credit + elapsed * s.rate);
    }

    fn replenish(s: *Slot, now_ns: i96) void {
        s.credit = s.available(now_ns);
        s.updated_ns = @max(s.updated_ns, now_ns);
    }
};

pub const Lease = struct {
    owner: *Controller,
    io: std.Io,
    slot: ?usize,
    released: bool = false,

    pub fn release(lease: *Lease) void {
        if (lease.released) return;
        lease.owner.mutex.lockUncancelable(lease.io);
        defer lease.owner.mutex.unlock(lease.io);
        std.debug.assert(lease.owner.inflight > 0);
        lease.owner.inflight -= 1;
        if (lease.slot) |i| {
            const slot = &lease.owner.slots[i];
            std.debug.assert(slot.count > 0);
            slot.count -= 1;
            if (slot.count == 0 and slot.rate == 0) slot.* = .{};
        }
        lease.released = true;
    }
};

pub const Result = union(enum) { accepted: Lease, rejected: Refusal };

pub const Controller = struct {
    mutex: std.Io.Mutex = .init,
    slots: [principal_capacity]Slot = @splat(.{}),
    inflight: u32 = 0,
    peak: u32 = 0,
    admitted: u64 = 0,
    rejected_global: u64 = 0,
    rejected_key: u64 = 0,
    rejected_capacity: u64 = 0,
    rejected_rate: u64 = 0,

    pub fn acquire(c: *Controller, io: std.Io, limits: Limits, principal: Principal) Result {
        return c.acquireAt(io, limits, principal, @import("clock.zig").nowNs(io));
    }

    /// Credit is integer nanosecond-tokens on the monotonic clock; an explicit time keeps the boundary tests exact.
    pub fn acquireAt(c: *Controller, io: std.Io, limits: Limits, principal: Principal, now_ns: i96) Result {
        c.mutex.lockUncancelable(io);
        defer c.mutex.unlock(io);
        if (limits.max_inflight > 0 and c.inflight >= limits.max_inflight) {
            c.rejected_global +|= 1;
            return .{ .rejected = .{ .reason = .global_limit } };
        }
        var index: ?usize = null;
        if (limits.max_inflight_per_key > 0 or limits.requests_per_second > 0) {
            var empty: ?usize = null;
            for (&c.slots, 0..) |*slot, i| {
                if (!slot.occupied) {
                    if (empty == null) empty = i;
                } else if (same(slot.principal, principal)) {
                    index = i;
                    break;
                } else if (empty == null and slot.count == 0 and slot.available(now_ns) == slot.capacity) {
                    empty = i; // idle with full credit: reusing it refunds nothing
                }
            }
            if (index) |i| {
                if (limits.max_inflight_per_key > 0 and c.slots[i].count >= limits.max_inflight_per_key) {
                    c.rejected_key +|= 1;
                    return .{ .rejected = .{ .reason = .key_limit } };
                }
            } else {
                index = empty;
                if (index == null) {
                    c.rejected_capacity +|= 1;
                    return .{ .rejected = .{ .reason = .principal_capacity } };
                }
                const burst = if (limits.request_burst == 0) limits.requests_per_second else limits.request_burst;
                const capacity = @as(u64, burst) * token;
                c.slots[index.?] = .{
                    .principal = principal,
                    .occupied = true,
                    .rate = limits.requests_per_second,
                    .capacity = capacity,
                    .credit = capacity,
                    .updated_ns = now_ns,
                };
            }
            const slot = &c.slots[index.?];
            if (slot.rate > 0) {
                slot.replenish(now_ns);
                if (slot.credit < token) {
                    c.rejected_rate +|= 1;
                    const wait_ns = std.math.divCeil(u64, token - slot.credit, slot.rate) catch unreachable;
                    const wait_s = std.math.divCeil(u64, wait_ns, token) catch unreachable;
                    return .{ .rejected = .{ .reason = .rate_limit, .retry_after_s = @intCast(@max(1, wait_s)) } };
                }
                slot.credit -= token;
            }
            slot.count += 1;
        }
        c.inflight += 1;
        c.peak = @max(c.peak, c.inflight);
        c.admitted +|= 1;
        return .{ .accepted = .{ .owner = c, .io = io, .slot = index } };
    }

    pub fn snapshot(c: *Controller, io: std.Io) Snapshot {
        c.mutex.lockUncancelable(io);
        defer c.mutex.unlock(io);
        var principals: u32 = 0;
        var rate_buckets: u32 = 0;
        for (c.slots) |slot| {
            if (slot.count > 0) principals += 1;
            if (slot.occupied and slot.rate > 0) rate_buckets += 1;
        }
        return .{
            .inflight = c.inflight,
            .peak = c.peak,
            .principals = principals,
            .admitted = c.admitted,
            .rejected_global = c.rejected_global,
            .rejected_key = c.rejected_key,
            .rejected_capacity = c.rejected_capacity,
            .rejected_rate = c.rejected_rate,
            .rate_buckets = rate_buckets,
        };
    }
};

fn same(a: Principal, b: Principal) bool {
    if (a == null or b == null) return a == null and b == null;
    return std.crypto.timing_safe.eql([32]u8, a.?, b.?);
}

pub fn code(reason: Reason) []const u8 {
    return switch (reason) {
        .global_limit => "max_inflight_exceeded",
        .key_limit => "max_inflight_per_key_exceeded",
        .principal_capacity => "admission_capacity_exceeded",
        .rate_limit => "requests_per_second_exceeded",
    };
}

pub fn message(reason: Reason) []const u8 {
    return switch (reason) {
        .global_limit => "too many model requests in progress; retry after a request finishes",
        .key_limit => "too many model requests in progress for this API key; retry after a request finishes",
        .principal_capacity => "request admission capacity reached; retry after the indicated interval",
        .rate_limit => "model request rate exceeded for this API key; retry after the indicated interval",
    };
}
