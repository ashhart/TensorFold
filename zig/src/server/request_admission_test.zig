//! Admission invariants: limits, bounded identities, released reservations and concurrent callers.
const std = @import("std");
const admission = @import("request_admission.zig");
const testing = std.testing;
const io = testing.io;

fn principal(n: u8) admission.Principal {
    return @as([32]u8, @splat(n));
}

fn take(c: *admission.Controller, limits: admission.Limits, key: admission.Principal) !admission.Lease {
    return switch (c.acquire(io, limits, key)) {
        .accepted => |lease| lease,
        .rejected => error.UnexpectedRejection,
    };
}

test "request admission: only a cap or a rate turns admission on; a burst alone does not" {
    try testing.expect(!(admission.Limits{}).enabled());
    try testing.expect(!(admission.Limits{ .request_burst = 9 }).enabled());
    for ([_]admission.Limits{ .{ .max_inflight = 1 }, .{ .max_inflight_per_key = 1 }, .{ .requests_per_second = 1 } }) |limits|
        try testing.expect(limits.enabled());
}

test "request admission: disabled limits preserve acceptance and reservations release once" {
    var c: admission.Controller = .{};
    var leases: [160]admission.Lease = undefined;
    for (&leases, 0..) |*lease, n| lease.* = try take(&c, .{}, principal(@intCast(n)));
    try testing.expectEqual(@as(u32, leases.len), c.snapshot(io).inflight);
    try testing.expectEqual(@as(u32, 0), c.snapshot(io).principals);
    for (&leases) |*lease| {
        lease.release();
        lease.release();
    }
    try testing.expectEqual(@as(u32, 0), c.snapshot(io).inflight);
    try testing.expectEqual(@as(u64, leases.len), c.snapshot(io).admitted);
}

test "request admission: global and per-key limits are independent and refusals reserve nothing" {
    var c: admission.Controller = .{};
    const limits: admission.Limits = .{ .max_inflight = 2, .max_inflight_per_key = 1 };
    var first = try take(&c, limits, principal(1));
    defer first.release();
    try testing.expectEqual(admission.Reason.key_limit, c.acquire(io, limits, principal(1)).rejected.reason);
    var second = try take(&c, limits, principal(2));
    try testing.expectEqual(admission.Reason.global_limit, c.acquire(io, limits, principal(3)).rejected.reason);
    second.release();
    var third = try take(&c, limits, principal(3));
    defer third.release();
    const snap = c.snapshot(io);
    try testing.expectEqual(@as(u32, 2), snap.inflight);
    try testing.expectEqual(@as(u32, 2), snap.principals);
    try testing.expectEqual(@as(u64, 3), snap.admitted);
    try testing.expectEqual(@as(u64, 1), snap.rejected_global);
    try testing.expectEqual(@as(u64, 1), snap.rejected_key);
}

test "request admission: anonymous requests share a bucket while authenticated keys remain independent" {
    var c: admission.Controller = .{};
    const limits: admission.Limits = .{ .max_inflight_per_key = 1 };
    var anonymous = try take(&c, limits, null);
    defer anonymous.release();
    try testing.expectEqual(admission.Reason.key_limit, c.acquire(io, limits, null).rejected.reason);
    var authenticated = try take(&c, limits, principal(0));
    defer authenticated.release();
    try testing.expectEqual(@as(u32, 2), c.snapshot(io).principals);
}

test "request admission: key churn cannot grow storage and an exhausted table recovers on release" {
    var c: admission.Controller = .{};
    const limits: admission.Limits = .{ .max_inflight_per_key = 2 };
    var leases: [admission.principal_capacity]admission.Lease = undefined;
    for (&leases, 0..) |*lease, n| lease.* = try take(&c, limits, principal(@intCast(n)));
    defer for (&leases) |*lease| lease.release();
    try testing.expectEqual(admission.Reason.principal_capacity, c.acquire(io, limits, principal(200)).rejected.reason);
    var same_key = try take(&c, limits, principal(0));
    same_key.release();
    leases[0].release();
    var replacement = try take(&c, limits, principal(200));
    defer replacement.release();
    try testing.expectEqual(@as(u32, admission.principal_capacity), c.snapshot(io).principals);
    try testing.expectEqual(@as(u64, 1), c.snapshot(io).rejected_capacity);
}

const Race = struct {
    controller: admission.Controller = .{},
    go: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    attempts: std.atomic.Value(u32) = .init(0),
    accepted: std.atomic.Value(u32) = .init(0),

    fn run(r: *Race, key: admission.Principal, limits: admission.Limits) void {
        while (!r.go.load(.acquire)) std.Thread.yield() catch {};
        var result = r.controller.acquireAt(io, limits, key, 0);
        if (result == .accepted) _ = r.accepted.fetchAdd(1, .monotonic);
        _ = r.attempts.fetchAdd(1, .release);
        if (result == .accepted) {
            while (!r.release.load(.acquire)) std.Thread.yield() catch {};
            result.accepted.release();
        }
    }
};

test "request admission: concurrent reservations never oversubscribe either cap" {
    for ([_]admission.Limits{ .{ .max_inflight = 4 }, .{ .max_inflight_per_key = 4 }, .{ .requests_per_second = 1, .request_burst = 4 } }) |limits| {
        var race: Race = .{};
        var threads: [16]std.Thread = undefined;
        var started: usize = 0;
        defer {
            race.go.store(true, .release);
            race.release.store(true, .release);
            for (threads[0..started]) |thread| thread.join();
        }
        for (&threads) |*thread| {
            thread.* = try std.Thread.spawn(.{}, Race.run, .{ &race, principal(1), limits });
            started += 1;
        }
        race.go.store(true, .release);
        while (race.attempts.load(.acquire) < threads.len) std.Thread.yield() catch {};
        try testing.expectEqual(@as(u32, 4), race.accepted.load(.acquire));
        try testing.expectEqual(@as(u32, 4), race.controller.snapshot(io).inflight);
        race.release.store(true, .release);
        for (threads) |thread| thread.join();
        started = 0;
        const snap = race.controller.snapshot(io);
        try testing.expectEqual(@as(u32, 0), snap.inflight);
        try testing.expectEqual(@as(u32, 0), snap.principals);
        try testing.expectEqual(@as(u32, 4), snap.peak);
    }
}

fn spend(c: *admission.Controller, limits: admission.Limits, key: admission.Principal, now: i96) !void {
    var lease = switch (c.acquireAt(io, limits, key, now)) {
        .accepted => |lease| lease,
        .rejected => return error.UnexpectedRejection,
    };
    lease.release();
}

test "request admission: burst credit refills at exact boundaries without floating-point drift" {
    var c: admission.Controller = .{};
    const limits: admission.Limits = .{ .requests_per_second = 2, .request_burst = 3 };
    for (0..3) |_| try spend(&c, limits, principal(1), 0);
    const full = c.acquireAt(io, limits, principal(1), 0).rejected;
    try testing.expectEqual(admission.Reason.rate_limit, full.reason);
    try testing.expectEqual(@as(u32, 1), full.retry_after_s);
    try testing.expectEqual(admission.Reason.rate_limit, c.acquireAt(io, limits, principal(1), 250_000_000).rejected.reason);
    try testing.expectEqual(admission.Reason.rate_limit, c.acquireAt(io, limits, principal(1), 499_999_999).rejected.reason);
    try spend(&c, limits, principal(1), 500_000_000);
    try testing.expectEqual(admission.Reason.rate_limit, c.acquireAt(io, limits, principal(1), 500_000_000).rejected.reason);
    try spend(&c, limits, principal(2), 500_000_000);
    const snap = c.snapshot(io);
    try testing.expectEqual(@as(u32, 0), snap.inflight);
    try testing.expectEqual(@as(u32, 2), snap.rate_buckets);
    try testing.expectEqual(@as(u64, 4), snap.rejected_rate);
}

test "request admission: clocks cannot refund spent credit and long idle refills only the burst" {
    var c: admission.Controller = .{};
    const limits: admission.Limits = .{ .requests_per_second = 1, .request_burst = 1 };
    try spend(&c, limits, null, 1_000_000_000);
    try testing.expectEqual(admission.Reason.rate_limit, c.acquireAt(io, limits, null, -1).rejected.reason);
    try testing.expectEqual(admission.Reason.rate_limit, c.acquireAt(io, limits, null, 1_999_999_999).rejected.reason);
    try spend(&c, limits, null, 2_000_000_000);
    const distant: i96 = std.math.maxInt(i96) / 2;
    try spend(&c, limits, null, distant);
    try testing.expectEqual(admission.Reason.rate_limit, c.acquireAt(io, limits, null, distant).rejected.reason);
}

test "request admission: depleted idle identities cannot be evicted to reset their rate allowance" {
    var c: admission.Controller = .{};
    const limits: admission.Limits = .{ .requests_per_second = 1, .request_burst = 1 };
    for (0..admission.principal_capacity) |n| try spend(&c, limits, principal(@intCast(n)), 0);
    try testing.expectEqual(@as(u32, 0), c.snapshot(io).inflight);
    try testing.expectEqual(admission.Reason.principal_capacity, c.acquireAt(io, limits, principal(200), 0).rejected.reason);
    try testing.expectEqual(admission.Reason.principal_capacity, c.acquireAt(io, limits, principal(200), 999_999_999).rejected.reason);
    try testing.expectEqual(admission.Reason.rate_limit, c.acquireAt(io, limits, principal(0), 999_999_999).rejected.reason);
    try spend(&c, limits, principal(200), 1_000_000_000);
    try testing.expectEqual(@as(u32, admission.principal_capacity), c.snapshot(io).rate_buckets);
    try testing.expectEqual(admission.Reason.rate_limit, c.acquireAt(io, limits, principal(200), 1_000_000_000).rejected.reason);
}

test "request admission: concurrency refusal consumes no rate credit and unspecified burst uses the rate" {
    var c: admission.Controller = .{};
    const limits: admission.Limits = .{ .max_inflight = 1, .requests_per_second = 2 };
    var held = c.acquireAt(io, limits, principal(1), 0).accepted;
    try testing.expectEqual(admission.Reason.global_limit, c.acquireAt(io, limits, principal(2), 0).rejected.reason);
    held.release();
    try spend(&c, limits, principal(2), 0);
    try spend(&c, limits, principal(2), 0);
    try testing.expectEqual(admission.Reason.rate_limit, c.acquireAt(io, limits, principal(2), 0).rejected.reason);
}
