//! Production budget and startup helpers run without Metal or OS-dependent memory readings.
const std = @import("std");
const cache = @import("cache_budget.zig");
const batch = @import("flashnext_batch_budget.zig");
const meta = @import("../families/flashnext/batch_meta.zig");
const GiB = cache.GiB;
const policy: cache.Policy = .{ .margin = 2 * GiB, .ceiling = 16 * GiB };
fn plan(cap: u64, ready: u64, gib: ?f64, over: bool) !batch.Reservation {
    const f = try cache.fitUnder(cap, ready, gib, over, policy);
    return batch.reserve(cap, ready, policy.margin, f.budget, gib != null, over);
}

test "T49 unset cache reserves the single-host default before automatic sessions" {
    const r = try plan(80 * GiB, 40 * GiB, null, false);
    try std.testing.expectEqual(@as(u64, 16 * GiB), r.cache);
    try std.testing.expectEqual(r.cache, r.reserved);
    try std.testing.expectEqual(@as(u64, 22 * GiB), r.sessions);
    try std.testing.expectEqual(@as(u32, 6), try meta.fit(16, false, r.sessions - 2 * GiB, 4 * GiB));
    const constrained = try plan(60 * GiB, 53 * GiB, null, false);
    try std.testing.expectEqual(@as(u64, 5 * GiB), constrained.cache);
    try std.testing.expectEqual(@as(u64, 0), constrained.sessions);
    try std.testing.expectEqual(@as(u32, 1), try meta.fit(16, false, constrained.sessions, GiB));
    const override_without_value = try plan(80 * GiB, 40 * GiB, null, true);
    try std.testing.expectEqual(@as(u64, 16 * GiB), override_without_value.reserved);
}

test "T49 explicit cache values include zero and retain override semantics" {
    const r = try plan(80 * GiB, 40 * GiB, 8, false);
    try std.testing.expectEqual(@as(u64, 8 * GiB), r.cache);
    try std.testing.expectEqual(@as(u64, 8 * GiB), r.reserved);
    try std.testing.expectEqual(@as(u64, 30 * GiB), r.sessions);
    const off = try plan(80 * GiB, 40 * GiB, 0, false);
    try std.testing.expectEqual(@as(u64, 0), off.cache);
    try std.testing.expectEqual(@as(u64, 38 * GiB), off.sessions);
    const large = try plan(80 * GiB, 40 * GiB, 24, false);
    try std.testing.expectEqual(@as(u64, 24 * GiB), large.cache);
    try std.testing.expectEqual(@as(u64, 14 * GiB), large.sessions);
    try std.testing.expectError(error.CacheOverCap, plan(80 * GiB, 40 * GiB, 48, false));
    const over = try plan(80 * GiB, 40 * GiB, 48, true);
    try std.testing.expectEqual(@as(u64, 48 * GiB), over.cache);
    try std.testing.expectEqual(@as(u64, 0), over.reserved);
    try std.testing.expectEqual(@as(u64, 38 * GiB), over.sessions);
}

test "T49 default is bounded by available room and cannot wrap at the ceiling" {
    for ([_]u64{ 0, 1, 2 * GiB, 4 * GiB, 16 * GiB, 17 * GiB, std.math.maxInt(u64) }) |room| {
        const r = try plan(std.math.maxInt(u64), std.math.maxInt(u64) - room, null, false);
        const available = room -| policy.margin;
        try std.testing.expectEqual(@min(available, policy.ceiling), r.cache);
        try std.testing.expectEqual(available - r.cache, r.sessions);
    }
    const past = try plan(4 * GiB, 8 * GiB, null, false);
    try std.testing.expectEqual(@as(u64, 0), past.cache);
    try std.testing.expectEqual(@as(u64, 0), past.sessions);
}

test "T49 startup names session and prompt-cache budgets on the served line" {
    var bytes: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&bytes);
    try batch.startup(&w, 6, "262112,262112", try plan(80 * GiB, 40 * GiB, null, false), false, true);
    const line = w.buffered();
    try std.testing.expect(std.mem.indexOf(u8, line, "session budget 22.0 GiB") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "prompt cache 16.0 GiB") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "6 session slots") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "262112,262112 tokens") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "a lone request") != null);
    w = .fixed(&bytes);
    try batch.startup(&w, 2, "16384", try plan(80 * GiB, 40 * GiB, 8, false), true, false);
    try std.testing.expect(std.mem.indexOf(u8, w.buffered(), "prompt cache 8.0 GiB") != null);
    try std.testing.expect(std.mem.indexOf(u8, w.buffered(), "serial MTP") != null);
}

test "T49 measured sessions cannot shrink the chosen cache or spend its reservation" {
    const r = try plan(80 * GiB, 40 * GiB, null, false);
    const fitted = try batch.afterSessions(r, 80 * GiB, 62 * GiB, policy.margin);
    try std.testing.expectEqual(@as(u64, 16 * GiB), fitted.cache);
    try std.testing.expectEqual(@as(u64, 16 * GiB), fitted.room);
    try std.testing.expect(!fitted.over_cap);
    try std.testing.expectError(error.OverMemoryLimit, batch.afterSessions(r, 80 * GiB, 62 * GiB + 1, policy.margin));
    try std.testing.expectError(error.OverMemoryLimit, batch.afterSessions(r, 80 * GiB, 80 * GiB, policy.margin));
    const over = try batch.afterSessions(try plan(80 * GiB, 40 * GiB, 48, true), 80 * GiB, 70 * GiB, policy.margin);
    try std.testing.expectEqual(@as(u64, 48 * GiB), over.cache);
    try std.testing.expectEqual(@as(u64, 8 * GiB), over.room);
    try std.testing.expect(over.over_cap);
    try std.testing.expectError(error.OverMemoryLimit, batch.afterSessions(r, std.math.maxInt(u64), std.math.maxInt(u64), policy.margin));
}
