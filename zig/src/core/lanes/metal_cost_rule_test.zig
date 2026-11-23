//! Fake GPU time exposes width-local slow stretches without changing the target arithmetic.
const std = @import("std");
const rule = @import("cost_rule.zig");
const Timer = struct {
    calls: usize = 0,
    total: f64 = 0,
    slow: bool = false,
    pub fn time(t: *@This(), i: usize) !f64 {
        const noisy = t.slow and t.calls >= 30 * 8 and t.calls < 31 * 8;
        const value = truth(i) * (if (noisy) @as(f64, 1.08) else 1);
        t.calls += 1;
        t.total += value;
        return value;
    }
};
fn truth(i: usize) f64 {
    const width = i + 1;
    return 10 + 0.5 * @as(f64, @floatFromInt(width)) + @as(f64, if (width > 16) 2 else 0) +
        @as(f64, if (width > 32) 3 else 0) + @as(f64, if (width > 48) 4 else 0);
}

test "Metal window prices: interleaved minima survive one whole-width slow stretch" {
    var timer: Timer = .{ .slow = true };
    var ms: [64]f64 = undefined;
    try rule.measure(&timer, &timer, &ms, null, false);
    try rule.smooth(&timer, &ms);
    for (ms, 0..) |m, i| try std.testing.expectApproxEqRel(truth(i), m, 0.01);
    var baseline: f64 = 0;
    for (0..64) |i| baseline += 8 * truth(i);
    try std.testing.expect(timer.total <= baseline * 1.2);
    std.debug.print("Metal timing fake: baseline_ms={d:.3} measured_ms={d:.3} calls={d} increase_pct={d:.3}\n", .{ baseline, timer.total, timer.calls, 100 * (timer.total / baseline - 1) });
}

test "Metal window prices: a high outlier below the dip threshold is retimed and real tile steps survive" {
    var timer: Timer = .{};
    var ms: [64]f64 = undefined;
    for (&ms, 0..) |*m, i| m.* = truth(i);
    ms[8] *= 1.04;
    try rule.smooth(&timer, &ms);
    for (ms, 0..) |m, i| try std.testing.expectApproxEqAbs(truth(i), m, 1e-9);
    for ([_]usize{ 16, 32, 48 }) |i| try std.testing.expect(ms[i] > ms[i - 1] + 1);
    try std.testing.expect(timer.calls <= ms.len);
}

test "Metal window prices: a bad saved table heals across bounded warm loads without raising costs" {
    const cache = @import("cost_cache.zig");
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const key = cache.buildKey("same-binary", &.{"same-shape"});
    const cursor_key = cache.buildKey(&key, &.{"metal-heal-v1"});
    var original: [64]f64 = undefined;
    var widths: [64]u32 = undefined;
    for (&original, &widths, 0..) |*m, *w, i| {
        m.* = truth(i) * 1.08;
        w.* = @intCast(i + 1);
    }
    cache.saveAt([64]f64, a, io, key, original, dir);
    var maximum_ms: f64 = 0;
    var maximum_reads: usize = 0;
    for (0..64) |_| {
        var kept = cache.loadAt([64]f64, a, io, key, dir).?;
        var cursor = cache.loadAt(u64, a, io, cursor_key, dir) orelse 0;
        var timer: Timer = .{};
        // Today's cache-hit branch returns the saved table; the shared healing policy is added with the fix.
        if (comptime @hasDecl(rule, "healWindows")) _ = try rule.healWindows(&timer, &kept, &widths, &cursor);
        for (kept, original) |m, before| try std.testing.expect(m <= before);
        try std.testing.expect(timer.total <= 200);
        try std.testing.expect(timer.calls <= 7);
        maximum_ms = @max(maximum_ms, timer.total);
        maximum_reads = @max(maximum_reads, timer.calls);
        cache.saveAt([64]f64, a, io, key, kept, dir);
        cache.saveAt(u64, a, io, cursor_key, cursor, dir);
    }
    const healed = cache.loadAt([64]f64, a, io, key, dir).?;
    for (healed, 0..) |m, i| try std.testing.expectApproxEqRel(truth(i), m, 0.01);
    try std.testing.expectEqual(@as(u64, 64), cache.loadAt(u64, a, io, cursor_key, dir).?);
    std.debug.print("Metal healing fake: loads=64 max_forward_ms={d:.3} max_reads={d} remaining_error_pct=0\n", .{ maximum_ms, maximum_reads });
}

test "Metal window prices: paired warm probes follow identical calls despite different clocks and saved state" {
    const a = std.testing.allocator;
    const Wire = struct {
        bits: std.ArrayList(bool) = .empty,
        at: usize = 0,
        lead: bool = true,
        pub fn decide(w: *@This(), mine: bool) !bool {
            if (w.lead) {
                try w.bits.append(a, mine);
                return mine;
            }
            const bit = w.bits.items[w.at];
            w.at += 1;
            return bit;
        }
    };
    const Reads = struct {
        calls: std.ArrayList(usize) = .empty,
        scale: f64,
        pub fn time(r: *@This(), i: usize) !f64 {
            try r.calls.append(a, i);
            return truth(i) * r.scale;
        }
    };
    var wire: Wire = .{};
    defer wire.bits.deinit(a);
    var lead: Reads = .{ .scale = 1 };
    defer lead.calls.deinit(a);
    var peer: Reads = .{ .scale = 2 };
    defer peer.calls.deinit(a);
    var ms: [64]f64 = undefined;
    var unused: [64]f64 = @splat(0);
    var widths: [64]u32 = undefined;
    for (&ms, &widths, 0..) |*m, *w, i| {
        m.* = truth(i) * 1.08;
        w.* = @intCast(i + 1);
    }
    var cursor: u64 = 31;
    var peer_cursor: u64 = 0;
    _ = try rule.warm_prices.synced(&lead, &ms, &widths, &cursor, &wire, true);
    wire.lead = false;
    _ = try rule.warm_prices.synced(&peer, &unused, &widths, &peer_cursor, &wire, false);
    try std.testing.expectEqualSlices(usize, lead.calls.items, peer.calls.items);
    try std.testing.expect(lead.calls.items.len > 1);
    try std.testing.expectEqual(wire.bits.items.len, wire.at);
    try std.testing.expectEqual(@as(u64, 32), cursor);
    try std.testing.expectEqual(@as(u64, 0), peer_cursor);
}

test "Metal window prices: stateful Qwen cold timing restores the serial saved-table oracle" {
    const Stateful = struct {
        calls: usize = 0,
        pub fn time(t: *@This(), i: usize) !f64 {
            const value = truth(i) + @as(f64, @floatFromInt(t.calls));
            t.calls += 1;
            return value;
        }
    };
    var timer: Stateful = .{};
    var ms: [16]f64 = undefined;
    try rule.serialWindows(&timer, &ms);
    var expected: [16]f64 = undefined;
    for (&expected, 0..) |*m, i| m.* = truth(i) + @as(f64, @floatFromInt(8 * i + 1));
    try std.testing.expectEqualSlices(f64, &expected, &ms);
    try std.testing.expectEqual(@as(usize, 16 * 8), timer.calls);
    const old_reference: [16]f64 = @splat(1);
    try std.testing.expect(rule.drifted(&ms, &old_reference));
    try rule.again(&timer, &ms);
    try std.testing.expectEqualSlices(f64, &expected, &ms);
    try std.testing.expectEqual(@as(usize, 2 * 16 * 8), timer.calls);
}
