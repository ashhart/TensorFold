//! Keyed draw equivalence and verification-row timing without duplicate draws or heap fallback.
const std = @import("std");
const t = std.testing;
const lanes = @import("lanes");
const be = lanes.backend;
const c = @import("config.zig");
const timing = @import("draw_timing.zig");
const backend = @import("backend.zig");
const mask_words = (c.vocab + 31) / 32;

const Clock = struct {
    ns: i128 = 900 * std.time.ns_per_s,
    step: i128 = 4 * std.time.ns_per_ms,
    reads: usize = 0,

    pub fn now(clock: *@This()) i128 {
        const ns = clock.ns;
        clock.ns += clock.step;
        clock.reads += 1;
        return ns;
    }
};

fn window(s: *lanes.Stream, rows: usize) be.Window {
    return .{ .stream = s, .pending = 1, .held = 0, .tokens = (&[_]u32{1})[0 .. rows - 1], .parents = null, .positions = (&[_]u64{ 13, 99 })[0..rows] };
}

pub fn check(comptime drawRow: anytype) !void {
    const gpa = t.allocator;
    const words = try gpa.alloc(u16, 2 * c.vocab);
    defer gpa.free(words);
    @memset(words, 0);
    for (0..2) |r| {
        words[r * c.vocab] = 0x4000; // 2, then 1: the two bounded candidates, with all others 0.
        words[r * c.vocab + 1] = 0x3f80;
    }
    var sampled: [2]u32 = undefined;
    var best = [_]u32{ 0, 0, 0, 0 };
    const settings: lanes.Sampling = .{ .seed = 1234, .temperature = 0.7, .top_k = 2, .top_p = 0.95 };
    var s = try lanes.Stream.init(gpa, .{ .id = "sampled", .prompt = &.{1}, .max_new = 16, .sampling = settings });
    defer s.deinit(gpa);
    var greedy = try lanes.Stream.init(gpa, .{ .id = "greedy", .prompt = &.{1}, .max_new = 16 });
    defer greedy.deinit(gpa);
    const w = window(&s, 2);
    var g = window(&greedy, 1);
    var clock: Clock = .{};

    // Unequal sampled/greedy windows in both orders: two clock reads per host span, none per GPU pick.
    for ([_]bool{ false, true }) |reversed| {
        const pair = if (reversed) [_]be.Window{ g, w } else [_]be.Window{ w, g };
        timing.reset(&pair);
        for (pair) |one| {
            clock.ns += 20 * std.time.ns_per_s; // prior forward/other-stream work is outside this window.
            try timing.drawWindow(drawRow, t.failing_allocator, one, sampled[0..one.rows()], words, &best, &clock);
            if (one.stream == &s) {
                for (sampled, w.positions) |got, p| {
                    const want = try lanes.sampling.choose(gpa, &.{ 2, 1 }, &.{ 0, 1 }, p, settings);
                    try t.expectEqual(@as(u32, @intCast(want)), got);
                }
            } else try t.expectEqual(@as(u32, 0), sampled[0]);
        }
        try t.expectEqual(@as(f64, 4), s.verify_host_ms);
        try t.expectEqual(@as(?f64, 2), s.verify_row_ms);
        try t.expectEqual(@as(f64, 0), greedy.verify_host_ms);
        try t.expectEqual(@as(?f64, 0), greedy.verify_row_ms);
    }
    try t.expectEqual(@as(usize, 4), clock.reads);

    // A later-row failure keeps the keyed draw already produced, but cannot publish partial work.
    timing.reset(&.{w});
    sampled[1] = 777;
    best[3] = 1;
    try t.expectError(error.NonfiniteLfmLogits, timing.drawWindow(drawRow, t.failing_allocator, w, &sampled, words, &best, &clock));
    try t.expectEqual(@as(u32, @intCast(try lanes.sampling.choose(gpa, &.{ 2, 1 }, &.{ 0, 1 }, w.positions[0], settings))), sampled[0]);
    try t.expectEqual(@as(u32, 777), sampled[1]);
    try t.expectEqual(@as(f64, 0), s.verify_host_ms);
    try t.expectEqual(@as(?f64, 2), s.verify_row_ms);
    try t.expectEqual(@as(usize, 5), clock.reads);
    best[3] = 0;
    clock.step = 6 * std.time.ns_per_ms;
    try timing.drawWindow(drawRow, t.failing_allocator, w, &sampled, words, &best, &clock);
    try t.expectEqual(@as(f64, 6), s.verify_host_ms);
    try t.expectEqual(@as(?f64, 3), s.verify_row_ms); // last success, not smoothed or divided by accepted rows.

    // Activate an inactive constraint: its initial follow does not call the rules.
    var constraint = lanes.grammar.Constraint.init(undefined, mask_words, 7);
    greedy.grammar = &constraint;
    timing.reset(&.{g});
    const before = clock.reads;
    try timing.drawWindow(drawRow, t.failing_allocator, g, sampled[0..1], words, &best, &clock);
    try t.expectEqual(before, clock.reads);
    try t.expectEqual(@as(?f64, null), greedy.verify_row_ms);
    try t.expect(try constraint.follow(7));
    try t.expect(constraint.active);
    var masks: [2 * mask_words]u32 = @splat(0);
    masks[0] = 2;
    masks[mask_words] = 2;
    g.masks = masks[0..mask_words];
    timing.reset(&.{g});
    try timing.drawWindow(drawRow, t.failing_allocator, g, sampled[0..1], words, &best, &clock);
    try t.expectEqual(@as(u32, 1), sampled[0]);
    try t.expectEqual(@as(f64, 6), greedy.verify_host_ms);
    try t.expectEqual(@as(?f64, null), greedy.verify_row_ms);
    try t.expectEqual(before + 2, clock.reads);

    // Explicit masks also cannot lend their measured slope to the request's unmasked sampler.
    var masked = w;
    masked.masks = &masks;
    timing.reset(&.{masked});
    try timing.drawWindow(drawRow, t.failing_allocator, masked, &sampled, words, &best, &clock);
    try t.expectEqualSlices(u32, &.{ 1, 1 }, &sampled);
    try t.expectEqual(@as(f64, 6), s.verify_host_ms);
    try t.expectEqual(@as(?f64, null), s.verify_row_ms);
    timing.reset(&.{masked});
    masks[mask_words] = 0;
    try t.expectError(error.NoAllowedLfmToken, timing.drawWindow(drawRow, t.failing_allocator, masked, &sampled, words, &best, &clock));
    try t.expectEqual(@as(f64, 0), s.verify_host_ms);
    try t.expectEqual(@as(?f64, null), s.verify_row_ms);
    timing.reset(&.{w});
    const one = window(&s, 1);
    try timing.drawWindow(drawRow, t.failing_allocator, one, sampled[0..1], words, &best, &clock);
    try t.expectEqual(@as(?f64, 6), s.verify_row_ms);

    // Unknown host work that fails allocation stays unknown, preserving the original sampler error.
    var unbounded = try lanes.Stream.init(gpa, .{ .id = "unbounded", .prompt = &.{1}, .max_new = 16, .sampling = .{ .seed = 7, .temperature = 1, .top_k = 0 } });
    defer unbounded.deinit(gpa);
    const unknown = window(&unbounded, 1);
    timing.reset(&.{unknown});
    try t.expectError(error.OutOfMemory, timing.drawWindow(drawRow, t.failing_allocator, unknown, sampled[0..1], words, &best, &clock));
    try t.expectEqual(@as(f64, 0), unbounded.verify_host_ms);
    try t.expectEqual(@as(?f64, null), unbounded.verify_row_ms);

    // Explicit zero-temperature settings use GPU picks too, but a failed first window is not measured free.
    var cold = try lanes.Stream.init(gpa, .{ .id = "zero", .prompt = &.{1}, .max_new = 16, .sampling = .{ .seed = 7, .temperature = 0 } });
    defer cold.deinit(gpa);
    const zero = window(&cold, 1);
    timing.reset(&.{zero});
    const before_zero = clock.reads;
    best[1] = 1;
    try t.expectError(error.NonfiniteLfmLogits, timing.drawWindow(drawRow, t.failing_allocator, zero, sampled[0..1], words, &best, &clock));
    try t.expectEqual(@as(?f64, null), cold.verify_row_ms);
    best[1] = 0;
    try timing.drawWindow(drawRow, t.failing_allocator, zero, sampled[0..1], words, &best, &clock);
    try t.expectEqual(@as(u32, 0), sampled[0]);
    try t.expectEqual(@as(f64, 0), cold.verify_host_ms);
    try t.expectEqual(@as(?f64, 0), cold.verify_row_ms);
    try t.expectEqual(before_zero, clock.reads);

    // The actual backend clears every stale span before rejecting a malformed batch, without a forward.
    var target: backend.Metal = .{ .gpa = t.failing_allocator, .model = undefined, .options = .{ .capacity = 32 }, .scratch = undefined };
    try t.expectError(error.InvalidLfmWindows, target.backend().verify(&.{ w, g }, &.{}));
    try t.expectEqual(@as(f64, 0), s.verify_host_ms);
    try t.expectEqual(@as(f64, 0), greedy.verify_host_ms);
    try t.expectEqual(@as(?f64, 6), s.verify_row_ms);
    try t.expectEqual(@as(?f64, null), greedy.verify_row_ms);
}

pub fn checkCandidates(comptime drawRow: anytype, comptime value: anytype) !void {
    const gpa = std.testing.allocator;
    const words = try gpa.alloc(u16, c.vocab);
    defer gpa.free(words);
    const values = try gpa.alloc(f64, c.vocab);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, c.vocab);
    defer gpa.free(ids);
    for (words, values, ids, 0..) |*w, *v, *id, i| {
        const x: f32 = @as(f32, @floatFromInt((i * 37) % 257)) / 8.0 - 16.0;
        w.* = @intCast(@as(u32, @bitCast(x)) >> 16);
        v.* = value(w.*);
        id.* = i;
    }
    for ([_]u32{ 0, 1, 20, 128, 257, c.vocab }) |k| {
        for ([_]u64{ 0, 128, 65536 }) |position| {
            const s: lanes.Sampling = .{ .seed = 1234, .top_k = k, .temperature = 0.7, .top_p = 0.95, .min_p = 0.1 };
            const want = try lanes.sampling.choose(gpa, values, ids, position, s);
            try std.testing.expectEqual(want, try drawRow(gpa, words, s, position, &.{}, .{ 0, 0 }));
        }
    }
    try std.testing.expectEqual(try backend.draw(gpa, words, null, 0, &.{}), try backend.draw(gpa, words, .{ .seed = 7, .temperature = 0 }, 99, &.{}));
    words[0] = 0x7fc0;
    try std.testing.expectError(error.NonfiniteLfmLogits, backend.draw(gpa, words, null, 0, &.{}));
}

pub fn checkMasked(comptime value: anytype) !void {
    const gpa = std.testing.allocator;
    const words = try gpa.alloc(u16, c.vocab);
    defer gpa.free(words);
    const values = try gpa.alloc(f64, c.vocab);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, c.vocab);
    defer gpa.free(ids);
    var mask: [mask_words]u32 = @splat(0);
    for (words, values, ids, 0..) |*w, *v, *id, i| {
        const x: f32 = @as(f32, @floatFromInt((i * 37) % 257)) / 8.0 - 16.0;
        w.* = @intCast(@as(u32, @bitCast(x)) >> 16);
        const on = i % 7 == 3 or i % 1000 == 1;
        if (on) mask[i / 32] |= @as(u32, 1) << @intCast(i % 32);
        v.* = if (on) value(w.*) else -std.math.inf(f64);
        id.* = i;
    }
    for ([_]u32{ 0, 1, 20, 257, c.vocab }) |k| {
        for ([_]u64{ 0, 128, 65536 }) |position| {
            const s: lanes.Sampling = .{ .seed = 99, .top_k = k, .temperature = 0.8, .top_p = 0.9, .min_p = 0.05 };
            const want = try lanes.sampling.choose(gpa, values, ids, position, s);
            try std.testing.expectEqual(want, try backend.draw(gpa, words, s, position, &mask));
        }
    }
    try std.testing.expectEqual(try lanes.sampling.choose(gpa, values, ids, 0, .{ .seed = 1, .temperature = 0 }), try backend.draw(gpa, words, null, 0, &mask));
    @memset(&mask, 0);
    try std.testing.expectError(error.NoAllowedLfmToken, backend.draw(gpa, words, null, 0, &mask));
}
