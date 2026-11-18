//! tf_sample_full against tf_gpu_sample's rule: the tail it reaches, its nucleus and top_k past 1,024, where both agree.
const std = @import("std");
const full = @import("gpu_full.zig");
const rule = @import("gpu_rule.zig");
const Sampling = @import("sampling.zig").Sampling;

const a = std.testing.allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

/// Draws at or past `from` in `n` positions, by the engine's rule or by tf_gpu_sample's alone.
fn tailDraws(logits: []const f32, s: Sampling, from: u32, n: u32, engine: bool) !u32 {
    var count: u32 = 0;
    for (0..n) |pos| {
        const p: u32 = @intCast(pos);
        const id = if (engine) try full.draw(a, logits, s, p, null) else try rule.sample(a, logits, s, p, null);
        count += @intFromBool(id >= from);
    }
    return count;
}

/// A token's rank in the (value desc, id asc) order of `logits` / T.
fn rank(logits: []const f32, inv_t: f32, id: u32) usize {
    const v = logits[id] * inv_t;
    var n: usize = 0;
    for (logits, 0..) |l, i| n += @intFromBool(l * inv_t > v or (l * inv_t == v and i < id));
    return n;
}

/// The nucleus's length in float64: the shortest prefix of the order whose probability reaches top_p.
fn nucleus64(logits: []const f32, inv_t: f32, top_p: f64) !usize {
    const order = try a.alloc(rule.Cand, logits.len);
    defer a.free(order);
    var m = -std.math.inf(f32);
    for (order, logits, 0..) |*c, l, i| {
        c.* = .{ .v = l * inv_t, .i = @intCast(i) };
        m = @max(m, c.v);
    }
    std.mem.sort(rule.Cand, order, {}, rule.better);
    var z: f64 = 0.0;
    for (order) |c| z += @exp(@as(f64, c.v - m));
    var cum: f64 = 0.0;
    for (order, 1..) |c, n| {
        cum += @exp(@as(f64, c.v - m)) / z;
        if (cum >= top_p) return n;
    }
    return order.len;
}

/// A row of `n` logits in [-3, 0] from a fixed generator, with ties every `levels` steps when `levels` is not 0.
fn spread(logits: []f32, levels: u32) void {
    var x: u64 = 0x9E37_79B9_7F4A_7C15;
    for (logits) |*l| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        const u: f32 = @floatFromInt(x >> 40);
        const r = u / 16777216.0;
        l.* = if (levels == 0) -3.0 * r else -3.0 * @floor(r * @as(f32, @floatFromInt(levels))) / @as(f32, @floatFromInt(levels));
    }
}

test "the Metal engine draws top_k off and above 1,024 with tf_sample_full, the rest as before" {
    try expect(full.fullVocabulary(.{ .seed = 1, .top_k = 0 }));
    try expect(full.fullVocabulary(.{ .seed = 1, .top_k = 1025 }));
    try expect(!full.fullVocabulary(.{ .seed = 1, .top_k = 1 }));
    try expect(!full.fullVocabulary(.{ .seed = 1, .top_k = 1024 }));
    var logits: [2048]f32 = undefined;
    spread(&logits, 0);
    const s: Sampling = .{ .seed = 2, .temperature = 0.8, .top_k = 40, .top_p = 0.9 };
    for (0..50) |pos| try expectEqual(try rule.sample(a, &logits, s, @intCast(pos), null), try full.draw(a, &logits, s, @intCast(pos), null));
    try expectEqual(rule.argmax(&logits), try full.draw(a, &logits, null, 0, null));
}

test "with top_k off and top_p 1.0 the whole vocabulary is drawn, not tf_gpu_sample's 1,024" {
    const logits: [2048]f32 = @splat(0.0);
    const s: Sampling = .{ .seed = 3, .temperature = 1.0, .top_k = 0, .top_p = 1.0 };
    try expectEqual(@as(u32, 0), try tailDraws(&logits, s, 1024, 400, false));
    const tail = try tailDraws(&logits, s, 1024, 400, true);
    try expect(tail >= 150 and tail <= 250);
}

test "a nucleus past 1,024 tokens ends where its mass reaches top_p" {
    const logits: [4096]f32 = @splat(0.0);
    const s: Sampling = .{ .seed = 5, .temperature = 1.0, .top_k = 0, .top_p = 0.5 };
    const p = try full.plan(a, &logits, s);
    try expectEqual(@as(u32, 2047), p.cut.id);
    var past: u32 = 0;
    for (0..400) |pos| {
        const id = full.race(&logits, p, s.seed, @intCast(pos), null);
        try expect(id <= 2047);
        past += @intFromBool(id >= 1024);
    }
    try expect(past >= 150 and past <= 250);
    try expectEqual(@as(u32, 0), try tailDraws(&logits, s, 1024, 100, false));
    var row: [4096]f32 = undefined;
    for ([_]u32{ 0, 7 }) |levels| {
        spread(&row, levels);
        const q = try full.plan(a, &row, .{ .seed = 5, .temperature = 1.0, .top_k = 0, .top_p = 0.9 });
        const want = try nucleus64(&row, 1.0, 0.9);
        const got = rank(&row, 1.0, q.cut.id) + 1;
        try expect(want > 1024 and got + 2 >= want and got <= want + 2);
    }
}

test "top_k above 1,024 keeps its top_k tokens and cuts its nucleus over them" {
    const logits: [4096]f32 = @splat(0.0);
    var s: Sampling = .{ .seed = 9, .temperature = 1.0, .top_k = 3000, .top_p = 1.0 };
    const p = try full.plan(a, &logits, s);
    try expectEqual(@as(u32, 2999), p.top.id);
    var past: u32 = 0;
    for (0..300) |pos| {
        const id = full.race(&logits, p, s.seed, @intCast(pos), null);
        try expect(id < 3000);
        past += @intFromBool(id >= 1024);
    }
    try expect(past >= 150);
    s.top_p = 0.5;
    const q = try full.plan(a, &logits, s);
    try expect(q.cut.id >= 1498 and q.cut.id <= 1500);
}

test "min_p keeps every token within ln(min_p) of the max, past 1,024 too" {
    var logits: [4096]f32 = @splat(-10.0);
    @memset(logits[0..2000], 0.0);
    const s: Sampling = .{ .seed = 11, .temperature = 1.0, .top_k = 0, .top_p = 1.0, .min_p = 0.01 };
    var past: u32 = 0;
    for (0..300) |pos| {
        const id = try full.draw(a, &logits, s, @intCast(pos), null);
        try expect(id < 2000);
        past += @intFromBool(id >= 1024);
    }
    try expect(past >= 100);
}

test "where tf_gpu_sample's 1,024 hold the kept tokens both rules draw the same token" {
    var logits: [4096]f32 = undefined;
    for (&logits, 0..) |*l, i| l.* = -0.05 * @as(f32, @floatFromInt(i));
    for ([_]Sampling{
        .{ .seed = 13, .temperature = 0.7, .top_k = 0, .top_p = 0.9 },
        .{ .seed = 14, .temperature = 1.0, .top_k = 0, .top_p = 1.0, .min_p = 0.05 },
    }) |s| for (0..200) |pos| {
        const p: u32 = @intCast(pos);
        try expectEqual(try rule.sample(a, &logits, s, p, null), try full.sample(a, &logits, s, p, null));
    };
    for (&logits, 0..) |*l, i| l.* = -0.001 * @as(f32, @floatFromInt(i));
    const s: Sampling = .{ .seed = 15, .temperature = 1.0, .top_k = 0, .top_p = 0.95 };
    const plan = try full.plan(a, &logits, s);
    try expect(plan.cut.id > 1024);
    for (0..200) |pos| {
        const id = full.race(&logits, plan, s.seed, @intCast(pos), null);
        if (id < 1024) try expectEqual(id, try rule.sample(a, &logits, s, @intCast(pos), null));
    }
}

test "a uniform that rounds to 1.0 does not win the race from the tail" {
    try expectEqual(@as(f32, 1.0), rule.uniform24(7, 66903620, 1500));
    try expect(full.raceUniform(7, 66903620, 1500) < 1.0);
    var logits: [2048]f32 = @splat(-40.0);
    @memset(logits[0..1024], 0.0);
    logits[1500] = -20.0;
    const p = try full.plan(a, &logits, .{ .seed = 7, .temperature = 1.0, .top_k = 0, .top_p = 1.0 });
    try expect(full.race(&logits, p, 7, 66903620, null) < 1024);
}
