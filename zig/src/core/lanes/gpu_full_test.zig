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

/// The pre-selection full sort, kept as an independent regression oracle.
const Sorted = struct {
    const Allocator = std.mem.Allocator;
    const Plan = full.Plan;
    const Mark = full.Mark;
    const end = full.end;
    const never = full.never;
    const start = Mark{ .key = 0xFFFF_FFFF, .id = 0xFFFF_FFFF };

    fn atOrBefore(k: u32, i: u32, m: Mark) bool {
        return k > m.key or (k == m.key and i <= m.id);
    }

    fn after(k: u32, i: u32, m: Mark) bool {
        return k < m.key or (k == m.key and i > m.id);
    }

    fn markOf(c: rule.Cand) Mark {
        return .{ .key = rule.key(c.v), .id = c.i };
    }

    /// The kept set for `logits` under `s`, found as tf_sample_full finds it.
    pub fn plan(gpa: Allocator, logits: []const f32, s: Sampling) !Plan {
        const r = rule.rule(s);
        const inv_t = r.inv_t;
        const order = try gpa.alloc(rule.Cand, logits.len);
        defer gpa.free(order);
        var m = -std.math.inf(f32);
        for (order, logits, 0..) |*c, l, i| {
            c.* = .{ .v = l * inv_t, .i = @intCast(i) };
            m = @max(m, c.v);
        }
        std.mem.sort(rule.Cand, order, {}, rule.better);
        const min_log = r.min_log;
        var p = Plan{ .inv_t = inv_t, .m = m, .lo = @max(m + min_log, m - never) };
        if (s.top_k != 0 and s.top_k < logits.len) p.top = markOf(order[s.top_k - 1]);
        const top_p = r.top_p;
        if (top_p > 0.0 and top_p < 1.0) p.cut = nucleus(logits, p, order, top_p);
        return p;
    }

    /// The old nucleus sums its first 1024 sorted candidates, then extends through the full row.
    fn nucleus(logits: []const f32, p: Plan, order: []const rule.Cand, top_p: f32) Mark {
        const norm = mass(logits, p, 1.0, -std.math.inf(f32), start, p.top, end, false);
        const n = @min(order.len, rule.candidates);
        var cum: f32 = 0.0;
        for (order[0..n]) |c| {
            cum += @exp(c.v - p.m) / norm;
            if (cum >= top_p) return markOf(c);
        }
        return extend(logits, p, norm, top_p, cum, markOf(order[n - 1]));
    }

    /// The first place past `b` whose prefix reaches top_p, found by bits of key then id; `end` when none can win.
    fn extend(logits: []const f32, p: Plan, norm: f32, top_p: f32, cum: f32, b: Mark) Mark {
        if (!(cum + mass(logits, p, norm, p.lo, b, p.top, end, false) >= top_p)) return end;
        var key: u32 = 0;
        var bit: u6 = 32;
        while (bit > 0) {
            bit -= 1;
            const cand = key | @as(u32, 1) << @intCast(bit);
            if (cum + mass(logits, p, norm, p.lo, b, p.top, .{ .key = cand, .id = 0xFFFF_FFFF }, false) >= top_p) key = cand;
        }
        var id: u32 = 0;
        bit = @intCast(32 - @clz(@as(u32, @intCast(logits.len - 1))));
        while (bit > 0) {
            bit -= 1;
            const cand = id | @as(u32, 1) << @intCast(bit);
            if (cand >= logits.len) continue;
            if (cum + mass(logits, p, norm, p.lo, b, p.top, .{ .key = key, .id = cand }, true) < top_p) id = cand;
        }
        return .{ .key = key, .id = id };
    }

    /// The old mass over the kept set, in the kernel's order.
    fn mass(logits: []const f32, p: Plan, norm: f32, lo: f32, b: Mark, top: Mark, cut: Mark, strict: bool) f32 {
        var lanes: [1024]f32 = @splat(0.0);
        for (logits, 0..) |l, i| {
            const v = l * p.inv_t;
            if (!(v >= lo)) continue;
            const k = rule.key(v);
            const id: u32 = @intCast(i);
            const inside = if (strict) k > cut.key or (k == cut.key and id < cut.id) else atOrBefore(k, id, cut);
            if (after(k, id, b) and atOrBefore(k, id, top) and inside) lanes[i % 1024] += @exp(v - p.m) / norm;
        }
        return rule.lanesSum(&lanes);
    }

    fn sample(gpa: Allocator, logits: []const f32, s: ?Sampling, position: u32, ids: ?[]const u32) !u32 {
        const st = s orelse {
            const at = rule.argmax(logits);
            return if (ids) |m| m[at] else at;
        };
        const r = rule.rule(st);
        const inv_t = r.inv_t;
        const top_p = r.top_p;
        const min_log = r.min_log;
        const kc = r.top_k;
        const cap: usize = if (kc == 0 or kc > rule.candidates) rule.candidates else kc;
        const all = try gpa.alloc(rule.Cand, logits.len);
        defer gpa.free(all);
        var m = -std.math.inf(f32);
        for (all, logits, 0..) |*c, l, i| {
            c.* = .{ .v = l * inv_t, .i = @intCast(i) };
            m = @max(m, c.v);
        }
        const z = threadSum(all, m);
        std.mem.sort(rule.Cand, all, {}, rule.better);
        const n: usize = @min(@min(all.len, rule.candidates), cap);
        var norm = z;
        if (kc != 0) {
            norm = 0.0;
            for (all[0..n]) |c| norm += @exp(c.v - m);
        }
        var keep: usize = n;
        if (top_p > 0.0 and top_p < 1.0) {
            var cum: f32 = 0.0;
            for (all[0..n], 0..) |c, j| {
                cum += @exp(c.v - m) / norm;
                if (cum >= top_p) {
                    keep = j + 1;
                    break;
                }
            }
        }
        if (min_log > -std.math.inf(f32)) {
            const floor = m + min_log;
            var j: usize = 0;
            while (j < keep and all[j].v >= floor) j += 1;
            keep = j;
        }
        var best: usize = 0;
        var best_score = -std.math.inf(f32);
        for (all[0..keep], 0..) |c, t| {
            const id = if (ids) |map| map[c.i] else c.i;
            const score = c.v - @log(-@log(rule.uniform24(st.seed, position, id)));
            if (t == 0 or score > best_score) {
                best = t;
                best_score = score;
            }
        }
        const pick = all[best].i;
        return if (ids) |map| map[pick] else pick;
    }

    /// The row's normalizer in the kernel's order: 1024 strided thread sums, a 32-lane butterfly, simdgroups in order.
    fn threadSum(all: []const rule.Cand, m: f32) f32 {
        var lanes: [1024]f32 = @splat(0.0);
        for (all, 0..) |c, i| lanes[i % 1024] += @exp(c.v - m);
        return rule.lanesSum(&lanes);
    }
};

fn samePlan(want: full.Plan, got: full.Plan) !void {
    inline for (.{ "inv_t", "m", "lo" }) |field| {
        try expectEqual(@as(u32, @bitCast(@field(want, field))), @as(u32, @bitCast(@field(got, field))));
    }
    try std.testing.expectEqualDeep(want.top, got.top);
    try std.testing.expectEqualDeep(want.cut, got.cut);
}

/// Rows by mode: continuous, 7-level ties, -inf masks, signed zeros, NaNs and infinities among values, all masked.
fn selectionRow(logits: []f32, mode: usize) void {
    const special = [_]u32{ 0x7fc00001, 0xffc00001, 0x7f800000, 0xff800000, 0x00000000, 0x80000000 };
    spread(logits, if (mode == 1) 7 else 0);
    for (logits, 0..) |*value, i| value.* = switch (mode) {
        2 => if (i % 3 == 0) -std.math.inf(f32) else value.*,
        3 => if (i % 2 == 0) 0.0 else -0.0,
        4 => if (i % 2 == 0) @bitCast(special[(i / 2) % special.len]) else value.*,
        5 => -std.math.inf(f32),
        else => value.*,
    };
}

test "verify selection: full-sort picks and every plan field survive ties masks zeros and NaNs" {
    var storage: [4103]f32 = undefined;
    var mapped: [4103]u32 = undefined;
    for (&mapped, 0..) |*id, i| id.* = @intCast(50000 - i);
    for ([_]usize{ 33, 2048, 4103 }) |width| for (0..6) |mode| {
        const logits = storage[0..width];
        selectionRow(logits, mode);
        for ([_]u32{ 1, 20, 40, 1024, 0, 1025, @intCast(width) }) |top_k| {
            for ([_][2]f64{ .{ 1.0, 0.0 }, .{ 0.9, 0.0 }, .{ 0.5, 0.05 }, .{ 0.0, 0.01 } }) |cuts| {
                var s: Sampling = .{ .seed = 0, .temperature = 0.7, .top_k = top_k, .top_p = cuts[0], .min_p = cuts[1] };
                const want = try Sorted.plan(a, logits, s);
                const got = try full.plan(a, logits, s);
                try samePlan(want, got);
                for ([_]u64{ 0, 7, 0x123456789abcdef }) |seed| {
                    s.seed = seed;
                    for ([_]u32{ 0, 37 }) |position| {
                        for ([_]?[]const u32{ null, mapped[0..width] }) |ids| {
                            try expectEqual(try Sorted.sample(a, logits, s, position, ids), try rule.sample(a, logits, s, position, ids));
                            try expectEqual(full.race(logits, want, seed, position, ids), full.race(logits, got, seed, position, ids));
                        }
                    }
                }
            }
        }
    };
}

fn allocationCase(gpa: std.mem.Allocator, top_k: u32, width: usize) !void {
    var logits: [4096]f32 = undefined;
    selectionRow(logits[0..width], 1);
    const s: Sampling = .{ .seed = 7, .top_k = top_k, .top_p = 0.9 };
    _ = try rule.sample(gpa, logits[0..width], s, 37, null);
    _ = try full.plan(gpa, logits[0..width], s);
}

test "verify selection: FailingAllocator covers heap and full-sort allocation failures without leaks" {
    for ([_]u32{ 1, 20, 1024, 0, 1025, 4096 }) |k| {
        try std.testing.checkAllAllocationFailures(a, allocationCase, .{ k, @as(usize, 4096) });
    }
    try std.testing.checkAllAllocationFailures(a, allocationCase, .{ @as(u32, 20), @as(usize, 33) });
    var logits: [4096]f32 = undefined;
    spread(&logits, 0);
    var greedy = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    try expectEqual(rule.argmax(&logits), try rule.sample(greedy.allocator(), &logits, null, 0, null));
    try expectEqual(@as(usize, 0), greedy.allocations);
    var tracked = std.testing.FailingAllocator.init(a, .{});
    _ = try rule.sample(tracked.allocator(), &logits, .{ .seed = 0, .top_k = 20 }, 0, null);
    try expectEqual(@as(usize, 20 * (@sizeOf(usize) + @sizeOf(rule.Cand))), tracked.allocated_bytes);
    try expectEqual(tracked.allocated_bytes, tracked.freed_bytes);
}

test "verify selection: large vocabulary keeps the same 1024 keys with bounded storage" {
    const selection = @import("sampling_selection.zig");
    const logits = try a.alloc(f32, 200000);
    defer a.free(logits);
    selectionRow(logits, 1);
    logits[150001] = @bitCast(@as(u32, 0x7fc00001));
    const order = try a.alloc(rule.Cand, logits.len);
    defer a.free(order);
    for (order, logits, 0..) |*c, value, i| c.* = .{ .v = value, .i = @intCast(i) };
    std.mem.sort(rule.Cand, order, {}, rule.better);
    var tracked = std.testing.FailingAllocator.init(a, .{});
    const selected = try selection.topK32(tracked.allocator(), logits, 1.0, 1024, rule.better);
    try expectEqual(@as(usize, 1024), selected.len);
    for (selected, order[0..selected.len]) |got, want| {
        try expectEqual(@as(u32, @bitCast(want.v)), @as(u32, @bitCast(got.v)));
        try expectEqual(want.i, got.i);
    }
    tracked.allocator().free(selected);
    try expectEqual(@as(usize, 1024 * (@sizeOf(usize) + @sizeOf(rule.Cand))), tracked.allocated_bytes);
    try expectEqual(tracked.allocated_bytes, tracked.freed_bytes);
}
