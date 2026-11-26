//! rule.topCands, rule.sample and full.plan against a reference that sorts the whole row, bit for bit.
const std = @import("std");
const full = @import("gpu_full.zig");
const rule = @import("gpu_rule.zig");
const Sampling = @import("sampling.zig").Sampling;

const a = std.testing.allocator;
const inf = std.math.inf(f32);
const nan = std.math.nan(f32);

/// The whole row scaled and sorted: the reference every selection must equal.
fn sortedRow(logits: []const f32, inv_t: f32, max: *f32) ![]rule.Cand {
    const order = try a.alloc(rule.Cand, logits.len);
    var m = -inf;
    for (order, logits, 0..) |*c, l, i| {
        c.* = .{ .v = l * inv_t, .i = @intCast(i) };
        m = @max(m, c.v);
    }
    std.mem.sort(rule.Cand, order, {}, rule.better);
    max.* = m;
    return order;
}

/// The row's normalizer as threadSum summed it: the scaled candidates in row order.
fn lanesNorm(logits: []const f32, inv_t: f32, m: f32) f32 {
    var lanes: [1024]f32 = @splat(0.0);
    for (logits, 0..) |l, i| {
        const c: rule.Cand = .{ .v = l * inv_t, .i = @intCast(i) };
        lanes[i % 1024] += @exp(c.v - m);
    }
    return rule.lanesSum(&lanes);
}

/// rule.sample over a full sort: the sum of every token's mass is taken in row order, the rest reads the sorted row.
fn fullSortSample(logits: []const f32, st: Sampling, position: u32) !u32 {
    const r = rule.rule(st);
    var m: f32 = undefined;
    const all = try sortedRow(logits, r.inv_t, &m);
    defer a.free(all);
    const cap: usize = if (r.top_k == 0 or r.top_k > rule.candidates) rule.candidates else r.top_k;
    const n: usize = @min(@min(all.len, rule.candidates), cap);
    var norm = lanesNorm(logits, r.inv_t, m);
    if (r.top_k != 0) {
        norm = 0.0;
        for (all[0..n]) |c| norm += @exp(c.v - m);
    }
    var keep: usize = n;
    if (r.top_p > 0.0 and r.top_p < 1.0) {
        var cum: f32 = 0.0;
        for (all[0..n], 0..) |c, j| {
            cum += @exp(c.v - m) / norm;
            if (cum >= r.top_p) {
                keep = j + 1;
                break;
            }
        }
    }
    if (r.min_log > -inf) {
        var j: usize = 0;
        while (j < keep and all[j].v >= m + r.min_log) j += 1;
        keep = j;
    }
    var best: usize = 0;
    var best_score = -inf;
    for (all[0..keep], 0..) |c, t| {
        const score = c.v - @log(-@log(rule.uniform24(st.seed, position, c.i)));
        if (t == 0 or score > best_score) {
            best = t;
            best_score = score;
        }
    }
    return all[best].i;
}

/// full.plan over a full sort.
fn fullSortPlan(logits: []const f32, s: Sampling) !full.Plan {
    const r = rule.rule(s);
    var m: f32 = undefined;
    const order = try sortedRow(logits, r.inv_t, &m);
    defer a.free(order);
    var p = full.Plan{ .inv_t = r.inv_t, .m = m, .lo = @max(m + r.min_log, m - full.never) };
    if (s.top_k != 0 and s.top_k < logits.len) p.top = .{ .key = rule.key(order[s.top_k - 1].v), .id = order[s.top_k - 1].i };
    if (r.top_p > 0.0 and r.top_p < 1.0) p.cut = full.nucleus(logits, p, order, r.top_p);
    return p;
}

const kinds = 12;

/// A row of 12 kinds: Gaussian, tied, equal, signed zeros, -inf, +inf, NaN, extremes, ramps, all -inf, plateaus.
fn fill(row: []f32, kind: usize, prng: *std.Random.DefaultPrng) void {
    const rnd = prng.random();
    for (row, 0..) |*x, i| {
        const g: f32 = @floatCast(rnd.floatNorm(f64) * 3.0);
        const n: f32 = @floatFromInt(row.len);
        const at: f32 = @floatFromInt(i);
        x.* = switch (kind) {
            0 => g,
            1 => @round(g) * 0.5,
            2 => 1.25,
            3 => if (i % 2 == 0) 0.0 else -0.0,
            4 => if (rnd.uintLessThan(u8, 3) == 0) -inf else g,
            5 => if (rnd.uintLessThan(u8, 40) == 0) inf else g,
            6 => if (rnd.uintLessThan(u8, 30) == 0) (if (rnd.boolean()) nan else -nan) else g,
            7 => switch (rnd.uintLessThan(u8, 5)) {
                0 => 1e30,
                1 => -1e30,
                2 => 1e-30,
                3 => 1.0e-42,
                else => g,
            },
            8 => at / n * 20.0,
            9 => (n - at) / n * 20.0,
            10 => -inf,
            else => -@floor(at / 7.0) * 0.001,
        };
    }
}

const temperatures = [_]f64{ 0.7, 1.0, 1e-9, 1e3 };
const settings = [_]struct { k: u32, p: f64, min: f64 }{
    .{ .k = 20, .p = 0.95, .min = 0 },
    .{ .k = 0, .p = 1.0, .min = 0 },
    .{ .k = 0, .p = 0.9, .min = 0 },
    .{ .k = 1, .p = 1.0, .min = 0 },
    .{ .k = 1024, .p = 0.9, .min = 0.05 },
    .{ .k = 1500, .p = 0.95, .min = 0 },
    .{ .k = 3000, .p = 0.5, .min = 0 },
    .{ .k = 50, .p = 1.0, .min = 0.02 },
    .{ .k = 7, .p = 0.0, .min = 0.9 },
};

fn bitsEqual(x: anytype, y: @TypeOf(x)) bool {
    return std.mem.eql(u8, std.mem.asBytes(&x), std.mem.asBytes(&y));
}

test "top leaves the full sort's prefix and maximum, bit for bit, on every kind of row" {
    var prng = std.Random.DefaultPrng.init(4337);
    var row: [3000]f32 = undefined;
    for ([_]usize{ 1, 2, 5, 63, 1024, 1025, 3000 }) |len| for (0..kinds) |kind| for ([_]f32{ 1.0 / 0.7, 1e6, 0.001 }) |inv_t| {
        fill(row[0..len], kind, &prng);
        var m: f32 = undefined;
        const want = try sortedRow(row[0..len], inv_t, &m);
        defer a.free(want);
        for ([_]usize{ 1, 2, 20, 1000, 1024, len }) |n| {
            if (n > len) continue;
            const got = try a.alloc(rule.Cand, n);
            defer a.free(got);
            const top_m = rule.topCands(row[0..len], inv_t, got);
            try std.testing.expect(bitsEqual(m, top_m));
            try std.testing.expect(bitsEqual(lanesNorm(row[0..len], inv_t, m), rule.threadSum(row[0..len], inv_t, top_m)));
            for (got, want[0..n]) |g, w| try std.testing.expect(bitsEqual(g, w));
        }
    };
}

test "sample and plan draw as the full sort does, on adversarial rows and every sampling rule" {
    var prng = std.Random.DefaultPrng.init(1234);
    var row: [3000]f32 = undefined;
    for ([_]usize{ 7, 1025, 3000 }) |len| for (0..kinds) |kind| {
        fill(row[0..len], kind, &prng);
        for (temperatures) |t| for (settings) |cfg| {
            const s: Sampling = .{ .seed = 99 + kind, .temperature = t, .top_k = cfg.k, .top_p = cfg.p, .min_p = cfg.min };
            const want_plan = try fullSortPlan(row[0..len], s);
            try std.testing.expect(bitsEqual(want_plan, try full.plan(a, row[0..len], s)));
            for ([_]u32{ 0, 17, 66903620 }) |position| {
                if (cfg.k <= rule.candidates) {
                    try std.testing.expectEqual(try fullSortSample(row[0..len], s, position), try rule.sample(a, row[0..len], s, position, null));
                }
                try std.testing.expectEqual(full.race(row[0..len], want_plan, s.seed, position, null), try full.sample(a, row[0..len], s, position, null));
            }
        };
    };
}

test "a vocabulary-sized row draws as its full sort does" {
    var prng = std.Random.DefaultPrng.init(201058);
    const row = try a.alloc(f32, 201058);
    defer a.free(row);
    for ([_]usize{ 0, 1, 8 }) |kind| {
        fill(row, kind, &prng);
        for (settings[0..5]) |cfg| {
            const s: Sampling = .{ .seed = 7, .temperature = 0.7, .top_k = cfg.k, .top_p = cfg.p, .min_p = cfg.min };
            try std.testing.expect(bitsEqual(try fullSortPlan(row, s), try full.plan(a, row, s)));
            if (cfg.k <= rule.candidates) try std.testing.expectEqual(try fullSortSample(row, s, 3), try rule.sample(a, row, s, 3, null));
        }
    }
}
