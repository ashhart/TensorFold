//! A sampled XPU row drawn from the card's top_k candidates equals the draw over its whole row, on the host.
const std = @import("std");
const api = @import("engine_api");
const native = @import("xpu.zig");
const Lanes = @import("xpu_lanes.zig").Lanes;

const gpa = std.testing.allocator;
const SEG = 1024; // ids one topk_part group takes
const KMAX = 64;

/// The card's two kernels on the host: each group's K best of its 1024 ids, then their merge; the row's flag beside.
fn card(row: []const f32, k: usize, vals: []f32, ids: []u32) bool {
    const parts = (row.len + SEG - 1) / SEG;
    var lv: [256][KMAX]f32 = undefined;
    var li: [256][KMAX]u32 = undefined;
    var bad = false;
    for (0..parts) |p| {
        const seg = row[p * SEG ..][0..@min(SEG, row.len - p * SEG)];
        for (seg) |x| bad = bad or !std.math.isFinite(x);
        var n: usize = 0;
        for (seg, 0..) |x, i| {
            const v: f32 = if (std.math.isNan(x)) -std.math.inf(f32) else x;
            if (n == k and !(v > lv[p][k - 1])) continue;
            var j = if (n < k) n else k - 1;
            if (n < k) n += 1;
            while (j > 0 and v > lv[p][j - 1]) : (j -= 1) {
                lv[p][j] = lv[p][j - 1];
                li[p][j] = li[p][j - 1];
            }
            lv[p][j] = v;
            li[p][j] = @intCast(p * SEG + i);
        }
        for (n..k) |j| {
            lv[p][j] = -std.math.inf(f32);
            li[p][j] = std.math.maxInt(i32);
        }
    }
    var ptr: [256]usize = @splat(0);
    for (0..k) |out| {
        var best: usize = 0;
        var found = false;
        for (0..parts) |p| {
            if (ptr[p] >= k) continue;
            const v = lv[p][ptr[p]];
            const i = li[p][ptr[p]];
            if (!found or v > lv[best][ptr[best]] or (v == lv[best][ptr[best]] and i < li[best][ptr[best]])) {
                best = p;
                found = true;
            }
        }
        vals[out] = lv[best][ptr[best]];
        ids[out] = li[best][ptr[best]];
        ptr[best] += 1;
    }
    return bad;
}

/// A row of `n` fp16 values from a few levels (so many tie), signed zeros among them.
fn fill(rng: std.Random, row: []f32, levels: u32) void {
    for (row) |*v| {
        const l = rng.uintLessThan(u32, levels);
        v.* = if (l == 0) (if (rng.boolean()) -0.0 else 0.0) else @as(f32, @floatFromInt(l)) / 4 - 3;
    }
}

const Cfg = struct { top_k: u32, top_p: f64, min_p: f64, temperature: f64 };
const cfgs = [_]Cfg{
    .{ .top_k = 20, .top_p = 0.95, .min_p = 0, .temperature = 1 },
    .{ .top_k = 1, .top_p = 1, .min_p = 0, .temperature = 0.7 },
    .{ .top_k = 64, .top_p = 0.5, .min_p = 0.05, .temperature = 1.7 },
    .{ .top_k = 33, .top_p = 1, .min_p = 0.1, .temperature = 0.3 },
    .{ .top_k = 7, .top_p = 0, .min_p = 0, .temperature = 1 },
};

test "draft sources: a draw from the card's candidates is the draw over the whole row, random rows and ties" {
    var prng = std.Random.DefaultPrng.init(0x2830);
    const rng = prng.random();
    const sizes = [_]usize{ 70, 1023, 1024, 1025, 3000, 5121 };
    var drawn: usize = 0;
    for (sizes) |n| for ([_]u32{ 3, 12, 400 }) |levels| {
        const row = try gpa.alloc(f32, n);
        defer gpa.free(row);
        for (0..3) |_| {
            if (levels == 400) {
                for (row) |*v| v.* = @as(f32, @floatCast(rng.floatNorm(f64) * 3));
            } else fill(rng, row, levels);
            for (cfgs) |cfg| {
                if (cfg.top_k > n) continue;
                var vals: [KMAX]f32 = undefined;
                var ids: [KMAX]u32 = undefined;
                const k = cfg.top_k;
                try std.testing.expect(!card(row, k, &vals, &ids));
                for (0..4) |pos| {
                    const s: api.Sampling = .{ .seed = rng.int(u63), .temperature = cfg.temperature, .top_k = cfg.top_k, .top_p = cfg.top_p, .min_p = cfg.min_p };
                    const want = try native.pickRow(gpa, row, 100 + pos, s);
                    const got = try native.pickCand(gpa, vals[0..k], ids[0..k], false, 100 + pos, s);
                    try std.testing.expectEqual(want, got);
                    drawn += 1;
                }
            }
        }
    };
    try std.testing.expect(drawn > 500);
}

test "draft sources: the candidates are the top_k of the whole row in main's order, ties to the lower id" {
    var prng = std.Random.DefaultPrng.init(0x2831);
    const rng = prng.random();
    const row = try gpa.alloc(f32, 4097);
    defer gpa.free(row);
    for ([_]u32{ 1, 2, 5 }) |levels| {
        fill(rng, row, levels);
        var vals: [KMAX]f32 = undefined;
        var ids: [KMAX]u32 = undefined;
        _ = card(row, KMAX, &vals, &ids);
        const order = try gpa.alloc(u32, row.len);
        defer gpa.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        std.mem.sort(u32, order, row, struct {
            fn before(r: []const f32, a: u32, b: u32) bool {
                return r[a] > r[b] or (r[a] == r[b] and a < b);
            }
        }.before);
        try std.testing.expectEqualSlices(u32, order[0..KMAX], &ids);
        for (vals, ids) |v, i| try std.testing.expectEqual(@as(u32, @bitCast(row[i])), @as(u32, @bitCast(v)));
    }
}

test "draft sources: a value that is not finite anywhere in the row fails the draw, as the whole-row scan does" {
    var prng = std.Random.DefaultPrng.init(0x2832);
    const rng = prng.random();
    const row = try gpa.alloc(f32, 3000);
    defer gpa.free(row);
    const bad = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
    for (bad) |x| for ([_]usize{ 0, 1023, 1024, 2999 }) |at| {
        fill(rng, row, 12);
        row[at] = x;
        var vals: [KMAX]f32 = undefined;
        var ids: [KMAX]u32 = undefined;
        const flag = card(row, 20, &vals, &ids);
        try std.testing.expect(flag);
        const s: api.Sampling = .{ .seed = 5, .top_k = 20 };
        try std.testing.expectError(error.NonfiniteLogits, native.pickRow(gpa, row, 9, s));
        try std.testing.expectError(error.NonfiniteLogits, native.pickCand(gpa, vals[0..20], ids[0..20], flag, 9, s));
    };
}

test "draft sources: the card keeps candidates only for a sampled row with a top_k from 1 to 64" {
    const base: api.Sampling = .{ .seed = 1 };
    try std.testing.expectEqual(@as(u32, 20), Lanes.candidateK(base));
    try std.testing.expectEqual(@as(u32, 1), Lanes.candidateK(.{ .seed = 1, .top_k = 1 }));
    try std.testing.expectEqual(@as(u32, 64), Lanes.candidateK(.{ .seed = 1, .top_k = 64 }));
    try std.testing.expectEqual(@as(u32, 0), Lanes.candidateK(.{ .seed = 1, .top_k = 65 }));
    try std.testing.expectEqual(@as(u32, 0), Lanes.candidateK(.{ .seed = 1, .top_k = 0 }));
    try std.testing.expectEqual(@as(u32, 0), Lanes.candidateK(.{ .seed = 1, .temperature = 0 }));
}
