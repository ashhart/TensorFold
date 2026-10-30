//! DeepSeek-V4.1's MoE route under the contract: the top-k of sqrt(softplus(logit)) + bias, weights scaled by 1.5.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const sources = @import("kernel_sources");

pub const Variant = enum { backbone, drafter };

const scale: f32 = 1.5; // the reference's routed_scaling_factor
const eps: f32 = 1e-20;

pub fn experts(v: Variant) usize {
    return if (v == .backbone) 384 else 128;
}

pub fn picks(v: Variant) usize {
    return if (v == .backbone) 6 else 3;
}

fn depth(sh: c.Shape) usize {
    return sh.k + 8;
}

pub fn spec(v: Variant) c.Spec {
    return .{
        .name = switch (v) {
            .backbone => "metal dsv41 route backbone tf_dsv41_route_select",
            .drafter => "metal dsv41 route drafter tf_dsv41_route_select",
        },
        .accum = .f32,
        .depth = depth,
        .ops = 4,
        .out = .f32,
        .nan = .{ .stated = "a NaN logit or bias is never picked; a +inf logit is picked at a NaN weight and its row's other weights 0" },
        .tiles = .{ .{ .size = 8, .step = 1, .min = 1 }, .{}, .{} },
        .served = if (v == .backbone) &.{
            .{ .rows = 4, .n = 384, .k = 6 },
            .{ .rows = 1, .n = 384, .k = 6 },
            .{ .rows = 8, .n = 384, .k = 6 },
            .{ .rows = 8192, .n = 384, .k = 6 },
        } else &.{
            .{ .rows = 4, .n = 128, .k = 3 },
            .{ .rows = 1, .n = 128, .k = 3 },
            .{ .rows = 8, .n = 128, .k = 3 },
            .{ .rows = 8192, .n = 128, .k = 3 },
        },
        .sums = .{ .exact = true, .why = "the stress picks' scores are roots of exact squares, so their sum is exact in any fp32 order" },
    };
}

fn softplus64(z: f64) f64 {
    if (std.math.isNan(z)) return z;
    const e = @exp(@min(z, 20.0));
    if (z > 20.0) return z;
    if (z < -10.0) return e - 0.5 * e * e;
    return @log(1.0 + e);
}

const Args = extern struct { rows: i32, topk: i32, scale: f32, eps: f32 };

/// The stress picks' scores: roots of exact squares (the logit is the square, above softplus's linear tail).
const stress_scores = [_]f64{ 32768, 9, 8, 7, 6, 5 };

pub const Route = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    variant: Variant,
    e: usize,
    k: usize,
    shape: c.Shape = undefined,
    logits: []f32 = &.{},
    bias: []f32 = &.{},
    bias_vl: []f32 = &.{},
    image: []u8 = &.{},
    previous: []u8 = &.{},
    runs: usize = 0,
    x: rig_.Ref = undefined,
    bias_r: rig_.Ref = undefined,
    bias_vl_r: rig_.Ref = undefined,
    image_r: rig_.Ref = undefined,
    picks: rig_.Ref = undefined,
    weights: rig_.Ref = undefined,
    msg: [240]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, variant: Variant) !Route {
        const e = experts(variant);
        const k = picks(variant);
        const src = try std.fmt.allocPrint(a, "#define TF_DSV41_E {d}\n#define TF_DSV41_TOPK {d}\n{s}", .{ e, k, sources.dsv41_route });
        defer a.free(src);
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .pipe = try rig_.pipeline(device, src, "tf_dsv41_route_select"),
            .variant = variant,
            .e = e,
            .k = k,
        };
    }

    pub fn deinit(t: *Route) void {
        t.a.free(t.logits);
        t.a.free(t.bias);
        t.a.free(t.bias_vl);
        t.a.free(t.image);
        t.a.free(t.previous);
        t.rig.deinit();
        t.pipe.deinit();
    }

    pub fn kernel(t: *Route) g.Kernel {
        return .{ .spec = spec(t.variant), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(Route);

    /// The two experts whose (logit, bias) the random fill makes exactly equal on every fourth row: a planted tie.
    fn tiePair(t: *const Route) [2]usize {
        return .{ 11, if (t.e == 384) 322 else 100 };
    }

    pub fn prepare(t: *Route, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != t.e or shape.k != t.k) return false;
        t.rig.reset();
        t.shape = shape;
        t.runs = 0;
        t.a.free(t.logits);
        t.logits = &.{};
        t.a.free(t.bias);
        t.bias = &.{};
        t.a.free(t.bias_vl);
        t.bias_vl = &.{};
        t.a.free(t.image);
        t.image = &.{};
        t.a.free(t.previous);
        t.previous = &.{};
        const rows = shape.rows;
        t.logits = try t.a.alloc(f32, rows * t.e);
        t.bias = try t.a.alloc(f32, t.e);
        t.bias_vl = try t.a.alloc(f32, t.e);
        t.image = try t.a.alloc(u8, rows);
        t.previous = try t.a.alloc(u8, t.count() * 4);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.bias) |*v| v.* = rng.float(f32) - 0.5;
        for (t.bias_vl) |*v| v.* = rng.float(f32) - 0.5;
        const tie = t.tiePair();
        t.bias[tie[0]] = 0.25;
        t.bias[tie[1]] = 0.25;
        t.bias_vl[tie[0]] = 0.25;
        t.bias_vl[tie[1]] = 0.25;
        for (t.logits) |*v| v.* = rng.float(f32) * 12 - 6;
        for (0..rows) |r| {
            t.image[r] = @intCast(r % 2);
            if (r % 4 == 1) {
                t.logits[r * t.e + tie[0]] = 0.5;
                t.logits[r * t.e + tie[1]] = 0.5;
            }
        }
        switch (fill) {
            .stress => {
                // The k picks' logits are exact squares; everything else sits far below, and the bias is zero.
                @memset(t.logits, -80.0);
                @memset(t.bias, 0);
                @memset(t.bias_vl, 0);
                for (0..rows) |r| for (0..t.k) |j| {
                    const s = stress_scores[j];
                    t.logits[r * t.e + j] = @floatCast(s * s);
                };
            },
            .nonfinite => {
                const values = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
                const plus = if (t.e == 384) @as(usize, 100) else 60;
                const minus = if (t.e == 384) @as(usize, 200) else 120;
                t.logits[0 * t.e + 7] = values[0];
                t.logits[1 * t.e + plus] = values[1];
                t.logits[2 * t.e + minus] = values[2];
                t.logits[3 * t.e + 3] = values[1];
                t.logits[3 * t.e + 4] = values[2];
                t.bias[13] = values[0];
                t.bias_vl[17] = values[0];
            },
            .random => {},
        }
        t.x = try t.rig.input(std.mem.sliceAsBytes(t.logits));
        t.bias_r = try t.rig.input(std.mem.sliceAsBytes(t.bias));
        t.bias_vl_r = try t.rig.input(std.mem.sliceAsBytes(t.bias_vl));
        t.image_r = try t.rig.input(t.image);
        t.picks = try t.rig.scratch(t.count() * 4);
        t.weights = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Route) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.x.buf, t.x.off, 0);
        p.e.setBuffer(t.bias_r.buf, t.bias_r.off, 1);
        p.e.setBuffer(t.bias_vl_r.buf, t.bias_vl_r.off, 2);
        p.e.setBuffer(t.image_r.buf, t.image_r.off, 3);
        p.e.setValue(Args{
            .rows = @intCast(t.shape.rows),
            .topk = @intCast(t.k),
            .scale = scale,
            .eps = eps,
        }, 4);
        p.e.setBuffer(t.picks.buf, t.picks.off, 5);
        p.e.setBuffer(t.weights.buf, t.weights.off, 6);
        p.e.dispatchGroups(mtl.Size.of(t.shape.rows, 1, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(p);
        const picked = t.rig.bytes(t.picks, t.count() * 4);
        if (t.runs > 0 and !std.mem.eql(u8, t.previous, picked)) return error.RoutePicksNotRepeatable;
        @memcpy(t.previous, picked);
        t.runs += 1;
    }

    pub fn count(t: *Route) usize {
        return t.shape.rows * t.k;
    }

    fn rowScores(t: *Route, r: usize, s: *[384]f64, key: *[384]f64) void {
        const bias = if (t.image[r] != 0) t.bias_vl else t.bias;
        for (0..t.e) |i| {
            s[i] = @sqrt(softplus64(t.logits[r * t.e + i]));
            key[i] = s[i] + bias[i];
        }
    }

    /// The k picks in fp64 (NaN keys never win, ties to the lower id), and their unbiased scores' sum in pick order.
    fn select(t: *Route, key: *const [384]f64, s: *const [384]f64, chosen: *[6]usize, total: *f64) void {
        var used: [384]bool = @splat(false);
        total.* = 0;
        for (0..t.k) |kk| {
            var best: usize = t.e; // none
            for (0..t.e) |i| {
                if (used[i] or key[i] != key[i]) continue;
                if (best == t.e or key[i] > key[best] or (key[i] == key[best] and i < best)) best = i;
            }
            if (best == t.e) {
                chosen[kk] = t.e;
            } else {
                used[best] = true;
                chosen[kk] = best;
                total.* += s[best];
            }
        }
    }

    pub fn reference(t: *Route, want: []f64, mag: []f64) !void {
        var s: [384]f64 = undefined;
        var key: [384]f64 = undefined;
        var chosen: [6]usize = undefined;
        for (0..t.shape.rows) |r| {
            t.rowScores(r, &s, &key);
            var total: f64 = 0;
            t.select(&key, &s, &chosen, &total);
            for (0..t.k) |kk| {
                const i = chosen[kk];
                const w = if (i == t.e) 0 else (s[i] / (total + 1e-20)) * scale;
                want[r * t.k + kk] = w;
                mag[r * t.k + kk] = @abs(w);
            }
        }
    }

    /// Gate 4: the picks as fp64 gives them, the sum in the accumulator and order, then the kernel's fp32 weight chain.
    pub fn simulate(t: *Route, acc: c.Accum, order: g.Order, out: []f64) !void {
        var s: [384]f64 = undefined;
        var key: [384]f64 = undefined;
        var chosen: [6]usize = undefined;
        var terms: [6]f64 = undefined;
        for (0..t.shape.rows) |r| {
            t.rowScores(r, &s, &key);
            var total: f64 = 0;
            t.select(&key, &s, &chosen, &total);
            var nt: usize = 0;
            for (chosen[0..t.k]) |i| if (i != t.e) {
                terms[nt] = s[i];
                nt += 1;
            };
            const sum = contract.rows.sumIn(acc, order, terms[0..nt]);
            for (0..t.k) |kk| {
                const i = chosen[kk];
                out[r * t.k + kk] = if (i == t.e) 0 else blk: {
                    const num: f32 = @floatCast(s[i]);
                    const den: f32 = @floatCast(sum + 1e-20);
                    break :blk @as(f64, num / den * scale);
                };
            }
        }
    }

    pub fn check(t: *Route, at: g.Gate) !?[]const u8 {
        const n = t.count();
        const picked = t.rig.bytes(t.picks, n * 4);
        const weights = t.rig.bytes(t.weights, n * 4);
        if (at == .nonfinite) {
            const buf = try t.a.alloc(f64, 2 * n);
            defer t.a.free(buf);
            const want = buf[0..n];
            const mag = buf[n..];
            try t.reference(want, mag);
            var nf: usize = 0;
            var differ: usize = 0;
            var first: usize = n;
            for (want, 0..) |w, i| {
                if (c.classOf(w) != .finite) nf += 1;
                if (c.classOf(c.read(.f32, weights, i)) != c.classOf(w)) {
                    differ += 1;
                    if (first == n) first = i;
                }
            }
            if (nf == 0 or nf == n) return try std.fmt.bufPrint(&t.msg, "the plants make {d} of {d} reference weights non-finite: no proof", .{ nf, n });
            if (differ > 0) return try std.fmt.bufPrint(&t.msg, "weight {d} is {t} where float64 gives {t}", .{ first, c.classOf(c.read(.f32, weights, first)), c.classOf(want[first]) });
            return null;
        }
        var s: [384]f64 = undefined;
        var key: [384]f64 = undefined;
        var chosen: [6]usize = undefined;
        for (0..t.shape.rows) |r| {
            t.rowScores(r, &s, &key);
            var used: [384]bool = @splat(false);
            for (0..t.k) |kk| {
                const i = r * t.k + kk;
                const pick = std.mem.readInt(i32, picked[i * 4 ..][0..4], .little);
                if (pick < 0) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} took nothing, but every expert is a candidate", .{ r, kk });
                const p: usize = @intCast(pick);
                if (p >= t.e) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} is expert {d} of {d}", .{ r, kk, p, t.e });
                if (used[p]) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} repeats expert {d}", .{ r, kk, p });
                used[p] = true;
                // The best remaining key in fp64: the pick must reach it within the fp32 arithmetic's slack.
                var bestk: f64 = -std.math.inf(f64);
                var bestid: usize = t.e;
                for (0..t.e) |j| {
                    if (used[j] or key[j] != key[j]) continue;
                    if (key[j] > bestk or (key[j] == bestk and j < bestid)) {
                        bestk = key[j];
                        bestid = j;
                    }
                }
                const slack = 1e-5 + 1e-6 * @abs(bestk);
                if (bestid != t.e and key[p] < bestk - slack) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} took expert {d} at key {d:.6} while expert {d} remains at {d:.6}", .{ r, kk, p, key[p], bestid, bestk });
                for (0..t.e) |j| {
                    if (used[j] or key[j] != key[p] or j >= p) continue;
                    return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} took expert {d} while expert {d} ties it at key {d:.6}", .{ r, kk, p, j, key[p] });
                }
            }
            // The weights against the fp64 formula over the kernel's own picks.
            var total: f64 = 0;
            for (0..t.k) |kk| {
                const pick = std.mem.readInt(i32, picked[(r * t.k + kk) * 4 ..][0..4], .little);
                chosen[kk] = if (pick < 0) t.e else @intCast(pick);
                if (chosen[kk] != t.e) total += s[chosen[kk]];
            }
            for (0..t.k) |kk| {
                const i = r * t.k + kk;
                const want = if (chosen[kk] == t.e) 0 else (s[chosen[kk]] / (total + 1e-20)) * scale;
                const got = c.read(.f32, weights, i);
                if (c.classOf(got) != c.classOf(want)) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d}: weight class {t} where the fp64 formula gives {t}", .{ r, kk, c.classOf(got), c.classOf(want) });
                if (at != .nonfinite and @abs(got - want) > c.bound(spec(t.variant), t.shape, want, @abs(want)))
                    return try std.fmt.bufPrint(&t.msg, "row {d} pick {d}: weight {e:.6} where the fp64 formula from its own picks gives {e:.6}", .{ r, kk, got, want });
            }
        }
        return null;
    }
};
