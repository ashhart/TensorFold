//! SAM3's FP16 trunk row norm (sam3_tile_norm_f16) against a float64 layer norm, gathering grid rows in tile order.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const u: f64 = 0x1p-24;
const eps: f32 = 1e-5;

pub const vt = g.hooks(Probe);

/// Grid row of tile-ordered row r: tiles column by column, row-major inside a tile (the spec's tile map).
pub fn gridRow(r: usize, grid: usize, tile: usize) usize {
    const n = grid / tile;
    const t = r / (tile * tile);
    const inner = r % (tile * tile);
    return ((t % n) * tile + inner / tile) * grid + (t / n) * tile + inner % tile;
}

fn gamma(n: f64) f64 {
    return n * u / (1 - n * u);
}

fn depth(s: c.Shape) usize {
    return (s.n + 255) / 256 + 12;
}

const Params = extern struct { width: u32, gather: u32, grid: u32, tile: u32, eps: f32 };

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    gather: bool,
    control: bool,
    rows: usize = 0,
    width: usize = 0,
    x: []f32 = &.{},
    weight: []f32 = &.{},
    bias: []f32 = &.{},
    allow: []f64 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn label(gather: bool, control: bool) []const u8 {
        if (control) return "control: sam3 trunk tile norm gathers row r";
        return if (gather) "metal sam3 trunk tile norm gathered" else "metal sam3 trunk tile norm";
    }

    pub fn spec(p: *const Probe) c.Spec {
        return .{
            .name = label(p.gather, p.control),
            .accum = .f32,
            .depth = depth,
            .out = .f16,
            .served = &.{.{ .rows = 5184, .n = 1024, .k = 1024 }},
            .tiles = .{ .{}, .{ .size = 256 }, .{} },
        };
    }

    pub fn reset(p: *Probe) void {
        for ([_][]f32{ p.x, p.weight, p.bias }) |x| p.a.free(x);
        p.a.free(p.allow);
        p.x = &.{};
        p.weight = &.{};
        p.bias = &.{};
        p.allow = &.{};
        p.rig.reset();
    }

    /// The gathered norm runs the whole 72 x 72 grid in nine 24 x 24 tiles whatever rows the harness asks for.
    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.n == 0 or (p.gather and s.rows > 5184)) return false;
        p.reset();
        p.rows = if (p.gather) 5184 else s.rows;
        p.width = s.n;
        p.x = try p.a.alloc(f32, p.rows * p.width);
        p.weight = try p.a.alloc(f32, p.width);
        p.bias = try p.a.alloc(f32, p.width);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // stress: 2054 and 2056 in turn (mean 2055, deviation 1); bf16 takes both to 2048, fp16 sums overflow
        for (p.x, 0..) |*v, i| v.* = if (fill == .stress) (if (i % 2 == 0) 2054 else 2056) else (rng.float(f32) * 2 - 1) * 4 + 0.5;
        for (p.weight) |*v| v.* = if (fill == .stress) 1 else rng.float(f32) * 2;
        for (p.bias) |*v| v.* = if (fill == .stress) 0 else rng.float(f32) - 0.5;
        if (fill == .nonfinite) {
            p.x[0] = std.math.nan(f32);
            p.x[p.width + 1] = std.math.inf(f32);
            p.x[2 * p.width + 2] = -std.math.inf(f32);
            p.x[3 * p.width] = std.math.inf(f32);
            p.x[3 * p.width + 1] = -std.math.inf(f32);
        }
        for ([_][]const f32{ p.x, p.weight, p.bias }, &p.ins) |values, *input| input.* = try p.rig.input(std.mem.sliceAsBytes(values));
        p.out = try p.rig.output(p.count() * 2);
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        for (p.ins, 0..) |r, i| pass.e.setBuffer(r.buf, r.off, i);
        pass.e.setBuffer(p.out.buf, p.out.off, 3);
        pass.e.setValue(Params{ .width = @intCast(p.width), .gather = @intFromBool(p.gather), .grid = 72, .tile = 24, .eps = eps }, 4);
        pass.e.dispatchThreads(mtl.Size.of(p.rows * 256, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return p.rows * p.width;
    }

    /// Float64's norm and FP32's reach: sum chains, the mean's error, eps, division, 2-ulp rsqrt, products, bias.
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        p.a.free(p.allow);
        p.allow = try p.a.alloc(f64, want.len);
        const w: f64 = @floatFromInt(p.width);
        const d1: f64 = @floatFromInt(depth(.{ .rows = 1, .n = p.width, .k = 1 }));
        for (0..p.rows) |r| {
            const src = p.x[(if (p.gather) gridRow(r, 72, 24) else r) * p.width ..][0..p.width];
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (src) |x| {
                sum += x;
                abs += @abs(@as(f64, x));
            }
            const mu = sum / w;
            const dmu = gamma(d1) * abs / w + u * (@abs(mu) + gamma(d1) * abs / w);
            var q: f64 = 0;
            var dq: f64 = 0;
            var sq: f64 = 0;
            for (src) |x| {
                const v = x - mu;
                const e = dmu + u * (@abs(v) + dmu);
                q += v * v;
                dq += (2 * @abs(v) + e) * e;
                sq += (@abs(v) + e) * (@abs(v) + e);
            }
            dq += gamma(d1 + 1) * sq;
            const t = q / w + eps;
            const dt = dq / w + 2 * u * (t + dq / w);
            const rate = 1 / @sqrt(t);
            const rho = if (dt < t) 0.5 * dt / (t - dt) + 4 * u else std.math.inf(f64);
            for (src, 0..) |x, col| {
                const at = r * p.width + col;
                const v = x - mu;
                const e = dmu + u * (@abs(v) + dmu);
                const prod = v * rate * p.weight[col];
                want[at] = prod + p.bias[col];
                mag[at] = @abs(want[at]);
                const err = @abs(p.weight[col]) * (e * rate * (1 + rho) + @abs(v) * rate * rho) + 2 * u * @abs(prod) + u * (@abs(want[at]) + @abs(prod));
                const work = err * (1 + 0x1p-20);
                p.allow[at] = if (std.math.isFinite(work)) work + 0x1p-11 * (@abs(want[at]) + work) + 0x1p-25 else std.math.floatMax(f64);
            }
        }
    }

    pub fn bounds(p: *Probe, _: []const f64, _: []const f64, allow: []f64) !void {
        @memcpy(allow, p.allow);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try p.a.alloc(f64, p.width);
        defer p.a.free(terms);
        const w: f64 = @floatFromInt(p.width);
        for (0..p.rows) |r| {
            const src = p.x[(if (p.gather) gridRow(r, 72, 24) else r) * p.width ..][0..p.width];
            for (terms, src) |*t, x| t.* = x;
            const mu = contract.rows.sumIn(acc, order, terms) / w;
            for (terms, src) |*t, x| t.* = (x - mu) * (x - mu);
            const rate = 1 / @sqrt(contract.rows.sumIn(acc, order, terms) / w + eps);
            for (src, 0..) |x, col| out[r * p.width + col] = @as(f16, @floatCast((x - mu) * rate * p.weight[col] + p.bias[col]));
        }
    }
};

test "the tile map takes tiles column by column, row-major inside each, and is a bijection on the grid" {
    try std.testing.expectEqual(@as(usize, 0), gridRow(0, 72, 24));
    try std.testing.expectEqual(@as(usize, 72), gridRow(24, 72, 24));
    try std.testing.expectEqual(@as(usize, 1728), gridRow(576, 72, 24));
    try std.testing.expectEqual(@as(usize, 24), gridRow(3 * 576, 72, 24));
    var seen: [5184]bool = @splat(false);
    for (0..5184) |r| seen[gridRow(r, 72, 24)] = true;
    for (seen) |s| try std.testing.expect(s);
}
