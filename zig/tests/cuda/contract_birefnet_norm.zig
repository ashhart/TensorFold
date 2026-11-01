//! tf_br_norm (by rows, by a window map, PatchMerging's) and tf_br_deform under the contract against float64.
const std = @import("std");
const cm = @import("contract_birefnet_common.zig");

const c = cm.c;
const g = cm.g;
const Range = cm.Range;
const Ends = cm.Ends;
const ks = cm.ks;
const uniform = cm.uniform;
const plant = cm.plant;
const U = cm.U;

// --- tf_br_norm: LayerNorm in two fp32 passes; a map's padding rows zeros; PatchMerging's 2 x 2 gather --------------

pub const NormForm = enum { rows, map, merge };

/// PatchMerging's source grid in the test: 8 tokens a row, so 4 output rows a source row pair.
const merge_w = 8;

pub const Norm = struct {
    a: std.mem.Allocator,
    rig: cm.rig_.Rig,
    k: *const ks.Kernels,
    name: []const u8,
    form: NormForm,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    map: []i32 = &.{},
    gamma: []f32 = &.{},
    beta: []f32 = &.{},
    allow: []f64 = &.{},
    ins: [4]u64 = @splat(0),
    out: u64 = 0,

    /// A thread's strided chain, the warp's five levels, the eight warps' chain.
    fn depth(s: c.Shape) usize {
        return (s.n + 255) / 256 + 5 + 8;
    }

    /// Rows: output rows; n: the width (PatchMerging's 4 C); k: the same, the sums' length.
    pub fn spec(name: []const u8, f: NormForm) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .tiles = if (f == .merge) .{ .{ .size = 8, .step = 4, .min = 4 }, .{}, .{} } else .{ .{ .size = 8 }, .{}, .{} },
            .served = switch (f) {
                .rows => &.{ .{ .rows = 300, .n = 192, .k = 192 }, .{ .rows = 130, .n = 384, .k = 384 }, .{ .rows = 70, .n = 768, .k = 768 }, .{ .rows = 40, .n = 1536, .k = 1536 } },
                .map => &.{ .{ .rows = 288, .n = 192, .k = 192 }, .{ .rows = 144, .n = 1536, .k = 1536 } },
                .merge => &.{ .{ .rows = 64, .n = 768, .k = 768 }, .{ .rows = 32, .n = 1536, .k = 1536 }, .{ .rows = 16, .n = 3072, .k = 3072 } },
            },
            .sums = .{ .exact = false },
        };
    }

    pub fn deinit(t: *Norm) void {
        t.free();
        t.rig.deinit();
    }
    pub fn free(t: *Norm) void {
        inline for (.{ &t.x, &t.gamma, &t.beta }) |p| {
            t.a.free(p.*);
            p.* = &.{};
        }
        t.a.free(t.map);
        t.map = &.{};
        t.a.free(t.allow);
        t.allow = &.{};
    }
    pub fn kernel(t: *Norm, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.form), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Norm);

    /// The map's padding: every seventh row (as torch pads after the norm, those rows are zeros).
    fn pad(r: usize) bool {
        return r % 7 == 3;
    }
    fn sourceRows(t: *const Norm) usize {
        return switch (t.form) {
            .rows => t.shape.rows,
            .map => t.shape.rows - (t.shape.rows + 3) / 7,
            .merge => 4 * t.shape.rows,
        };
    }
    fn width(t: *const Norm) usize {
        return if (t.form == .merge) t.shape.n / 4 else t.shape.n;
    }

    pub fn fill(t: *Norm, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 4 != 0 or shape.k != shape.n) return false;
        if (t.form == .merge and shape.rows % (merge_w / 2) != 0) return false;
        t.free();
        t.shape = shape;
        t.x = try t.a.alloc(f32, t.sourceRows() * t.width());
        t.gamma = try t.a.alloc(f32, shape.n);
        t.beta = try t.a.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (mode == .stress) {
            // integers to 2041 a row: its sum exact in fp32, past fp16's range in a few dozen terms
            for (t.x) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, 0, 2041));
        } else {
            uniform(rng, t.x, -4, 4);
            // a row far off zero: the mean's rounding then dominates the deviations
            for (t.x[0..t.width()]) |*v| v.* += 300;
            if (mode == .nonfinite) plant(t.x, t.width());
        }
        uniform(rng, t.gamma, -2, 2);
        uniform(rng, t.beta, -1, 1);
        if (t.form == .map) {
            const order = try t.a.alloc(i32, t.sourceRows());
            defer t.a.free(order);
            for (order, 0..) |*v, i| v.* = @intCast(i);
            rng.shuffle(i32, order);
            t.map = try t.a.alloc(i32, shape.rows);
            var next: usize = 0;
            for (t.map, 0..) |*m, r| {
                if (pad(r)) {
                    m.* = -1;
                } else {
                    m.* = order[next];
                    next += 1;
                }
            }
        }
        return true;
    }

    pub fn prepare(t: *Norm, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        t.rig.reset();
        if (!try t.fill(shape, mode, seed)) return false;
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(if (t.map.len > 0) t.map else &[_]i32{0}));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.gamma));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.beta));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Norm) !void {
        const mode: ks.NormMode = switch (t.form) {
            .rows => .rows,
            .map => .map,
            .merge => .merge,
        };
        try ks.norm(t.k, t.rig.s, t.ins[0], if (t.form == .map) t.ins[1] else 0, t.ins[2], t.ins[3], t.out, t.shape.rows, t.shape.n, mode, merge_w, t.shape.n, 0, 1e-5);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Norm) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Row r's values in the kernel's order; false for a map's padding.
    fn row(t: *const Norm, r: usize, out: []f64) bool {
        const n = t.shape.n;
        switch (t.form) {
            .rows => for (out, 0..) |*v, ch| {
                v.* = t.x[r * n + ch];
            },
            .map => {
                if (t.map[r] < 0) return false;
                const src: usize = @intCast(t.map[r]);
                for (out, 0..) |*v, ch| v.* = t.x[src * n + ch];
            },
            .merge => {
                // [x0 | x1 | x2 | x3]: (2i, 2j), (2i + 1, 2j), (2i, 2j + 1), (2i + 1, 2j + 1)
                const q = n / 4;
                const half = merge_w / 2;
                for (out, 0..) |*v, ch| {
                    const part = ch / q;
                    const sy = (r / half) * 2 + (part & 1);
                    const sx = (r % half) * 2 + (part >> 1);
                    v.* = t.x[(sy * merge_w + sx) * q + ch % q];
                }
            },
        }
        return true;
    }

    /// The outputs in float64, each one's allowance from fp32's analysis of both passes and the scale and shift.
    fn compute(t: *Norm, acc: ?struct { c.Accum, g.Order }, want: []f64, allow: ?[]f64) !void {
        const n = t.shape.n;
        const buf = try t.a.alloc(f64, 2 * n);
        defer t.a.free(buf);
        const v = buf[0..n];
        const sq = buf[n..];
        const nf: f64 = @floatFromInt(n);
        const D: f64 = @floatFromInt(depth(t.shape));
        for (0..t.shape.rows) |r| {
            if (!t.row(r, v)) {
                @memset(want[r * n ..][0..n], 0);
                if (allow) |al| @memset(al[r * n ..][0..n], 0);
                continue;
            }
            var s: f64 = 0;
            var s1: f64 = 0;
            for (v) |x| {
                s += x;
                s1 += @abs(x);
            }
            if (acc) |ao| s = cm.rows.sumIn(ao[0], ao[1], v);
            const mu = s / nf;
            var s2: f64 = 0;
            var sd: f64 = 0;
            var most: f64 = 0;
            for (v, sq) |x, *q| {
                q.* = (x - mu) * (x - mu);
                s2 += q.*;
                sd += @abs(x - mu);
                most = @max(most, @abs(x - mu));
            }
            if (acc) |ao| s2 = cm.rows.sumIn(ao[0], ao[1], sq);
            const variance = s2 / nf;
            const rs = 1 / @sqrt(variance + 1e-5);
            for (v, 0..) |x, ch| {
                const i = r * n + ch;
                const gm: f64 = t.gamma[ch];
                const y = (x - mu) * rs * gm + t.beta[ch];
                want[i] = y;
                const al = allow orelse continue;
                const e_mu = c.gamma(D, U) * s1 / nf + 2 * cm.div_ulps * U * @abs(mu);
                const e_d = e_mu + U * most;
                const e_var = (2 * e_d * sd + nf * e_d * e_d) / nf + c.gamma(D + 1, U) * s2 / nf + (2 * cm.div_ulps + 1) * U * (variance + 1e-5);
                const rel_r = e_var / (2 * (variance + 1e-5)) * 1.01 + 2 * (cm.sqrt_ulps + cm.div_ulps) * U;
                const d = @abs(x - mu);
                const e = @abs(gm) * rs * ((e_mu + U * d) * (1 + rel_r) + d * (rel_r + 2 * U)) + 2 * U * @abs(y) + U * @abs(@as(f64, t.beta[ch]));
                al[i] = if (std.math.isFinite(y)) 1.01 * e + 0x1p-149 else 0;
            }
        }
    }

    pub fn reference(t: *Norm, want: []f64, mag: []f64) !void {
        t.a.free(t.allow);
        t.allow = try t.a.alloc(f64, t.count());
        try t.compute(null, want, t.allow);
        for (mag, want) |*m, w| m.* = if (std.math.isFinite(w)) @abs(w) else 0;
    }
    pub fn bounds(t: *Norm, _: []const f64, _: []const f64, allow: []f64) !void {
        @memcpy(allow, t.allow);
    }
    pub fn simulate(t: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        try t.compute(.{ acc, order }, out, null);
    }
};

// --- tf_br_deform: torchvision's modulated deform_conv2d columns (stride 1, padding k / 2) ---------------------------

const grid_h = 16;
const grid_w = 12;
const first = 5;

pub const Deform = struct {
    a: std.mem.Allocator,
    rig: cm.rig_.Rig,
    k: *const ks.Kernels,
    name: []const u8,
    side: usize,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    offs: []f32 = &.{},
    ins: [2]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    /// Rows: the chunk's pixels (from pixel 5 of a 16 x 12 grid); n: its columns, k x k taps of 64; k: 64.
    pub fn spec(name: []const u8, side: usize) c.Spec {
        return .{ .name = name, .accum = .none, .depth = cm.noSums, .out = .f32, .tiles = .{ .{ .size = 32 }, .{}, .{} }, .served = switch (side) {
            1 => &comptime served(1),
            3 => &comptime served(3),
            else => &comptime served(7),
        } };
    }
    fn served(comptime side: usize) [2]c.Shape {
        return .{ .{ .rows = grid_h * grid_w - first, .n = side * side * 64, .k = 64 }, .{ .rows = 64, .n = side * side * 64, .k = 64 } };
    }

    pub fn deinit(t: *Deform) void {
        t.a.free(t.x);
        t.a.free(t.offs);
        t.ends.deinit();
        t.rig.deinit();
    }
    pub fn kernel(t: *Deform, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.side), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Deform);

    fn taps(t: *const Deform) usize {
        return t.side * t.side;
    }
    fn ldo(t: *const Deform) usize {
        return std.mem.alignForward(usize, 3 * t.taps(), 64);
    }

    pub fn fill(t: *Deform, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > grid_h * grid_w - first or shape.k != 64 or shape.n != t.taps() * 64) return false;
        t.a.free(t.x);
        t.a.free(t.offs);
        t.shape = shape;
        t.x = try t.a.alloc(f32, grid_h * grid_w * 64);
        t.offs = try t.a.alloc(f32, grid_h * grid_w * t.ldo());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        uniform(rng, t.x, -3, 3);
        // offsets to 2.5 pixels: taps fall outside, on the border and between pixels; whole-pixel ones too
        uniform(rng, t.offs, -2.5, 2.5);
        for (t.offs, 0..) |*v, i| {
            if (i % t.ldo() >= 3 * t.taps()) v.* = 0 else if (i % 11 == 0) v.* = @round(v.*);
        }
        if (mode == .nonfinite) plant(t.x, 64);
        return true;
    }

    pub fn prepare(t: *Deform, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        t.rig.reset();
        if (!try t.fill(shape, mode, seed)) return false;
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.offs));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Deform) !void {
        try ks.deform(t.k, t.rig.s, t.ins[0], t.ins[1], t.out, .{ .h = grid_h, .w = grid_w }, t.side, first, t.shape.rows, t.ldo());
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Deform) usize {
        return t.shape.rows * t.shape.n;
    }

    fn px(t: *const Deform, y: i64, x: i64, ch: usize) f64 {
        return t.x[(@as(usize, @intCast(y)) * grid_w + @as(usize, @intCast(x))) * 64 + ch];
    }

    /// The sample's position and weights as the kernel forms them in fp32; products, sums and the mask's ulps.
    pub fn reference(t: *Deform, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const nt = t.taps();
        const pad: i64 = @intCast(t.side / 2);
        for (0..t.shape.rows) |r| {
            const pix = first + r;
            const oy: i64 = @intCast(pix / grid_w);
            const ox: i64 = @intCast(pix % grid_w);
            const off = t.offs[pix * t.ldo() ..];
            for (0..nt) |tap| {
                const m: f64 = off[2 * nt + tap];
                const mask = 2 / (1 + @exp(-m));
                const er = Range.of(@exp(-m)).fp32(cm.exp_ulps).add(Range.of(1)).fp32(0.5);
                const mr = er.recip().fp32(cm.div_ulps).mul(Range.of(2));
                const h: f32 = @as(f32, @floatFromInt(oy - pad + @as(i64, @intCast(tap / t.side)))) + off[2 * tap];
                const w: f32 = @as(f32, @floatFromInt(ox - pad + @as(i64, @intCast(tap % t.side)))) + off[2 * tap + 1];
                const inside = h > -1 and h < grid_h and w > -1 and w < grid_w;
                const hl: i64 = @intFromFloat(@floor(h));
                const wl: i64 = @intFromFloat(@floor(w));
                const lh: f32 = h - @as(f32, @floatFromInt(hl));
                const lw: f32 = w - @as(f32, @floatFromInt(wl));
                const hh: f32 = 1 - lh;
                const hw: f32 = 1 - lw;
                const wt = [4]f32{ hh * hw, hh * lw, lh * hw, lh * lw };
                const ok = [4]bool{ hl >= 0 and wl >= 0, hl >= 0 and wl + 1 <= grid_w - 1, hl + 1 <= grid_h - 1 and wl >= 0, hl + 1 <= grid_h - 1 and wl + 1 <= grid_w - 1 };
                const ys = [4]i64{ hl, hl, hl + 1, hl + 1 };
                const xs = [4]i64{ wl, wl + 1, wl, wl + 1 };
                for (0..64) |ch| {
                    const i = r * t.shape.n + tap * 64 + ch;
                    if (!inside) {
                        cm.put(&t.ends, want, mag, i, mask * 0, Range.of(0));
                        continue;
                    }
                    var s: f64 = 0;
                    var sr = Range.of(0);
                    for (0..4) |q| {
                        const v: f64 = if (ok[q]) t.px(ys[q], xs[q], ch) else 0;
                        const term = @as(f64, wt[q]) * v;
                        s += term;
                        sr = if (q == 0) Range.of(term).fp32(0.5) else sr.add(Range.of(term).fp32(0.5)).fp32(0.5);
                    }
                    cm.put(&t.ends, want, mag, i, mask * s, sr.mul(mr).fp32(0.5));
                }
            }
        }
    }
    pub fn bounds(t: *Deform, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Deform, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
};
