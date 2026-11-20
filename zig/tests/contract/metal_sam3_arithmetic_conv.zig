//! sam3_depthwise and sam3_group_norm against float64.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const arithmetic = @import("metal_sam3_arithmetic.zig");
const c = contract.spec;
const g = contract.gates;

const u: f64 = 0x1p-24;
const eps: f32 = 1e-5;

const ConvShape = extern struct { in_h: u32, in_w: u32, in_c: u32, out_c: u32, k: u32, out_h: u32, out_w: u32, stride: u32, pad: u32, first: u32, rows: u32 };

fn taps(s: c.Shape) usize {
    return s.k * s.k + 1;
}

/// The memory fuser's 7 x 7 depthwise convolution (pad 3) on 72 x 72 x 256: rows = side, n = channels, k = taps.
const Depthwise = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    control: bool,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    w: []f32 = &.{},
    b: []f32 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn label(control: bool) []const u8 {
        return if (control) "control: sam3 arithmetic depthwise transposes its taps" else "metal sam3 arithmetic depthwise";
    }

    fn spec(p: *const Depthwise) c.Spec {
        return .{
            .name = label(p.control),
            .accum = .f32,
            .depth = taps,
            .out = .f32,
            .tiles = .{ .{ .size = 7 }, .{}, .{} },
            .served = &.{.{ .rows = 72, .n = 256, .k = 7 }},
        };
    }

    fn reset(p: *Depthwise) void {
        for ([_][]f32{ p.x, p.w, p.b }) |v| p.a.free(v);
        p.x = &.{};
        p.w = &.{};
        p.b = &.{};
        p.rig.reset();
    }

    pub fn prepare(p: *Depthwise, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k % 2 == 0) return false;
        p.reset();
        p.shape = shape;
        const side = shape.rows;
        p.x = try p.a.alloc(f32, side * side * shape.n);
        p.w = try p.a.alloc(f32, shape.n * shape.k * shape.k);
        p.b = try p.a.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (p.x) |*v| v.* = rng.float(f32) * 4 - 2;
        for (p.w) |*v| v.* = rng.float(f32) * 2 - 1;
        for (p.b) |*v| v.* = rng.float(f32) * 2 - 1;
        if (fill == .stress) {
            // Integer taps whose partial sums stay below 2^18: exact in fp32 in any order, not in fp16 or bf16.
            @memset(p.x, 1);
            @memset(p.b, 0);
            for (p.w, 0..) |*v, i| v.* = if (i % 2 == 0) 4097 else -4096;
        } else if (fill == .nonfinite) {
            const n = shape.n;
            p.x[0] = std.math.nan(f32);
            p.x[(2 * side + 2) * n + 1] = std.math.inf(f32);
            p.x[(4 * side + 4) * n + 2] = -std.math.inf(f32);
            p.x[(1 * side + 3) * n + 3] = std.math.inf(f32);
            p.x[(1 * side + 4) * n + 3] = -std.math.inf(f32);
            p.b[n - 1] = std.math.nan(f32);
        }
        for ([_][]const f32{ p.x, p.w, p.b }, &p.ins) |v, *r| r.* = try p.rig.input(std.mem.sliceAsBytes(v));
        p.out = try p.rig.output(p.count() * 4);
        return true;
    }

    pub fn run(p: *Depthwise) !void {
        const side: u32 = @intCast(p.shape.rows);
        const n: u32 = @intCast(p.shape.n);
        const k: u32 = @intCast(p.shape.k);
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        for (p.ins, 0..) |r, i| pass.e.setBuffer(r.buf, r.off, i);
        pass.e.setBuffer(p.out.buf, p.out.off, 3);
        pass.e.setValue(ConvShape{ .in_h = side, .in_w = side, .in_c = n, .out_c = n, .k = k, .out_h = side, .out_w = side, .stride = 1, .pad = k / 2, .first = 0, .rows = 0 }, 4);
        pass.e.dispatchThreads(mtl.Size.of(p.count(), 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Depthwise) usize {
        return p.shape.rows * p.shape.rows * p.shape.n;
    }

    /// Output i's terms in the kernel's order: the bias, then the taps inside the map, row by row.
    fn terms(p: *const Depthwise, i: usize, out: []f64) []f64 {
        const n = p.shape.n;
        const k = p.shape.k;
        const side = p.shape.rows;
        const ch = i % n;
        const x = i / n % side;
        const y = i / n / side;
        out[0] = p.b[ch];
        var len: usize = 1;
        for (0..k) |ky| for (0..k) |kx| {
            const sy = @as(isize, @intCast(y + ky)) - @as(isize, @intCast(k / 2));
            const sx = @as(isize, @intCast(x + kx)) - @as(isize, @intCast(k / 2));
            const edge: isize = @intCast(side);
            if (sy < 0 or sx < 0 or sy >= edge or sx >= edge) continue;
            const at = (@as(usize, @intCast(sy)) * side + @as(usize, @intCast(sx))) * n + ch;
            out[len] = @as(f64, p.x[at]) * p.w[(ch * k + ky) * k + kx];
            len += 1;
        };
        return out[0..len];
    }

    pub fn reference(p: *Depthwise, want: []f64, mag: []f64) !void {
        const buf = try p.a.alloc(f64, taps(p.shape));
        defer p.a.free(buf);
        for (want, mag, 0..) |*w, *m, i| {
            w.* = 0;
            m.* = 0;
            for (p.terms(i, buf)) |t| {
                w.* += t;
                m.* += @abs(t);
            }
        }
    }

    pub fn simulate(p: *Depthwise, acc: c.Accum, order: g.Order, out: []f64) !void {
        const buf = try p.a.alloc(f64, taps(p.shape));
        defer p.a.free(buf);
        for (out, 0..) |*w, i| w.* = contract.rows.sumIn(acc, order, p.terms(i, buf));
    }
};

fn statsDepth(s: c.Shape) usize {
    return (s.rows * (s.n / s.k) + 255) / 256 + 9;
}

/// Group i's statistics in float64 and the magnitudes the bound needs.
const Stats = struct { n: f64, mean: f64, variance: f64, sum_abs: f64, sum_squares: f64 };

/// The pixel decoder's group norm, 8 groups at 144² and 288² x 256: rows = pixels, n = channels, k = groups.
const GroupNorm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    control: bool,
    stress: bool = false,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    w: []f32 = &.{},
    b: []f32 = &.{},
    stats: []Stats = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn label(control: bool) []const u8 {
        return if (control) "control: sam3 arithmetic group norm drops the tree's last level" else "metal sam3 arithmetic group norm";
    }

    fn spec(p: *const GroupNorm) c.Spec {
        return .{
            .name = label(p.control),
            .accum = .f32,
            .depth = statsDepth,
            .out = .f32,
            .tiles = .{ .{ .size = 8 }, .{}, .{} },
            .served = &.{ .{ .rows = 144 * 144, .n = 256, .k = 8 }, .{ .rows = 288 * 288, .n = 256, .k = 8 } },
        };
    }

    fn reset(p: *GroupNorm) void {
        for ([_][]f32{ p.x, p.w, p.b }) |v| p.a.free(v);
        p.a.free(p.stats);
        p.x = &.{};
        p.w = &.{};
        p.b = &.{};
        p.stats = &.{};
        p.rig.reset();
    }

    pub fn prepare(p: *GroupNorm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.n % shape.k != 0) return false;
        p.reset();
        p.shape = shape;
        p.stress = fill == .stress;
        p.x = try p.a.alloc(f32, shape.rows * shape.n);
        p.w = try p.a.alloc(f32, shape.n);
        p.b = try p.a.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // An offset mean makes E[x^2] - mean^2 cancel, as the formula must survive.
        for (p.x) |*v| v.* = 3 + rng.float(f32) * 4 - 2;
        for (p.w) |*v| v.* = (0.5 + rng.float(f32)) * @as(f32, if (rng.boolean()) 1 else -1);
        for (p.b) |*v| v.* = rng.float(f32) - 0.5;
        if (fill == .stress) {
            // Integers: S is 0 and Q stays below 2^24, exact in fp32 in any order; fp16 overflows Q, bf16 rounds 2025.
            for (p.x, 0..) |*v, i| v.* = if (i % 2 == 0) 45 else -45;
        } else if (fill == .nonfinite) {
            const n = shape.n;
            const width = n / shape.k;
            p.x[0] = std.math.nan(f32);
            p.x[1 * n + width] = std.math.inf(f32);
            p.x[2 * n + 2 * width] = -std.math.inf(f32);
            p.x[3 * n + 3 * width] = std.math.inf(f32);
            p.x[4 * n + 3 * width + 1] = -std.math.inf(f32);
        }
        for ([_][]const f32{ p.x, p.w, p.b }, &p.ins) |v, *r| r.* = try p.rig.input(std.mem.sliceAsBytes(v));
        p.out = try p.rig.output(p.count() * 4);
        p.stats = try p.a.alloc(Stats, shape.k);
        p.measure();
        return true;
    }

    pub fn run(p: *GroupNorm) !void {
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        for (p.ins, 0..) |r, i| pass.e.setBuffer(r.buf, r.off, i);
        pass.e.setBuffer(p.out.buf, p.out.off, 3);
        pass.e.setValue([3]u32{ @intCast(p.shape.rows), @intCast(p.shape.n), @intCast(p.shape.k) }, 4);
        pass.e.dispatchThreads(mtl.Size.of(p.shape.k * 256, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const GroupNorm) usize {
        return p.shape.rows * p.shape.n;
    }

    /// Mean and variance as the kernel forms them (max(0, Q/n - mean^2)) where that is not finite, else two-pass.
    fn measure(p: *GroupNorm) void {
        const width = p.shape.n / p.shape.k;
        for (p.stats, 0..) |*st, group| {
            var s: f64 = 0;
            var q: f64 = 0;
            var sa: f64 = 0;
            for (0..p.shape.rows) |pixel| for (0..width) |j| {
                const v: f64 = p.x[pixel * p.shape.n + group * width + j];
                s += v;
                q += v * v;
                sa += @abs(v);
            };
            const n: f64 = @floatFromInt(p.shape.rows * width);
            const mean = s / n;
            var variance = @max(0, q / n - mean * mean);
            if (std.math.isFinite(s) and std.math.isFinite(q)) {
                var d: f64 = 0;
                for (0..p.shape.rows) |pixel| for (0..width) |j| {
                    const v = p.x[pixel * p.shape.n + group * width + j] - mean;
                    d += v * v;
                };
                variance = d / n;
            }
            st.* = .{ .n = n, .mean = mean, .variance = variance, .sum_abs = sa, .sum_squares = q };
        }
    }

    fn normed(p: *const GroupNorm, i: usize, mean: f64, variance: f64) f64 {
        const ch = i % p.shape.n;
        return (p.x[i] - mean) * (1 / @sqrt(variance + eps)) * p.w[ch] + p.b[ch];
    }

    fn statsOf(p: *const GroupNorm, i: usize) Stats {
        return p.stats[i % p.shape.n / (p.shape.n / p.shape.k)];
    }

    pub fn reference(p: *GroupNorm, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            const st = p.statsOf(i);
            w.* = p.normed(i, st.mean, st.variance);
            m.* = @abs(w.*);
        }
    }

    /// First-order error through S, Q (thread chains, 8 tree levels), mean, Q/n - mean², eps, 2-ulp rsqrt, the output.
    pub fn bounds(p: *GroupNorm, want: []const f64, _: []const f64, allow: []f64) !void {
        for (want, allow, 0..) |w, *l, i| {
            if (!std.math.isFinite(w)) continue;
            const st = p.statsOf(i);
            const chain = @ceil(st.n / 256) + 8;
            const ds = if (p.stress) 0 else c.gamma(chain, u) * st.sum_abs;
            const dq = if (p.stress) 0 else c.gamma(chain + 1, u) * st.sum_squares;
            const mean = @abs(st.mean);
            const dm = ds / st.n + u * (mean + ds / st.n);
            const q = st.sum_squares / st.n;
            const dqn = dq / st.n + u * (q + dq / st.n);
            const dmm = 2 * mean * dm + dm * dm + u * (mean + dm) * (mean + dm);
            const dv = dqn + dmm + u * (q + dqn + (mean + dm) * (mean + dm));
            const arg = st.variance + eps;
            const da = dv + u * (arg + dv);
            const rho = da / arg;
            if (rho >= 0.5) {
                l.* = std.math.floatMax(f64);
                continue;
            }
            const dr = rho / (2 * (1 - rho)) + 0x1p-22;
            const r = 1 / @sqrt(arg);
            const ch = i % p.shape.n;
            const d = @abs(p.x[i] - st.mean);
            const dd = dm + u * (d + dm);
            const et = @abs(p.w[ch]) * r * ((dd * (1 + dr) + d * dr) * (1 + 2 * u) + 2 * u * d);
            l.* = (et + u * (@abs(w) + et)) * (1 + 0x1p-20) + 0x1p-149;
        }
    }

    pub fn simulate(p: *GroupNorm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const width = p.shape.n / p.shape.k;
        const n = p.shape.rows * width;
        const buf = try p.a.alloc(f64, 2 * n);
        defer p.a.free(buf);
        var means: [64]f64 = undefined;
        var variances: [64]f64 = undefined;
        for (0..p.shape.k) |group| {
            for (0..p.shape.rows) |pixel| for (0..width) |j| {
                const v: f64 = p.x[pixel * p.shape.n + group * width + j];
                buf[pixel * width + j] = v;
                buf[n + pixel * width + j] = v * v;
            };
            const nf: f64 = @floatFromInt(n);
            means[group] = contract.rows.sumIn(acc, order, buf[0..n]) / nf;
            variances[group] = @max(0, contract.rows.sumIn(acc, order, buf[n..]) / nf - means[group] * means[group]);
        }
        for (out, 0..) |*w, i| {
            const group = i % p.shape.n / width;
            w.* = p.normed(i, means[group], variances[group]);
        }
    }
};

const vt_depthwise = g.hooks(Depthwise);
const vt_norm = g.hooks(GroupNorm);

pub const Suite = struct {
    a: std.mem.Allocator,
    depthwise: [2]?*Depthwise = @splat(null),
    norm: [2]?*GroupNorm = @splat(null),

    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for ([_]bool{ false, true }) |control| {
            const gate: ?g.Gate = if (control) .shapes else null;
            if (if (only) |needle| std.mem.indexOf(u8, Depthwise.label(control), needle) != null else true) {
                try kernels.ensureUnusedCapacity(s.a, 1);
                const p = try s.a.create(Depthwise);
                errdefer s.a.destroy(p);
                p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = try arithmetic.pipeline(s.a, device, "sam3_depthwise", control), .control = control };
                s.depthwise[@intFromBool(control)] = p;
                kernels.appendAssumeCapacity(.{ .spec = p.spec(), .ctx = p, .vt = &vt_depthwise, .control = gate });
            }
            if (if (only) |needle| std.mem.indexOf(u8, GroupNorm.label(control), needle) != null else true) {
                try kernels.ensureUnusedCapacity(s.a, 1);
                const p = try s.a.create(GroupNorm);
                errdefer s.a.destroy(p);
                p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = try arithmetic.pipeline(s.a, device, "sam3_group_norm", control), .control = control };
                s.norm[@intFromBool(control)] = p;
                kernels.appendAssumeCapacity(.{ .spec = p.spec(), .ctx = p, .vt = &vt_norm, .control = gate });
            }
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.depthwise) |maybe| if (maybe) |p| {
            p.reset();
            p.rig.deinit();
            p.pipe.deinit();
            s.a.destroy(p);
        };
        for (s.norm) |maybe| if (maybe) |p| {
            p.reset();
            p.rig.deinit();
            p.pipe.deinit();
            s.a.destroy(p);
        };
    }
};
