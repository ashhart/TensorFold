//! MiniMax-M3.1-Flash's drafter kernels under the contract: norms and taps here, the rest in metal_minimax_draft_*.zig.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const attn = @import("metal_minimax_draft_attn.zig");
const pick = @import("metal_minimax_draft_pick.zig");
const walk = @import("metal_minimax_draft_walk.zig");
const act = @import("metal_minimax_draft_act.zig");

const c = contract.spec;
const g = contract.gates;
const Texts = @import("metal_minimax.zig").Texts;

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// A value as fp32 then bf16 rounds it (to nearest even, twice).
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// A value as fp32 rounds it.
pub fn f32r(v: f64) f64 {
    return @as(f32, @floatCast(v));
}

/// A value's spread through fp32 steps and bf16 rounding points.
pub const Range = struct {
    lo: f64,
    hi: f64,
    pub fn point(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    /// Each end moved by `ulps` fp32 roundings of its own size.
    pub fn fp32(a: Range, ulps: f64) Range {
        const u = ulps * c.unit(.f32);
        return .{ .lo = a.lo - u * @abs(a.lo) - 0x1p-149, .hi = a.hi + u * @abs(a.hi) + 0x1p-149 };
    }
    pub fn bf16(a: Range) Range {
        return .{ .lo = bf(a.lo), .hi = bf(a.hi) };
    }
    pub fn add(a: Range, b: Range) Range {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    pub fn neg(a: Range) Range {
        return .{ .lo = -a.hi, .hi = -a.lo };
    }
    pub fn mul(a: Range, b: Range) Range {
        const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
};

/// The allowance a range gives about `want`: the farther end.
pub fn spread(want: f64, r: Range) f64 {
    return @max(@abs(want - r.lo), @abs(r.hi - want));
}

/// The builds as dspark.zig, model.zig and dflash2.zig compile them: DSpark's, the target's (MLX's options), DFlash2's.
pub const Build = enum { dspark, target, dflash2 };

pub fn pipeline(device: mtl.Device, b: Build, name: []const u8) !mtl.Pipeline {
    const text = if (b == .dflash2) sources.minimax_m3_dflash2 else sources.minimax_m3_dspark;
    const opts: mtl.CompileOptions = if (b == .target) mtl.CompileOptions.mlx() else .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise };
    const lib = try mtl.Library.fromSource(device, text, opts);
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, name, false);
}

pub const nan = std.math.nan(f32);
pub const inf = std.math.inf(f32);

/// Random bf16 bits in [lo, hi).
pub fn fillBf(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
}

/// Gate 4's norm rows (the rows family's): integers in [-32, 32], so fp32 sums of squares are exact in any order.
fn stressRows(rng: std.Random, x: []f32) void {
    for (x) |*v| v.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 65))) - 32;
}

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    owned: std.ArrayList(Owned) = .empty,

    const Owned = struct { ctx: *anyopaque, free: *const fn (*anyopaque, std.mem.Allocator) void };

    pub fn add(s: *Suite, comptime T: type, only: ?[]const u8, v: T.Variant) !void {
        if (only) |o| if (std.mem.indexOf(u8, T.spec(v).name, o) == null) return;
        const t = try s.a.create(T);
        errdefer s.a.destroy(t);
        t.* = try T.init(s.a, s.device, s.queue, v);
        const free = struct {
            fn f(p: *anyopaque, a: std.mem.Allocator) void {
                const x: *T = @ptrCast(@alignCast(p));
                x.deinit();
                a.destroy(x);
            }
        }.f;
        try s.owned.append(s.a, .{ .ctx = t, .free = free });
        try s.kernels.append(s.a, t.kernel());
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(Norm.Variant)) |v| try s.add(Norm, only, v);
        for (std.enums.values(Tap.Variant)) |v| try s.add(Tap, only, v);
        try act.collect(s, only);
        try attn.collect(s, only);
        try pick.collect(s, only);
        try walk.collect(s, only);
    }
    pub fn deinit(s: *Suite) void {
        for (s.owned.items) |o| o.free(o.ctx, s.a);
        s.owned.deinit(s.a);
    }
};

/// Gate 5's texts: a `minimax_m3 draft ` registration compiles dflash2.metal when it names dflash2, else dspark.metal.
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !?Texts {
    _ = a;
    _ = device;
    if (!std.mem.startsWith(u8, name, "minimax_m3 draft ")) return null;
    const text = if (std.mem.indexOf(u8, name, " dflash2 ") != null) sources.minimax_m3_dflash2 else sources.minimax_m3_dspark;
    return .{ .here = text, .pre_m5 = text };
}

// --- mm_norm16, df_norm16, df_norm: rms norms of a row, 256 threads -----------------------------------------------

/// Each thread's fma chain over width / 256 inputs, a 5-level simd_sum, then the 8 simdgroups' parts in turn.
fn normDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 5 + 8;
}

pub const norm_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

pub const Norm = struct {
    /// DSpark's Gemma norm of bf16 rows; DFlash2's Qwen3 norm of bf16 or fp32 rows, or of bf16(R + X) written to R.
    pub const Variant = enum { mm_norm16, df_norm16, df_norm, df_norm_add };
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Variant,
    shape: c.Shape = undefined,
    /// The rows as the norm reads them (after the add), the fp32 residual before it, X and w as bf16.
    x: []f32 = &.{},
    r0: []f32 = &.{},
    xb: []u16 = &.{},
    w: []u16 = &.{},
    res: rig_.Ref = undefined,
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn spec(v: Variant) c.Spec {
        // DFlash2's df_norm: the draft block's 8 rows; the bf16 norms: a context absorb's fc rows (2,048), and a few
        const block: []const c.Shape = &.{.{ .rows = 8, .n = 1, .k = 6144 }};
        const context: []const c.Shape = &.{ .{ .rows = 2048, .n = 1, .k = 6144 }, .{ .rows = 7, .n = 1, .k = 6144 } };
        return .{
            .name = switch (v) {
                .mm_norm16 => "minimax_m3 draft dspark mm_norm16",
                .df_norm16 => "minimax_m3 draft dflash2 df_norm16",
                .df_norm => "minimax_m3 draft dflash2 df_norm fp32 rows",
                .df_norm_add => "minimax_m3 draft dflash2 df_norm bf16(R + X) rows",
            },
            .accum = .f32,
            .depth = normDepth,
            // the mean's division, + eps, precise rsqrt (2), x * inv, then (1 + w) and its product, or w's product
            .ops = if (v == .mm_norm16) 7 else 6,
            // Qwen3's bf16(x * inv) before w's product
            .mids = if (v == .mm_norm16) 0 else 1,
            .out = .bf16,
            .tiles = .{ .{ .size = 1 }, .{}, .{ .size = 256, .step = 1, .min = 1 } },
            .served = if (v == .df_norm or v == .df_norm_add) block else context, // n unused
            .sums = .{ .bf16 = 0, .why = norm_why },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant) !Norm {
        const name = switch (v) {
            .mm_norm16 => "mm_norm16",
            .df_norm16 => "df_norm16",
            .df_norm, .df_norm_add => "df_norm",
        };
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(device, if (v == .mm_norm16) .dspark else .dflash2, name), .v = v };
    }
    pub fn deinit(t: *Norm) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Norm) void {
        inline for (.{ "x", "r0", "xb", "w" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Norm) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Norm);

    fn fp32Rows(t: *const Norm) bool {
        return t.v == .df_norm or t.v == .df_norm_add;
    }

    pub fn prepare(t: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = shape.rows * shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(f32, n);
        t.r0 = try t.a.alloc(f32, n);
        t.xb = try t.a.alloc(u16, n);
        t.w = try t.a.alloc(u16, shape.k);
        fillBf(rng, t.w, -0.5, 0.5);
        for (t.r0) |*v| v.* = (rng.float(f32) * 2 - 1) * (if (rng.uintLessThan(u32, 64) == 7) @as(f32, 20) else 1);
        fillBf(rng, t.xb, -2, 2);
        if (fill == .stress) {
            stressRows(rng, t.r0);
            for (t.xb, t.r0) |*b, r| b.* = if (t.v == .df_norm_add) 0 else c.toBf16(r);
        }
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, shape.k / 3 }, .{ 1, shape.k / 2 }, .{ 2, shape.k - 1 }, .{ 3, 0 }, .{ 3, shape.k - 1 } };
            const put = [_]f32{ nan, inf, -inf, inf, -inf };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                const i = p[0] * shape.k + p[1];
                if (t.v == .df_norm) t.r0[i] = v else t.xb[i] = c.toBf16(v);
            };
        }
        for (t.x, t.r0, t.xb) |*x, r, b| x.* = switch (t.v) {
            .mm_norm16, .df_norm16 => c.fromBf16(b),
            .df_norm => r,
            .df_norm_add => @floatCast(bf(@as(f32, r + c.fromBf16(b)))),
        };
        if (t.fp32Rows()) t.res = try t.rig.scratch(n * 4);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.xb));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Norm) !void {
        const n = t.shape.rows * t.shape.k;
        if (t.fp32Rows()) @memcpy(t.rig.bytes(t.res, n * 4), std.mem.sliceAsBytes(t.r0));
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        const args = extern struct { rows: u32, width: u32, eps: f32, add: u32 }{ .rows = @intCast(t.shape.rows), .width = @intCast(t.shape.k), .eps = 1e-6, .add = @intFromBool(t.v == .df_norm_add) };
        if (t.fp32Rows()) {
            p.e.setBuffer(t.res.buf, t.res.off, 0);
            p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 1);
            p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 2);
            p.e.setValue(args, 3);
            p.e.setBuffer(t.out.buf, t.out.off, 4);
        } else {
            p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
            p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
            if (t.v == .mm_norm16) p.e.setValue(extern struct { rows: u32, width: u32, eps: f32 }{ .rows = args.rows, .width = args.width, .eps = args.eps }, 2) else p.e.setValue(args, 2);
            p.e.setBuffer(t.out.buf, t.out.off, 3);
        }
        p.e.dispatchGroups(mtl.Size.of(t.shape.rows, 1, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
        if (!t.fp32Rows()) return;
        // R as the run left it: the rows the norm read, bit for bit (a NaN as a NaN)
        const got = std.mem.bytesAsSlice(f32, t.rig.bytes(t.res, n * 4));
        for (got, t.x) |gv, want| if (!(@as(u32, @bitCast(gv)) == @as(u32, @bitCast(want)) or (std.math.isNan(gv) and std.math.isNan(want)))) return error.ResidualRowsDiffer;
    }

    pub fn count(t: *Norm) usize {
        return t.shape.rows * t.shape.k;
    }

    const Sim = struct { c.Accum, g.Order };

    fn inv(t: *const Norm, row: usize, acc: ?Sim) !f64 {
        const k = t.shape.k;
        const sq = try t.a.alloc(f64, k);
        defer t.a.free(sq);
        for (sq, t.x[row * k ..][0..k]) |*s, x| s.* = @as(f64, x) * x;
        var ss: f64 = 0;
        for (sq) |s| ss += s;
        if (acc) |o| ss = contract.rows.sumIn(o[0], o[1], sq);
        return 1 / @sqrt(ss / @as(f64, @floatFromInt(k)) + @as(f32, 1e-6));
    }

    fn outputs(t: *Norm, out: []f64, mag: ?[]f64, acc: ?Sim) !void {
        const k = t.shape.k;
        for (0..t.shape.rows) |r| {
            const s = try t.inv(r, acc);
            for (0..k) |i| {
                const w: f64 = c.fromBf16(t.w[i]);
                const x: f64 = t.x[r * k + i];
                const y = if (t.v == .mm_norm16) x * s * (1 + w) else w * (x * s);
                out[r * k + i] = y;
                if (mag) |m| m[r * k + i] = if (std.math.isFinite(y)) @abs(y) / 2 else 0; // y ~ ss^-1/2
            }
        }
    }

    pub fn reference(t: *Norm, want: []f64, mag: []f64) !void {
        try t.outputs(want, mag, null);
    }
    pub fn simulate(t: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        try t.outputs(out, null, .{ acc, order });
    }
};

// --- mm_tap: the target's residual after a tapped layer, as bf16, into its slot among the taps -------------------

pub const Tap = struct {
    pub const Variant = enum { dspark_taps, dflash2_taps };
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Variant,
    shape: c.Shape = undefined,
    r: []f32 = &.{},
    old: []u16 = &.{},
    in: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: positions; n: the hidden width; k: the taps (DSpark 5, DFlash2 6); the other taps' slots stay put.
    pub fn spec(v: Variant) c.Spec {
        const taps: usize = if (v == .dspark_taps) 5 else 6;
        return .{
            .name = if (v == .dspark_taps) "minimax_m3 draft target mm_tap 5 taps" else "minimax_m3 draft target mm_tap 6 taps",
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .tiles = .{ .{ .size = 1 }, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            // a decode window and a prompt chunk's rows (4,096 launch the same grid; 512 keep the float64 arrays small)
            .served = if (taps == 5) &.{ .{ .rows = 8, .n = 6144, .k = 5 }, .{ .rows = 512, .n = 6144, .k = 5 } } else &.{ .{ .rows = 8, .n = 6144, .k = 6 }, .{ .rows = 512, .n = 6144, .k = 6 } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant) !Tap {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(device, .target, "mm_tap"), .v = v };
    }
    pub fn deinit(t: *Tap) void {
        t.a.free(t.r);
        t.a.free(t.old);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Tap) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Tap);

    fn tap(t: *const Tap) usize {
        return t.shape.k - 2;
    }

    pub fn prepare(t: *Tap, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k < 2) return false;
        t.rig.reset();
        t.a.free(t.r);
        t.a.free(t.old);
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.r = try t.a.alloc(f32, shape.rows * shape.n);
        for (t.r) |*v| v.* = (rng.float(f32) * 2 - 1) * 64;
        if (fill == .nonfinite) for ([_]f32{ nan, inf, -inf }, 0..) |v, i| {
            t.r[(i * 7) % t.r.len] = v;
        };
        t.old = try t.a.alloc(u16, t.count());
        fillBf(rng, t.old, -1, 1);
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.r));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Tap) !void {
        @memcpy(t.rig.bytes(t.out, t.count() * 2), std.mem.sliceAsBytes(t.old));
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.in.buf, t.in.off, 0);
        p.e.setValue([4]u32{ @intCast(t.shape.rows), @intCast(t.shape.n), @intCast(t.shape.k), @intCast(t.tap()) }, 1);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Tap) usize {
        return t.shape.rows * t.shape.k * t.shape.n;
    }

    pub fn reference(t: *Tap, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        for (want, mag, 0..) |*w, *m, i| {
            const r = i / (t.shape.k * n);
            const slot = i / n % t.shape.k;
            w.* = if (slot == t.tap()) t.r[r * n + i % n] else c.fromBf16(t.old[i]);
            m.* = 0;
        }
    }
};
