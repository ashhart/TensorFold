//! DFlash2's activation (df_silu) and grouped block convolution (df_conv) under the contract.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const d = @import("metal_minimax_draft.zig");

const c = contract.spec;
const g = contract.gates;
const bf = d.bf;
const noSums = d.noSums;
const pipeline = d.pipeline;
const fillBf = d.fillBf;
const nan = d.nan;
const inf = d.inf;

pub fn collect(s: *d.Suite, only: ?[]const u8) !void {
    try s.add(Silu, only, .df_silu);
    for (std.enums.values(Conv.Variant)) |v| try s.add(Conv, only, v);
}

// --- df_silu: Qwen3's SiLU of the gate times up, at its bf16 points -----------------------------------------------

pub const Silu = struct {
    pub const Variant = enum { df_silu };
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    gu: []u16 = &.{},
    in: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: positions; n: the ffn width (k unused).
    pub fn spec(_: Variant) c.Spec {
        return .{
            .name = "minimax_m3 draft dflash2 df_silu",
            .accum = .none,
            .depth = noSums,
            // precise exp (2 ulps), the add, the division, the sigmoid's own 1 - y (1), g * s, then * up
            .ops = 6,
            .mids = 1,
            .out = .bf16,
            .tiles = .{ .{ .size = 1 }, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            .served = &.{ .{ .rows = 8, .n = 12288, .k = 1 }, .{ .rows = 1, .n = 12288, .k = 1 } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, _: Variant) !Silu {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(device, .dflash2, "df_silu") };
    }
    pub fn deinit(t: *Silu) void {
        t.a.free(t.gu);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Silu) g.Kernel {
        return .{ .spec = spec(.df_silu), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Silu);

    pub fn prepare(t: *Silu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.gu);
        t.shape = shape;
        const n = shape.n;
        t.gu = try t.a.alloc(u16, shape.rows * 2 * n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // gates past +-20 reach silu's flat ends; ups both signs
        for (0..shape.rows) |r| {
            fillBf(rng, t.gu[r * 2 * n ..][0..n], -24, 24);
            fillBf(rng, t.gu[r * 2 * n + n ..][0..n], -4, 4);
        }
        if (fill == .nonfinite) for ([_]f32{ nan, inf, -inf, nan, inf, -inf }, 0..) |v, i| {
            const r = i % shape.rows;
            t.gu[r * 2 * n + (if (i < 3) 0 else n) + (i * 7) % n] = c.toBf16(v);
        };
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.gu));
        t.out = try t.rig.output(shape.rows * n * 2);
        return true;
    }

    pub fn run(t: *Silu) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.in.buf, t.in.off, 0);
        p.e.setValue([2]u32{ @intCast(t.shape.rows), @intCast(t.shape.n) }, 1);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Silu) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Silu, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        for (want, mag, 0..) |*w, *m, i| {
            const r = i / n;
            const gate: f64 = c.fromBf16(t.gu[r * 2 * n + i % n]);
            w.* = gate / (1 + @exp(-gate)) * c.fromBf16(t.gu[r * 2 * n + n + i % n]);
            m.* = 0;
        }
    }
};

// --- df_conv: the grouped two-tap block convolution, exact at its bf16 points ----------------------------------------

pub const Conv = struct {
    /// Side 0 (the sublayer input's prepare) and side 1 (its output), each with its own taps of base and the deltas.
    pub const Variant = enum { side0, side1 };
    const block = 8;
    const group = 16;
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Variant,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    d: []u16 = &.{},
    base: []u16 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: block positions; n: the hidden width (groups of 16 channels).
    pub fn spec(v: Variant) c.Spec {
        return .{
            .name = if (v == .side0) "minimax_m3 draft dflash2 df_conv side 0" else "minimax_m3 draft dflash2 df_conv side 1",
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .tiles = .{ .{ .size = 8, .step = 1, .min = 1 }, .{ .size = 256, .step = 16, .min = 16 }, .{} },
            .served = &.{ .{ .rows = 8, .n = 6144, .k = 2 }, .{ .rows = 16, .n = 6144, .k = 2 } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant) !Conv {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(device, .dflash2, "df_conv"), .v = v };
    }
    pub fn deinit(t: *Conv) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Conv) void {
        inline for (.{ "x", "d", "base" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Conv) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Conv);

    fn groups(t: *const Conv) usize {
        return t.shape.n / group;
    }
    fn side(t: *const Conv) usize {
        return @backingInt(t.v);
    }

    pub fn prepare(t: *Conv, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % group != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, shape.rows * shape.n);
        t.d = try t.a.alloc(u16, shape.rows * 4 * t.groups());
        t.base = try t.a.alloc(u16, 4 * shape.n);
        fillBf(rng, t.x, -4, 4);
        fillBf(rng, t.d, -0.5, 0.5);
        fillBf(rng, t.base, -1.5, 1.5);
        if (fill == .nonfinite) for ([_]f32{ nan, inf, -inf }, 0..) |v, i| {
            t.x[(1 + i) * shape.n + 5 * i] = c.toBf16(v);
            t.d[(i * 4 + t.side() * 2) * t.groups() % t.d.len] = c.toBf16(v);
        };
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.d));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.base));
        t.out = try t.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(t: *Conv) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setValue([5]u32{ @intCast(t.shape.rows), @intCast(t.shape.n), block, @intCast(t.groups()), @intCast(t.side()) }, 3);
        p.e.setBuffer(t.out.buf, t.out.off, 4);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Conv) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Conv, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        const gs = t.groups();
        const s = t.side();
        for (want, mag, 0..) |*w, *m, i| {
            const r = i / n;
            const ch = i % n;
            const gi = ch / group;
            const dyn = t.d[r * 4 * gs + s * 2 * gs ..];
            const c0 = bf(@as(f64, c.fromBf16(t.base[s * 2 * n + ch])) + c.fromBf16(dyn[gi]));
            const c1 = bf(@as(f64, c.fromBf16(t.base[(s * 2 + 1) * n + ch])) + c.fromBf16(dyn[gs + gi]));
            var y = bf(c0 * c.fromBf16(t.x[i]));
            if (r % block >= 1) y = bf(y + bf(c1 * c.fromBf16(t.x[i - n])));
            w.* = y;
            m.* = 0;
        }
    }
};
