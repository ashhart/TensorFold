//! The drafter's activation (rc_df_silu) and grouped four-tap block convolution (rc_df_conv) under the contract.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_ricursive_controls.zig");
const d = @import("metal_ricursive_dflash2.zig");

const c = contract.spec;
const g = contract.gates;
const D = d.D;
const Mutation = d.Mutation;
const bf = d.bf;
const fillBf = d.fillBf;
const nan = d.nan;
const inf = d.inf;

pub fn collect(s: *d.Suite, x: d.Ctx) !void {
    try s.add(x, Silu, .silu, .none);
    for (std.enums.values(Conv.Variant)) |v| try s.add(x, Conv, v, .none);
}

// --- rc_df_silu: the release's SiLU of the gate times up, at its bf16 points ---------------------------------------

pub const Silu = struct {
    pub const Variant = enum { silu };
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    shape: c.Shape = undefined,
    gu: []u16 = &.{},
    in: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: positions; n: the ffn width (k unused).
    pub fn spec(_: Variant) c.Spec {
        return .{
            .name = "ricursive dflash2 rc_df_silu",
            .accum = .none,
            .depth = d.noSums,
            // precise exp (2 ulps), the add, the division, the sigmoid's own e * y (1), g * s, then * up
            .ops = 6,
            .mids = 1,
            .out = .bf16,
            .tiles = .{ .{ .size = 1 }, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            .served = &.{.{ .rows = D.block, .n = D.ffn, .k = 1 }},
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, _: Variant, m: Mutation) !Silu {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try d.pipeline(device, a, "rc_df_silu", m), .m = m };
    }
    pub fn deinit(t: *Silu) void {
        t.a.free(t.gu);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Silu) g.Kernel {
        return controls.mark(.{ .spec = spec(.silu), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Silu);

    pub fn prepare(t: *Silu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.gu);
        t.gu = &.{};
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

// --- rc_df_conv: the grouped four-tap block convolution, exact at its bf16 points -----------------------------------

pub const Conv = struct {
    /// Side 0 (the sublayer input's prepare) and side 1 (its output), each with its own taps of base and the deltas.
    pub const Variant = enum { side0, side1 };
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Variant,
    m: Mutation,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    d: []u16 = &.{},
    base: []u16 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: block positions; n: the hidden width (groups of 16 channels); k: the taps.
    pub fn spec(v: Variant) c.Spec {
        return .{
            .name = if (v == .side0) "ricursive dflash2 rc_df_conv side 0" else "ricursive dflash2 rc_df_conv side 1",
            .accum = .none,
            .depth = d.noSums,
            .out = .bf16,
            .tiles = .{ .{ .size = D.block, .step = 1, .min = 1 }, .{ .size = 256, .step = D.group, .min = D.group }, .{} },
            .served = &.{.{ .rows = D.block, .n = D.hidden, .k = D.conv }},
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, m: Mutation) !Conv {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try d.pipeline(device, a, "rc_df_conv", m), .v = v, .m = m };
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
        return controls.mark(.{ .spec = spec(t.v), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Conv);

    fn groups(t: *const Conv) usize {
        return t.shape.n / D.group;
    }
    fn side(t: *const Conv) usize {
        return @backingInt(t.v);
    }
    /// Row r's coefficient delta for (side, tap, group): dyn's [side][tap][group] layout.
    fn delta(t: *const Conv, r: usize, s: usize, tap: usize, gi: usize) f64 {
        const k = t.shape.k;
        return c.fromBf16(t.d[((r * 2 + s) * k + tap) * t.groups() + gi]);
    }

    pub fn prepare(t: *Conv, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % D.group != 0 or shape.k == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, shape.rows * shape.n);
        t.d = try t.a.alloc(u16, shape.rows * 2 * shape.k * t.groups());
        t.base = try t.a.alloc(u16, 2 * shape.k * shape.n);
        fillBf(rng, t.x, -4, 4);
        fillBf(rng, t.d, -0.5, 0.5);
        fillBf(rng, t.base, -1.5, 1.5);
        if (fill == .nonfinite) for ([_]f32{ nan, inf, -inf }, 0..) |v, i| {
            t.x[(1 + i) * shape.n + 5 * i] = c.toBf16(v);
            t.d[((i * 2 + t.side()) * shape.k) * t.groups() % t.d.len] = c.toBf16(v);
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
        p.e.setValue([6]u32{ @intCast(t.shape.rows), @intCast(t.shape.n), D.block, @intCast(t.groups()), @intCast(t.side()), @intCast(t.shape.k) }, 3);
        p.e.setBuffer(t.out.buf, t.out.off, 4);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Conv) usize {
        return t.shape.rows * t.shape.n;
    }

    /// z-lab's order per tap inside the block: y = bf16(y + bf16(base x[r - t])), then y = bf16(y + delta x[r - t]).
    pub fn reference(t: *Conv, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        const s = t.side();
        for (want, mag, 0..) |*w, *m, i| {
            const r = i / n;
            const ch = i % n;
            const gi = ch / D.group;
            var y: f64 = 0;
            for (0..t.shape.k) |tap| {
                if (r % D.block < tap) break;
                const x: f64 = c.fromBf16(t.x[i - tap * n]);
                y = bf(y + bf(@as(f64, c.fromBf16(t.base[(s * t.shape.k + tap) * n + ch])) * x));
                y = bf(y + t.delta(r, s, tap, gi) * x);
            }
            w.* = y;
            m.* = 0;
        }
    }
};
