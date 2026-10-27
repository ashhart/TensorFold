//! mi_situ, mi_gate_add and mi_bias under the contract: elementwise steps at the publisher's bf16 points.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const mi = @import("metal_moss_image_common.zig");

const c = mi.c;
const g = mi.g;
const ks = mi.ks;
const cfg = mi.cfg;
const D = mi.D;
const Range = mi.Range;
const Ends = mi.Ends;
const bf = mi.bf;
const kref = mi.kref;
const kat = mi.kat;

// --- mi_situ: (4 tanh(g / 4) sigmoid(g)) (25 tanh(h / 25)), each op rounded to bf16 ---------------------------------

pub const Situ = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    /// The joint blocks' form: hidden and gate halves of one (rows, 2 x width) projection; else two buffers.
    halves: bool,
    name: []const u8,
    shape: c.Shape = undefined,
    h: []u16 = &.{},
    gv: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// Rows: positions; n: the MLP's width (k unused).
    pub fn spec(name: []const u8, halves: bool) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = mi.noSums,
            .ops = 2 * (mi.exp_ulps + 2 * mi.tanh_ulps + mi.div_ulps) + 4,
            .mids = 8,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 256 }, .{} },
            .served = if (halves)
                &.{ .{ .rows = 4188, .n = cfg.ffn, .k = 1 }, .{ .rows = 4130, .n = cfg.ffn, .k = 1 } }
            else
                &.{ .{ .rows = 63, .n = cfg.ffn, .k = 1 }, .{ .rows = 10, .n = cfg.ffn, .k = 1 } },
        };
    }

    pub fn deinit(t: *Situ) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Situ) void {
        t.a.free(t.h);
        t.a.free(t.gv);
        t.h = &.{};
        t.gv = &.{};
    }
    pub fn kernel(t: *Situ, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.halves), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Situ);

    pub fn prepare(t: *Situ, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = shape.rows * shape.n;
        t.h = try t.a.alloc(u16, n);
        t.gv = try t.a.alloc(u16, n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // gates past +-20 reach the sigmoid's flat ends, hidden values past +-50 the clamp's
        mi.randomBf16(rng, t.h, -80, 80);
        mi.randomBf16(rng, t.gv, -24, 24);
        if (fill == .nonfinite) {
            const at = [_]usize{ 0, 1, 2 };
            for (at, [_]u16{ mi.nan16, mi.inf16, mi.ninf16 }) |p, v| {
                t.h[(p * 7) % n] = v;
                t.gv[(p * 11 + 3) % n] = v;
            }
        }
        if (t.halves) {
            const both = try t.a.alloc(u16, 2 * n);
            defer t.a.free(both);
            for (0..shape.rows) |r| {
                @memcpy(both[2 * r * shape.n ..][0..shape.n], t.h[r * shape.n ..][0..shape.n]);
                @memcpy(both[(2 * r + 1) * shape.n ..][0..shape.n], t.gv[r * shape.n ..][0..shape.n]);
            }
            t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(both));
        } else {
            t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.h));
            t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.gv));
        }
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Situ) !void {
        const p = t.rig.begin();
        const w = t.shape.n;
        const gate = if (t.halves) kat(t.ins[0], w * 2) else kref(t.ins[1]);
        ks.situ(p.e, t.pipe, kref(t.ins[0]), gate, kref(t.out), t.shape.rows, w, if (t.halves) 2 * w else w);
        try t.rig.finish(p);
    }

    pub fn count(t: *Situ) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Situ, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const fifth: f64 = @as(f32, 1.0 / 25.0);
        for (t.h, t.gv, 0..) |hb, gb, i| {
            const h: f64 = c.fromBf16(hb);
            const gv: f64 = c.fromBf16(gb);
            const sig = bf(1 / (1 + @exp(-gv)));
            const left = bf(bf(4 * bf(mi.tanh(bf(gv * 0.25)))) * sig);
            const right = bf(25 * bf(mi.tanh(bf(h * fifth))));
            const sr = Range.of(@exp(-gv)).fp32(mi.exp_ulps).add(Range.of(1)).fp32(0.5).recip().fp32(mi.div_ulps).bf16();
            const lr = Range.of(gv * 0.25).fp32(0.5).bf16().up(mi.tanh).fp32(mi.tanh_ulps).bf16().mul(Range.of(4)).fp32(0.5).bf16().mul(sr).fp32(0.5).bf16();
            const rr = Range.of(h * fifth).fp32(0.5).bf16().up(mi.tanh).fp32(mi.tanh_ulps).bf16().mul(Range.of(25)).fp32(0.5).bf16();
            t.ends.put(want, mag, i, bf(left * right), lr.mul(rr).fp32(0.5).bf16());
        }
    }
    pub fn bounds(t: *Situ, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Situ, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
};

// --- mi_gate_add: x = bf16(x + bf16(g y)), in place --------------------------------------------------------------

pub const GateAdd = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    name: []const u8,
    shape: c.Shape = undefined,
    old: []u16 = &.{},
    y: []u16 = &.{},
    gv: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// Rows: positions; n: the stream's width (k unused).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = mi.noSums,
            .ops = 1,
            .mids = 1,
            .out = .bf16,
            .in_place = true,
            .tiles = .{ .{}, .{ .size = 256 }, .{} },
            .served = &.{ .{ .rows = 4188, .n = cfg.hidden, .k = 1 }, .{ .rows = 63, .n = cfg.hidden, .k = 1 } },
        };
    }

    pub fn deinit(t: *GateAdd) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *GateAdd) void {
        t.a.free(t.old);
        t.a.free(t.y);
        t.a.free(t.gv);
        t.old = &.{};
        t.y = &.{};
        t.gv = &.{};
    }
    pub fn kernel(t: *GateAdd, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = blk: {
        var hooks = g.hooks(GateAdd);
        hooks.overlap = overlap;
        break :blk hooks;
    };
    /// The stream is its own output.
    fn overlap(_: *anyopaque) bool {
        return true;
    }

    pub fn prepare(t: *GateAdd, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = shape.rows * shape.n;
        t.old = try t.a.alloc(u16, n);
        t.y = try t.a.alloc(u16, n);
        t.gv = try t.a.alloc(u16, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        mi.randomBf16(rng, t.old, -8, 8);
        mi.randomBf16(rng, t.y, -4, 4);
        mi.randomBf16(rng, t.gv, -2, 2);
        if (fill == .nonfinite) {
            for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, 0..) |v, i| {
                t.old[(i * 13) % n] = v;
                t.y[(i * 17 + 1) % n] = v;
            }
            if (shape.n > 6) t.gv[5] = mi.nan16;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.y));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.gv));
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *GateAdd) !void {
        // in place: the stream's rows go back before each independent run
        @memcpy(t.rig.bytes(t.out, t.count() * 2), std.mem.sliceAsBytes(t.old));
        const p = t.rig.begin();
        ks.gateAdd(p.e, t.pipe, kref(t.out), kref(t.ins[0]), kref(t.ins[1]), t.shape.rows, t.shape.n);
        try t.rig.finish(p);
    }

    pub fn count(t: *GateAdd) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *GateAdd, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (t.old, t.y, 0..) |xb, yb, i| {
            const x: f64 = c.fromBf16(xb);
            const y: f64 = c.fromBf16(yb);
            const gv: f64 = c.fromBf16(t.gv[i % t.shape.n]);
            const r = Range.of(gv * y).fp32(0.5).bf16().add(Range.of(x)).fp32(0.5).bf16();
            t.ends.put(want, mag, i, bf(x + bf(gv * y)), r);
        }
    }
    pub fn bounds(t: *GateAdd, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *GateAdd, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
};

// --- mi_bias: y = bf16(c + b), a projection's fp32 sums and its bias rounded once --------------------------------

pub const Bias = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    name: []const u8,
    shape: c.Shape = undefined,
    sums: []f32 = &.{},
    b: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// Rows: positions; n: the projection's outputs (k unused).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = mi.noSums,
            .ops = 1,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 256 }, .{} },
            .served = &.{ .{ .rows = 4125, .n = cfg.hidden, .k = 1 }, .{ .rows = 63, .n = 3 * cfg.hidden, .k = 1 }, .{ .rows = 63, .n = cfg.ffn, .k = 1 }, .{ .rows = 63, .n = cfg.hidden, .k = 1 } },
        };
    }

    pub fn deinit(t: *Bias) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Bias) void {
        t.a.free(t.sums);
        t.a.free(t.b);
        t.sums = &.{};
        t.b = &.{};
    }
    pub fn kernel(t: *Bias, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Bias);

    pub fn prepare(t: *Bias, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = shape.rows * shape.n;
        t.sums = try t.a.alloc(f32, n);
        t.b = try t.a.alloc(u16, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.sums) |*v| v.* = (rng.float(f32) * 2 - 1) * 16;
        mi.randomBf16(rng, t.b, -2, 2);
        if (fill == .nonfinite) {
            for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, i| t.sums[(i * 13) % n] = v;
            if (shape.n > 6) t.b[5] = mi.inf16;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.sums));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.b));
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Bias) !void {
        const p = t.rig.begin();
        ks.bias(p.e, t.pipe, kref(t.ins[0]), kref(t.ins[1]), kref(t.out), t.shape.rows, t.shape.n);
        try t.rig.finish(p);
    }

    pub fn count(t: *Bias) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Bias, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (t.sums, 0..) |s, i| {
            const b: f64 = c.fromBf16(t.b[i % t.shape.n]);
            t.ends.put(want, mag, i, bf(s + b), Range.of(s).add(Range.of(b)).fp32(0.5).bf16());
        }
    }
    pub fn bounds(t: *Bias, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Bias, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
};
