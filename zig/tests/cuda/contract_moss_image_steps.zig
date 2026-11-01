//! tf_moss_situ and tf_moss_gate_add under the contract: elementwise steps at the publisher's bf16 points.
const std = @import("std");
const cuda = @import("cuda");
const moss = @import("moss_image_cuda");
const rig_ = @import("contract_rig.zig");
const mi = @import("contract_moss_image_ref.zig");

const c = mi.c;
const g = mi.g;
const Range = mi.Range;
const Ends = mi.Ends;
const bf = mi.bf;

// --- tf_moss_situ: (4 tanh(g / 4) sigmoid(g)) (25 tanh(h / 25)), each op rounded to bf16 ------------------------

pub const Situ = struct {
    rig: rig_.Rig,
    k: moss.Kernels,
    /// The joint blocks' form: hidden and gate halves of one (rows, 2 x width) projection; else two buffers.
    halves: bool,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    h: []u16 = &.{},
    gv: []u16 = &.{},
    ins: [2]u64 = @splat(0),
    out: u64 = 0,
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
                &.{ .{ .rows = 4188, .n = mi.ffn, .k = 1 }, .{ .rows = 4130, .n = mi.ffn, .k = 1 } }
            else
                &.{ .{ .rows = 63, .n = mi.ffn, .k = 1 }, .{ .rows = 10, .n = mi.ffn, .k = 1 } },
        };
    }

    pub fn deinit(t: *Situ) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Situ) void {
        const a = t.rig.gpu.gpa;
        a.free(t.h);
        a.free(t.gv);
        t.h = &.{};
        t.gv = &.{};
    }
    pub fn kernel(t: *Situ) g.Kernel {
        return .{ .spec = spec(t.name, t.halves), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Situ);

    pub fn prepare(t: *Situ, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const n = shape.rows * shape.n;
        t.h = try a.alloc(u16, n);
        t.gv = try a.alloc(u16, n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // gates past +-20 reach the sigmoid's flat ends, hidden values past +-50 the tanh's
        mi.randomBf16(rng, t.h, -80, 80);
        mi.randomBf16(rng, t.gv, -24, 24);
        if (fill == .nonfinite) {
            for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, 0..) |v, p| {
                t.h[(p * 7) % n] = v;
                t.gv[(p * 11 + 3) % n] = v;
            }
        }
        if (t.halves) {
            const both = try a.alloc(u16, 2 * n);
            defer a.free(both);
            for (0..shape.rows) |r| {
                @memcpy(both[2 * r * shape.n ..][0..shape.n], t.h[r * shape.n ..][0..shape.n]);
                @memcpy(both[(2 * r + 1) * shape.n ..][0..shape.n], t.gv[r * shape.n ..][0..shape.n]);
            }
            t.ins[0] = try mi.input(&t.rig, both);
        } else {
            t.ins[0] = try mi.input(&t.rig, t.h);
            t.ins[1] = try mi.input(&t.rig, t.gv);
        }
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Situ) !void {
        const w = t.shape.n;
        const gate = if (t.halves) t.ins[0] + w * 2 else t.ins[1];
        try moss.situ(&t.k, t.rig.s, t.ins[0], gate, t.out, t.shape.rows, w, if (t.halves) 2 * w else w);
        try t.rig.s.synchronize();
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

// --- tf_moss_gate_add: x = bf16(x + bf16(g y)), in place ---------------------------------------------------------

pub const GateAdd = struct {
    rig: rig_.Rig,
    k: moss.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    old: []u16 = &.{},
    y: []u16 = &.{},
    gv: []u16 = &.{},
    ins: [3]u64 = @splat(0),
    out: u64 = 0,
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
            .served = &.{ .{ .rows = 4188, .n = mi.hidden, .k = 1 }, .{ .rows = 63, .n = mi.hidden, .k = 1 } },
        };
    }

    pub fn deinit(t: *GateAdd) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *GateAdd) void {
        const a = t.rig.gpu.gpa;
        a.free(t.old);
        a.free(t.y);
        a.free(t.gv);
        t.old = &.{};
        t.y = &.{};
        t.gv = &.{};
    }
    pub fn kernel(t: *GateAdd) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(GateAdd);

    pub fn prepare(t: *GateAdd, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const n = shape.rows * shape.n;
        t.old = try a.alloc(u16, n);
        t.y = try a.alloc(u16, n);
        t.gv = try a.alloc(u16, shape.n);
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
        t.ins[0] = try mi.input(&t.rig, t.old);
        t.ins[1] = try mi.input(&t.rig, t.y);
        t.ins[2] = try mi.input(&t.rig, t.gv);
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *GateAdd) !void {
        // in place: the stream's rows go back before each independent run
        const bytes = t.count() * 2;
        try (cuda.DeviceBuffer{ .d = t.rig.gpu.d, .ptr = t.out, .len = bytes }).copyFrom(0, t.ins[0], bytes, t.rig.s.handle);
        try moss.gateAdd(&t.k, t.rig.s, t.out, t.ins[1], t.ins[2], t.shape.rows, t.shape.n);
        try t.rig.s.synchronize();
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
