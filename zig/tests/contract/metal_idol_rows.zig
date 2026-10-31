//! id_gate_add, id_embed, id_cond and id_bias32 under the contract: elementwise steps at idol-sdk's fp16 points.
const std = @import("std");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const id = @import("metal_idol_common.zig");

const c = id.c;
const g = id.g;
const cfg = id.cfg;
const H = id.H;
const Range = id.Range;
const Ends = id.Ends;
const hf = id.hf;
const kref = id.kref;

/// The steps' common parts: the outputs' ends, and its bounds and intervals from them.
fn Common(comptime T: type) type {
    return struct {
        pub fn bounds(t: *T, want: []const f64, _: []const f64, allow: []f64) !void {
            t.ends.bounds(want, allow);
        }
        pub fn intervals(t: *T, low: []f64, high: []f64) !void {
            t.ends.intervals(low, high);
        }
    };
}

// --- id_gate_add: x = fp16(x + fp16(g y)), in place ------------------------------------------------------------

pub const GateAdd = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *id.Lib,
    name: []const u8,
    rows: usize = 0,
    old: []f16 = &.{},
    y: []f16 = &.{},
    gv: []f16 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// Rows: positions; n: the stream's width, 1,024 (the launcher's).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = id.noSums,
            .ops = 1,
            .mids = 1,
            .out = .f16,
            .in_place = true,
            .served = &.{ .{ .rows = 12294, .n = H, .k = 1 }, .{ .rows = 8192, .n = H, .k = 1 }, .{ .rows = 4101, .n = H, .k = 1 } },
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
    pub const bounds = Common(GateAdd).bounds;
    pub const intervals = Common(GateAdd).intervals;

    pub fn prepare(t: *GateAdd, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != H) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        const n = shape.rows * H;
        t.old = try t.a.alloc(f16, n);
        t.y = try t.a.alloc(f16, n);
        t.gv = try t.a.alloc(f16, H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        id.randomF16(rng, t.old, -8, 8);
        id.randomF16(rng, t.y, -4, 4);
        id.randomF16(rng, t.gv, -2, 2);
        if (fill == .nonfinite) {
            for ([_]f16{ id.nan16, id.inf16, -id.inf16 }, 0..) |v, i| {
                t.old[(i * 13) % n] = v;
                t.y[(i * 17 + 1) % n] = v;
            }
            t.gv[5] = id.nan16;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.y));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.gv));
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *GateAdd) !void {
        try t.lib.need("id_gate_add");
        // in place: the stream's rows go back before each independent run
        @memcpy(t.rig.bytes(t.out, t.count() * 2), std.mem.sliceAsBytes(t.old));
        const p = t.rig.begin();
        t.lib.k.gateAdd(p.e, kref(t.out), kref(t.ins[0]), kref(t.ins[1]), t.rows);
        try t.rig.finish(p);
    }

    pub fn count(t: *GateAdd) usize {
        return t.rows * H;
    }

    pub fn reference(t: *GateAdd, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (t.old, t.y, 0..) |xb, yb, i| {
            const x: f64 = xb;
            const y: f64 = yb;
            const gv: f64 = t.gv[i % H];
            const r = Range.of(gv * y).fp32(0.5).fp16().add(Range.of(x)).fp32(0.5).fp16();
            t.ends.put(want, mag, i, hf(x + hf(gv * y)), r);
        }
    }
};

// --- id_embed: x = fp16(fp16(z w^T + b) + pe), z [8192, 16] fp32 --------------------------------------------------

/// The 16 inputs' FMA chain.
fn embedDepth(_: c.Shape) usize {
    return cfg.latent_channels + 1;
}

pub const Embed = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *id.Lib,
    name: []const u8,
    stress: bool = false,
    z: []f32 = &.{},
    /// w [1024, 16] then b [1024]
    wb: []f32 = &.{},
    pe: []f16 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    const R = cfg.latent_rows;
    const K = cfg.latent_channels;

    /// Rows: the latent's 8,192, which the launcher fixes (a gate's fewer rows run all of them); k: its 16 channels.
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = embedDepth, .ops = 2, .mids = 1, .out = .f16, .served = &.{.{ .rows = R, .n = H, .k = K }} };
    }

    pub fn deinit(t: *Embed) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Embed) void {
        t.a.free(t.z);
        t.a.free(t.wb);
        t.a.free(t.pe);
        t.z = &.{};
        t.wb = &.{};
        t.pe = &.{};
    }
    pub fn kernel(t: *Embed, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Embed);
    pub const bounds = Common(Embed).bounds;
    pub const intervals = Common(Embed).intervals;

    pub fn prepare(t: *Embed, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != H or shape.k != K) return false;
        t.rig.reset();
        t.free();
        t.stress = fill == .stress;
        t.z = try t.a.alloc(f32, R * K);
        t.wb = try t.a.alloc(f32, H * K + H);
        t.pe = try t.a.alloc(f16, R * H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (t.stress) {
            // products to 2^16 whose halves cancel but for the first: partial sums near 2^19, exact in fp32
            for (0..R) |r| for (0..K / 2) |k| {
                t.z[r * K + k] = id.int(rng, 1, 255);
                t.z[r * K + K / 2 + k] = t.z[r * K + k] + @as(f32, if (k == 0) 1 else 0);
            };
            for (0..H) |o| for (0..K / 2) |k| {
                t.wb[o * K + k] = id.sign(rng) * id.int(rng, 1, 255);
                t.wb[o * K + K / 2 + k] = -t.wb[o * K + k];
            };
            for (t.wb[H * K ..]) |*v| v.* = id.int(rng, -8, 8);
            for (t.pe) |*v| v.* = @floatCast(id.int(rng, -8, 8));
        } else {
            id.randomF32(rng, t.z, -3, 3);
            id.randomF32(rng, t.wb, -0.5, 0.5);
            id.randomF16(rng, t.pe, -1, 1);
        }
        if (fill == .nonfinite) {
            for ([_]f32{ id.nan32, id.inf32, -id.inf32 }, 0..) |v, i| t.z[i * K + 3 * i] = v;
            t.wb[H * K + 9] = id.nan32;
            t.pe[5 * H + 11] = id.inf16;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.z));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.wb[0 .. H * K]));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.wb[H * K ..]));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.pe));
        t.out = try t.rig.output(R * H * 2);
        return true;
    }

    pub fn run(t: *Embed) !void {
        try t.lib.need("id_embed");
        const p = t.rig.begin();
        t.lib.k.embed(p.e, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.ins[3]), kref(t.out));
        try t.rig.finish(p);
    }

    pub fn count(_: *Embed) usize {
        return R * H;
    }

    fn each(t: *Embed, ends: *Ends, want: []f64, mag: []f64, acc: ?c.Accum, order: g.Order) void {
        const gd = c.gamma(@floatFromInt(embedDepth(.{ .rows = 1, .n = 1, .k = 1 })), c.unit(.f32));
        var terms: [K]f64 = undefined;
        for (0..R) |r| for (0..H) |o| {
            var s: f64 = 0;
            var abs: f64 = 0;
            for (&terms, t.z[r * K ..][0..K], t.wb[o * K ..][0..K]) |*v, z, w| {
                v.* = @as(f64, z) * @as(f64, w);
                s += v.*;
                abs += @abs(v.*);
            }
            var sr = Range.of(s).widen(gd * abs);
            if (acc) |x| {
                s = contract.rows.sumIn(x, order, &terms);
                sr = Range.of(s);
            } else if (t.stress) sr = Range.of(s);
            const b: f64 = t.wb[H * K + o];
            const pe: f64 = t.pe[r * H + o];
            const y = sr.add(Range.of(b)).fp32(0.5).fp16().add(Range.of(pe)).fp32(0.5).fp16();
            ends.put(want, mag, r * H + o, hf(hf(s + b) + pe), y);
        };
    }

    pub fn reference(t: *Embed, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        t.each(&t.ends, want, mag, null, .sequential);
    }

    pub fn simulate(t: *Embed, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        var scratch: Ends = .{ .a = t.a };
        defer scratch.deinit();
        try scratch.size(t.count());
        t.each(&scratch, out, mag, acc, order);
    }
};

// --- id_cond: y = fp16((a + ba) + (c + bc)), the two condition projections' fp32 sums and biases -----------------

pub const Cond = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *id.Lib,
    name: []const u8,
    rows: usize = 0,
    /// a, c [rows, 1024], then ba, bc [1024]
    v: []f32 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// Rows: the condition's 4,101; n: its width, 1,024 (the launcher's).
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .none, .depth = id.noSums, .ops = 3, .out = .f16, .served = &.{.{ .rows = cfg.cond_rows, .n = H, .k = 1 }} };
    }

    pub fn deinit(t: *Cond) void {
        t.a.free(t.v);
        t.v = &.{};
        t.ends.deinit();
        t.rig.deinit();
    }
    pub fn kernel(t: *Cond, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Cond);
    pub const bounds = Common(Cond).bounds;
    pub const intervals = Common(Cond).intervals;

    pub fn prepare(t: *Cond, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != H) return false;
        t.rig.reset();
        t.a.free(t.v);
        t.rows = shape.rows;
        const n = shape.rows * H;
        t.v = try t.a.alloc(f32, 2 * n + 2 * H);
        var prng = std.Random.DefaultPrng.init(seed);
        id.randomF32(prng.random(), t.v, -8, 8);
        if (fill == .nonfinite) {
            for ([_]f32{ id.nan32, id.inf32, -id.inf32 }, 0..) |v, i| {
                t.v[(i * 13) % n] = v;
                t.v[n + (i * 17 + 1) % n] = v;
            }
            t.v[2 * n + 5] = id.inf32;
            t.v[2 * n + H + 6] = id.nan32;
        }
        for (0..4) |i| {
            const part = if (i < 2) t.v[i * n ..][0..n] else t.v[2 * n + (i - 2) * H ..][0..H];
            t.ins[i] = try t.rig.input(std.mem.sliceAsBytes(part));
        }
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Cond) !void {
        try t.lib.need("id_cond");
        const p = t.rig.begin();
        t.lib.k.cond(p.e, kref(t.ins[0]), kref(t.ins[2]), kref(t.ins[1]), kref(t.ins[3]), kref(t.out), t.rows);
        try t.rig.finish(p);
    }

    pub fn count(t: *Cond) usize {
        return t.rows * H;
    }

    pub fn reference(t: *Cond, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const n = t.count();
        for (0..n) |i| {
            const av: f64 = t.v[i];
            const cv: f64 = t.v[n + i];
            const ba: f64 = t.v[2 * n + i % H];
            const bc: f64 = t.v[2 * n + H + i % H];
            const r = Range.of(av + ba).fp32(0.5).add(Range.of(cv + bc).fp32(0.5)).fp32(0.5).fp16();
            t.ends.put(want, mag, i, hf((av + ba) + (cv + bc)), r);
        }
    }
};

// --- id_bias32: y += b by column, fp32, in place --------------------------------------------------------------

pub const Bias32 = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *id.Lib,
    name: []const u8,
    shape: c.Shape = undefined,
    old: []f32 = &.{},
    b: []f32 = &.{},
    ins: [1]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// Rows: the latent's; n: the velocity's 16 channels.
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .none, .depth = id.noSums, .ops = 1, .out = .f32, .in_place = true, .served = &.{.{ .rows = cfg.latent_rows, .n = cfg.latent_channels, .k = 1 }} };
    }

    pub fn deinit(t: *Bias32) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Bias32) void {
        t.a.free(t.old);
        t.a.free(t.b);
        t.old = &.{};
        t.b = &.{};
    }
    pub fn kernel(t: *Bias32, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = blk: {
        var hooks = g.hooks(Bias32);
        hooks.overlap = overlap;
        break :blk hooks;
    };
    /// The sums are their own output.
    fn overlap(_: *anyopaque) bool {
        return true;
    }
    pub const bounds = Common(Bias32).bounds;
    pub const intervals = Common(Bias32).intervals;

    pub fn prepare(t: *Bias32, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = shape.rows * shape.n;
        t.old = try t.a.alloc(f32, n);
        t.b = try t.a.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        id.randomF32(rng, t.old, -16, 16);
        id.randomF32(rng, t.b, -2, 2);
        if (fill == .nonfinite) {
            for ([_]f32{ id.nan32, id.inf32, -id.inf32 }, 0..) |v, i| t.old[(i * 13) % n] = v;
            if (shape.n > 6) t.b[5] = id.inf32;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.b));
        t.out = try t.rig.output(n * 4);
        return true;
    }

    pub fn run(t: *Bias32) !void {
        try t.lib.need("id_bias32");
        @memcpy(t.rig.bytes(t.out, t.count() * 4), std.mem.sliceAsBytes(t.old));
        const p = t.rig.begin();
        t.lib.k.bias32(p.e, kref(t.out), kref(t.ins[0]), t.shape.rows, t.shape.n);
        try t.rig.finish(p);
    }

    pub fn count(t: *Bias32) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Bias32, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (t.old, 0..) |y, i| {
            const b: f64 = t.b[i % t.shape.n];
            t.ends.put(want, mag, i, y + b, Range.of(y + b).fp32(0.5));
        }
    }
};
