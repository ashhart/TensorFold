//! MiniMax-M3.1-Flash's glue under the contract: embedding, SwiGLU-OAI, argmax and the tower's copies (adds: _rows).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const rows_k = @import("metal_minimax_glue_rows.zig");

const c = contract.spec;
const g = contract.gates;
const Texts = @import("metal_minimax.zig").Texts;

pub const source = sources.minimax_m3_glue;

/// MiniMax-M3.1-Flash's served widths (its config): the residual, the experts' and the dense layers' SwiGLU, the tower.
pub const hidden = 6144;
pub const moe_inter = 3072;
pub const dense_inter = 12288;
pub const topk = 4;
pub const vocab = 200064;
pub const tower = 1280;

/// glue.metal's argument block (forward.zig's Args).
pub const MArgs = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, dim: u32 = 0, pos: u32 = 0, stride: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 = 0, alpha: f32 = 0, limit: f32 = 0, scale: f32 = 0 };

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// A value as fp32 then bf16 rounds it (to nearest even, twice): an fp32 step on exact inputs, then the bf16 point.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// A value's spread through fp32 steps and bf16 rounding points.
pub const Range = struct {
    lo: f64,
    hi: f64,
    pub fn point(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    /// Each end moved by `e` either way.
    pub fn widen(a: Range, e: f64) Range {
        return .{ .lo = a.lo - e, .hi = a.hi + e };
    }
    pub fn bf16(a: Range) Range {
        return .{ .lo = bf(a.lo), .hi = bf(a.hi) };
    }
    /// Times an exact value of either sign.
    pub fn times(a: Range, v: f64) Range {
        return .{ .lo = @min(a.lo * v, a.hi * v), .hi = @max(a.lo * v, a.hi * v) };
    }
};

/// The allowance a range gives about `want`: the farther end.
pub fn spread(want: f64, r: Range) f64 {
    return @max(@abs(want - r.lo), @abs(r.hi - want));
}

/// One of glue.metal's kernels, compiled as model.zig compiles it (MLX's language, precise math functions).
pub fn pipeline(device: mtl.Device, name: []const u8) !mtl.Pipeline {
    const lib = try mtl.Library.fromSource(device, source, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, name, false);
}

pub const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };

// --- mm_embed: an 8-bit affine row, bf16(scale * q + bias) held in fp32 -------------------------------------------

/// Table rows the test's ids index (the gates need distinct rows, not a 200k vocabulary).
const table = 48;
const ebits = 8;
const egroup = 64;

pub const Embed = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    ids: []u32 = &.{},
    words: []u32 = &.{},
    s: []u16 = &.{},
    b: []u16 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: tokens; n: the hidden width (k unused). The embedding is the checkpoint's 8-bit groups of 64.
    pub fn spec() c.Spec {
        return .{
            .name = "metal minimax_m3 glue mm_embed q8 g64",
            .accum = .none,
            .depth = noSums,
            .out = .f32,
            .tiles = .{ .{}, .{ .size = 256, .step = 64, .min = 64 }, .{} },
            .served = &.{ .{ .rows = 32, .n = hidden, .k = 1 }, .{ .rows = 4096, .n = hidden, .k = 1 }, .{ .rows = 1, .n = hidden, .k = 1 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Embed {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(device, "mm_embed") };
    }
    pub fn deinit(t: *Embed) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Embed) void {
        inline for (.{ "ids", "words", "s", "b" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Embed) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Embed);

    pub fn prepare(t: *Embed, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % egroup != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.ids = try t.a.alloc(u32, shape.rows);
        for (t.ids, 0..) |*v, r| v.* = if (r < 4) @intCast(r) else rng.uintLessThan(u32, table);
        t.words = try t.a.alloc(u32, table * n / 4);
        for (t.words) |*v| v.* = rng.int(u32);
        t.s = try t.a.alloc(u16, table * n / egroup);
        t.b = try t.a.alloc(u16, table * n / egroup);
        // MLX's affine groups: a positive scale, the bias the group's least value
        for (t.s, t.b) |*s, *b| {
            const sc = 0.001 + rng.float(f32) * 0.02;
            s.* = c.toBf16(sc);
            b.* = c.toBf16(-sc * (rng.float(f32) * 255));
        }
        if (fill == .nonfinite) {
            const gpr = n / egroup;
            t.s[0 * gpr] = c.toBf16(planted[0]);
            t.b[1 * gpr] = c.toBf16(planted[1]);
            t.b[2 * gpr] = c.toBf16(planted[2]);
            t.s[3 * gpr] = c.toBf16(planted[1]);
            if (gpr > 1) t.b[3 * gpr + 1] = c.toBf16(planted[2]);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.ids));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.words));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.s));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.b));
        t.out = try t.rig.output(shape.rows * n * 4);
        return true;
    }

    pub fn run(t: *Embed) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setValue(MArgs{ .rows = @intCast(t.shape.rows), .width = @intCast(t.shape.n), .bits = ebits, .group = egroup }, 4);
        p.e.setBuffer(t.out.buf, t.out.off, 5);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Embed) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Element (r, j)'s scale times its code, and its bias.
    fn parts(t: *Embed, r: usize, j: usize) [2]f64 {
        const n = t.shape.n;
        const row = t.ids[r];
        const q: f64 = @floatFromInt((t.words[(row * n + j) / 4] >> @intCast((j % 4) * 8)) & 0xff);
        const gi = row * (n / egroup) + j / egroup;
        return .{ @as(f64, c.fromBf16(t.s[gi])) * q, c.fromBf16(t.b[gi]) };
    }

    pub fn reference(t: *Embed, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            const p = t.parts(i / t.shape.n, i % t.shape.n);
            w.* = bf(p[0] + p[1]);
            m.* = 0;
        }
    }

    /// The scale-and-add in fp32 (fused or not, the compiler's choice): two roundings at most, then the bf16 point.
    pub fn intervals(t: *Embed, low: []f64, high: []f64) !void {
        for (low, high, 0..) |*l, *h, i| {
            const p = t.parts(i / t.shape.n, i % t.shape.n);
            const e = c.unit(.f32) * (@abs(p[0]) + 2 * @abs(p[0] + p[1])) + 0x1p-149;
            const r = Range.point(p[0] + p[1]).widen(e).bf16();
            l.* = r.lo;
            h.* = r.hi;
        }
    }

    pub fn bounds(t: *Embed, want: []const f64, _: []const f64, allow: []f64) !void {
        const buf = try t.a.alloc(f64, 2 * t.count());
        defer t.a.free(buf);
        try t.intervals(buf[0..t.count()], buf[t.count()..]);
        for (allow, want, buf[0..t.count()], buf[t.count()..]) |*l, w, lo, hi| l.* = if (std.math.isFinite(w)) spread(w, .{ .lo = lo, .hi = hi }) else 0;
    }
};

// --- mm_swiglu: SwiGLU-OAI in bf16 steps, (clamp(up) + 1) * g * sigmoid(g * alpha), g = min(gate, limit) -----------

const alpha: f32 = 1.702;
const limit: f32 = 7.0;
const u32_ulp = 0x1p-24;

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// mm_sigmoid's fp32 error: 1 / (1 + exp(-|x|)) within 8 ulps; below zero exp(-|x|) (4) times that, rounded (1).
fn sigmoidRange(x: f64) Range {
    const s = sigmoid(x);
    const e = @as(f64, if (x >= 0) 8 else 13) * u32_ulp * s;
    return .{ .lo = @max(0, s - e), .hi = s + e };
}

/// torch's clamp: NaN stays NaN.
fn clampNan(v: f64, lo: f64, hi: f64) f64 {
    return if (std.math.isNan(v)) v else std.math.clamp(v, lo, hi);
}

pub const Swiglu = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    gu: []u16 = &.{},
    in: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: slots or tokens; n: the SwiGLU width (gate|up rows twice as wide).
    pub fn spec() c.Spec {
        return .{
            .name = "metal minimax_m3 glue mm_swiglu (SwiGLU-OAI, alpha 1.702, limit 7)",
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            // a window's 32 x 4 expert slots, a prompt chunk's 4,096 x 4, the shared expert's and dense layers' rows
            .served = &.{ .{ .rows = 128, .n = moe_inter, .k = 1 }, .{ .rows = 16384, .n = moe_inter, .k = 1 }, .{ .rows = 32, .n = dense_inter, .k = 1 }, .{ .rows = 1, .n = moe_inter, .k = 1 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Swiglu {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(device, "mm_swiglu") };
    }
    pub fn deinit(t: *Swiglu) void {
        t.a.free(t.gu);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Swiglu) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Swiglu);

    pub fn prepare(t: *Swiglu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.gu);
        t.shape = shape;
        const n = shape.n;
        t.gu = try t.a.alloc(u16, shape.rows * 2 * n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // gates past the limit and deep into the sigmoid's tail, ups past the clamp on both sides
        for (0..shape.rows) |r| {
            for (t.gu[r * 2 * n ..][0..n]) |*v| v.* = c.toBf16(rng.float(f32) * 48 - 32);
            for (t.gu[r * 2 * n + n ..][0..n]) |*v| v.* = c.toBf16(rng.float(f32) * 20 - 10);
        }
        if (fill == .nonfinite) for (0..@min(shape.rows, 6)) |r| {
            const at = r * 2 * n + (if (r < 3) 0 else n) + (r * 7) % n;
            t.gu[at] = c.toBf16(planted[r % 3]);
        };
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.gu));
        t.out = try t.rig.output(shape.rows * n * 2);
        return true;
    }

    pub fn run(t: *Swiglu) !void {
        const n: u32 = @intCast(t.shape.n);
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.in.buf, t.in.off, 0);
        p.e.setValue(MArgs{ .rows = @intCast(t.shape.rows), .width = n, .stride = 2 * n, .alpha = alpha, .limit = limit }, 1);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Swiglu) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Output i's clamped gate, bf16(alpha g), and bf16(clamp(up) + 1).
    fn parts(t: *Swiglu, i: usize) [3]f64 {
        const n = t.shape.n;
        const r = i / n;
        const j = i % n;
        const gate: f64 = c.fromBf16(t.gu[r * 2 * n + j]);
        const up: f64 = c.fromBf16(t.gu[r * 2 * n + n + j]);
        const gl = if (std.math.isNan(gate)) gate else @min(gate, limit);
        return .{ gl, bf(gl * alpha), bf(clampNan(up, -limit, limit) + 1) };
    }

    pub fn reference(t: *Swiglu, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            const p = t.parts(i);
            w.* = bf(p[2] * bf(p[0] * bf(sigmoid(p[1]))));
            m.* = 0;
        }
    }

    pub fn intervals(t: *Swiglu, low: []f64, high: []f64) !void {
        for (low, high, 0..) |*l, *h, i| {
            const p = t.parts(i);
            const r = sigmoidRange(p[1]).bf16().times(p[0]).bf16().times(p[2]).bf16();
            l.* = r.lo;
            h.* = r.hi;
        }
    }

    pub fn bounds(t: *Swiglu, want: []const f64, _: []const f64, allow: []f64) !void {
        const buf = try t.a.alloc(f64, 2 * t.count());
        defer t.a.free(buf);
        try t.intervals(buf[0..t.count()], buf[t.count()..]);
        for (allow, want, buf[0..t.count()], buf[t.count()..]) |*l, w, lo, hi| l.* = if (std.math.isFinite(w)) spread(w, .{ .lo = lo, .hi = hi }) else 0;
    }
};

// --- mm_widen and mm_narrow: the tower's copies between bf16 and fp32 rows -----------------------------------------

pub const Copy = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    narrow: bool,
    total: usize = 0,
    x: []u8 = &.{},
    in: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: the tower's patches (2,304 at most) or an image's rows; n: their width. The kernel sees rows * n values.
    pub fn spec(narrow: bool) c.Spec {
        return .{
            .name = if (narrow) "metal minimax_m3 glue mm_narrow fp32 to bf16" else "metal minimax_m3 glue mm_widen bf16 to fp32",
            .accum = .none,
            .depth = noSums,
            .out = if (narrow) .bf16 else .f32,
            .tiles = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            .served = &.{ .{ .rows = 16, .n = tower, .k = 1 }, .{ .rows = 2304, .n = tower, .k = 1 }, .{ .rows = 576, .n = hidden, .k = 1 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, narrow: bool) !Copy {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(device, if (narrow) "mm_narrow" else "mm_widen"), .narrow = narrow };
    }
    pub fn deinit(t: *Copy) void {
        t.a.free(t.x);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Copy) g.Kernel {
        return .{ .spec = spec(t.narrow), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Copy);

    pub fn prepare(t: *Copy, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.total = shape.rows * shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (t.narrow) {
            const x = try t.a.alloc(f32, t.total);
            // full fp32 mantissas and round-to-even halves; normal values only (Apple GPUs flush fp32 subnormals)
            for (x, 0..) |*v, i| v.* = if (i % 5 == 0) c.fromBf16((rng.int(u16) & 0x477f) | 0x0080) * (1 + 0x1p-8) else (rng.float(f32) * 2 - 1) * 64;
            if (fill == .nonfinite) for (planted, 0..) |v, i| {
                x[i * 3 % t.total] = v;
            };
            t.x = std.mem.sliceAsBytes(x);
        } else {
            const x = try t.a.alloc(u16, t.total);
            for (x) |*v| v.* = rng.int(u16);
            for (x) |*v| if (std.math.isNan(c.fromBf16(v.*)) or std.math.isInf(c.fromBf16(v.*))) {
                v.* = 0x3f80;
            };
            if (fill == .nonfinite) for (planted, 0..) |v, i| {
                x[i * 3 % t.total] = c.toBf16(v);
            };
            t.x = std.mem.sliceAsBytes(x);
        }
        t.in = try t.rig.input(t.x);
        t.out = try t.rig.output(t.total * @as(usize, if (t.narrow) 2 else 4));
        return true;
    }

    pub fn run(t: *Copy) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.in.buf, t.in.off, 0);
        p.e.setBuffer(t.out.buf, t.out.off, 1);
        p.e.setValue(@as(u32, @intCast(t.total)), 2);
        p.e.dispatchThreads(mtl.Size.of(t.total, 1, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Copy) usize {
        return t.total;
    }

    pub fn reference(t: *Copy, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            w.* = if (t.narrow) bf(std.mem.bytesAsSlice(f32, t.x)[i]) else c.fromBf16(std.mem.bytesAsSlice(u16, t.x)[i]);
            m.* = 0;
        }
    }
};

// --- mm_argmax: the greedy draw over a bf16 logits row, ties to the lower id ---------------------------------------

/// mm_better: NaN beats any number (the lower id among NaNs), else the larger value, ties to the lower id.
fn better(v: f32, i: usize, best: f32, at: usize) bool {
    if (std.math.isNan(v)) return !std.math.isNan(best) or i < at;
    if (std.math.isNan(best)) return false;
    return v > best or (v == best and i < at);
}

pub const Argmax = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    logits: []u16 = &.{},
    in: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [160]u8 = undefined,

    /// n: the vocabulary (rows and k unused: one row a launch).
    pub fn spec() c.Spec {
        return .{
            .name = "metal minimax_m3 glue mm_argmax",
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .nan = .{ .stated = "a NaN logit wins (the lowest id among NaNs), as torch.argmax takes NaN as the largest; else ties to the lower id" },
            .tiles = .{ .{}, .{ .size = 1024, .step = 1, .min = 1 }, .{} },
            .served = &.{.{ .rows = 1, .n = vocab, .k = 1 }},
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Argmax {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(device, "mm_argmax") };
    }
    pub fn deinit(t: *Argmax) void {
        t.a.free(t.logits);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Argmax) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Argmax);

    pub fn prepare(t: *Argmax, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.logits);
        t.logits = try t.a.alloc(u16, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // few distinct values, so the largest is tied many times over
        for (t.logits) |*v| v.* = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) / 4 - 8);
        if (fill == .nonfinite) {
            const n = shape.n;
            t.logits[n / 3] = c.toBf16(planted[1]);
            t.logits[n / 2] = c.toBf16(planted[0]);
            t.logits[n - 1] = c.toBf16(planted[0]);
            t.logits[0] = c.toBf16(planted[2]);
        }
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.logits));
        t.out = try t.rig.output(4);
        return true;
    }

    pub fn run(t: *Argmax) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.in.buf, t.in.off, 0);
        p.e.setBuffer(t.out.buf, t.out.off, 1);
        p.e.setValue(@as(u32, @intCast(t.logits.len)), 2);
        p.e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(_: *Argmax) usize {
        return 1;
    }
    pub fn reference(_: *Argmax, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    pub fn check(t: *Argmax, _: g.Gate) !?[]const u8 {
        var best: f32 = -std.math.inf(f32);
        var at: usize = std.math.maxInt(u32);
        for (t.logits, 0..) |v, i| if (better(c.fromBf16(v), i, best, at)) {
            best = c.fromBf16(v);
            at = i;
        };
        const got = std.mem.bytesToValue(u32, t.rig.bytes(t.out, 4));
        if (got != at) return try std.fmt.bufPrint(&t.msg, "drew id {d} where the rule picks {d} of {d}", .{ got, at, t.logits.len });
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    embed: ?*Embed = null,
    swiglu: ?*Swiglu = null,
    copies: std.ArrayList(*Copy) = .empty,
    argmax: ?*Argmax = null,
    rows: rows_k.Suite = undefined,

    fn wanted(only: ?[]const u8, name: []const u8) bool {
        const o = only orelse return true;
        return std.mem.indexOf(u8, name, o) != null;
    }

    fn one(s: *Suite, comptime T: type, only: ?[]const u8) !?*T {
        if (!wanted(only, T.spec().name)) return null;
        const x = try s.a.create(T);
        x.* = try T.init(s.a, s.device, s.queue);
        try s.kernels.append(s.a, x.kernel());
        return x;
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        s.embed = try s.one(Embed, only);
        s.swiglu = try s.one(Swiglu, only);
        for ([_]bool{ false, true }) |narrow| {
            if (!wanted(only, Copy.spec(narrow).name)) continue;
            const x = try s.a.create(Copy);
            x.* = try Copy.init(s.a, s.device, s.queue, narrow);
            try s.copies.append(s.a, x);
            try s.kernels.append(s.a, x.kernel());
        }
        s.argmax = try s.one(Argmax, only);
        s.rows = .{ .a = s.a, .device = s.device, .queue = s.queue, .kernels = s.kernels };
        try s.rows.collect(only);
    }
    pub fn deinit(s: *Suite) void {
        if (s.embed) |x| x.deinit();
        if (s.swiglu) |x| x.deinit();
        for (s.copies.items) |x| x.deinit();
        s.copies.deinit(s.a);
        if (s.argmax) |x| x.deinit();
        s.rows.deinit();
    }
};

/// Gate 5's texts for this group's registrations (`minimax_m3 glue ...`): glue.metal as is, on every chip.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?Texts {
    if (!std.mem.startsWith(u8, name, "minimax_m3 glue ")) return null;
    return .{ .here = source, .pre_m5 = source };
}
