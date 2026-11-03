//! Unlimited-OCR-2's rows under the contract: the tower's adds, bias, norms and GELUs, the decoder's norms and rotary.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const tf = @import("tensorfold");
const rig_ = @import("rig.zig");
const uocr = @import("metal_uocr.zig");

const c = contract.spec;
const g = contract.gates;
const bf = uocr.bf;
const cfg = tf.uocr.config;

/// fp32's unit roundoff: one correctly rounded fp32 step moves a value by at most this share of it.
const U: f64 = 0x1p-24;
/// x / y within MSL's fast-math 2.5 ulps (5 U), an upper bound for the safe math the family compiles with.
const div_u: f64 = 5;
/// Metal's precise exp, sqrt and rsqrt within 4 ulps (8 U of the value).
const fn_u: f64 = 8;
/// 1 / (1 + exp(-z)) and z / (1 + exp(-z)): exp's 8 U, the add's 1 and the division's 5, rounded up.
const sig_u: f64 = 16;
/// precise::sin and cos within 4 ulps of 1: an absolute spread (their values are at most 1).
const trig_abs: f64 = 0x1p-21;
/// 1 + erf(x * f32(1 / sqrt 2)): uv_erf's faithful value, the argument's rounding and the add, each under 2^-24.
const erf_abs: f64 = 0x1p-22;

/// Each kernel by its function's name.
const Kind = enum { uv_add, uv_bias, uv_layer_norm, uv_layer_norm2d, uv_gelu, uv_quick_gelu, ud_rms_norm, ud_rope, ud_router_norm, ud_silu_mul, ud_add };

const Case = struct { kind: Kind, name: []const u8, eps: f32 = 0, drop: bool = false };

const cases = [_]Case{
    .{ .kind = .uv_add, .name = "metal uocr rows uv_add (residual in place)" },
    .{ .kind = .uv_bias, .name = "metal uocr rows uv_bias" },
    .{ .kind = .uv_layer_norm, .name = "metal uocr rows uv_layer_norm SAM (eps 1e-6)", .eps = cfg.sam_eps },
    .{ .kind = .uv_layer_norm, .name = "metal uocr rows uv_layer_norm CLIP (eps 1e-5)", .eps = cfg.clip_eps },
    .{ .kind = .uv_layer_norm2d, .name = "metal uocr rows uv_layer_norm2d neck (eps 1e-6)", .eps = cfg.sam_eps },
    .{ .kind = .uv_gelu, .name = "metal uocr rows uv_gelu (erf, in place)" },
    .{ .kind = .uv_quick_gelu, .name = "metal uocr rows uv_quick_gelu (in place)" },
    .{ .kind = .ud_rms_norm, .name = "metal uocr rows ud_rms_norm", .eps = cfg.eps },
    .{ .kind = .ud_rope, .name = "metal uocr rows ud_rope page2d (in place)" },
    .{ .kind = .ud_router_norm, .name = "metal uocr rows ud_router_norm (Gemma norm, clamp, page2d)" },
    .{ .kind = .ud_silu_mul, .name = "metal uocr rows ud_silu_mul" },
    .{ .kind = .ud_add, .name = "metal uocr rows ud_add (residual in place)" },
    .{ .kind = .uv_layer_norm2d, .name = "metal uocr rows uv_layer_norm2d control: bf16((x - u) / den) left in fp32", .eps = cfg.sam_eps, .drop = true },
    .{ .kind = .uv_quick_gelu, .name = "metal uocr rows uv_quick_gelu control: bf16(sigmoid) left in fp32", .drop = true },
    .{ .kind = .ud_rms_norm, .name = "metal uocr rows ud_rms_norm control: bf16(x rsqrt) left in fp32", .eps = cfg.eps, .drop = true },
    .{ .kind = .ud_rope, .name = "metal uocr rows ud_rope control: bf16 products left in fp32", .drop = true },
    .{ .kind = .ud_router_norm, .name = "metal uocr rows ud_router_norm control: the bf16 normed row left in fp32", .drop = true },
    .{ .kind = .ud_silu_mul, .name = "metal uocr rows ud_silu_mul control: bf16(silu) left in fp32", .drop = true },
};

fn inPlace(k: Kind) bool {
    return k == .uv_add or k == .ud_add or k == .uv_gelu or k == .uv_quick_gelu or k == .ud_rope;
}

fn sums(k: Kind) bool {
    return k == .uv_layer_norm or k == .uv_layer_norm2d or k == .ud_rms_norm or k == .ud_router_norm;
}

/// A row's sums: each of 256 threads' strided chain, the simd_sum tree, then the 8 simdgroups' partials.
fn rowDepth(s: c.Shape) usize {
    return (s.n + 255) / 256 + 5 + 8;
}

/// ud_router_norm's: one square a thread, the simd_sum tree, then 4 partials.
fn routerDepth(_: c.Shape) usize {
    return 1 + 5 + 4;
}

const norm_why = "bf16 outputs after the sums' rsqrt: a bf16 sum's error is a few output ulps, not provably 100x";

fn spec(v: Case) c.Spec {
    // the engine's shapes, an odd prompt chunk or the CLIP rows first (their tails fall off a 256-thread group)
    const served: []const c.Shape = switch (v.kind) {
        .uv_add => &.{ .{ .rows = 257, .n = cfg.clip_width, .k = 1 }, .{ .rows = 4096, .n = cfg.sam_width, .k = 1 }, .{ .rows = 1600, .n = cfg.sam_width, .k = 1 }, .{ .rows = 101, .n = cfg.clip_width, .k = 1 } },
        .uv_bias => &.{ .{ .rows = 257, .n = 3 * cfg.clip_width, .k = 1 }, .{ .rows = 4096, .n = cfg.sam_width, .k = 1 }, .{ .rows = 4096, .n = cfg.sam_ffn, .k = 1 }, .{ .rows = 257, .n = cfg.clip_ffn, .k = 1 }, .{ .rows = 100, .n = cfg.hidden, .k = 1 } },
        .uv_layer_norm => if (v.eps == cfg.clip_eps) &.{ .{ .rows = 257, .n = cfg.clip_width, .k = 1 }, .{ .rows = 101, .n = cfg.clip_width, .k = 1 } } else &.{ .{ .rows = 1600, .n = cfg.sam_width, .k = 1 }, .{ .rows = 4096, .n = cfg.sam_width, .k = 1 } },
        .uv_layer_norm2d => &.{ .{ .rows = 1600, .n = cfg.neck_width, .k = 1 }, .{ .rows = 4096, .n = cfg.neck_width, .k = 1 } },
        .uv_gelu => &.{ .{ .rows = 1600, .n = cfg.sam_ffn, .k = 1 }, .{ .rows = 4096, .n = cfg.sam_ffn, .k = 1 } },
        .uv_quick_gelu => &.{ .{ .rows = 257, .n = cfg.clip_ffn, .k = 1 }, .{ .rows = 101, .n = cfg.clip_ffn, .k = 1 } },
        .ud_rms_norm, .ud_add, .ud_rope => &.{ .{ .rows = 37, .n = cfg.hidden, .k = 1 }, .{ .rows = 2048, .n = cfg.hidden, .k = 1 }, .{ .rows = 1, .n = cfg.hidden, .k = 1 } },
        .ud_router_norm => &.{ .{ .rows = 37, .n = cfg.router_dim, .k = 1 }, .{ .rows = 2048, .n = cfg.router_dim, .k = 1 }, .{ .rows = 1, .n = cfg.router_dim, .k = 1 } },
        .ud_silu_mul => &.{ .{ .rows = 37, .n = cfg.dense_ffn, .k = 1 }, .{ .rows = 2048, .n = cfg.dense_ffn, .k = 1 }, .{ .rows = 1, .n = cfg.shared_ffn, .k = 1 }, .{ .rows = 2048, .n = cfg.shared_ffn, .k = 1 }, .{ .rows = 300, .n = cfg.expert_ffn, .k = 1 } },
    };
    var s: c.Spec = .{ .name = v.name, .accum = .none, .depth = uocr.noSums, .out = .bf16, .in_place = inPlace(v.kind), .served = served };
    s.tiles[1] = switch (v.kind) {
        .uv_bias => .{ .size = 256, .step = 64, .min = 64 },
        .ud_rope => .{ .size = 128, .step = 128, .min = 128 },
        .ud_router_norm => .{},
        else => .{ .size = 256, .step = 1, .min = 1 },
    };
    if (sums(v.kind)) {
        s.accum = .f32;
        s.depth = if (v.kind == .ud_router_norm) routerDepth else rowDepth;
        s.sums = .{ .bf16 = 0, .why = norm_why };
    }
    if (v.kind == .ud_router_norm) {
        s.out = .f32;
        s.nan = .{ .stated = "a non-finite input reads as 0 and a non-finite result is written as 0 (dash_attention.py's nan_to_num)" };
    }
    return s;
}

/// A value's spread through fp32 steps and bf16 rounding points.
const R = struct {
    lo: f64,
    hi: f64,
    fn pt(v: f64) R {
        return .{ .lo = v, .hi = v };
    }
    fn wid(a: R, e: f64) R {
        return .{ .lo = a.lo - e, .hi = a.hi + e };
    }
    /// Widened by k fp32 units of its larger end (k = 1: one correctly rounded step).
    fn rel(a: R, k: f64) R {
        return a.wid(k * U * @max(@abs(a.lo), @abs(a.hi)));
    }
    fn bf16(a: R) R {
        return .{ .lo = bf(a.lo), .hi = bf(a.hi) };
    }
    fn add(a: R, b: R) R {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    fn neg(a: R) R {
        return .{ .lo = -a.hi, .hi = -a.lo };
    }
    fn mul(a: R, b: R) R {
        const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    fn times(a: R, v: f64) R {
        return a.mul(pt(v));
    }
    fn sq(a: R) R {
        if (a.lo >= 0) return .{ .lo = a.lo * a.lo, .hi = a.hi * a.hi };
        if (a.hi <= 0) return .{ .lo = a.hi * a.hi, .hi = a.lo * a.lo };
        return .{ .lo = 0, .hi = @max(a.lo * a.lo, a.hi * a.hi) };
    }
    fn rsqrt(a: R) R {
        return .{ .lo = 1 / @sqrt(a.hi), .hi = 1 / @sqrt(a.lo) };
    }
    fn clamp(a: R, m: f64) R {
        return .{ .lo = std.math.clamp(a.lo, -m, m), .hi = std.math.clamp(a.hi, -m, m) };
    }
};

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// torch.clamp: NaN stays NaN.
fn clampNan(v: f64, m: f64) f64 {
    return if (std.math.isNan(v)) v else std.math.clamp(v, -m, m);
}

/// erf in float64 to about 1e-15: its Maclaurin series below 3, erfc's continued fraction (60 terms) above, 1 past 6.
fn erf(z: f64) f64 {
    if (std.math.isNan(z)) return z;
    if (z < 0) return -erf(-z);
    if (z > 6) return 1;
    if (z < 3) {
        var term = z;
        var s: f64 = z;
        var n: f64 = 1;
        while (n < 200) : (n += 1) {
            term *= -z * z / n;
            const t = term / (2 * n + 1);
            s += t;
            if (@abs(t) < 1e-18 * @abs(s)) break;
        }
        return s * 2 / @sqrt(std.math.pi);
    }
    var f = z;
    var k: f64 = 60;
    while (k >= 1) : (k -= 1) f = z + (k / 2) / f;
    return 1 - @exp(-z * z) / (@sqrt(std.math.pi) * f);
}

/// A simulated accumulator for gate 4, or none: float64's own sums.
const Sim = ?struct { acc: c.Accum, order: g.Order };

fn sumOf(sim: Sim, terms: []const f64) f64 {
    if (sim) |s| return contract.rows.sumIn(s.acc, s.order, terms);
    var s: f64 = 0;
    for (terms) |v| s += v;
    return s;
}

fn uni(r: std.Random, lo: f32, hi: f32) u16 {
    return c.toBf16(lo + r.float(f32) * (hi - lo));
}

fn val(h: u16) f64 {
    return c.fromBf16(h);
}

const Norm = extern struct { width: u32, eps: f32 };
const Rope = extern struct { heads: u32, stride: u32 };

pub const Rows = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Case,
    shape: c.Shape = undefined,
    /// The first bf16 operand (a, x or the gate), the second (b, the weight or up), the layer norm's bias.
    x: []u16 = &.{},
    y: []u16 = &.{},
    z: []u16 = &.{},
    /// uv_bias's fp32 sums; the rows' token positions then their pages.
    f: []f32 = &.{},
    pos: []u32 = &.{},
    inv: [cfg.rope_token_pairs]f32 = cfg.invTok(),
    /// The buffers in the kernel's order (the in-place output where it reads its operand), and how many.
    refs: [6]rig_.Ref = undefined,
    nrefs: usize = 0,
    io: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Case) !Rows {
        const name = @tagName(v.kind);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try uocr.pipeline(device, if (name[1] == 'v') .vision else .decoder, name), .v = v };
    }
    pub fn deinit(t: *Rows) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Rows) void {
        inline for (.{ "x", "y", "z", "f", "pos" }) |name| {
            t.a.free(@field(t, name));
            @field(t, name) = &.{};
        }
    }
    pub fn kernel(t: *Rows) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = if (t.v.kind == .ud_router_norm) &vt_router else &vt, .control = if (t.v.drop) .shapes else null };
    }
    const vt = blk: {
        var h = g.hooks(Rows);
        h.overlap = overlap;
        h.check = null;
        break :blk h;
    };
    const vt_router = g.hooks(Rows);
    /// The in-place kernels write over the input they read (the engine binds one buffer as both).
    fn overlap(ctx: *anyopaque) bool {
        const t: *Rows = @ptrCast(@alignCast(ctx));
        return inPlace(t.v.kind);
    }

    pub fn prepare(t: *Rows, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const k = t.v.kind;
        if (shape.rows == 0 or shape.n == 0 or (k == .ud_rope and shape.n % 128 != 0) or (k == .ud_router_norm and shape.n != 128)) return false;
        if (fill == .stress and shape.n % 2 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const rows = shape.rows;
        const n = shape.n;
        const total = rows * n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const ln = k == .uv_layer_norm or k == .uv_layer_norm2d;
        const pairs = k == .uv_add or k == .ud_add or k == .ud_silu_mul;
        const rotary = k == .ud_rope or k == .ud_router_norm;
        t.x = try t.a.alloc(u16, if (k == .uv_bias) 0 else total);
        t.y = try t.a.alloc(u16, if (pairs) total else if (sums(k) or k == .uv_bias) n else 0);
        t.z = try t.a.alloc(u16, if (ln) n else 0);
        t.f = try t.a.alloc(f32, if (k == .uv_bias) total else 0);
        t.pos = try t.a.alloc(u32, if (rotary) 2 * rows else 0);
        const span: [2]f32 = switch (k) {
            .uv_layer_norm, .uv_layer_norm2d => .{ -2.25, 3.75 },
            .uv_gelu => .{ -6, 6 },
            .uv_quick_gelu => .{ -8, 8 },
            .ud_silu_mul => .{ -20, 12 },
            else => .{ -4, 4 },
        };
        for (t.x) |*v| v.* = uni(rng, span[0], span[1]);
        for (t.y) |*v| v.* = if (sums(k)) uni(rng, -1.5, 1.5) else uni(rng, -2, 2);
        for (t.z) |*v| v.* = uni(rng, -0.5, 0.5);
        for (t.f) |*v| v.* = (rng.float(f32) * 2 - 1) * 8;
        for (t.pos, 0..) |*v, i| v.* = rng.uintLessThan(u32, if (i < rows) cfg.max_context else 64);
        // a router weight past the clamp: (1 + w) times the normed value leaves +-1e4
        if (k == .ud_router_norm and n > 5) t.y[5] = c.toBf16(2000);
        if (fill == .stress) {
            // integers whose squares sum exactly in fp32 and overflow fp16 in any order; mirrored halves (mean 0)
            for (t.x) |*v| v.* = c.toBf16(@as(f32, @floatFromInt(24 + rng.uintLessThan(u32, 17))) * (if (rng.boolean()) @as(f32, 1) else -1));
            if (ln) for (0..rows) |r| for (0..n / 2) |j| {
                t.x[r * n + n / 2 + j] = t.x[r * n + j] ^ 0x8000;
            };
            for (t.y) |*v| v.* = c.toBf16(if (k == .ud_router_norm) 0 else 1);
            @memset(t.z, 0);
        }
        if (fill == .nonfinite) {
            // a row each, one row holding both infinities, in the first operand; then the second's and the bias's
            const at = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 }, .{ 3, 5 } };
            const vals = [_]f32{ uocr.planted[0], uocr.planted[1], uocr.planted[2], uocr.planted[1], uocr.planted[2] };
            for (at, vals) |p, v| if (p[0] < rows and p[1] < n) {
                if (k == .uv_bias) t.f[p[0] * n + p[1]] = v else t.x[p[0] * n + p[1]] = c.toBf16(v);
            };
            if (rows > 4 and n > 7 and k != .ud_router_norm) {
                if (t.y.len == total) t.y[4 * n + 6] = c.toBf16(uocr.planted[0]) else if (t.y.len == n) t.y[6] = c.toBf16(uocr.planted[0]);
                if (t.z.len > 0) t.z[7] = c.toBf16(uocr.planted[1]);
            }
        }
        // the buffers in the order the kernel takes them, as vision.zig and decoder.zig bind them
        t.io = try t.rig.output(total * @as(usize, if (k == .ud_router_norm) 4 else 2));
        const b = std.mem.sliceAsBytes;
        const ins: []const []const u8 = switch (k) {
            .uv_add, .ud_add => &.{ &.{}, b(t.y), &.{} },
            .uv_bias => &.{ b(t.f), b(t.y), &.{} },
            .uv_gelu, .uv_quick_gelu => &.{&.{}},
            .ud_rope => &.{ &.{}, b(t.pos[0..rows]), b(t.pos[rows..]), b(&t.inv) },
            .ud_router_norm => &.{ b(t.x), b(t.y), b(t.pos[0..rows]), b(t.pos[rows..]), b(&t.inv), &.{} },
            .ud_rms_norm, .ud_silu_mul => &.{ b(t.x), b(t.y), &.{} },
            .uv_layer_norm, .uv_layer_norm2d => &.{ b(t.x), b(t.y), b(t.z), &.{} },
        };
        // an empty entry is the output: the last buffer, or the operand an in-place kernel overwrites
        for (ins, 0..) |data, i| t.refs[i] = if (data.len == 0) t.io else try t.rig.input(data);
        t.nrefs = ins.len;
        return true;
    }

    /// One launch as the family's vision.zig and decoder.zig dispatch it.
    pub fn run(t: *Rows) !void {
        const k = t.v.kind;
        const rows = t.shape.rows;
        const n = t.shape.n;
        const total: u32 = @intCast(rows * n);
        if (inPlace(k)) @memcpy(t.rig.bytes(t.io, t.x.len * 2), std.mem.sliceAsBytes(t.x));
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        for (t.refs[0..t.nrefs], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        const at = t.nrefs;
        const by256 = mtl.Size.of(256, 1, 1);
        switch (k) {
            .uv_add, .ud_add, .ud_silu_mul, .uv_gelu, .uv_quick_gelu => {
                e.setValue(total, at);
                e.dispatchThreads(mtl.Size.of(total, 1, 1), by256);
            },
            .uv_bias => {
                e.setValue(@as(u32, @intCast(n)), at);
                e.dispatchThreads(mtl.Size.of(n, rows, 1), by256);
            },
            .uv_layer_norm, .uv_layer_norm2d, .ud_rms_norm => {
                e.setValue(Norm{ .width = @intCast(n), .eps = t.v.eps }, at);
                e.dispatchGroups(mtl.Size.of(rows, 1, 1), by256);
            },
            .ud_rope => {
                e.setValue(Rope{ .heads = @intCast(n / 128), .stride = @intCast(n) }, at);
                e.dispatchThreads(mtl.Size.of(128, n / 128, rows), mtl.Size.of(128, 1, 1));
            },
            .ud_router_norm => e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(128, 1, 1)),
        }
        try t.rig.finish(p);
    }

    pub fn count(t: *Rows) usize {
        return t.shape.rows * t.shape.n;
    }

    /// A bf16 rounding point, which a control leaves out.
    fn keep(t: *const Rows, v: f64) f64 {
        return if (t.v.drop) v else bf(v);
    }
    fn pr(t: *const Rows, r: R) R {
        return if (t.v.drop) r else r.bf16();
    }

    /// The page2d angle of row r's column d: an fp32 product (torch's float32 freqs), or the page's parity.
    fn angle(t: *const Rows, r: usize, d: usize) f64 {
        const j = d % 64;
        if (j == 63) return @floatFromInt(t.pos[t.shape.rows + r] % 2);
        return @as(f32, @floatCast(@as(f64, @floatFromInt(t.pos[r])) * t.inv[j]));
    }

    /// Row r's float64 values at the kernel's rounding points (gate 4: its sums in `sim`) and their fp32 spreads.
    fn row(t: *Rows, r: usize, want: []f64, rg: []R, buf: []f64, rbuf: []R, sim: Sim) void {
        const n = t.shape.n;
        const nf: f64 = @floatFromInt(n);
        const gam = c.gamma(@floatFromInt(spec(t.v).depth(t.shape)), U);
        const eps: f64 = t.v.eps;
        const x = if (t.x.len > 0) t.x[r * n ..][0..n] else &[_]u16{};
        const terms = buf[0..n];
        switch (t.v.kind) {
            .uv_add, .ud_add, .uv_bias, .uv_gelu, .uv_quick_gelu, .ud_silu_mul => for (want, rg, 0..) |*w, *o, j| {
                const i = r * n + j;
                switch (t.v.kind) {
                    .uv_add, .ud_add, .uv_bias => {
                        const s = (if (t.v.kind == .uv_bias) @as(f64, t.f[i]) else val(t.x[i])) + val(t.y[if (t.v.kind == .uv_bias) j else i]);
                        w.* = bf(s);
                        o.* = R.pt(s).rel(1).bf16();
                    },
                    .uv_gelu => {
                        const v = val(x[j]);
                        const e1 = 1 + erf(v / std.math.sqrt2);
                        w.* = bf(0.5 * v * e1);
                        o.* = R.pt(e1).wid(erf_abs).times(0.5 * v).rel(1).bf16();
                    },
                    .uv_quick_gelu => {
                        const v = val(x[j]);
                        const z = @as(f64, @as(f32, 1.702)) * v;
                        w.* = bf(v * t.keep(sigmoid(bf(z))));
                        const zr = R.pt(z).rel(1).bf16();
                        o.* = t.pr(R.rel(.{ .lo = sigmoid(zr.lo), .hi = sigmoid(zr.hi) }, sig_u)).times(v).bf16();
                    },
                    else => {
                        const v = val(x[j]);
                        const u = val(t.y[i]);
                        w.* = bf(t.keep(v * sigmoid(v)) * u);
                        o.* = t.pr(R.pt(v * sigmoid(v)).rel(sig_u)).times(u).bf16();
                    },
                }
            },
            .ud_rms_norm => {
                for (terms, x) |*q, v| q.* = val(v) * val(v);
                const se = sumOf(null, terms);
                const inv = 1 / @sqrt(sumOf(sim, terms) / nf + eps);
                const ir = R.pt(se).wid(gam * se).times(1 / nf).rel(div_u).add(R.pt(eps)).rel(1).rsqrt().rel(fn_u);
                for (want, rg, x, 0..) |*w, *o, xv, j| {
                    w.* = bf(val(t.y[j]) * t.keep(val(xv) * inv));
                    o.* = t.pr(ir.times(val(xv)).rel(1)).times(val(t.y[j])).bf16();
                }
            },
            .uv_layer_norm, .uv_layer_norm2d => {
                for (terms, x) |*q, v| q.* = val(v);
                var abs: f64 = 0;
                for (x) |v| abs += @abs(val(v));
                const mr = R.pt(sumOf(null, terms)).wid(gam * abs).times(1 / nf).rel(div_u);
                const two = t.v.kind == .uv_layer_norm2d;
                const m = sumOf(sim, terms) / nf;
                const u = if (two) bf(m) else m;
                const ur = if (two) mr.bf16() else mr;
                const ds = buf[n .. 2 * n];
                var vr = R.pt(0);
                for (x, terms, ds, rbuf) |v, *q, *d, *dr| {
                    d.* = if (two) bf(val(v) - u) else val(v) - u;
                    q.* = if (two) bf(d.* * d.*) else d.* * d.*;
                    dr.* = R.pt(val(v)).add(ur.neg()).rel(1);
                    if (two) dr.* = dr.bf16();
                    vr = vr.add(if (two) dr.sq().bf16() else dr.sq().rel(1));
                }
                const vs = sumOf(sim, terms) / nf;
                vr = vr.wid(gam * vr.hi).times(1 / nf).rel(div_u);
                // LayerNorm2d rounds var, var + eps, sqrt and the quotient; LayerNorm takes rsqrt of the fp32 variance
                const den = bf(@sqrt(bf(bf(vs) + eps)));
                const dn = vr.bf16().add(R.pt(eps)).rel(1).bf16();
                const dnr = R.rel(.{ .lo = @sqrt(dn.lo), .hi = @sqrt(dn.hi) }, fn_u).bf16();
                const ir = vr.add(R.pt(eps)).rel(1).rsqrt().rel(fn_u);
                for (want, rg, ds, rbuf, 0..) |*w, *o, d, dr, j| {
                    const wt = val(t.y[j]);
                    const b = val(t.z[j]);
                    if (two) {
                        w.* = bf(bf(wt * t.keep(d / den)) + b);
                        o.* = t.pr(dr.mul(.{ .lo = 1 / dnr.hi, .hi = 1 / dnr.lo }).rel(div_u)).times(wt).bf16().add(R.pt(b)).rel(1).bf16();
                    } else {
                        w.* = bf(d / @sqrt(vs + eps) * wt + b);
                        o.* = dr.mul(ir).rel(1).times(wt).rel(1).add(R.pt(b)).rel(1).bf16();
                    }
                }
            },
            .ud_rope => for (want, rg, x, 0..) |*w, *o, xv, d| {
                const dd = d % 128;
                const rh = if (dd < 64) -val(x[d + 64]) else val(x[d - 64]);
                const ang = t.angle(r, dd);
                const cr = R.pt(@cos(ang)).wid(trig_abs).bf16();
                const sr = R.pt(@sin(ang)).wid(trig_abs).bf16();
                w.* = bf(t.keep(val(xv) * bf(@cos(ang))) + t.keep(rh * bf(@sin(ang))));
                o.* = t.pr(cr.times(val(xv))).add(t.pr(sr.times(rh))).rel(1).bf16();
            },
            .ud_router_norm => {
                const lim = 1e4;
                const reps: f64 = @as(f32, 1e-6);
                const vs = buf[n .. 2 * n];
                for (vs, terms, x) |*v, *q, xv| {
                    v.* = if (std.math.isFinite(val(xv))) val(xv) else 0;
                    q.* = v.* * v.*;
                }
                const se = sumOf(null, terms);
                const inv = 1 / @sqrt(sumOf(sim, terms) / 128 + reps);
                // s / 128 is exact; then + eps, rsqrt, the product, 1 + w and its product, each clamped as torch does
                const ir = R.pt(se).wid(gam * se).times(1.0 / 128.0).add(R.pt(reps)).rel(1).rsqrt().rel(fn_u);
                const nw = buf[2 * n .. 3 * n];
                for (vs, nw, rbuf, 0..) |v, *w, *nr, j| {
                    const w1 = 1 + val(t.y[j]);
                    w.* = t.keep(clampNan(clampNan(v * inv, lim) * w1, lim));
                    nr.* = t.pr(ir.times(v).rel(1).clamp(lim).mul(R.pt(w1).rel(1)).rel(1).clamp(lim));
                }
                for (want, rg, 0..) |*w, *o, d| {
                    const pair = if (d < 64) d + 64 else d - 64;
                    const ang = t.angle(r, d);
                    const cr = R.pt(@cos(ang)).wid(trig_abs).bf16();
                    const sr = R.pt(@sin(ang)).wid(trig_abs).bf16();
                    const rh = if (d < 64) -nw[pair] else nw[pair];
                    const rhr = if (d < 64) rbuf[pair].neg() else rbuf[pair];
                    const out = bf(bf(nw[d] * bf(@cos(ang))) + bf(rh * bf(@sin(ang))));
                    w.* = if (std.math.isFinite(out)) out else 0;
                    o.* = rbuf[d].mul(cr).bf16().add(rhr.mul(sr).bf16()).rel(1).bf16();
                }
            },
        }
    }

    /// Every output's value and spread (freed by the caller).
    fn eval(t: *Rows, sim: Sim) !struct { want: []f64, rg: []R } {
        const n = t.shape.n;
        const want = try t.a.alloc(f64, t.count() + 3 * n);
        errdefer t.a.free(want);
        const rg = try t.a.alloc(R, t.count() + n);
        for (0..t.shape.rows) |r| t.row(r, want[r * n ..][0..n], rg[r * n ..][0..n], want[t.count()..], rg[t.count()..], sim);
        return .{ .want = want, .rg = rg };
    }
    fn done(t: *Rows, e: anytype) void {
        t.a.free(e.want);
        t.a.free(e.rg);
    }

    pub fn reference(t: *Rows, want: []f64, mag: []f64) !void {
        const e = try t.eval(null);
        defer t.done(e);
        @memcpy(want, e.want[0..want.len]);
        @memset(mag, 0);
    }

    pub fn intervals(t: *Rows, low: []f64, high: []f64) !void {
        const e = try t.eval(null);
        defer t.done(e);
        for (low, high, e.rg[0..low.len]) |*l, *h, o| {
            l.* = o.lo;
            h.* = o.hi;
        }
    }

    /// The farther end of each output's spread from its float64 value.
    pub fn bounds(t: *Rows, want: []const f64, _: []const f64, allow: []f64) !void {
        const e = try t.eval(null);
        defer t.done(e);
        for (allow, want, e.rg[0..want.len]) |*l, w, o| l.* = if (std.math.isFinite(w)) @max(@abs(w - o.lo), @abs(o.hi - w)) else 0;
    }

    pub fn simulate(t: *Rows, acc: c.Accum, order: g.Order, out: []f64) !void {
        const e = try t.eval(.{ .acc = acc, .order = order });
        defer t.done(e);
        @memcpy(out, e.want[0..out.len]);
    }

    /// ud_router_norm's stated rule: every output finite, within its spread about the sanitized float64 value.
    pub fn check(t: *Rows, _: g.Gate) !?[]const u8 {
        const e = try t.eval(null);
        defer t.done(e);
        const out = try t.rig.outputBytes();
        for (0..t.count()) |i| {
            const got = c.read(.f32, out, i);
            const o = e.rg[i];
            if (!std.math.isFinite(got) or got < o.lo or got > o.hi)
                return try std.fmt.bufPrint(&t.msg, "output {d} is {e:.6} where the rule gives {e:.6} in [{e:.6}, {e:.6}]", .{ i, got, e.want[i], o.lo, o.hi });
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    tests: std.ArrayList(*Rows) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (cases) |v| {
            if (only) |o| if (std.mem.indexOf(u8, v.name, o) == null) continue;
            const x = try s.a.create(Rows);
            x.* = try Rows.init(s.a, s.device, s.queue, v);
            try s.tests.append(s.a, x);
            try s.kernels.append(s.a, x.kernel());
        }
    }
    pub fn deinit(s: *Suite) void {
        for (s.tests.items) |x| {
            x.deinit();
            s.a.destroy(x);
        }
        s.tests.deinit(s.a);
    }
};

/// Gate 5's texts for `uocr rows ...`: the tower's source for uv_* functions, the decoder's for ud_*, on every chip.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?uocr.Texts {
    if (!std.mem.startsWith(u8, name, "uocr rows ")) return null;
    const fun = name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..];
    const text = uocr.source(if (std.mem.startsWith(u8, fun, "uv_")) .vision else .decoder);
    return .{ .here = text, .pre_m5 = text };
}
