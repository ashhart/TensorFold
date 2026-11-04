//! Unlimited-OCR-2's CUDA rows under the contract: the tower's adds, norms and GELUs, the decoder's norms and rotary.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const uocr = @import("contract_uocr.zig");
const ref = @import("contract_uocr_rows_ref.zig");

const c = ref.c;
const g = ref.g;
const R = ref.R;
const bf = ref.bf;
const val = ref.val;
const U = ref.U;

/// Each kernel by its function's name (tf_uocr_*).
const Kind = enum { add, layer_norm, layer_norm2d, gelu, quick_gelu, rms_norm, rope, router_norm, silu_mul };

const Case = struct { kind: Kind, name: []const u8, eps: f32 = 0, ctl: ?[:0]const u8 = null };

const cases = [_]Case{
    .{ .kind = .add, .name = "cuda uocr rows tf_uocr_add (residual in place)" },
    .{ .kind = .layer_norm, .name = "cuda uocr rows tf_uocr_layer_norm SAM (eps 1e-6)", .eps = ref.sam_eps },
    .{ .kind = .layer_norm, .name = "cuda uocr rows tf_uocr_layer_norm CLIP (eps 1e-5)", .eps = ref.clip_eps },
    .{ .kind = .layer_norm2d, .name = "cuda uocr rows tf_uocr_layer_norm2d neck (eps 1e-6)", .eps = ref.sam_eps },
    .{ .kind = .layer_norm2d, .name = "control: uocr tf_uocr_layer_norm2d drops bf16((x - u) / den)", .eps = ref.sam_eps, .ctl = "tf_ctl_uocr_layer_norm2d_quotient" },
    .{ .kind = .gelu, .name = "cuda uocr rows tf_uocr_gelu (erf, in place)" },
    .{ .kind = .quick_gelu, .name = "cuda uocr rows tf_uocr_quick_gelu (in place)" },
    .{ .kind = .quick_gelu, .name = "control: uocr tf_uocr_quick_gelu drops bf16(sigmoid)", .ctl = "tf_ctl_uocr_quick_gelu_sigmoid" },
    .{ .kind = .rms_norm, .name = "cuda uocr rows tf_uocr_rms_norm", .eps = ref.eps },
    .{ .kind = .rms_norm, .name = "control: uocr tf_uocr_rms_norm drops bf16(x rsqrt)", .eps = ref.eps, .ctl = "tf_ctl_uocr_rms_norm_scaled" },
    .{ .kind = .rope, .name = "cuda uocr rows tf_uocr_rope page2d (in place)" },
    .{ .kind = .rope, .name = "control: uocr tf_uocr_rope drops its bf16 products", .ctl = "tf_ctl_uocr_rope_products" },
    .{ .kind = .router_norm, .name = "cuda uocr rows tf_uocr_router_norm (Gemma norm, clamp, page2d)" },
    .{ .kind = .router_norm, .name = "control: uocr tf_uocr_router_norm drops the bf16 normed row", .ctl = "tf_ctl_uocr_router_norm_normed" },
    .{ .kind = .silu_mul, .name = "cuda uocr rows tf_uocr_silu_mul" },
    .{ .kind = .silu_mul, .name = "control: uocr tf_uocr_silu_mul drops bf16(silu)", .ctl = "tf_ctl_uocr_silu_mul_silu" },
};

fn inPlace(k: Kind) bool {
    return k == .add or k == .gelu or k == .quick_gelu or k == .rope;
}

fn sums(k: Kind) bool {
    return k == .layer_norm or k == .layer_norm2d or k == .rms_norm or k == .router_norm;
}

/// A row's sums (block_sum): each of 256 threads' strided chain, the warp butterfly, then warp 0's over the partials.
fn rowDepth(s: c.Shape) usize {
    return (s.n + 255) / 256 + 5 + 5;
}

/// The router's: one square a thread, the warp butterfly, then warp 0's over 4 partials.
fn routerDepth(_: c.Shape) usize {
    return 1 + 5 + 5;
}

const norm_why = "bf16 outputs after the sums' rsqrt: a bf16 sum's error is a few output ulps, not provably 100x";

fn spec(v: Case) c.Spec {
    const H = ref.hidden;
    // the engine's shapes, an odd prompt chunk or the CLIP rows first (their tails fall off a 256-thread block)
    const served: []const c.Shape = switch (v.kind) {
        .add => &.{ .{ .rows = 257, .n = ref.clip_width, .k = 1 }, .{ .rows = 4096, .n = ref.sam_width, .k = 1 }, .{ .rows = 1600, .n = ref.sam_width, .k = 1 }, .{ .rows = 37, .n = H, .k = 1 }, .{ .rows = 2048, .n = H, .k = 1 }, .{ .rows = 1, .n = H, .k = 1 } },
        .layer_norm => if (v.eps == ref.clip_eps) &.{ .{ .rows = 257, .n = ref.clip_width, .k = 1 }, .{ .rows = 101, .n = ref.clip_width, .k = 1 } } else &.{ .{ .rows = 1600, .n = ref.sam_width, .k = 1 }, .{ .rows = 4096, .n = ref.sam_width, .k = 1 } },
        .layer_norm2d => &.{ .{ .rows = 1600, .n = ref.neck_width, .k = 1 }, .{ .rows = 4096, .n = ref.neck_width, .k = 1 } },
        .gelu => &.{ .{ .rows = 1600, .n = ref.sam_ffn, .k = 1 }, .{ .rows = 4096, .n = ref.sam_ffn, .k = 1 } },
        .quick_gelu => &.{ .{ .rows = 257, .n = ref.clip_ffn, .k = 1 }, .{ .rows = 101, .n = ref.clip_ffn, .k = 1 } },
        .rms_norm, .rope => &.{ .{ .rows = 37, .n = H, .k = 1 }, .{ .rows = 2048, .n = H, .k = 1 }, .{ .rows = 1, .n = H, .k = 1 } },
        .router_norm => &.{ .{ .rows = 37, .n = ref.router_dim, .k = 1 }, .{ .rows = 2048, .n = ref.router_dim, .k = 1 }, .{ .rows = 1, .n = ref.router_dim, .k = 1 } },
        .silu_mul => &.{ .{ .rows = 37, .n = ref.dense_ffn, .k = 1 }, .{ .rows = 2048, .n = ref.dense_ffn, .k = 1 }, .{ .rows = 1, .n = ref.shared_ffn, .k = 1 }, .{ .rows = 2048, .n = ref.shared_ffn, .k = 1 }, .{ .rows = 300, .n = ref.expert_ffn, .k = 1 } },
    };
    var s: c.Spec = .{ .name = v.name, .accum = .none, .depth = ref.noSums, .out = .bf16, .in_place = inPlace(v.kind), .served = served };
    s.tiles[1] = switch (v.kind) {
        .rope => .{ .size = 128, .step = 128, .min = 128 },
        .router_norm => .{},
        else => .{ .size = 256, .step = 1, .min = 1 },
    };
    if (sums(v.kind)) {
        s.accum = .f32;
        s.depth = if (v.kind == .router_norm) routerDepth else rowDepth;
        s.sums = .{ .bf16 = 0, .why = norm_why };
    }
    if (v.kind == .router_norm) {
        s.out = .f32;
        s.nan = .{ .stated = "a non-finite input reads as 0 and a non-finite result is written as 0 (dash_attention.py's nan_to_num)" };
    }
    return s;
}

pub const Rows = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    k: uocr.kernels.Kernels,
    v: Case,
    shape: c.Shape = undefined,
    /// The first bf16 operand (a, x or the gate), the second (b, the weight or up), the layer norm's bias.
    x: []u16 = &.{},
    y: []u16 = &.{},
    z: []u16 = &.{},
    /// The rows' token positions then their pages.
    pos: []u32 = &.{},
    inv: [63]f32 = undefined,
    /// Device inputs in the kernel's order (the in-place operand's copy first), and the output.
    ins: [6]u64 = @splat(0),
    io: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(t: *Rows) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Rows) void {
        inline for (.{ "x", "y", "z", "pos" }) |name| {
            t.a.free(@field(t, name));
            @field(t, name) = &.{};
        }
    }
    pub fn kernel(t: *Rows) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = if (t.v.kind == .router_norm) &vt_router else &vt, .control = if (t.v.ctl != null) .shapes else null };
    }
    const vt = blk: {
        var h = g.hooks(Rows);
        h.check = null;
        break :blk h;
    };
    const vt_router = g.hooks(Rows);

    pub fn prepare(t: *Rows, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const k = t.v.kind;
        if (shape.rows == 0 or shape.n == 0 or (k == .rope and shape.n % 128 != 0) or (k == .router_norm and shape.n != 128)) return false;
        if (fill == .stress and shape.n % 2 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const rows = shape.rows;
        const n = shape.n;
        const total = rows * n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const ln = k == .layer_norm or k == .layer_norm2d;
        const pairs = k == .add or k == .silu_mul;
        const rotary = k == .rope or k == .router_norm;
        t.x = try t.a.alloc(u16, total);
        t.y = try t.a.alloc(u16, if (pairs) total else if (sums(k)) n else 0);
        t.z = try t.a.alloc(u16, if (ln) n else 0);
        t.pos = try t.a.alloc(u32, if (rotary) 2 * rows else 0);
        const span: [2]f32 = switch (k) {
            .layer_norm, .layer_norm2d => .{ -2.25, 3.75 },
            .gelu => .{ -6, 6 },
            .quick_gelu => .{ -8, 8 },
            .silu_mul => .{ -20, 12 },
            else => .{ -4, 4 },
        };
        for (t.x) |*v| v.* = ref.uni(rng, span[0], span[1]);
        for (t.y) |*v| v.* = if (sums(k)) ref.uni(rng, -1.5, 1.5) else ref.uni(rng, -2, 2);
        for (t.z) |*v| v.* = ref.uni(rng, -0.5, 0.5);
        for (t.pos, 0..) |*v, i| v.* = rng.uintLessThan(u32, if (i < rows) ref.max_context else 64);
        // a router weight past the clamp: (1 + w) times the normed value leaves +-1e4
        if (k == .router_norm and n > 5) t.y[5] = c.toBf16(2000);
        if (fill == .stress) {
            // integers whose squares sum exactly in fp32 and overflow fp16 in any order; mirrored halves (mean 0)
            for (t.x) |*v| v.* = c.toBf16(@as(f32, @floatFromInt(24 + rng.uintLessThan(u32, 17))) * (if (rng.boolean()) @as(f32, 1) else -1));
            if (ln) for (0..rows) |r| for (0..n / 2) |j| {
                t.x[r * n + n / 2 + j] = t.x[r * n + j] ^ 0x8000;
            };
            for (t.y) |*v| v.* = c.toBf16(if (k == .router_norm) 0 else 1);
            @memset(t.z, 0);
        }
        if (fill == .nonfinite) {
            // a row each, one row holding both infinities, in the first operand; then the second's and the bias's
            const at = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 }, .{ 3, 5 } };
            const vals = [_]f32{ ref.planted[0], ref.planted[1], ref.planted[2], ref.planted[1], ref.planted[2] };
            for (at, vals) |p, v| if (p[0] < rows and p[1] < n) {
                t.x[p[0] * n + p[1]] = c.toBf16(v);
            };
            if (rows > 4 and n > 7 and k != .router_norm) {
                if (t.y.len == total) t.y[4 * n + 6] = c.toBf16(ref.planted[0]) else if (t.y.len == n) t.y[6] = c.toBf16(ref.planted[0]);
                if (t.z.len > 0) t.z[7] = c.toBf16(ref.planted[1]);
            }
        }
        const b = std.mem.sliceAsBytes;
        t.ins[0] = try t.rig.input(b(t.x));
        if (t.y.len > 0) t.ins[1] = try t.rig.input(b(t.y));
        if (t.z.len > 0) t.ins[2] = try t.rig.input(b(t.z));
        if (rotary) {
            t.ins[3] = try t.rig.input(b(t.pos[0..rows]));
            t.ins[4] = try t.rig.input(b(t.pos[rows..]));
            t.ins[5] = try t.rig.input(b(&t.inv));
        }
        t.io = try t.rig.output(total * @as(usize, if (k == .router_norm) 4 else 2));
        return true;
    }

    /// One launch as the family's cuda_vision.zig and cuda_decoder.zig dispatch it.
    pub fn run(t: *Rows) !void {
        const k = t.v.kind;
        const rows = t.shape.rows;
        const n = t.shape.n;
        const total = rows * n;
        const s = t.rig.s;
        const in = t.ins;
        // an in-place kernel works on its own copy of the operand, so every independent run starts from the input
        if (inPlace(k)) try (cuda.DeviceBuffer{ .d = t.rig.gpu.d, .ptr = t.io, .len = total * 2 }).copyFrom(0, in[0], total * 2, s.handle);
        const blocks = (total + 255) / 256;
        const all: i64 = @intCast(total);
        switch (k) {
            .add => try t.k.run("add", s, .{ blocks, 1, 1 }, 256, .{ t.io, in[1], t.io, all }),
            .gelu => try t.k.run("gelu", s, .{ blocks, 1, 1 }, 256, .{ t.io, all }),
            .quick_gelu => try t.k.run("quick_gelu", s, .{ blocks, 1, 1 }, 256, .{ t.io, all }),
            .silu_mul => try t.k.run("silu_mul", s, .{ blocks, 1, 1 }, 256, .{ in[0], in[1], t.io, all }),
            .layer_norm => try t.k.run("layer_norm", s, .{ rows, 1, 1 }, 256, .{ in[0], in[1], in[2], t.io, n, t.v.eps }),
            .layer_norm2d => try t.k.run("layer_norm2d", s, .{ rows, 1, 1 }, 256, .{ in[0], in[1], in[2], t.io, n, t.v.eps }),
            .rms_norm => try t.k.run("rms_norm", s, .{ rows, 1, 1 }, 256, .{ in[0], in[1], t.io, n, t.v.eps }),
            .rope => try t.k.run("rope", s, .{ rows, n / 128, 1 }, 128, .{ t.io, in[3], in[4], in[5], n }),
            .router_norm => try t.k.run("router_norm", s, .{ rows, 1, 1 }, 128, .{ in[0], in[1], in[3], in[4], in[5], t.io }),
        }
        try s.synchronize();
    }

    pub fn count(t: *Rows) usize {
        return t.shape.rows * t.shape.n;
    }

    /// The page2d angle of row r's column d: an fp32 product (torch's float32 freqs), or the page's parity.
    fn angle(t: *const Rows, r: usize, d: usize) f64 {
        const j = d % 64;
        if (j == 63) return @floatFromInt(t.pos[t.shape.rows + r] % 2);
        return @as(f32, @floatCast(@as(f64, @floatFromInt(t.pos[r])) * t.inv[j]));
    }

    /// Row r's float64 values at the kernel's rounding points (gate 4: its sums in `sim`) and their fp32 spreads.
    fn row(t: *Rows, r: usize, want: []f64, rg: []R, buf: []f64, rbuf: []R, sim: ref.Sim) void {
        const n = t.shape.n;
        const nf: f64 = @floatFromInt(n);
        const gam = c.gamma(@floatFromInt(spec(t.v).depth(t.shape)), U);
        const eps: f64 = t.v.eps;
        const x = t.x[r * n ..][0..n];
        const terms = buf[0..n];
        switch (t.v.kind) {
            .add, .gelu, .quick_gelu, .silu_mul => for (want, rg, 0..) |*w, *o, j| {
                const i = r * n + j;
                switch (t.v.kind) {
                    .add => {
                        const s = val(t.x[i]) + val(t.y[i]);
                        w.* = bf(s);
                        o.* = R.pt(s).rel(1).bf16();
                    },
                    .gelu => {
                        const v = val(x[j]);
                        const e1 = 1 + ref.erf(v / std.math.sqrt2);
                        w.* = bf(0.5 * v * e1);
                        o.* = R.pt(e1).wid(ref.erf_abs).times(0.5 * v).rel(1).bf16();
                    },
                    .quick_gelu => {
                        const v = val(x[j]);
                        const z = @as(f64, @as(f32, 1.702)) * v;
                        w.* = bf(v * bf(ref.sigmoid(bf(z))));
                        const zr = R.pt(z).rel(1).bf16();
                        o.* = R.rel(.{ .lo = ref.sigmoid(zr.lo), .hi = ref.sigmoid(zr.hi) }, ref.sig_u).bf16().times(v).bf16();
                    },
                    else => {
                        const v = val(x[j]);
                        const u = val(t.y[i]);
                        w.* = bf(bf(v * ref.sigmoid(v)) * u);
                        o.* = R.pt(v * ref.sigmoid(v)).rel(ref.sig_u).bf16().times(u).bf16();
                    },
                }
            },
            .rms_norm => {
                for (terms, x) |*q, v| q.* = val(v) * val(v);
                const se = ref.sumOf(null, terms);
                const inv = 1 / @sqrt(ref.sumOf(sim, terms) / nf + eps);
                const ir = R.pt(se).wid(gam * se).times(1 / nf).rel(ref.div_u).add(R.pt(eps)).rel(1).rsqrt().rel(ref.rsqrt_u);
                for (want, rg, x, 0..) |*w, *o, xv, j| {
                    w.* = bf(val(t.y[j]) * bf(val(xv) * inv));
                    o.* = ir.times(val(xv)).rel(1).bf16().times(val(t.y[j])).bf16();
                }
            },
            .layer_norm, .layer_norm2d => {
                for (terms, x) |*q, v| q.* = val(v);
                var abs: f64 = 0;
                for (x) |v| abs += @abs(val(v));
                const mr = R.pt(ref.sumOf(null, terms)).wid(gam * abs).times(1 / nf).rel(ref.div_u);
                const two = t.v.kind == .layer_norm2d;
                const m = ref.sumOf(sim, terms) / nf;
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
                const vs = ref.sumOf(sim, terms) / nf;
                vr = vr.wid(gam * vr.hi).times(1 / nf).rel(ref.div_u);
                // LayerNorm2d rounds var, var + eps, sqrt and the quotient; LayerNorm takes rsqrt of the fp32 variance
                const den = bf(@sqrt(bf(bf(vs) + eps)));
                const dn = vr.bf16().add(R.pt(eps)).rel(1).bf16();
                const dnr = R.rel(.{ .lo = @sqrt(dn.lo), .hi = @sqrt(dn.hi) }, ref.sqrt_u).bf16();
                const ir = vr.add(R.pt(eps)).rel(1).rsqrt().rel(ref.rsqrt_u);
                for (want, rg, ds, rbuf, 0..) |*w, *o, d, dr, j| {
                    const wt = val(t.y[j]);
                    const b = val(t.z[j]);
                    if (two) {
                        w.* = bf(bf(wt * bf(d / den)) + b);
                        o.* = dr.mul(.{ .lo = 1 / dnr.hi, .hi = 1 / dnr.lo }).rel(ref.div_u).bf16().times(wt).bf16().add(R.pt(b)).rel(1).bf16();
                    } else {
                        w.* = bf(d / @sqrt(vs + eps) * wt + b);
                        o.* = dr.mul(ir).rel(1).times(wt).rel(1).add(R.pt(b)).rel(1).bf16();
                    }
                }
            },
            .rope => for (want, rg, x, 0..) |*w, *o, xv, d| {
                const dd = d % 128;
                const rh = if (dd < 64) -val(x[d + 64]) else val(x[d - 64]);
                const ang = t.angle(r, dd);
                const cr = R.pt(@cos(ang)).wid(ref.trig_abs).bf16();
                const sr = R.pt(@sin(ang)).wid(ref.trig_abs).bf16();
                w.* = bf(bf(val(xv) * bf(@cos(ang))) + bf(rh * bf(@sin(ang))));
                o.* = cr.times(val(xv)).bf16().add(sr.times(rh).bf16()).rel(1).bf16();
            },
            .router_norm => {
                const lim = 1e4;
                const reps: f64 = @as(f32, 1e-6);
                const vs = buf[n .. 2 * n];
                for (vs, terms, x) |*v, *q, xv| {
                    v.* = if (std.math.isFinite(val(xv))) val(xv) else 0;
                    q.* = v.* * v.*;
                }
                const se = ref.sumOf(null, terms);
                const inv = 1 / @sqrt(ref.sumOf(sim, terms) / 128 + reps);
                // s / 128 is exact; then + eps, rsqrt, the product, 1 + w and its product, each clamped as torch does
                const ir = R.pt(se).wid(gam * se).times(1.0 / 128.0).add(R.pt(reps)).rel(1).rsqrt().rel(ref.rsqrt_u);
                const nw = buf[2 * n .. 3 * n];
                for (vs, nw, rbuf, 0..) |v, *w, *nr, j| {
                    const w1 = 1 + val(t.y[j]);
                    w.* = bf(ref.clampNan(ref.clampNan(v * inv, lim) * w1, lim));
                    nr.* = ir.times(v).rel(1).clamp(lim).mul(R.pt(w1).rel(1)).rel(1).clamp(lim).bf16();
                }
                for (want, rg, 0..) |*w, *o, d| {
                    const pair = if (d < 64) d + 64 else d - 64;
                    const ang = t.angle(r, d);
                    const cr = R.pt(@cos(ang)).wid(ref.trig_abs).bf16();
                    const sr = R.pt(@sin(ang)).wid(ref.trig_abs).bf16();
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
    fn eval(t: *Rows, sim: ref.Sim) !struct { want: []f64, rg: []R } {
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

    /// tf_uocr_router_norm's stated rule: every output finite, within its spread about the sanitized float64 value.
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

/// The group's registrations, on the heap: the gates hold pointers to them.
pub const Group = struct {
    gpa: std.mem.Allocator,
    tests: std.ArrayList(*Rows) = .empty,

    pub fn init(ctx: uocr.Ctx, list: *std.ArrayList(g.Kernel)) !*Group {
        const a = ctx.gpu.gpa;
        const grp = try a.create(Group);
        grp.* = .{ .gpa = a };
        errdefer grp.deinit();
        for (cases) |v| {
            const k = if (v.ctl) |name| try withControl(ctx, v.kind, name) else ctx.k.*;
            const x = try a.create(Rows);
            x.* = .{ .a = a, .rig = ctx.rig(), .k = k, .v = v, .inv = ref.invTok() };
            try grp.tests.append(a, x);
            try list.append(a, x.kernel());
        }
        return grp;
    }

    pub fn deinit(grp: *Group) void {
        for (grp.tests.items) |x| {
            x.deinit();
            grp.gpa.destroy(x);
        }
        grp.tests.deinit(grp.gpa);
        grp.gpa.destroy(grp);
    }
};

/// The family's kernels with `kind`'s swapped for its planted control.
fn withControl(ctx: uocr.Ctx, kind: Kind, control: [:0]const u8) !uocr.kernels.Kernels {
    return switch (kind) {
        inline else => |kd| ctx.with(@tagName(kd), control),
    };
}
