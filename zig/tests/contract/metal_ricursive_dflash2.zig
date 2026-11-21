//! Ripple's DFlash2 drafter kernels (dflash2.metal) under the contract: norms, embed and taps here, the rest in _*.zig.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_ricursive_controls.zig");
const attn = @import("metal_ricursive_dflash2_attn.zig");
const act = @import("metal_ricursive_dflash2_act.zig");
const pick = @import("metal_ricursive_dflash2_pick.zig");

const c = contract.spec;
const g = contract.gates;
const family = tf.ricursive.contracts;
const dc = tf.ricursive.dflash2_config;
pub const Mutation = controls.Mutation;

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

pub const nan = std.math.nan(f32);
pub const inf = std.math.inf(f32);

/// Random bf16 bits in [lo, hi).
pub fn fillBf(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
}

pub const norm_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

/// The published drafter's geometry beside Ripple-65B (dflash2_config.zig's qualified check).
pub const D = struct {
    pub const hidden: u32 = @intCast(family.ripple.hidden);
    pub const vocab: u32 = @intCast(family.ripple.vocab);
    pub const heads: u32 = 80;
    pub const kv: u32 = 8;
    pub const ffn: u32 = 40960;
    pub const window: u32 = 4096;
    pub const global: u32 = dc.global_window;
    pub const block: u32 = 16;
    pub const rank: u32 = 256;
    pub const top: u32 = 32;
    pub const group: u32 = 16;
    pub const conv: u32 = 4;
    pub const taps: u32 = dc.max_taps;
    pub const theta: f32 = 10000;
    pub const eps: f32 = 1e-6;
    /// The longest context the engine serves (ricursive's --context cap): positions stay below it.
    pub const context: u32 = 262144;

    /// A layer's ring: its span and a block past it, whole 128-key blocks (Config.ring).
    pub fn ring(span: u32) u32 {
        return std.mem.alignForward(u32, span + block, 128);
    }
};

/// dflash2.metal's pipeline `name`, built with the control's edit.
pub fn pipeline(device: mtl.Device, a: std.mem.Allocator, name: [:0]const u8, m: Mutation) !mtl.Pipeline {
    const lib = try controls.dflash2Library(device, a, m);
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, name, false);
}

/// What a probe's registration needs: the run's allocator, GPU, the kernel list and the --only filter.
pub const Ctx = struct { a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8 };

/// The drafter's probes, freed together.
pub const Suite = struct {
    owned: std.ArrayList(Owned) = .empty,

    const Owned = struct { ctx: *anyopaque, free: *const fn (*anyopaque, std.mem.Allocator) void };

    /// T's probe for `v` built with `m`, kept and registered when its name holds `only`.
    pub fn add(s: *Suite, x: Ctx, comptime T: type, v: T.Variant, m: Mutation) !void {
        const name = if (m == .none) T.spec(v).name else m.name();
        if (x.only) |o| if (std.mem.indexOf(u8, name, o) == null) return;
        const t = try x.a.create(T);
        t.* = T.init(x.a, x.device, x.queue, v, m) catch |err| {
            x.a.destroy(t);
            return err;
        };
        const free = struct {
            fn f(p: *anyopaque, a: std.mem.Allocator) void {
                const y: *T = @ptrCast(@alignCast(p));
                y.deinit();
                a.destroy(y);
            }
        }.f;
        s.owned.append(x.a, .{ .ctx = t, .free = free }) catch |err| {
            free(t, x.a);
            return err;
        };
        try x.kernels.append(x.a, t.kernel());
    }

    /// Every kernel's registrations; dflash2.metal's pipelines go on the roll call.
    pub fn collect(s: *Suite, x: Ctx, functions: *std.ArrayList([]const u8)) !void {
        for (family.dflash2_names) |n| try functions.append(x.a, n);
        try s.add(x, Tap, .tap, .none);
        for (std.enums.values(Norm.Variant)) |v| try s.add(x, Norm, v, .none);
        try s.add(x, Embed, .embed, .none);
        try attn.collect(s, x);
        try act.collect(s, x);
        try pick.collect(s, x);
    }

    /// One control: its kernel built with the planted fault, at the variant the fault shows in.
    pub fn plant(s: *Suite, x: Ctx, m: Mutation) !void {
        switch (m) {
            .df_tap_next_slot => try s.add(x, Tap, .tap, m),
            .df_norm_half_mean => try s.add(x, Norm, .norm_add, m),
            .df_norm16_no_weight => try s.add(x, Norm, .norm16, m),
            .df_embed_own_row => try s.add(x, Embed, .embed, m),
            .df_prep_ropes_nope => try s.add(x, attn.Prep, .global_block, m),
            .df_conv_three_taps => try s.add(x, act.Conv, .side1, m),
            .df_silu_no_negative => try s.add(x, act.Silu, .silu, m),
            .df_vec_no_floor => try s.add(x, attn.Vec, .floor, m),
            .df_topk_ascending => try s.add(x, pick.TopK, .logits, m),
            .df_walk_from_pending => try s.add(x, pick.Walk, .walk, m),
            else => return error.NotADflash2Control,
        }
    }

    pub fn deinit(s: *Suite, a: std.mem.Allocator) void {
        for (s.owned.items) |o| o.free(o.ctx, a);
        s.owned.deinit(a);
    }
};

// --- rc_df_norm, rc_df_norm16: the release's RMSNorm of fp32 rows (bf16(R + X) first) or bf16 rows, 256 threads ---

/// Each thread's fma chain over width / 256 inputs, a 5-level simd_sum, then the 8 simdgroups' parts in turn.
fn normDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 5 + 8;
}

/// Gate 4's norm rows: integers in [-32, 32], so fp32 sums of squares are exact in any order.
fn stressRows(rng: std.Random, x: []f32) void {
    for (x) |*v| v.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 65))) - 32;
}

pub const Norm = struct {
    /// The block's norms of the fp32 residual (layer 0's input, then each after bf16(R + X)); the context's bf16 rows.
    pub const Variant = enum { norm, norm_add, norm16 };
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Variant,
    m: Mutation,
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
        // the draft block's 16 rows; fc's rows of a context absorb (2,048), and a few
        const block: []const c.Shape = &.{.{ .rows = D.block, .n = 1, .k = D.hidden }};
        const context: []const c.Shape = &.{ .{ .rows = 2048, .n = 1, .k = D.hidden }, .{ .rows = 7, .n = 1, .k = D.hidden } };
        return .{
            .name = switch (v) {
                .norm => "ricursive dflash2 rc_df_norm fp32 rows",
                .norm_add => "ricursive dflash2 rc_df_norm bf16(R + X) rows",
                .norm16 => "ricursive dflash2 rc_df_norm16",
            },
            .accum = .f32,
            .depth = normDepth,
            // the mean's division, + eps, precise rsqrt (2), x * inv, then w's product
            .ops = 6,
            // bf16(x * inv) before w's product
            .mids = 1,
            .out = .bf16,
            .tiles = .{ .{ .size = 1 }, .{}, .{ .size = 256, .step = 1, .min = 1 } },
            .served = if (v == .norm16) context else block, // n unused
            .sums = .{ .bf16 = 0, .why = norm_why },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, m: Mutation) !Norm {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(device, a, if (v == .norm16) "rc_df_norm16" else "rc_df_norm", m), .v = v, .m = m };
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
        return controls.mark(.{ .spec = spec(t.v), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Norm);

    fn fp32Rows(t: *const Norm) bool {
        return t.v != .norm16;
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
            for (t.xb, t.r0) |*b, r| b.* = if (t.v == .norm_add) 0 else c.toBf16(r);
        }
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, shape.k / 3 }, .{ 1, shape.k / 2 }, .{ 2, shape.k - 1 }, .{ 3, 0 }, .{ 3, shape.k - 1 } };
            const put = [_]f32{ nan, inf, -inf, inf, -inf };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                const i = p[0] * shape.k + p[1];
                if (t.v == .norm) t.r0[i] = v else t.xb[i] = c.toBf16(v);
            };
        }
        for (t.x, t.r0, t.xb) |*x, r, b| x.* = switch (t.v) {
            .norm16 => c.fromBf16(b),
            .norm => r,
            .norm_add => @floatCast(bf(@as(f32, r + c.fromBf16(b)))),
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
        const args = extern struct { rows: u32, width: u32, eps: f32, add: u32 }{ .rows = @intCast(t.shape.rows), .width = @intCast(t.shape.k), .eps = D.eps, .add = @intFromBool(t.v == .norm_add) };
        if (t.fp32Rows()) {
            p.e.setBuffer(t.res.buf, t.res.off, 0);
            p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 1);
            p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 2);
            p.e.setValue(args, 3);
            p.e.setBuffer(t.out.buf, t.out.off, 4);
        } else {
            p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
            p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
            p.e.setValue(args, 2);
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
        return 1 / @sqrt(ss / @as(f64, @floatFromInt(k)) + @as(f64, D.eps));
    }

    fn outputs(t: *Norm, out: []f64, mag: ?[]f64, acc: ?Sim) !void {
        const k = t.shape.k;
        for (0..t.shape.rows) |r| {
            const s = try t.inv(r, acc);
            for (0..k) |i| {
                const w: f64 = c.fromBf16(t.w[i]);
                const y = w * (@as(f64, t.x[r * k + i]) * s);
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

// --- rc_df_embed: the block's ids through the target's table into the fp32 residual, unscaled --------------------

pub const Embed = struct {
    pub const Variant = enum { embed };
    /// Table rows the probe holds: ids only index it.
    const table = 64;
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    shape: c.Shape = undefined,
    e: []u16 = &.{},
    ids: []u32 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: the block's positions; n: the hidden width (k unused).
    pub fn spec(_: Variant) c.Spec {
        return .{
            .name = "ricursive dflash2 rc_df_embed",
            .accum = .none,
            .depth = noSums,
            .out = .f32,
            .tiles = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            .served = &.{.{ .rows = D.block, .n = D.hidden, .k = 1 }},
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, _: Variant, m: Mutation) !Embed {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(device, a, "rc_df_embed", m), .m = m };
    }
    pub fn deinit(t: *Embed) void {
        t.a.free(t.e);
        t.a.free(t.ids);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Embed) g.Kernel {
        return controls.mark(.{ .spec = spec(.embed), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Embed);

    pub fn prepare(t: *Embed, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > table or shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.e);
        t.a.free(t.ids);
        t.e = &.{};
        t.ids = &.{};
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.e = try t.a.alloc(u16, table * shape.n);
        fillBf(rng, t.e, -2, 2);
        t.ids = try t.a.alloc(u32, shape.rows);
        // no row reads its own table row
        for (t.ids, 0..) |*id, i| id.* = @intCast((i * 5 + 3) % table);
        if (fill == .nonfinite) for ([_]f32{ nan, inf, -inf }, 0..) |v, r| {
            if (r < shape.rows) t.e[t.ids[r] * shape.n + (r * 7) % shape.n] = c.toBf16(v);
        };
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.ids));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.e));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Embed) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setValue(extern struct { rows: u32, width: u32, eps: f32, add: u32 }{ .rows = @intCast(t.shape.rows), .width = @intCast(t.shape.n), .eps = 0, .add = 0 }, 2);
        p.e.setBuffer(t.out.buf, t.out.off, 3);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Embed) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Embed, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        for (want, mag, 0..) |*w, *m, i| {
            w.* = c.fromBf16(t.e[t.ids[i / n] * n + i % n]);
            m.* = 0;
        }
    }
};

// --- rc_tap: the target's residual after a tapped layer, as bf16, into its slot among the taps -------------------

pub const Tap = struct {
    pub const Variant = enum { tap };
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    shape: c.Shape = undefined,
    r: []u16 = &.{},
    old: []u16 = &.{},
    in: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: positions; n: the hidden width; k: the taps; the other taps' slots stay put.
    pub fn spec(_: Variant) c.Spec {
        return .{
            .name = "ricursive dflash2 rc_tap",
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .tiles = .{ .{ .size = 1 }, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            // a decode window's rows and a prompt chunk's
            .served = &.{ .{ .rows = D.block, .n = D.hidden, .k = D.taps }, .{ .rows = 2048, .n = D.hidden, .k = D.taps } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, _: Variant, m: Mutation) !Tap {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(device, a, "rc_tap", m), .m = m };
    }
    pub fn deinit(t: *Tap) void {
        t.a.free(t.r);
        t.a.free(t.old);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Tap) g.Kernel {
        return controls.mark(.{ .spec = spec(.tap), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Tap);

    /// The tap written: the next to last, so a slot past it is still inside the features.
    fn tap(t: *const Tap) usize {
        return t.shape.k - 2;
    }

    pub fn prepare(t: *Tap, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k < 2) return false;
        t.rig.reset();
        t.a.free(t.r);
        t.a.free(t.old);
        t.r = &.{};
        t.old = &.{};
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.r = try t.a.alloc(u16, shape.rows * shape.n);
        fillBf(rng, t.r, -64, 64);
        if (fill == .nonfinite) for ([_]f32{ nan, inf, -inf }, 0..) |v, i| {
            t.r[(i * 7) % t.r.len] = c.toBf16(v);
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
            w.* = c.fromBf16(if (slot == t.tap()) t.r[r * n + i % n] else t.old[i]);
            m.* = 0;
        }
    }
};
