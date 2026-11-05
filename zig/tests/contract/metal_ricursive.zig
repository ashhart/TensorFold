//! Ripple's elementwise glue under the contract, in float64's range at bf16 points; the interval helpers it shares.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_ricursive_controls.zig");
const c = contract.spec;
const g = contract.gates;
pub const family = tf.ricursive.contracts;
pub const R = family.ripple;
pub const DH = tf.ricursive.config.head_dim;

/// glue.metal's rc_args, as forward.zig fills it.
pub const Args = extern struct { rows: u32 = 0, width: u32 = 0, flags: u32 = 0, eps: f32 = 0, scale: f32 = 0, blend: f32 = 0 };
/// The release's embed_scale and ffn_clamp_threshold (config.json), as config.zig reads them into fp32.
pub const embed_scale: f32 = @floatCast(71.55417527999327);
pub const clamp: f32 = 20;
pub const eps: f32 = 1e-6;

/// fp32's unit roundoff.
pub const u: f64 = 0x1p-24;
/// A value's range: the least and most the kernel's fp32 steps allow.
pub const Range = [2]f64;

/// A value through fp32 and then bf16, as the kernels round.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

pub fn point(v: f64) Range {
    return .{ v, v };
}

/// The range moved out by `rel` of its larger end: fp32 steps' error.
pub fn widen(r: Range, rel: f64) Range {
    const m = @max(@abs(r[0]), @abs(r[1]));
    return .{ r[0] - rel * m - 0x1p-148, r[1] + rel * m + 0x1p-148 };
}

/// The bf16 values an fp32 result inside `r` rounds to; each end first moved an fp32 ulp out for fp32's own rounding.
pub fn round(r: Range) Range {
    return .{ bf(r[0] - @abs(r[0]) * 0x1p-23 - 0x1p-148), bf(r[1] + @abs(r[1]) * 0x1p-23 + 0x1p-148) };
}

pub fn mul(a: Range, b: Range) Range {
    const p = [4]f64{ a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1] };
    return .{ @min(@min(p[0], p[1]), @min(p[2], p[3])), @max(@max(p[0], p[1]), @max(p[2], p[3])) };
}

pub fn add(a: Range, b: Range) Range {
    return .{ a[0] + b[0], a[1] + b[1] };
}

/// The least and most of x^2 over a range.
pub fn square(r: Range) Range {
    const hi = @max(r[0] * r[0], r[1] * r[1]);
    return .{ if (r[0] <= 0 and r[1] >= 0) 0 else @min(r[0] * r[0], r[1] * r[1]), hi };
}

/// rsqrt(ss / width + eps) over a sum of squares in `ss`: the fp32 sum's gamma(depth), the mean, the eps and rsqrt.
pub fn invRms(ss: Range, width: usize, depth: usize) Range {
    const gm = c.gamma(@floatFromInt(depth), u);
    const w: f64 = @floatFromInt(width);
    return .{ (1 - 6 * u) / @sqrt(ss[1] * (1 + gm) / w + eps), (1 + 6 * u) / @sqrt(@max(ss[0] * (1 - gm), 0) / w + eps) };
}

/// Each output's allowance: its distance to the far end of its range, so the range alone decides.
pub fn allowRange(lo: []const f64, hi: []const f64, want: []const f64, allow: []f64) void {
    for (allow, lo, hi, want) |*l, a, b, w| l.* = @max(w - a, b - w, 0) + 0x1p-133;
}

pub fn none(_: c.Shape) usize {
    return 0;
}

pub const no_sums: c.Sums = .{ .f16 = 0, .bf16 = 0, .why = "no sums: elementwise, a gather or a copy" };

pub fn pipelineOf(lib: mtl.Library, device: mtl.Device, name: []const u8) !mtl.Pipeline {
    var buf: [48]u8 = undefined;
    const z = try std.fmt.bufPrintSentinel(&buf, "{s}", .{name}, 0);
    return mtl.Pipeline.init(device, lib, z, false);
}

pub fn random16(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
}

/// NaN, +Inf and -Inf into rows 0, 1 and 2 of `v` (rows `w` wide), as the input families plant them.
pub fn plant(v: []u16, w: usize, rows: usize) void {
    const at = [_][2]usize{ .{ 0, w / 3 }, .{ 1, w / 2 }, .{ 2, w - 1 } };
    const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
    for (at, put) |x, f| if (x[0] < rows) {
        v[x[0] * w + x[1]] = c.toBf16(f);
    };
}

pub fn xv(v: []const u16, i: usize) f64 {
    return c.fromBf16(v[i]);
}

pub const Which = enum { embed, gate, silu_clamp };

pub fn function(w: Which) []const u8 {
    return switch (w) {
        .embed => "rc_embed",
        .gate => "rc_gate",
        .silu_clamp => "rc_silu_clamp",
    };
}

/// Prompt chunks to 2,048 rows and decode windows of 16; the FFN's 40,960 wide rows to 512 (the grid's rows repeat).
pub fn spec(w: Which) c.Spec {
    const q = R.heads * DH;
    return switch (w) {
        .embed => .{ .name = "ricursive glue rc_embed", .accum = .none, .depth = none, .ops = 1, .out = .bf16, .served = &.{ .{ .rows = 16, .n = R.hidden, .k = R.hidden }, .{ .rows = 2048, .n = R.hidden, .k = R.hidden } }, .sums = no_sums },
        .gate => .{ .name = "ricursive glue rc_gate", .accum = .none, .depth = none, .ops = 8, .mids = 1, .out = .bf16, .in_place = true, .served = &.{ .{ .rows = 16, .n = q, .k = q }, .{ .rows = 2048, .n = q, .k = q } }, .sums = no_sums },
        .silu_clamp => .{ .name = "ricursive glue rc_silu_clamp", .accum = .none, .depth = none, .ops = 8, .mids = 1, .out = .bf16, .served = &.{ .{ .rows = 16, .n = R.intermediate, .k = R.intermediate }, .{ .rows = 512, .n = R.intermediate, .k = R.intermediate } }, .sums = no_sums },
    };
}

/// The elementwise glue: the width is the shape's k, the rows its rows.
pub const Pointwise = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    mutation: controls.Mutation,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    x: []u16 = &.{}, // the table (embed), the attention rows (gate), the stacked gate|up rows (silu_clamp)
    y: []u16 = &.{}, // the gate rows (gate)
    ids: []u32 = &.{},
    lo: []f64 = &.{},
    hi: []f64 = &.{},
    in: [2]rig_.Ref = undefined,
    out: [2]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: Which, mutation: controls.Mutation) !Pointwise {
        const lib = try controls.glueLibrary(device, a, mutation);
        errdefer lib.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = which, .mutation = mutation, .lib = lib, .pipe = try pipelineOf(lib, device, function(which)) };
    }

    fn drop(p: *Pointwise) void {
        for ([_]*[]u16{ &p.x, &p.y }) |s| {
            p.a.free(s.*);
            s.* = &.{};
        }
        for ([_]*[]f64{ &p.lo, &p.hi }) |s| {
            p.a.free(s.*);
            s.* = &.{};
        }
        p.a.free(p.ids);
        p.ids = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Pointwise) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
        p.lib.deinit();
    }

    const vt = g.hooks(Pointwise);

    pub fn kernel(p: *Pointwise) g.Kernel {
        return controls.mark(.{ .spec = spec(p.which), .ctx = p, .vt = &vt }, p.mutation);
    }

    pub fn prepare(p: *Pointwise, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k == 0) return false;
        p.drop();
        p.shape = s;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const w = s.k;
        const n = s.rows * w;
        switch (p.which) {
            .embed => {
                p.x = try p.a.alloc(u16, 8 * w);
                random16(rng, p.x, -2, 2);
                if (fill == .nonfinite) plant(p.x, w, 8);
                p.ids = try p.a.alloc(u32, s.rows);
                for (p.ids, 0..) |*id, i| id.* = @intCast((i * 5 + 3) % 8);
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.x));
                p.out[0] = try p.rig.output(n * 2);
                p.out[1] = try p.rig.output(n * 2);
            },
            .gate => {
                p.x = try p.a.alloc(u16, n);
                p.y = try p.a.alloc(u16, n);
                random16(rng, p.x, -4, 4);
                random16(rng, p.y, -8, 8);
                if (fill == .nonfinite) {
                    plant(p.y, w, s.rows);
                    if (s.rows > 4) { // an infinite attention value gated to 0: NaN; a NaN value
                        p.x[3 * w] = c.toBf16(std.math.inf(f32));
                        p.y[3 * w] = c.toBf16(-std.math.inf(f32));
                        p.x[4 * w + 1] = c.toBf16(std.math.nan(f32));
                    }
                }
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.y));
                p.out[0] = try p.rig.output(n * 2);
            },
            .silu_clamp => {
                p.x = try p.a.alloc(u16, 2 * n);
                random16(rng, p.x, -40, 40); // both sides of the clamp
                if (fill == .nonfinite) {
                    plant(p.x, 2 * w, s.rows);
                    // up's NaN and infinities in rows 3 to 5, where each row's gate is finite
                    if (s.rows > 5) for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 3..) |f, r| {
                        p.x[r * 2 * w + w + 7] = c.toBf16(f);
                    };
                }
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
                p.out[0] = try p.rig.output(n * 2);
            },
        }
        p.lo = try p.a.alloc(f64, p.count());
        p.hi = try p.a.alloc(f64, p.count());
        return true;
    }

    pub fn count(p: *Pointwise) usize {
        const n = p.shape.rows * p.shape.k;
        return if (p.which == .embed) 2 * n else n;
    }

    pub fn run(p: *Pointwise) !void {
        const s = p.shape;
        const w: u32 = @intCast(s.k);
        const rows: u32 = @intCast(s.rows);
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        switch (p.which) {
            .embed => {
                e.setBuffer(p.in[0].buf, p.in[0].off, 0);
                e.setBuffer(p.in[1].buf, p.in[1].off, 1);
                e.setValue(Args{ .rows = rows, .width = w, .scale = embed_scale }, 2);
                e.setBuffer(p.out[0].buf, p.out[0].off, 3);
                e.setBuffer(p.out[1].buf, p.out[1].off, 4);
            },
            .gate => {
                @memcpy(p.rig.bytes(p.out[0], p.x.len * 2), std.mem.sliceAsBytes(p.x)); // the rows before the gate
                e.setBuffer(p.out[0].buf, p.out[0].off, 0);
                e.setBuffer(p.in[0].buf, p.in[0].off, 1);
                e.setValue(Args{ .rows = rows, .width = w }, 2);
            },
            .silu_clamp => {
                e.setBuffer(p.in[0].buf, p.in[0].off, 0);
                e.setValue(Args{ .rows = rows, .width = w, .scale = clamp }, 1);
                e.setBuffer(p.out[0].buf, p.out[0].off, 2);
            },
        }
        e.dispatchThreads(mtl.Size.of(s.k, s.rows, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    /// The release's formula in float64 at its rounding points (want), and the range its fp32 steps allow (lo, hi).
    pub fn reference(p: *Pointwise, want: []f64, mag: []f64) !void {
        const s = p.shape;
        const w = s.k;
        const n = s.rows * w;
        for (0..s.rows) |r| for (0..w) |i| {
            const at = r * w + i;
            switch (p.which) {
                .embed => {
                    const v = xv(p.x, p.ids[r] * w + i) * @as(f64, embed_scale);
                    const e = round(widen(point(v), u));
                    for ([_]usize{ at, n + at }) |o| {
                        want[o] = bf(v);
                        p.lo[o] = e[0];
                        p.hi[o] = e[1];
                    }
                },
                .gate => {
                    const gv = xv(p.y, at);
                    const sig = 1.0 / (1.0 + @exp(-gv)); // exp, 1 + e and the division: 8 fp32 ulps
                    const e = round(mul(point(xv(p.x, at)), round(widen(point(sig), 8 * u))));
                    want[at] = bf(xv(p.x, at) * bf(sig));
                    p.lo[at] = e[0];
                    p.hi[at] = e[1];
                },
                .silu_clamp => {
                    const g0 = xv(p.x, r * 2 * w + i);
                    const up0 = xv(p.x, r * 2 * w + w + i);
                    // torch's clamp keeps a NaN
                    const gv = if (std.math.isNan(g0)) g0 else @min(g0, clamp);
                    const up = if (std.math.isNan(up0)) up0 else std.math.clamp(up0, -clamp, clamp);
                    const silu = gv / (1.0 + @exp(-gv));
                    const e = round(mul(round(widen(point(silu), 8 * u)), point(up)));
                    want[at] = bf(bf(silu) * up);
                    p.lo[at] = e[0];
                    p.hi[at] = e[1];
                },
            }
        };
        for (mag, want) |*m, v| m.* = @abs(v);
    }

    pub fn bounds(p: *Pointwise, want: []const f64, _: []const f64, allow: []f64) !void {
        allowRange(p.lo, p.hi, want, allow);
    }

    pub fn intervals(p: *Pointwise, low: []f64, high: []f64) !void {
        @memcpy(low, p.lo);
        @memcpy(high, p.hi);
    }
};
