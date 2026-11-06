//! The image encoder's glue under the contract: adds, norms, GELUs and CLIP's embedding and concat rows.
const std = @import("std");
const cuda = @import("cuda");
const rig_ = @import("contract_rig.zig");
const hub = @import("contract_uocr.zig");
const v = @import("contract_uocr_vision.zig");
const mi = @import("contract_moss_image_ref.zig");

const c = v.c;
const g = v.g;
const rows = v.rows;
const ks = hub.kernels;
const S = hub.sam_width;
const C = hub.clip_width;

pub const Kind = enum { add, concat, embed, gelu, quick_gelu, norm };

pub fn function(k: Kind) []const u8 {
    return switch (k) {
        .add => "add",
        .concat => "concat",
        .embed => "clip_embed",
        .gelu => "gelu",
        .quick_gelu => "quick_gelu",
        .norm => "layer_norm",
    };
}

fn control(fun: [:0]const u8, what: []const u8, gate: g.Gate) v.Control {
    return .{ .function = fun, .name = what, .gate = gate };
}

/// Each kernel's planted controls (contract_uocr_glue.cuh).
pub const controls = [_]struct { Kind, v.Control }{
    .{ .add, control("tf_ctl_uocr_add_subtracts", "control: cuda uocr vision tf_uocr_add, the rows are subtracted", .shapes) },
    .{ .add, control("tf_ctl_uocr_add_writes_input", "control: cuda uocr vision tf_uocr_add, each sum also written over its first input", .repeat) },
    .{ .norm, control("tf_ctl_uocr_layer_norm_threads", "control: cuda uocr vision tf_uocr_layer_norm, the mean divides by the threads", .shapes) },
    .{ .gelu, control("tf_ctl_uocr_gelu_scale", "control: cuda uocr vision tf_uocr_gelu, erf's argument is x times 0.75", .shapes) },
    .{ .quick_gelu, control("tf_ctl_uocr_quick_gelu_unrounded", "control: cuda uocr vision tf_uocr_quick_gelu, the sigmoid is not rounded to bf16", .shapes) },
    .{ .concat, control("tf_ctl_uocr_concat_left_twice", "control: cuda uocr vision tf_uocr_concat, the right half repeats the left rows", .shapes) },
    .{ .embed, control("tf_ctl_uocr_clip_embed_row", "control: cuda uocr vision tf_uocr_clip_embed, row r adds SAM's row r - 2", .shapes) },
};

/// Four inputs at most a thread in order (a 768- to 1,024-wide row over 256 threads), then two 32-way warp trees.
fn normDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 10;
}

pub fn spec(k: Kind) c.Spec {
    const flat: [3]c.Tile = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} };
    return switch (k) {
        // SAM's map (1024 view and tile) and CLIP's rows (1024 view and tile)
        .add => .{ .name = "cuda uocr vision tf_uocr_add", .accum = .none, .depth = v.noSums, .ops = 1, .out = .bf16, .tiles = flat, .served = &.{ .{ .rows = 4096, .n = S, .k = 1 }, .{ .rows = 1600, .n = S, .k = 1 }, .{ .rows = 257, .n = C, .k = 1 }, .{ .rows = 101, .n = C, .k = 1 } } },
        .concat => .{ .name = "cuda uocr vision tf_uocr_concat", .accum = .none, .depth = v.noSums, .out = .bf16, .served = &.{ .{ .rows = 256, .n = 2 * C, .k = 1 }, .{ .rows = 100, .n = 2 * C, .k = 1 } } },
        .embed => .{ .name = "cuda uocr vision tf_uocr_clip_embed", .accum = .none, .depth = v.noSums, .ops = 1, .out = .bf16, .served = &.{ .{ .rows = 257, .n = C, .k = 1 }, .{ .rows = 101, .n = C, .k = 1 } } },
        .gelu => .{ .name = "cuda uocr vision tf_uocr_gelu (erf)", .accum = .none, .depth = v.noSums, .ops = 6, .out = .bf16, .tiles = flat, .served = &.{ .{ .rows = 4096, .n = 4 * S, .k = 1 }, .{ .rows = 1600, .n = 4 * S, .k = 1 } } },
        .quick_gelu => .{ .name = "cuda uocr vision tf_uocr_quick_gelu", .accum = .none, .depth = v.noSums, .out = .bf16, .tiles = flat, .served = &.{ .{ .rows = 257, .n = 4 * C, .k = 1 }, .{ .rows = 101, .n = 4 * C, .k = 1 } } },
        // SAM's blocks (eps 1e-6), its neck's channel norms on rows of 256, CLIP's rows (eps 1e-5)
        .norm => .{ .name = "cuda uocr vision tf_uocr_layer_norm", .accum = .f32, .depth = normDepth, .ops = 8, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 256, .step = 1, .min = 1 } }, .served = &.{ .{ .rows = 4096, .n = S, .k = S }, .{ .rows = 4096, .n = 256, .k = 256 }, .{ .rows = 257, .n = C, .k = C }, .{ .rows = 1600, .n = 256, .k = 256 } } },
    };
}

/// The served width a launch's eps belongs to: CLIP's 1,024-wide rows take 1e-5, SAM's 1e-6.
fn epsOf(width: usize) f32 {
    return if (width == C) 1e-5 else 1e-6;
}

pub const Glue = struct {
    rig: rig_.Rig,
    k: ks.Kernels,
    kind: Kind,
    control: ?v.Control,
    shape: c.Shape = undefined,
    host: [3][]u16 = .{ &.{}, &.{}, &.{} },
    r: ?rows.Rows = null,
    ins: [3]u64 = @splat(0),
    out: u64 = 0,

    pub fn init(rig: rig_.Rig, k: ks.Kernels, kind: Kind, ctl: ?v.Control) Glue {
        return .{ .rig = rig, .k = k, .kind = kind, .control = ctl };
    }
    pub fn deinit(t: *Glue) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Glue) void {
        for (&t.host) |*h| {
            t.rig.gpu.gpa.free(h.*);
            h.* = &.{};
        }
        if (t.r) |r| r.deinit();
        t.r = null;
    }
    pub fn kernel(t: *Glue) g.Kernel {
        return v.mark(.{ .spec = spec(t.kind), .ctx = t, .vt = &vt }, t.control);
    }
    const vt = g.hooks(Glue);

    fn bf16s(t: *Glue, i: usize, n: usize) ![]u16 {
        t.host[i] = try t.rig.gpu.gpa.alloc(u16, n);
        return t.host[i];
    }

    pub fn prepare(t: *Glue, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        if (t.kind == .concat and shape.n % 2 != 0) return false;
        if (t.kind == .embed and shape.rows < 2) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * shape.n;
        const nf = fill == .nonfinite;
        const P = v.planted;
        switch (t.kind) {
            .add => {
                const x = try t.bf16s(0, n);
                const y = try t.bf16s(1, n);
                v.fill(rng, x, -4, 4);
                v.fill(rng, y, -4, 4);
                // a NaN, both infinities, and +inf meeting -inf (NaN)
                if (nf and n > 4) {
                    x[1] = c.toBf16(P[0]);
                    x[2] = c.toBf16(P[1]);
                    y[3] = c.toBf16(P[2]);
                    x[4] = c.toBf16(P[1]);
                    y[4] = c.toBf16(P[2]);
                }
            },
            .concat => {
                for (0..2) |i| v.fill(rng, try t.bf16s(i, n / 2), -4, 4);
                if (nf) {
                    t.host[0][0] = c.toBf16(P[0]);
                    t.host[1][0] = c.toBf16(P[1]);
                    t.host[1][t.host[1].len - 1] = c.toBf16(P[2]);
                }
            },
            .embed => {
                v.fill(rng, try t.bf16s(0, shape.n), -1, 1);
                v.fill(rng, try t.bf16s(1, (shape.rows - 1) * shape.n), -4, 4);
                v.fill(rng, try t.bf16s(2, n), -1, 1);
                if (nf) {
                    t.host[0][0] = c.toBf16(P[0]);
                    t.host[1][1] = c.toBf16(P[1]);
                    t.host[2][shape.n + 2] = c.toBf16(P[2]);
                }
            },
            .gelu, .quick_gelu => {
                const x = try t.bf16s(0, n);
                v.fill(rng, x, -8, 8);
                if (nf) for (P, 0..) |val, i| {
                    if (i < n) x[i * 7 % n] = c.toBf16(val);
                };
            },
            .norm => {
                var r = try rows.Rows.init(t.rig.gpu.gpa, .layer_norm, shape);
                r.eps = epsOf(shape.k);
                r.fill(fill, seed);
                t.r = r;
                t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(r.x));
                t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(r.w));
                t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(r.b));
                t.out = try t.rig.output(r.outputs() * 2);
                return true;
            },
        }
        for (t.host, 0..) |h, i| if (h.len > 0) {
            t.ins[i] = try t.rig.input(std.mem.sliceAsBytes(h));
        };
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Glue) !void {
        const sh = t.shape;
        const s = t.rig.s;
        const n = sh.rows * sh.n;
        switch (t.kind) {
            .add => try t.k.run("add", s, .{ ks.blocks(n), 1, 1 }, 256, .{ t.ins[0], t.ins[1], t.out, @as(i64, @intCast(n)) }),
            .concat => try t.k.run("concat", s, .{ sh.rows, ks.blocks(sh.n), 1 }, 256, .{ t.ins[0], t.ins[1], t.out, sh.rows, sh.n / 2, sh.n / 2 }),
            .embed => try t.k.run("clip_embed", s, .{ sh.rows, ks.blocks(sh.n), 1 }, 256, .{ t.ins[0], t.ins[1], t.ins[2], t.out, sh.n }),
            inline .gelu, .quick_gelu => |kd| {
                // the activations run in place: each launch starts from the prepared input copied into the output
                try (cuda.DeviceBuffer{ .d = t.rig.gpu.d, .ptr = t.out, .len = n * 2 }).copyFrom(0, t.ins[0], n * 2, s.handle);
                try t.k.run(comptime function(kd), s, .{ ks.blocks(n), 1, 1 }, 256, .{ t.out, @as(i64, @intCast(n)) });
            },
            .norm => try t.k.run("layer_norm", s, .{ sh.rows, 1, 1 }, 256, .{ t.ins[0], t.ins[1], t.ins[2], t.out, sh.k, epsOf(sh.k) }),
        }
        try s.synchronize();
    }

    pub fn count(t: *Glue) usize {
        return if (t.kind == .norm) t.shape.rows * t.shape.k else t.shape.rows * t.shape.n;
    }

    fn hv(t: *Glue, i: usize, at: usize) f64 {
        return c.fromBf16(t.host[i][at]);
    }

    /// quick_gelu's three bf16 points: z = bf16(fp32(1.702 x)), s = bf16(sigmoid z), bf16(x s).
    fn quick(x: f64) [2]f64 {
        const z = mi.bf(@as(f32, @floatCast(@as(f64, @as(f32, 1.702)) * x)));
        return .{ z, mi.bf(1 / (1 + @exp(-z))) };
    }

    pub fn reference(t: *Glue, want: []f64, mag: []f64) !void {
        if (t.r) |r| return r.reference(want, mag);
        const w = t.shape.n;
        for (want, mag, 0..) |*o, *m, i| {
            m.* = 0;
            o.* = switch (t.kind) {
                .add => t.hv(0, i) + t.hv(1, i),
                .concat => if (i % w < w / 2) t.hv(0, (i / w) * (w / 2) + i % w) else t.hv(1, (i / w) * (w / 2) + i % w - w / 2),
                .embed => (if (i < w) t.hv(0, i) else t.hv(1, i - w)) + t.hv(2, i),
                .gelu => rows.gelu(t.hv(0, i)),
                .quick_gelu => blk: {
                    const x = t.hv(0, i);
                    break :blk mi.bf(x * quick(x)[1]);
                },
                .norm => unreachable,
            };
        }
    }

    /// GELU's erf (under one ulp) and its fp32 steps: an absolute slack of x / 2 times 2^-22, where 1 + erf cancels.
    pub fn bounds(t: *Glue, _: []const f64, _: []const f64, allow: []f64) !void {
        if (t.kind != .gelu) return;
        for (allow, 0..) |*l, i| l.* += rows.geluSlack(t.hv(0, i));
    }

    /// quick_gelu's range: the sigmoid's fp32 exp, add and divide (16 ulps), each bf16 point, the product's rounding.
    pub fn intervals(t: *Glue, low: []f64, high: []f64) !void {
        for (low, high, 0..) |*l, *hi, i| {
            if (t.kind != .quick_gelu) {
                l.* = -std.math.floatMax(f64);
                hi.* = std.math.floatMax(f64);
                continue;
            }
            const x = t.hv(0, i);
            const z = quick(x)[0];
            const s = 1 / (1 + @exp(-z));
            const sr = mi.Range.of(s).fp32(8).bf16().mul(mi.Range.of(x));
            const r = (mi.Range{ .lo = sr.lo - 0x1p-24 * @abs(x), .hi = sr.hi + 0x1p-24 * @abs(x) }).bf16();
            l.* = r.lo;
            hi.* = r.hi;
        }
    }

    pub fn simulate(t: *Glue, acc: c.Accum, order: g.Order, out: []f64) !void {
        return (t.r orelse return error.NoSums).simulate(acc, order, out);
    }
};

/// The glue kernels and their controls, appended to `list`.
pub fn register(ctx: hub.Ctx, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !hub.Held {
    return v.register(@This(), Glue, ctx, a, list);
}
