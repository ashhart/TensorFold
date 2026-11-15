//! The image encoder's glue under the contract: adds, biases, norms, GELUs and CLIP's embedding and concat rows.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const hub = @import("metal_uocr.zig");
const v = @import("metal_uocr_vision.zig");

const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const Mutation = v.Mutation;

pub const Kind = enum { add, bias, concat, embed, gelu, quick_gelu, norm };

fn function(k: Kind) []const u8 {
    return switch (k) {
        .add => "uv_add",
        .bias => "uv_bias",
        .concat => "uv_concat",
        .embed => "uv_clip_embed",
        .gelu => "uv_gelu",
        .quick_gelu => "uv_quick_gelu",
        .norm => "uv_layer_norm",
    };
}

/// Four inputs at most a thread in order (a 768- to 1,024-wide row over 256 threads), then two 32-way simd_sum trees.
fn normDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 10;
}

const S = hub.sam_width;
const C = hub.clip_width;

pub fn spec(k: Kind) c.Spec {
    const flat: [3]c.Tile = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} };
    return switch (k) {
        // SAM's map (1024 view and tile) and CLIP's rows (1024 view and tile)
        .add => .{ .name = "uocr vision uv_add", .accum = .none, .depth = hub.noSums, .ops = 1, .out = .bf16, .tiles = flat, .served = &.{ .{ .rows = 4096, .n = S, .k = 1 }, .{ .rows = 1600, .n = S, .k = 1 }, .{ .rows = 257, .n = C, .k = 1 }, .{ .rows = 101, .n = C, .k = 1 } } },
        // SAM's qkv over the padded windows, its MLP, CLIP's qkv and MLP, the projector
        .bias => .{ .name = "uocr vision uv_bias", .accum = .none, .depth = hub.noSums, .ops = 1, .out = .bf16, .tiles = flat, .served = &.{ .{ .rows = 4900, .n = 3 * S, .k = 1 }, .{ .rows = 4096, .n = 4 * S, .k = 1 }, .{ .rows = 257, .n = 4 * C, .k = 1 }, .{ .rows = 100, .n = hub.hidden, .k = 1 } } },
        .concat => .{ .name = "uocr vision uv_concat", .accum = .none, .depth = hub.noSums, .out = .bf16, .served = &.{ .{ .rows = 256, .n = 2 * C, .k = 1 }, .{ .rows = 100, .n = 2 * C, .k = 1 } } },
        .embed => .{ .name = "uocr vision uv_clip_embed", .accum = .none, .depth = hub.noSums, .ops = 1, .out = .bf16, .served = &.{ .{ .rows = 257, .n = C, .k = 1 }, .{ .rows = 101, .n = C, .k = 1 } } },
        .gelu => .{ .name = "uocr vision uv_gelu (erf)", .accum = .none, .depth = hub.noSums, .ops = 6, .out = .bf16, .tiles = flat, .served = &.{ .{ .rows = 4096, .n = 4 * S, .k = 1 }, .{ .rows = 1600, .n = 4 * S, .k = 1 } } },
        .quick_gelu => .{ .name = "uocr vision uv_quick_gelu", .accum = .none, .depth = hub.noSums, .out = .bf16, .tiles = flat, .served = &.{ .{ .rows = 257, .n = 4 * C, .k = 1 }, .{ .rows = 101, .n = 4 * C, .k = 1 } } },
        // SAM's blocks (eps 1e-6), its neck's channel norms on rows of 256, CLIP's rows (eps 1e-5)
        .norm => .{ .name = "uocr vision uv_layer_norm", .accum = .f32, .depth = normDepth, .ops = 8, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 256, .step = 1, .min = 1 } }, .served = &.{ .{ .rows = 4096, .n = S, .k = S }, .{ .rows = 4096, .n = 256, .k = 256 }, .{ .rows = 257, .n = C, .k = C }, .{ .rows = 1600, .n = 256, .k = 256 } } },
    };
}

/// The served width a launch's eps belongs to: CLIP's 1,024-wide rows take 1e-5, SAM's 1e-6.
fn epsOf(width: usize) f32 {
    return if (width == C) 1e-5 else 1e-6;
}

pub const Glue = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    kind: Kind,
    m: Mutation,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    /// Host inputs as bytes: up to three (bias: fp32 sums then bf16 bias; norm: x, w, b).
    host: [3][]u8 = .{ &.{}, &.{}, &.{} },
    r: ?rows.Rows = null,
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, m: Mutation) !Glue {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .kind = kind, .m = m, .pipe = try v.pipeline(a, device, function(kind), m) };
    }
    pub fn deinit(t: *Glue) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Glue) void {
        for (&t.host) |*h| {
            t.a.free(h.*);
            h.* = &.{};
        }
        if (t.r) |r| r.deinit();
        t.r = null;
    }
    pub fn kernel(t: *Glue) g.Kernel {
        return v.mark(.{ .spec = spec(t.kind), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Glue);

    fn halves(t: *Glue, i: usize) []u16 {
        return @alignCast(std.mem.bytesAsSlice(u16, t.host[i]));
    }

    fn bf16s(t: *Glue, i: usize, n: usize) ![]u16 {
        t.host[i] = std.mem.sliceAsBytes(try t.a.alloc(u16, n));
        return t.halves(i);
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
        const P = hub.planted;
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
            .bias => {
                const cs = try t.a.alloc(f32, n);
                for (cs) |*s| s.* = (rng.float(f32) * 2 - 1) * 8;
                if (nf) for ([_]usize{ 1, 2 }, [_]f32{ P[0], P[1] }) |i, val| {
                    if (i < n) cs[i] = val;
                };
                t.host[0] = std.mem.sliceAsBytes(cs);
                const b = try t.bf16s(1, shape.n);
                v.fill(rng, b, -2, 2);
                if (nf) b[shape.n - 1] = c.toBf16(P[2]);
            },
            .concat => {
                for (0..2) |i| v.fill(rng, try t.bf16s(i, shape.rows * shape.n / 2), -4, 4);
                if (nf) {
                    t.halves(0)[0] = c.toBf16(P[0]);
                    t.halves(1)[0] = c.toBf16(P[1]);
                    t.halves(1)[t.halves(1).len - 1] = c.toBf16(P[2]);
                }
            },
            .embed => {
                v.fill(rng, try t.bf16s(0, shape.n), -1, 1);
                v.fill(rng, try t.bf16s(1, (shape.rows - 1) * shape.n), -4, 4);
                v.fill(rng, try t.bf16s(2, n), -1, 1);
                if (nf) {
                    t.halves(0)[0] = c.toBf16(P[0]);
                    t.halves(1)[1] = c.toBf16(P[1]);
                    t.halves(2)[shape.n + 2] = c.toBf16(P[2]);
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
                var r = try rows.Rows.init(t.a, .layer_norm, shape);
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
            t.ins[i] = try t.rig.input(h);
        };
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Glue) !void {
        const s = t.shape;
        const n: u32 = @intCast(s.rows * s.n);
        const w: u32 = @intCast(s.n);
        // the activations run in place: each launch starts from the prepared input copied into the output
        if (t.kind == .gelu or t.kind == .quick_gelu) @memcpy(t.rig.bytes(t.out, n * 2), t.host[0]);
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        const tg = mtl.Size.of(256, 1, 1);
        switch (t.kind) {
            .add => {
                for (t.ins[0..2], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.setBuffer(t.out.buf, t.out.off, 2);
                e.setValue(n, 3);
                e.dispatchThreads(mtl.Size.of(n, 1, 1), tg);
            },
            .bias => {
                for (t.ins[0..2], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.setBuffer(t.out.buf, t.out.off, 2);
                e.setValue(w, 3);
                e.dispatchThreads(mtl.Size.of(s.n, s.rows, 1), tg);
            },
            .concat => {
                for (t.ins[0..2], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.setBuffer(t.out.buf, t.out.off, 2);
                e.setValue([3]u32{ @intCast(s.rows), w / 2, w / 2 }, 3);
                e.dispatchThreads(mtl.Size.of(s.n, s.rows, 1), tg);
            },
            .embed => {
                for (t.ins, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.setBuffer(t.out.buf, t.out.off, 3);
                e.setValue(w, 4);
                e.dispatchThreads(mtl.Size.of(s.n, s.rows, 1), tg);
            },
            .gelu, .quick_gelu => {
                e.setBuffer(t.out.buf, t.out.off, 0);
                e.setValue(n, 1);
                e.dispatchThreads(mtl.Size.of(n, 1, 1), tg);
            },
            .norm => {
                for (t.ins, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.setBuffer(t.out.buf, t.out.off, 3);
                e.setValue(extern struct { width: u32, eps: f32 }{ .width = @intCast(s.k), .eps = epsOf(s.k) }, 4);
                e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), tg);
            },
        }
        try t.rig.finish(p);
    }

    pub fn count(t: *Glue) usize {
        return if (t.kind == .norm) t.shape.rows * t.shape.k else t.shape.rows * t.shape.n;
    }

    fn hv(t: *Glue, i: usize, at: usize) f64 {
        return c.fromBf16(t.halves(i)[at]);
    }

    /// quick_gelu's three bf16 points: z = bf16(fp32(1.702 x)), s = bf16(sigmoid z), bf16(x s).
    fn quick(x: f64) [2]f64 {
        const z = hub.bf(@as(f32, @floatCast(@as(f64, @as(f32, 1.702)) * x)));
        return .{ z, hub.bf(1 / (1 + @exp(-z))) };
    }

    pub fn reference(t: *Glue, want: []f64, mag: []f64) !void {
        if (t.r) |r| return r.reference(want, mag);
        const w = t.shape.n;
        for (want, mag, 0..) |*o, *m, i| {
            m.* = 0;
            o.* = switch (t.kind) {
                .add => t.hv(0, i) + t.hv(1, i),
                .bias => @as(f64, std.mem.bytesAsSlice(f32, t.host[0])[i]) + t.hv(1, i % w),
                .concat => if (i % w < w / 2) t.hv(0, (i / w) * (w / 2) + i % w) else t.hv(1, (i / w) * (w / 2) + i % w - w / 2),
                .embed => (if (i < w) t.hv(0, i) else t.hv(1, i - w)) + t.hv(2, i),
                .gelu => rows.gelu(t.hv(0, i)),
                .quick_gelu => blk: {
                    const x = t.hv(0, i);
                    break :blk hub.bf(x * quick(x)[1]);
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
            const r = v.Range.point(s).widen(16 * 0x1p-24 * s).bf16().times(x).widen(0x1p-24 * @abs(x)).bf16();
            l.* = r.lo;
            hi.* = r.hi;
        }
    }

    pub fn simulate(t: *Glue, acc: c.Accum, order: g.Order, out: []f64) !void {
        return (t.r orelse return error.NoSums).simulate(acc, order, out);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    made: std.ArrayList(*Glue) = .empty,

    const controls = [_]struct { Kind, Mutation }{
        .{ .add, .add_subtracts },           .{ .add, .add_writes_input },  .{ .bias, .bias_next_column },
        .{ .norm, .norm_mean_over_threads }, .{ .gelu, .gelu_wrong_scale }, .{ .quick_gelu, .quick_gelu_unrounded },
        .{ .concat, .concat_left_twice },    .{ .embed, .embed_own_row },
    };

    fn one(s: *Suite, kind: Kind, m: Mutation, only: ?[]const u8) !void {
        if (!v.wanted(only, if (m == .none) spec(kind).name else m.name())) return;
        const x = try s.a.create(Glue);
        x.* = try Glue.init(s.a, s.device, s.queue, kind, m);
        try s.made.append(s.a, x);
        try s.kernels.append(s.a, x.kernel());
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(Kind)) |kind| try s.one(kind, .none, only);
        for (controls) |ctl| try s.one(ctl[0], ctl[1], only);
    }
    pub fn deinit(s: *Suite) void {
        for (s.made.items) |x| x.deinit();
        s.made.deinit(s.a);
    }
};
