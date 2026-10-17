//! Affine projections under the contract on Metal: the row projection, affine_mm, simd_qmm (2-8 bits), lane kernels.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const row = tf.row_projection;
const sq = row.simd;
const sqb = row.simd_bits;
const mm = tf.affine_mm;
const lane = tf.lane_projection;

pub const Kind = enum { row, affine_mm, simd, simd_bits, lane, lane_reg };

/// One registered variant: the kernel, its code width and group, its output, and the lane kernel's K slices.
pub const Variant = struct { kind: Kind, bits: u8 = 4, group: usize = 64, f32_out: bool = false, sk: usize = 1, coop: bool = false };

/// Each 16-input piece's products, two roundings a piece, a lane's pieces in turn, then a simd_sum tree.
fn rowDepth(s: c.Shape) usize {
    return 16 + 3 * ((s.k + 511) / 512) + 8;
}

/// A group's products in the tile op, its scale and bias, the groups in turn (tf-affine-ref's bound).
fn groupDepth64(s: c.Shape) usize {
    return 64 + s.k / 32 + 8;
}

/// A group's 64 products in eight MMA steps, its sum, the groups a chunk takes, then the chunks' fixed tree.
fn simdDepth(s: c.Shape) usize {
    return 64 + 11 + s.k / 64 + 8;
}

/// A group's products, two FMAs a group, then the K slices in order (core_lane_projection.zig's oracle).
fn laneDepth(s: c.Shape) usize {
    return 128 + 2 * (s.k / 32) + 32;
}

/// affine_mm's 4-bit codes enter as q + 128 and its bias as b - 128 s, so an infinite input meets opposite signs.
const offset_inf = c.Known{ .gate = .nonfinite, .finding = "an infinite input gives NaN where MLX's s*sum(q x) + b*sum(x) keeps the infinity (the +128 code offset moves the bias to b - 128 s)" };

pub fn spec(v: Variant, name: []const u8) c.Spec {
    const out: c.Elem = if (v.f32_out) .f32 else .bf16;
    const step: usize = @max(16, v.group);
    return switch (v.kind) {
        .row => .{ .name = name, .accum = .f32, .depth = rowDepth, .out = out, .tiles = .{ .{ .size = 2 }, .{ .size = 8, .step = 8, .min = 8 }, .{ .size = 512, .step = step, .min = step } }, .served = &.{ .{ .rows = 16, .n = 2688, .k = 4096 }, .{ .rows = 3, .n = 4608, .k = 2688 } } },
        .affine_mm => .{ .name = name, .accum = .f32, .depth = groupDepth64, .out = out, .known = if (v.bits == 4) offset_inf else null, .tiles = .{ .{ .size = 64 }, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 128, .step = 64, .min = 64 } }, .served = &.{ .{ .rows = 128, .n = 2048, .k = 4096 }, .{ .rows = 65, .n = 1024, .k = 1024 } } },
        .simd, .simd_bits => .{ .name = name, .accum = .f32, .depth = simdDepth, .out = out, .tiles = .{ .{ .size = 8 }, .{ .size = 32, .step = 8, .min = 8 }, .{ .size = 512, .step = 64, .min = 64 } }, .served = &.{ .{ .rows = 16, .n = 2688, .k = 4096 }, .{ .rows = 9, .n = 6400, .k = 256 } } },
        .lane, .lane_reg => .{ .name = name, .accum = .f32, .depth = laneDepth, .out = out, .replaces = if (v.kind == .lane_reg) "the cooperative lane kernel" else null, .tiles = .{ .{ .size = 16 }, .{ .size = 32, .step = 16, .min = 32 }, .{ .size = 4 * v.group, .step = v.group, .min = v.group } }, .served = &.{ .{ .rows = 16, .n = 2048, .k = 2048 }, .{ .rows = 5, .n = 512, .k = 896 } } },
    };
}

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    row_p: ?row.Pipelines = null,
    sq_p: ?sq.Pipelines = null,
    sqb_p: ?sqb.Pipelines = null,
    mm_p: ?mm.Pipes = null,
    lane_p: ?lane.Projection = null,
    coop_p: ?lane.Projection = null,
    order: ?lane.RegOrder = null,
    f: ?Affine = null,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    s: rig_.Ref = undefined,
    b: rig_.Ref = undefined,
    sums: rig_.Ref = undefined,
    tiled: rig_.Ref = undefined,
    base: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        var p = Proj{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name };
        switch (v.kind) {
            .row => p.row_p = try row.Pipelines.load(device),
            .simd => p.sq_p = try sq.Pipelines.load(device),
            .simd_bits => p.sqb_p = try sqb.Pipelines.load(device),
            .affine_mm => {
                const src = try mm.source(device, a, .{ .bits = v.bits, .group = @intCast(v.group), .out_f32 = v.f32_out });
                defer a.free(src);
                const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
                defer lib.deinit();
                var pipes: mm.Pipes = undefined;
                for (mm.names, 0..) |n, i| pipes[i] = try mtl.Pipeline.init(device, lib, n, false);
                p.mm_p = pipes;
            },
            .lane => {},
            .lane_reg => p.order = try lane.RegOrder.init(device),
        }
        return p;
    }

    pub fn deinit(p: *Proj) void {
        p.drop();
        if (p.row_p) |*x| x.deinit();
        if (p.sq_p) |*x| x.deinit();
        if (p.sqb_p) |*x| x.deinit();
        if (p.mm_p) |pipes| for (pipes) |x| x.deinit();
        if (p.order) |o| o.deinit();
        p.rig.deinit();
    }

    fn drop(p: *Proj) void {
        if (p.f) |f| f.deinit();
        p.f = null;
        if (p.lane_p) |x| x.deinit();
        p.lane_p = null;
        if (p.coop_p) |x| x.deinit();
        p.coop_p = null;
        p.rig.reset();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        return .{ .spec = spec(p.v, p.name), .ctx = p, .vt = &vt };
    }

    const vt = g.hooks(Proj);

    fn layout(p: *const Proj, shape: c.Shape, coop: bool) lane.Layout {
        const reg = p.v.kind == .lane_reg and !coop;
        return .{ .n = shape.n, .k = shape.k, .format = .{ .bits = p.v.bits, .group = p.v.group }, .sk = p.v.sk, .output = if (p.v.f32_out) .f32 else .bf16, .precompute_sums = p.v.kind == .lane_reg or p.v.coop, .cooperative = coop or (p.v.coop and !reg), .reg = reg };
    }

    /// Whether the kernel takes this shape, as the engine's own validation says.
    fn takes(p: *Proj, shape: c.Shape) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % p.v.group != 0) return false;
        return switch (p.v.kind) {
            .row => if ((row.Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = shape.n, .k = shape.k, .group = p.v.group, .bits = p.v.bits }).validate()) |_| true else |_| false,
            .simd, .simd_bits => shape.n % 8 == 0 and shape.k % 64 == 0,
            .affine_mm => shape.n % 64 == 0 and shape.k % 64 == 0,
            .lane, .lane_reg => shape.rows <= 16 and if (p.layout(shape, false).validate()) |_| true else |_| false,
        };
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (!try p.takes(shape)) return false;
        var f = try Affine.init(p.a, shape, p.v.bits, p.v.group);
        if (p.v.kind == .affine_mm and p.v.bits == 4) f.off = 128;
        p.f = f;
        try f.fill(fill, seed);
        const words = try p.a.alloc(u32, shape.n * shape.k * p.v.bits / 32);
        defer p.a.free(words);
        f.words(words);
        p.x = try p.rig.input(std.mem.sliceAsBytes(f.x));
        const out_bytes = f.outputs() * c.size(spec(p.v, p.name).out);
        switch (p.v.kind) {
            .row, .simd, .simd_bits, .affine_mm => {
                p.w = try p.rig.input(std.mem.sliceAsBytes(words));
                p.s = try p.rig.input(std.mem.sliceAsBytes(f.scales));
                p.b = try p.rig.input(std.mem.sliceAsBytes(f.biases));
                if (p.v.kind == .affine_mm) p.sums = try p.rig.scratch(shape.rows * (shape.k / p.v.group) * 4);
            },
            .lane, .lane_reg => {
                const l = p.layout(shape, false);
                p.lane_p = try lane.Projection.init(p.a, p.rig.device, l);
                const pw = try p.a.alloc(u32, l.weightWords());
                defer p.a.free(pw);
                const psb = try p.a.alloc(u16, l.metadataElements());
                defer p.a.free(psb);
                try lane.pack(l, words, f.scales, f.biases, pw, psb);
                p.w = try p.rig.input(std.mem.sliceAsBytes(pw));
                p.s = try p.rig.input(std.mem.sliceAsBytes(psb));
                if (p.v.kind == .lane_reg) {
                    p.tiled = try p.rig.input(std.mem.sliceAsBytes(pw));
                    try p.order.?.run(&.{.{ .w = .{ .buf = p.w.buf, .off = p.w.off }, .n = shape.n, .k = shape.k }});
                    p.coop_p = try lane.Projection.init(p.a, p.rig.device, p.layout(shape, true));
                    p.base = try p.rig.scratch(out_bytes);
                }
            },
        }
        p.out = try p.rig.output(out_bytes);
        return true;
    }

    fn ref(r: rig_.Ref) lane.Ref {
        return .{ .buf = r.buf, .off = r.off };
    }

    pub fn run(p: *Proj) !void {
        const f = p.f.?;
        const shape = f.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        switch (p.v.kind) {
            .row => {
                const cl = try row.call(&p.row_p.?, .{ .w = p.w.buf, .w_off = p.w.off, .scales = p.s.buf, .s_off = p.s.off, .biases = p.b.buf, .b_off = p.b.off, .n = shape.n, .k = shape.k, .group = p.v.group, .bits = p.v.bits }, shape.rows, false);
                e.setPipeline(cl.pipeline);
                for ([_]rig_.Ref{ p.x, p.w, p.s, p.b }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.setValue(cl.dims, 4);
                e.setBuffer(p.out.buf, p.out.off, 5);
                e.dispatchGroups(mtl.Size.of(cl.groups, cl.slots, 1), mtl.Size.of(cl.threads, 1, 1));
            },
            .simd => {
                const cl = try sq.call(&p.sq_p.?, shape.n, shape.k, shape.rows, false);
                sq.encode(e, cl, .{ .w = p.w.buf, .w_off = p.w.off, .scales = p.s.buf, .s_off = p.s.off, .biases = p.b.buf, .b_off = p.b.off, .n = shape.n, .k = shape.k }, p.x.buf, p.x.off, p.out.buf, p.out.off);
            },
            .simd_bits => {
                const cl = try sqb.call(&p.sqb_p.?, p.v.bits, shape.n, shape.k, shape.rows, shape.n, false);
                sqb.encode(e, cl, .{ .w = p.w.buf, .w_off = p.w.off, .scales = p.s.buf, .s_off = p.s.off, .biases = p.b.buf, .b_off = p.b.off }, p.x.buf, p.x.off, p.out.buf, p.out.off);
            },
            .affine_mm => {
                const x = .{ .buf = p.x.buf, .off = p.x.off };
                const xs = .{ .buf = p.sums.buf, .off = p.sums.off };
                const q = .{ .w = .{ .buf = p.w.buf, .off = p.w.off }, .s = .{ .buf = p.s.buf, .off = p.s.off }, .b = .{ .buf = p.b.buf, .off = p.b.off }, .n = @as(u32, @intCast(shape.n)), .k = @as(u32, @intCast(shape.k)) };
                mm.rowSums(e, p.mm_p.?, @intCast(p.v.group), x, xs, @intCast(shape.rows), @intCast(shape.k));
                mm.dense(e, p.mm_p.?, x, xs, q, .{ .buf = p.out.buf, .off = p.out.off }, @intCast(shape.rows));
            },
            .lane, .lane_reg => try p.lane_p.?.encode(e, ref(p.x), ref(p.w), ref(p.s), ref(p.out), @intCast(shape.rows)),
        }
        try p.rig.finish(pass);
    }

    /// The cooperative kernel's bytes on the same inputs, from the weights before the reg order.
    pub fn replaced(p: *Proj) ![]const u8 {
        if (p.v.kind != .lane_reg) return error.NotAReplacement;
        const f = p.f.?;
        const len = f.outputs() * c.size(spec(p.v, p.name).out);
        @memset(p.rig.bytes(p.base, len), 0xff);
        const pass = p.rig.begin();
        try p.coop_p.?.encode(pass.e, ref(p.x), ref(p.tiled), ref(p.s), ref(p.base), @intCast(f.shape.rows));
        try p.rig.finish(pass);
        return p.rig.bytes(p.base, len);
    }

    pub fn count(p: *Proj) usize {
        return p.f.?.outputs();
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        return p.f.?.reference(want, mag);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.f.?.simulate(acc, order, out);
    }
};
