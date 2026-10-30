//! DeepSeek V4.1's hyper-connection boundary under the contract on Metal: the mixes, the collapse, the expand.

//! Each baseline is checked bit-for-bit against the phase-1 glue's hcmix/hcpre/hcpost on the same bytes.

//! Every registration is checked against the float64 model in dsv41_hc_ref.zig.

//! Controls: 1 non-finite inputs flushed to zero, 2 a formula term lost, 3 a sum in fp16, 4 a write past the end.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_hc_ref.zig");

const c = contract.spec;
const g = contract.gates;
const hcm = tf.dsv41_hc;
const glue = tf.dsv41_glue_tmp;
const base = "dsv41 hc ";
const mult = ref.mult;
const dim: usize = 4 * 5120; // hc_mult * hidden: the flattened stream the mixes and the collapse read

pub const Kind = ref.Kind;

fn elems(k: Kind, rows: usize) usize {
    return ref.count(k, rows, dim);
}

fn bytesOf(k: Kind, rows: usize) usize {
    return elems(k, rows) * c.size(if (k == .mix) .f32 else .bf16);
}

fn stream(k: Kind) usize {
    return if (k == .pre) mult else 1;
}

fn outs(k: Kind) usize {
    return if (k == .mix) 3 else 1; // the mix writes pre, post and comb; the collapse and the expand one buffer
}

/// The collapse: four products, three adds, one round to bf16.
fn preDepth(_: c.Shape) usize {
    return 7;
}

/// The expand: five products, four adds, one round to bf16.
fn postDepth(_: c.Shape) usize {
    return 9;
}

/// A lane's chain over every 256th element, the 5-level butterfly, the eight partials, then the divides after.
fn mixDepth(s: c.Shape) usize {
    return s.k / ref.tg + 14;
}

const mix_why = "the norm's sum of squares and the 24 matvecs, each lane chain, butterfly and partial in its fixed order";

pub fn spec(k: Kind, name: []const u8) c.Spec {
    const t = [_]c.Tile{ .{ .size = 16 }, .{}, .{} };
    return switch (k) {
        .mix => .{ .name = name, .accum = .f32, .depth = mixDepth, .ops = 260, .out = .f32, .tiles = t, .served = &.{
            .{ .rows = 16, .n = 1, .k = dim }, .{ .rows = 1, .n = 1, .k = dim }, .{ .rows = 7, .n = 1, .k = dim },
        }, .sums = .{ .why = mix_why }, .replaces = "tf_dsv41_g_hcmix, the temporary glue's mixes" },
        .pre => .{ .name = name, .accum = .f32, .depth = preDepth, .out = .bf16, .tiles = t, .served = &.{
            .{ .rows = 16, .n = 1, .k = dim }, .{ .rows = 1, .n = 1, .k = dim }, .{ .rows = 5, .n = 1, .k = dim },
        }, .sums = .{ .why = "four weighted streams added in stream order 0, 1, 2, 3, one round to bf16" }, .replaces = "tf_dsv41_g_hcpre" },
        .post => .{ .name = name, .accum = .f32, .depth = postDepth, .out = .bf16, .tiles = t, .served = &.{
            .{ .rows = 16, .n = 1, .k = dim }, .{ .rows = 1, .n = 1, .k = dim }, .{ .rows = 3, .n = 1, .k = dim },
        }, .sums = .{ .why = "the residual through comb in j order, then the post-weighted branch last, one round to bf16" }, .replaces = "tf_dsv41_g_hcpost" },
    };
}

fn hRef(r: rig_.Ref, extra: usize) hcm.Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

fn gRef(r: rig_.Ref, extra: usize) glue.Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

const vt = g.hooks(Proj);
const plain_vt = blk: {
    var v = g.hooks(Proj);
    v.replaced = null; // a control is proven at its own gate, not against the baseline
    break :blk v;
};

pub const Proj = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: Kind,
    ctl: u8,
    name: []const u8,
    pipes: hcm.Pipelines,
    glue_pipes: ?glue.Pipelines = null,
    shape: c.Shape = undefined,
    i: ?ref.In = null,
    in_x: rig_.Ref = undefined,
    in_fn: rig_.Ref = undefined,
    in_scale: rig_.Ref = undefined,
    in_base: rig_.Ref = undefined,
    in_mix: rig_.Ref = undefined,
    in_post: rig_.Ref = undefined,
    in_comb: rig_.Ref = undefined,
    in_res: rig_.Ref = undefined,
    out: [3]rig_.Ref = undefined,
    out_len: [3]usize = .{ 0, 0, 0 },

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, k: Kind, ctl: u8, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .kind = k, .ctl = ctl, .name = name, .pipes = undefined };
        if (ctl == 0) {
            p.pipes = try hcm.Pipelines.load(device);
            p.glue_pipes = try glue.Pipelines.load(device, sources.deepseek_v41_glue_tmp);
        } else {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.deepseek_v41_hc });
            defer a.free(text);
            p.pipes = try hcm.Pipelines.loadSource(device, text);
        }
        return p;
    }

    pub fn deinit(p: *Proj) void {
        if (p.glue_pipes) |*b| b.deinit();
        p.pipes.deinit();
        p.rig.deinit();
        p.arena.deinit();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const control: ?g.Gate = switch (p.ctl) {
            1 => .nonfinite,
            2 => .shapes,
            3 => .sums,
            else => .repeat,
        };
        var s = spec(p.kind, p.name);
        if (p.ctl == 0) return .{ .spec = s, .ctx = p, .vt = &vt, .control = null }; // the baseline passes every gate
        s.replaces = null; // a control is proven at its own gate, not against the baseline
        return .{ .spec = s, .ctx = p, .vt = &plain_vt, .control = control };
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != dim or shape.rows == 0) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        p.shape = shape;
        var i = try ref.In.init(p.arena.allocator(), p.kind, dim, shape.rows);
        errdefer i.deinit();
        i.fill(fill, seed);
        p.i = i;
        const n = shape.rows;
        const spare = try p.rig.input(std.mem.sliceAsBytes(i.scale[0..1])); // one small input every family can bind
        p.in_x = try p.rig.input(std.mem.sliceAsBytes(i.x));
        p.in_fn = if (p.kind == .mix) try p.rig.input(std.mem.sliceAsBytes(i.fnw)) else spare;
        p.in_scale = if (p.kind == .mix) try p.rig.input(std.mem.sliceAsBytes(i.scale)) else spare;
        p.in_base = if (p.kind == .mix) try p.rig.input(std.mem.sliceAsBytes(i.base)) else spare;
        p.in_mix = if (p.kind == .mix) spare else try p.rig.input(std.mem.sliceAsBytes(i.mixf));
        p.in_post = if (p.kind == .post) try p.rig.input(std.mem.sliceAsBytes(i.postf)) else spare;
        p.in_comb = if (p.kind == .post) try p.rig.input(std.mem.sliceAsBytes(i.comb)) else spare;
        p.in_res = if (p.kind == .post) try p.rig.input(std.mem.sliceAsBytes(i.res)) else spare;
        if (p.kind == .mix) {
            p.out_len = .{ n * mult * 4, n * mult * 4, n * 16 * 4 }; // pre, then post, then comb: the gather order
            p.out[0] = try p.rig.output(p.out_len[0]);
            p.out[1] = try p.rig.output(p.out_len[1]);
            p.out[2] = try p.rig.output(p.out_len[2]);
        } else {
            p.out_len = .{ bytesOf(p.kind, n), 0, 0 };
            p.out[0] = try p.rig.output(p.out_len[0]);
        }
        return true;
    }

    pub fn count(p: *Proj) usize {
        return elems(p.kind, p.shape.rows);
    }

    /// The hc call over rows r0 to r0 + n of the prepared buffers; the row-invariance probe splits by this.
    pub fn encode(p: *Proj, e: anytype, r0: usize, n: usize) void {
        const rows: u32 = @intCast(n);
        const h = hcm.Hc{ .dim = @intCast(dim), .iters = 20, .eps = 1e-20, .hc_eps = 1e-6, .alpha = 2.0 };
        const x = hRef(p.in_x, r0 * stream(p.kind) * dim * 2);
        switch (p.kind) {
            .mix => hcm.mix(&p.pipes, e, x, .{ hRef(p.in_fn, 0), hRef(p.in_scale, 0), hRef(p.in_base, 0) },
                .{ hRef(p.out[0], r0 * mult * 4), hRef(p.out[1], r0 * mult * 4), hRef(p.out[2], r0 * 16 * 4) }, rows, h),
            .pre => hcm.pre(&p.pipes, e, x, hRef(p.in_mix, r0 * mult * 4), hRef(p.out[0], r0 * dim * 2), rows, @intCast(dim)),
            .post => hcm.post(&p.pipes, e, x, hRef(p.in_res, r0 * mult * dim * 2), hRef(p.in_post, r0 * mult * 4), hRef(p.in_comb, r0 * 16 * 4), hRef(p.out[0], r0 * mult * dim * 2), rows, @intCast(dim)),
        }
    }

    /// The phase-1 glue over the same bytes and the same buffers, so gate 3 compares bits on identical inputs.
    fn encodeGlue(p: *Proj, e: anytype) void {
        const bp = p.glue_pipes.?;
        const rows: u32 = @intCast(p.shape.rows);
        const h = glue.Hc{ .dim = @intCast(dim), .iters = 20, .eps = 1e-20, .hc_eps = 1e-6, .alpha = 2.0 };
        switch (p.kind) {
            .mix => glue.hcmix(&bp, e, gRef(p.in_x, 0), .{ gRef(p.in_fn, 0), gRef(p.in_scale, 0), gRef(p.in_base, 0) },
                .{ gRef(p.out[0], 0), gRef(p.out[1], 0), gRef(p.out[2], 0) }, rows, h),
            .pre => glue.hcpre(&bp, e, gRef(p.in_x, 0), gRef(p.in_mix, 0), gRef(p.out[0], 0), rows, @intCast(dim)),
            .post => glue.hcpost(&bp, e, gRef(p.in_x, 0), gRef(p.in_res, 0), gRef(p.in_post, 0), gRef(p.in_comb, 0), gRef(p.out[0], 0), rows, @intCast(dim)),
        }
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        p.encode(pass.e, 0, p.shape.rows);
        try p.rig.finish(pass);
    }

    /// The glue's bytes for the prepared inputs: run into the very buffers the new kernel writes, then compared.
    pub fn replaced(p: *Proj) ![]const u8 {
        const pass = p.rig.begin();
        p.encodeGlue(pass.e);
        try p.rig.finish(pass);
        return p.rig.outputBytes();
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        p.i.?.compute(.none, .sequential, want, mag);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        const scratch = try p.arena.allocator().alloc(f64, p.count());
        p.i.?.compute(acc, order, out, scratch);
    }
};

/// Every registration and its four controls, plus the row-invariance probe that splits a launch into pieces.
pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Proj) = .empty,

    const E = struct { k: Kind, ctl: u8, name: []const u8 };
    const list = [_]E{
        .{ .k = .mix, .ctl = 0, .name = base ++ "mix tf_dsv41_hc_mix" },
        .{ .k = .pre, .ctl = 0, .name = base ++ "pre tf_dsv41_hc_pre" },
        .{ .k = .post, .ctl = 0, .name = base ++ "post tf_dsv41_hc_post" },
        .{ .k = .mix, .ctl = 1, .name = base ++ "control: mix non-finite inputs flushed to zero" },
        .{ .k = .mix, .ctl = 2, .name = base ++ "control: mix takes post from the pre slot and skips Sinkhorn" },
        .{ .k = .mix, .ctl = 3, .name = base ++ "control: mix lane totals flushed to fp16" },
        .{ .k = .mix, .ctl = 4, .name = base ++ "control: mix writes past its comb" },
        .{ .k = .pre, .ctl = 1, .name = base ++ "control: pre non-finite inputs flushed to zero" },
        .{ .k = .pre, .ctl = 2, .name = base ++ "control: pre rotates its four weights" },
        .{ .k = .pre, .ctl = 3, .name = base ++ "control: pre running total flushed to fp16" },
        .{ .k = .pre, .ctl = 4, .name = base ++ "control: pre writes past its last row" },
        .{ .k = .post, .ctl = 1, .name = base ++ "control: post non-finite inputs flushed to zero" },
        .{ .k = .post, .ctl = 2, .name = base ++ "control: post adds the branch without its weight" },
        .{ .k = .post, .ctl = 3, .name = base ++ "control: post running total flushed to fp16" },
        .{ .k = .post, .ctl = 4, .name = base ++ "control: post writes past its last stream" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.k, entry.ctl, entry.name);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }

    /// Launches of 1, 7 and 11 rows a launch equal one launch's bytes, every buffer: no sum crosses a row.
    pub fn invariance(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        var launches: usize = 0;
        var kernels: usize = 0;
        for (s.probes.items) |p| {
            if (p.ctl != 0) continue;
            kernels += 1;
            const shape = c.Shape{ .rows = 33, .n = 1, .k = dim };
            if (!try p.prepare(shape, .random, seed)) return error.Refused;
            try p.run();
            const whole = try s.a.dupe(u8, try p.rig.outputBytes());
            defer s.a.free(whole);
            for ([_]usize{ 1, 7, 11 }) |step| {
                for (p.out[0..outs(p.kind)], p.out_len[0..outs(p.kind)]) |o, len| @memset(p.rig.bytes(o, len), 0x55);
                const pass = p.rig.begin();
                var r0: usize = 0;
                while (r0 < shape.rows) : (r0 += step) {
                    p.encode(pass.e, r0, @min(step, shape.rows - r0));
                    launches += 1;
                }
                try p.rig.finish(pass);
                if (!std.mem.eql(u8, whole, try p.rig.outputBytes())) {
                    bad += 1;
                    std.debug.print("FAIL row invariance {s}: {d} rows a launch differs from one launch\n", .{ p.name, step });
                }
            }
        }
        if (kernels == 0) return 0;
        std.debug.print("{s} row invariance {s}: {d} kernels at 1, 7 and 11 rows a launch ({d} launches) byte-equal to one launch, {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", base, kernels, launches, bad });
        return bad;
    }
};
