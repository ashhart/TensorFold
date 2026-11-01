//! DeepSeek V4.1's fused hc mix pair on Metal, bit-equal to the glue's hcmix and checked against float64; controls 1-4.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_hc_ref.zig");

const c = contract.spec;
const g = contract.gates;
const fu = tf.dsv41_metal.fused;
const gl = tf.dsv41_metal.glue_tmp;
const base = "dsv41 fused ";
const mult = ref.mult;

/// hc_mult * hidden, as in metal_dsv41_hc.zig: the flattened stream the mixes read.
const dim: usize = 4 * 5120;

/// A lane's chain over every 256th element, the 5-level butterfly, the eight partials, then the divides after.
fn mixDepth(s: c.Shape) usize {
    return s.k / ref.tg + 14;
}

const mix_why = "the norm's sum of squares and the 24 matvecs, each lane chain, butterfly and partial in its fixed order";

pub const Fam = enum { hcmix };

fn spec(f: Fam, name: []const u8) c.Spec {
    const t = [_]c.Tile{ .{ .size = 16 }, .{}, .{} };
    return switch (f) {
        .hcmix => .{ .name = name, .accum = .f32, .depth = mixDepth, .ops = 260, .out = .f32, .tiles = t, .served = &.{
            .{ .rows = 16, .n = 1, .k = dim }, .{ .rows = 1, .n = 1, .k = dim }, .{ .rows = 7, .n = 1, .k = dim },
        }, .sums = .{ .why = mix_why }, .replaces = "tf_dsv41_g_hcmix, the temporary glue's mixes" },
    };
}

/// The gate each planted control must go red at; the baseline (0) has none.
fn controlGate(f: Fam, ctl: u8) ?g.Gate {
    return switch (f) {
        .hcmix => switch (ctl) {
            1 => .nonfinite,
            2 => .shapes,
            3 => .sums,
            else => .repeat,
        },
    };
}

fn at(r: rig_.Ref, extra: usize) fu.Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

fn atGlue(r: rig_.Ref, extra: usize) gl.Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

const vt = g.hooks(Proj);
const plain_vt = blk: {
    var v = g.hooks(Proj);
    v.replaced = null; // a control is proven at its own gate, not against the baseline
    break :blk v;
};

/// The fused pipelines hold 20 pipelines from one source; every probe of a control shares one compiled copy.
var shared: [5]?*fu.Pipelines = .{ null, null, null, null, null };

fn sharedPipes(a: std.mem.Allocator, device: mtl.Device, ctl: u8) !*fu.Pipelines {
    if (shared[ctl]) |p| return p;
    const p = try a.create(fu.Pipelines);
    if (ctl == 0) {
        p.* = try fu.Pipelines.load(device);
    } else {
        const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.dsv41.fused });
        defer a.free(text);
        p.* = try fu.Pipelines.loadSource(device, text);
    }
    shared[ctl] = p;
    return p;
}

pub const Proj = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    fam: Fam,
    ctl: u8,
    name: []const u8,
    pipes: *fu.Pipelines,
    glue: ?gl.Pipelines = null,
    shape: c.Shape = undefined,
    i: ?ref.In = null,
    in_x: rig_.Ref = undefined,
    in_fn: rig_.Ref = undefined,
    in_scale: rig_.Ref = undefined,
    in_base: rig_.Ref = undefined,
    out: [3]rig_.Ref = undefined, // pre, post and comb: the gather order
    out_len: [3]usize = .{ 0, 0, 0 },
    part: rig_.Ref = undefined, // the sums between the pair's two launches

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, f: Fam, ctl: u8, name: []const u8) !Proj {
        var p: Proj = .{
            .a = a,
            .arena = std.heap.ArenaAllocator.init(a),
            .rig = .{ .a = a, .device = device, .queue = queue },
            .fam = f,
            .ctl = ctl,
            .name = name,
            .pipes = try sharedPipes(a, device, ctl),
        };
        if (ctl == 0) p.glue = try gl.Pipelines.load(device, sources.dsv41.glue_tmp);
        return p;
    }

    pub fn deinit(p: *Proj) void {
        if (p.glue) |*b| b.deinit();
        p.arena.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const s = spec(p.fam, p.name);
        if (p.ctl == 0) return .{ .spec = s, .ctx = p, .vt = &vt, .control = null };
        var sc = s;
        sc.replaces = null; // a control is proven at its own gate, not against the baseline
        return .{ .spec = sc, .ctx = p, .vt = &plain_vt, .control = controlGate(p.fam, p.ctl) };
    }

    pub fn count(p: *Proj) usize {
        return ref.count(.mix, p.shape.rows, dim);
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != dim or shape.rows == 0) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        p.shape = shape;
        const a = p.arena.allocator();
        var i = try ref.In.init(a, .mix, dim, shape.rows);
        i.fill(fill, seed);
        if (fill == .stress) widenStream(&i); // gate 4's totals must pass fp16's ceiling, where control 3's flush drops bits
        p.i = i;
        const n = shape.rows;
        p.in_x = try p.rig.input(std.mem.sliceAsBytes(i.x));
        p.in_fn = try p.rig.input(std.mem.sliceAsBytes(i.fnw));
        p.in_scale = try p.rig.input(std.mem.sliceAsBytes(i.scale));
        p.in_base = try p.rig.input(std.mem.sliceAsBytes(i.base));
        p.out_len = .{ n * mult * 4, n * mult * 4, n * 16 * 4 };
        p.out[0] = try p.rig.output(p.out_len[0]);
        p.out[1] = try p.rig.output(p.out_len[1]);
        p.out[2] = try p.rig.output(p.out_len[2]);
        p.part = try p.rig.scratch(fu.mixScratch(n));
        return true;
    }

    /// Rows r0 to r0 + n of the prepared buffers as the pair: the sums, a barrier, then the tail.
    pub fn encode(p: *Proj, e: mtl.ComputeEncoder, r0: usize, n: usize) void {
        const rows: u32 = @intCast(n);
        const h = fu.Hc{ .dim = @intCast(dim), .iters = 20, .eps = 1e-20, .hc_eps = 1e-6, .alpha = 2.0 };
        fu.hcmix(p.pipes, e, .{
            .x = at(p.in_x, r0 * dim * 2),
            .w = .{ at(p.in_fn, 0), at(p.in_scale, 0), at(p.in_base, 0) },
            .out = .{ at(p.out[0], r0 * mult * 4), at(p.out[1], r0 * mult * 4), at(p.out[2], r0 * 16 * 4) },
            .part = at(p.part, r0 * fu.mix_sums * fu.mix_parts * 4),
            .h = h,
        }, rows);
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        p.encode(pass.e, 0, p.shape.rows);
        try p.rig.finish(pass);
    }

    /// The temporary glue's hcmix over the same bytes and buffers, so gate 3 compares bits on identical inputs.
    pub fn replaced(p: *Proj) ![]const u8 {
        const pass = p.rig.begin();
        const h = gl.Hc{ .dim = @intCast(dim), .iters = 20, .eps = 1e-20, .hc_eps = 1e-6, .alpha = 2.0 };
        gl.hcmix(&p.glue.?, pass.e, atGlue(p.in_x, 0), .{ atGlue(p.in_fn, 0), atGlue(p.in_scale, 0), atGlue(p.in_base, 0) }, .{ atGlue(p.out[0], 0), atGlue(p.out[1], 0), atGlue(p.out[2], 0) }, @intCast(p.shape.rows), h);
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

/// Gate 4's stress widened sixteen-fold: fp32 stays exact but totals pass fp16's ceiling, where control 3 flushes.
fn widenStream(i: *ref.In) void {
    for (i.x) |*v| v.* = ref.toBf(@floatCast(ref.bf(v.*) * 16));
}

/// Every registration and its controls, plus the row-invariance probe that splits a launch into pieces.
pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Proj) = .empty,

    const E = struct { f: Fam, ctl: u8, name: []const u8 };
    const list = [_]E{
        .{ .f = .hcmix, .ctl = 0, .name = base ++ "hcmix tf_dsv41_f_hcmix_part" },
        .{ .f = .hcmix, .ctl = 1, .name = base ++ "control: hc part reads every non-finite input as zero" },
        .{ .f = .hcmix, .ctl = 2, .name = base ++ "control: hc part halves the normed stream and hc fin writes the comb transposed" },
        .{ .f = .hcmix, .ctl = 3, .name = base ++ "control: hc part lane totals flushed to fp16" },
        .{ .f = .hcmix, .ctl = 4, .name = base ++ "control: hc part writes a partial past its row and hc fin past its comb" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.f, entry.ctl, entry.name);
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
        for (&shared) |*p| if (p.*) |pp| {
            pp.deinit();
            s.a.destroy(pp);
            p.* = null;
        };
        s.a.destroy(s);
    }

    /// Launches of 1, 7 and 11 rows a launch equal one launch's bytes: no sum crosses a row.
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
                for (p.out[0..3], p.out_len[0..3]) |o, len| @memset(p.rig.bytes(o, len), 0x55);
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
