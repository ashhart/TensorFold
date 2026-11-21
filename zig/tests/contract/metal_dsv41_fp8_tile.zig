//! DeepSeek V4.1 FP8 rows on the MMA tile under the contract: the fp8_rows order's bits, forms, controls, equality.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_fp8_ref.zig");
const eq = @import("metal_dsv41_fp8_tile_eq.zig");

const c = contract.spec;
const g = contract.gates;
const tile = tf.dsv41_metal.fp8_tile;
const rows_k = tf.dsv41_metal.fp8_rows;
const base = "dsv41_fp8_tile ";

/// A registration: output, forced form (null: the shape's), real fixture, the row kernel as baseline, planted fault.
pub const Variant = struct { out: tile.Out = .f32, form: ?tile.Form = null, fixture: bool = false, replaces: bool = false, ctl: u8 = 0 };

const real = [_]c.Shape{
    .{ .rows = 16, .n = 1280, .k = 5120 }, // wq_a
    .{ .rows = 6, .n = 512, .k = 5120 }, // wkv
    .{ .rows = 8, .n = 512, .k = 8192 }, // wo_b's K at a few columns
    .{ .rows = 9, .n = 5120, .k = 2304 }, // shared w2
    .{ .rows = 16, .n = 256, .k = 1280 }, // wq_b's and ix_wq_b's K
    .{ .rows = 7, .n = 1024, .k = 4096 }, // wo_a
    .{ .rows = 12, .n = 64, .k = 15360 }, // the drafter's main_proj K
};
const served = blk: {
    var out: [real.len + 16]c.Shape = undefined;
    for (real, 0..) |s, i| out[i] = s;
    for (1..17) |r| out[real.len + r - 1] = .{ .rows = r, .n = 64, .k = 1536 }; // every row count
    break :blk out;
};

pub fn spec(v: Variant, name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = ref.depth,
        .out = if (v.out == .f32) .f32 else .bf16,
        .tiles = if (v.fixture) .{ .{}, .{}, .{} } else .{ .{ .size = 16 }, .{ .size = 32, .step = 32, .min = 32 }, .{ .size = 256, .step = 32, .min = 32 } },
        .served = if (v.fixture) &.{ref.fixture.shape} else &served,
        .replaces = if (v.replaces) "tf_dsv41_fp8_w* one row a launch" else null,
    };
}

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    pipes: tile.Pipelines,
    rows: rows_k.Pipelines,
    f: ?ref.Fp8 = null,
    fill: g.Fill = .random,
    real_fixture: bool = false,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    s: rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    yb: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipes = undefined, .rows = undefined };
        if (v.ctl == 0) {
            p.pipes = try tile.Pipelines.load(device);
        } else {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ v.ctl, sources.dsv41.fp8_tile });
            defer a.free(text);
            p.pipes = try tile.Pipelines.loadSource(device, text);
        }
        errdefer p.pipes.deinit();
        p.rows = try rows_k.Pipelines.load(device);
        return p;
    }

    pub fn deinit(p: *Proj) void {
        p.drop();
        p.pipes.deinit();
        p.rows.deinit();
        p.rig.deinit();
    }

    fn drop(p: *Proj) void {
        if (p.f) |f| f.deinit();
        p.f = null;
        p.rig.reset();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const control: ?g.Gate = switch (p.v.ctl) {
            1 => .nonfinite,
            2 => .shapes,
            3 => .sums,
            4 => .repeat,
            else => null,
        };
        return .{ .spec = spec(p.v, p.name), .ctx = p, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Proj);

    fn call(p: *const Proj) tile.Call {
        const f = p.f.?;
        return .{ .k = @intCast(f.shape.k), .n = @intCast(f.shape.n), .rows = @intCast(f.shape.rows), .out = p.v.out };
    }

    fn form(p: *const Proj) tile.Form {
        const cl = p.call();
        const want = p.v.form orelse return tile.formOf(cl.k, cl.n);
        // a forced form where the shape admits it, else the shape's own
        const pad = std.math.ceilPowerOfTwo(u32, cl.k / 32) catch unreachable;
        if (cl.n % (8 * want.nj) != 0 or want.sgs > pad or pad / want.sgs > 512) return tile.formOf(cl.k, cl.n);
        return want;
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (!tile.fits(.{ .k = @intCast(shape.k), .n = @intCast(shape.n), .rows = @intCast(shape.rows) })) return false;
        const f = try ref.Fp8.init(p.a, shape);
        p.f = f;
        p.fill = fill;
        p.real_fixture = p.v.fixture and fill == .random and std.meta.eql(shape, ref.fixture.shape);
        if (p.real_fixture) try ref.fixture.load(f, null) else f.fill(fill, seed);
        const bytes = f.shape.rows * f.shape.n * c.size(spec(p.v, p.name).out);
        p.x = try p.rig.input(std.mem.sliceAsBytes(f.x));
        p.w = try p.rig.input(f.codes);
        p.s = try p.rig.input(f.scales);
        p.y = try p.rig.output(bytes);
        if (p.v.replaces) p.yb = try p.rig.scratch(bytes);
        return true;
    }

    fn at(r: rig_.Ref, extra: usize) tile.Ref {
        return .{ .buf = r.buf, .off = r.off + extra };
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        try tile.encodeAs(&p.pipes, pass.e, at(p.x, 0), at(p.w, 0), at(p.s, 0), at(p.y, 0), p.call(), .{}, p.form());
        try p.rig.finish(pass);
    }

    /// The row kernel's bytes on the same inputs, each row its own one-row launch.
    pub fn replaced(p: *Proj) ![]const u8 {
        const f = p.f.?;
        const es = c.size(spec(p.v, p.name).out);
        const pass = p.rig.begin();
        for (0..f.shape.rows) |r| {
            const one: rows_k.Call = .{ .k = @intCast(f.shape.k), .n = @intCast(f.shape.n), .rows = 1, .split = rows_k.splitOf(@intCast(f.shape.k)), .out = if (p.v.out == .f32) .f32 else .bf16 };
            const rr = struct {
                fn at(x: rig_.Ref, extra: usize) rows_k.Ref {
                    return .{ .buf = x.buf, .off = x.off + extra };
                }
            }.at;
            try rows_k.encodeWhole(&p.rows, pass.e, rr(p.x, 2 * r * f.shape.k), rr(p.w, 0), rr(p.s, 0), rr(p.yb, es * r * f.shape.n), one, .{});
        }
        try p.rig.finish(pass);
        return p.rig.bytes(p.yb, f.shape.rows * f.shape.n * es);
    }

    pub fn count(p: *Proj) usize {
        return p.f.?.outputs();
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        const f = p.f.?;
        if (!p.real_fixture) return f.reference(want, mag, p.v.out == .bf16);
        try ref.npyF64(ref.fixture.ref_npy, want);
        f.magnitudes(mag);
    }

    /// Random inputs must reproduce the fp8_rows order bit for bit; stress inputs and the real fixture keep the bound.
    pub fn bounds(p: *Proj, _: []const f64, _: []const f64, allow: []f64) !void {
        if (p.real_fixture or p.fill == .stress) return;
        @memset(allow, 0);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.f.?.simulate(acc, order, out);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Proj) = .empty,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{}, .name = base ++ "f32" },
        .{ .v = .{ .out = .bf16 }, .name = base ++ "bf16" },
        .{ .v = .{ .form = .{ .nj = 2, .sgs = 8 } }, .name = base ++ "f32 16 columns" },
        .{ .v = .{ .out = .bf16, .form = .{ .nj = 1, .sgs = 1 } }, .name = base ++ "bf16 one simdgroup" },
        .{ .v = .{ .out = .bf16, .replaces = true }, .name = base ++ "bf16 against fp8_rows one row a launch" },
        .{ .v = .{ .fixture = true }, .name = base ++ "release wkv fixture" },
        .{ .v = .{ .ctl = 1 }, .name = base ++ "control: non-finite inputs flushed" },
        .{ .v = .{ .ctl = 2 }, .name = base ++ "control: group and slice roots added in a chain" },
        .{ .v = .{ .ctl = 3 }, .name = base ++ "control: group roots rounded to fp16" },
        .{ .v = .{ .ctl = 4 }, .name = base ++ "control: writes past its output" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device, .queue = queue };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.v, entry.name);
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

    /// Every model FP8 shape at 1-16 rows, every form, the 16/17 seam, NaN and Inf: the row kernel's bytes.
    pub fn equality(s: *Suite, seed: u64) !usize {
        const p = if (s.probes.items.len > 0) s.probes.items[0] else return 0;
        if (p.v.ctl != 0) return 0;
        return eq.run(s.a, &p.rig, &p.pipes, &p.rows, seed);
    }
};
