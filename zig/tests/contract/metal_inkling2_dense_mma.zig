//! Inkling 2 Mini's projections on simdgroup matrices: each output equals the matvec kernel it replaces, byte for byte.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const tf = @import("tensorfold").inkling2_metal_host;
const dense = tf.dense;
const loop = tf.loop;
const controls = @import("metal_inkling2_dense_mma_controls.zig");
const src = @import("inkling2_metal");

/// `project`'s widths, or the loop gate's down projection over the normalized [h | m] row.
pub const Kind = enum { project, zeta };

const name = "metal inkling2 mma tf_ink2_dense_mma";

fn label(k: Kind, control: bool) []const u8 {
    if (control) return (if (k == .project) controls.Mutation.halves else controls.Mutation.groups).name();
    return if (k == .project) name else name ++ " as loop zeta";
}

fn depth(s: c.Shape) usize {
    return s.k;
}

/// The matvec's arguments and the loop's: rows, n, k and the output stride; rows and the row stride.
const MatvecArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };
const ZetaArgs = extern struct { rows: u32, stride: u32 };

const project_shapes = [_]c.Shape{ .{ .rows = 16, .n = 24, .k = 2304 }, .{ .rows = 45, .n = 64, .k = 3072 }, .{ .rows = 100, .n = 16, .k = 2304 } };
const zeta_shapes = [_]c.Shape{ .{ .rows = 16, .n = 64, .k = 4608 }, .{ .rows = 37, .n = 64, .k = 4608 } };

pub fn spec(k: Kind, control: bool) c.Spec {
    return .{
        .name = label(k, control),
        .accum = .none,
        .depth = depth,
        .out = .bf16,
        .replaces = if (k == .project) "the bf16 matvec kernels sp_matvec_*, on the same weights and rows" else "tf_inkling2_loop_zeta, on the same weights and rows",
        .nan = .{ .stated = "every output equals the replaced kernel's: a non-finite sum has the same class" },
        // Rows go 16 a tile, outputs 16 a tile and columns 256 a step; 45 rows and 24 outputs leave a tile part full.
        .tiles = .{ .{ .size = 16 }, .{ .size = 16, .step = 8, .min = 8 }, .{} },
        .served = if (k == .project) &project_shapes else &zeta_shapes,
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "the replaced kernel's fp32 sums are the baseline, bit for bit" },
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    mma: dense.Mma,
    matvec: [6]mtl.Pipeline,
    zeta: loop.Pipelines,
    kind: Kind,
    control: bool,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u16 = &.{},
    refs: [3]rig_.Ref = undefined,
    baseline: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, control: bool) !Probe {
        const text = try controls.apply(a, if (!control) .none else if (kind == .project) .halves else .groups);
        defer a.free(text);
        const mma = try dense.Mma.loadSource(device, text);
        errdefer mma.deinit();
        // The loader's library: precise functions, one pipeline for each row count `dense` rounds up to.
        const lib = try mtl.Library.fromSource(device, sources.spark25_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer lib.deinit();
        var matvec: [6]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (matvec[0..made]) |p| p.deinit();
        for ([_]u32{ 1, 2, 4, 8, 16, 32 }, 0..) |rows, i| {
            var buf: [32]u8 = undefined;
            matvec[i] = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrintSentinel(&buf, "sp_matvec_{d}", .{rows}, 0), false);
            made += 1;
        }
        var zeta = try loop.Pipelines.loadSource(device, src.loop);
        errdefer zeta.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .mma = mma, .matvec = matvec, .zeta = zeta, .kind = kind, .control = control };
    }

    fn clear(p: *Probe) void {
        p.a.free(p.x);
        p.a.free(p.w);
        p.x = &.{};
        p.w = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.clear();
        p.zeta.deinit();
        for (p.matvec) |m| m.deinit();
        p.mma.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.kind, p.control), .ctx = p, .vt = &vt, .control = if (p.control) .shapes else null };
    }
    const vt = g.hooks(Probe);

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const widths: []const usize = if (p.kind == .project) &.{ 2304, 3072 } else &.{4608};
        const fits = for (widths) |k| {
            if (shape.k == k) break true;
        } else false;
        // The loop's zeta kernel stores 64 ranks a row whatever the call.
        if (!fits or shape.rows == 0 or shape.rows > 128 or shape.n == 0 or shape.n % 8 != 0 or shape.n > 128) return false;
        if (p.kind == .zeta and shape.n != 64) return false;
        p.clear();
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.x = try p.a.alloc(u16, shape.rows * shape.k);
        p.w = try p.a.alloc(u16, shape.n * shape.k);
        for (p.x) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (p.w) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        // Huge terms that cancel (+2^20, -2^20) leave the small terms' sum less what each order of adding lost.
        var marked = try p.a.alloc(bool, shape.k);
        defer p.a.free(marked);
        @memset(marked, false);
        for (0..8) |j| {
            const a = (37 + 331 * j) % shape.k;
            const b = (a + 163) % shape.k;
            if (marked[a] or marked[b]) continue;
            marked[a] = true;
            marked[b] = true;
            for (0..shape.rows) |r| p.x[r * shape.k + a] = c.toBf16(1048576);
            for (0..shape.rows) |r| p.x[r * shape.k + b] = c.toBf16(1048576);
            for (0..shape.n) |n| {
                p.w[n * shape.k + a] = c.toBf16(1);
                p.w[n * shape.k + b] = c.toBf16(-1);
            }
        }
        if (fill == .nonfinite) {
            p.x[0] = c.toBf16(std.math.nan(f32));
            p.x[p.x.len / 2] = c.toBf16(std.math.inf(f32));
            p.x[p.x.len - 1] = c.toBf16(-std.math.inf(f32));
        }
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.w));
        p.refs[2] = try p.rig.output(p.count() * 2);
        p.baseline = try p.rig.scratch(p.count() * 2);
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        p.mma.launch(pass.e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), .{ .rows = @intCast(p.shape.rows), .n = @intCast(p.shape.n), .k = @intCast(p.shape.k), .ldy = @intCast(p.shape.n) });
        try p.rig.finish(pass);
    }

    fn hr(r: rig_.Ref) dense.Ref {
        return .{ .buf = r.buf, .off = r.off };
    }

    /// The kernels it replaces: `dense`'s launches of up to 32 rows, or the loop's zeta.
    pub fn replaced(p: *Probe) ![]const u8 {
        const pass = p.rig.begin();
        const e = pass.e;
        const rows = p.shape.rows;
        const k = p.shape.k;
        const n = p.shape.n;
        if (p.kind == .zeta) {
            e.setPipeline(p.zeta.zeta);
            e.setBuffer(p.refs[0].buf, p.refs[0].off, 0);
            e.setBuffer(p.refs[1].buf, p.refs[1].off, 1);
            e.setBuffer(p.baseline.buf, p.baseline.off, 2);
            e.setValue(ZetaArgs{ .rows = @intCast(rows), .stride = 0 }, 3);
            e.dispatchGroups(mtl.Size.of(n / 8, rows, 1), mtl.Size.of(256, 1, 1));
        } else {
            var base: usize = 0;
            while (base < rows) {
                const take: usize = @min(rows - base, 32);
                const variant: usize = if (take > 16) 5 else if (take > 8) 4 else if (take > 4) 3 else if (take > 2) 2 else if (take > 1) 1 else 0;
                e.setPipeline(p.matvec[variant]);
                e.setBuffer(p.refs[0].buf, p.refs[0].off + base * k * 2, 0);
                e.setBuffer(p.refs[1].buf, p.refs[1].off, 1);
                e.setValue(MatvecArgs{ .rows = @intCast(take), .n = @intCast(n), .k = @intCast(k), .ldy = @intCast(n) }, 2);
                e.setBuffer(p.baseline.buf, p.baseline.off + base * n * 2, 3);
                e.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                base += take;
            }
        }
        try p.rig.finish(pass);
        return p.rig.bytes(p.baseline, p.count() * 2);
    }

    pub fn count(p: *const Probe) usize {
        return p.shape.rows * p.shape.n;
    }

    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Non-finite inputs: every output has the replaced kernel's class, and a finite one its bytes.
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const got = try p.rig.outputBytes();
        const want = try p.replaced();
        for (0..p.count()) |i| {
            const a = c.fromBf16(std.mem.readInt(u16, got[2 * i ..][0..2], .little));
            const b = c.fromBf16(std.mem.readInt(u16, want[2 * i ..][0..2], .little));
            if (std.math.isNan(a) and std.math.isNan(b)) continue;
            if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) return "an output differs from the replaced kernel's";
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: [4]Probe = undefined,
    made: usize = 0,

    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.made]) |*probe| probe.deinit();
    }

    /// The entries exist where the GPU has no tensor units; the units kernels serve the projections on the others.
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (s.device.tensorUnits()) return;
        for (std.enums.values(Kind)) |kind| for ([_]bool{ false, true }) |control| {
            if (only) |needle| if (std.mem.indexOf(u8, label(kind, control), needle) == null) continue;
            s.probes[s.made] = try Probe.init(s.a, s.device, s.queue, kind, control);
            s.made += 1;
            try s.kernels.append(s.a, s.probes[s.made - 1].kernel());
        };
    }
};
