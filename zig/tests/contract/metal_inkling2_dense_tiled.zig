//! Inkling 2 Mini's register-tiled bf16 projections: 0-ULP bit-equal to sp_matvec_* across all shapes and tails.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const dense_in = contract.dense;
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const tf = @import("tensorfold").inkling2_metal_host;
const dense = tf.dense;
const controls = @import("metal_inkling2_dense_tiled_controls.zig");

const name = "metal inkling2 dense tf_ink2_dense_tiled";

fn label(control: ?controls.Mutation) []const u8 {
    return if (control) |m| m.name() else name;
}

/// A lane's dots over its 8-wide slices of the row, then the simd tree.
fn depth(s: c.Shape) usize {
    return s.k / 32 + 8;
}

/// The matvec's arguments: rows, n, k and the output stride.
const MatvecArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };

/// Every Inkling 2 Mini dense width (24..3072) and row tails (5, 7, 9, 17, 45); first k=3072 is n=24 for fast Gate 4.
const shapes = [_]c.Shape{
    .{ .rows = 5, .n = 24, .k = 2304 },
    .{ .rows = 7, .n = 24, .k = 3072 },
    .{ .rows = 7, .n = 384, .k = 2304 },
    .{ .rows = 9, .n = 1024, .k = 2304 },
    .{ .rows = 17, .n = 2304, .k = 3072 },
    .{ .rows = 45, .n = 3072, .k = 2304 },
};

pub fn spec(control: ?controls.Mutation) c.Spec {
    return .{
        .name = label(control),
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .replaces = "the bf16 matvec kernels sp_matvec_*, on the same weights and rows",
        .nan = .{ .stated = "every output equals the replaced kernel's: a non-finite sum has the same class" },
        // Rows go 16 a tile (with an 8-row tail path), outputs 16 a tile in multiples of 8, columns 256 a step.
        .tiles = .{ .{ .size = 16 }, .{ .size = 16, .step = 8, .min = 8 }, .{} },
        .served = &shapes,
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    tiled: dense.Tiled,
    matvec: [6]mtl.Pipeline,
    control: ?controls.Mutation,
    shape: c.Shape = undefined,
    d: ?dense_in.Dense = null,
    x: []u16 = &.{},
    w: []u16 = &.{},
    refs: [3]rig_.Ref = undefined,
    baseline: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: ?controls.Mutation) !Probe {
        const text = try controls.apply(a, control orelse .none);
        defer a.free(text);
        const tiled = try dense.Tiled.loadSource(device, text);
        errdefer tiled.deinit();
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
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .tiled = tiled, .matvec = matvec, .control = control };
    }

    fn clear(p: *Probe) void {
        if (p.d) |d| d.deinit();
        p.d = null;
        p.a.free(p.x);
        p.a.free(p.w);
        p.x = &.{};
        p.w = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.clear();
        for (p.matvec) |m| m.deinit();
        p.tiled.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.control), .ctx = p, .vt = &vt, .control = if (p.control) |m| m.gate() else null };
    }
    const vt = g.hooks(Probe);

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if ((shape.k != 2304 and shape.k != 3072) or shape.rows == 0 or shape.rows > 256 or shape.n == 0 or shape.n % 8 != 0 or shape.n > 3072) return false;
        p.clear();
        p.shape = shape;
        if (fill == .stress) {
            const d = try dense_in.Dense.init(p.a, shape, 1);
            p.d = d;
            d.fillX(.stress, seed, 2);
            p.w = try p.a.alloc(u16, shape.n * shape.k);
            const h = shape.k / 2;
            for (0..shape.n) |j| for (0..shape.k) |i| {
                const v: f32 = if (i < h) @floatFromInt(1 + (i * 7 + j) % 6) else -@as(f32, @floatFromInt(1 + ((i - h) * 7 + j) % 6));
                p.w[j * shape.k + i] = c.toBf16(v);
                d.w[j * shape.k + i] = c.fromBf16(p.w[j * shape.k + i]);
            };
            p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(d.x));
            p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.w));
            p.refs[2] = try p.rig.output(p.count() * 2);
            p.baseline = try p.rig.scratch(p.count() * 2);
            return true;
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.x = try p.a.alloc(u16, shape.rows * shape.k);
        p.w = try p.a.alloc(u16, shape.n * shape.k);
        for (p.x) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (p.w) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        // Huge cancelling +/-2^20 terms across lanes and octet halves catch any rounding-order change on every column.
        var marked = try p.a.alloc(bool, shape.k);
        defer p.a.free(marked);
        @memset(marked, false);
        for (0..8) |j| {
            const a = (37 + 331 * j) % shape.k;
            const b = (a + 163) % shape.k;
            if (!marked[a] and !marked[b]) {
                marked[a] = true;
                marked[b] = true;
                for (0..shape.rows) |r| p.x[r * shape.k + a] = c.toBf16(1048576);
                for (0..shape.rows) |r| p.x[r * shape.k + b] = c.toBf16(1048576);
                for (0..shape.n) |n| {
                    p.w[n * shape.k + a] = c.toBf16(1);
                    p.w[n * shape.k + b] = c.toBf16(-1);
                }
            }
            const lo = 256 * (j + 1);
            const hi = lo + 4;
            if (hi < shape.k and !marked[lo] and !marked[hi]) {
                marked[lo] = true;
                marked[hi] = true;
                for (0..shape.rows) |r| p.x[r * shape.k + lo] = c.toBf16(1048576);
                for (0..shape.rows) |r| p.x[r * shape.k + hi] = c.toBf16(1048576);
                for (0..shape.n) |n| {
                    p.w[n * shape.k + lo] = c.toBf16(1);
                    p.w[n * shape.k + hi] = c.toBf16(-1);
                }
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
        p.tiled.launch(pass.e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), .{ .rows = @intCast(p.shape.rows), .n = @intCast(p.shape.n), .k = @intCast(p.shape.k), .ldy = @intCast(p.shape.n) });
        try p.rig.finish(pass);
    }

    fn hr(r: rig_.Ref) dense.Ref {
        return .{ .buf = r.buf, .off = r.off };
    }

    /// The kernels it replaces: `dense`'s ceiling-power-of-two launches of up to 32 rows.
    pub fn replaced(p: *Probe) ![]const u8 {
        const pass = p.rig.begin();
        const e = pass.e;
        const rows = p.shape.rows;
        const k = p.shape.k;
        const n = p.shape.n;
        var base: usize = 0;
        while (base < rows) {
            const take: usize = @min(rows - base, 32);
            const variant: usize = if (take <= 1) 0 else if (take <= 2) 1 else if (take <= 4) 2 else if (take <= 8) 3 else if (take <= 16) 4 else 5;
            e.setPipeline(p.matvec[variant]);
            e.setBuffer(p.refs[0].buf, p.refs[0].off + base * k * 2, 0);
            e.setBuffer(p.refs[1].buf, p.refs[1].off, 1);
            e.setValue(MatvecArgs{ .rows = @intCast(take), .n = @intCast(n), .k = @intCast(k), .ldy = @intCast(n) }, 2);
            e.setBuffer(p.baseline.buf, p.baseline.off + base * n * 2, 3);
            e.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            base += take;
        }
        try p.rig.finish(pass);
        return p.rig.bytes(p.baseline, p.count() * 2);
    }

    pub fn count(p: *const Probe) usize {
        return p.shape.rows * p.shape.n;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        if (p.d) |d| return d.reference(want, mag);
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.d.?.simulate(acc, order, out);
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
    probes: std.ArrayList(*Probe) = .empty,

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }

    /// Runs on both M5+ and pre-M5 GPUs so the register-tiled kernel is contract-verified everywhere.
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]?controls.Mutation{ null, .halves, .second_col, .bf16_acc }) |control| {
            if (only) |needle| if (std.mem.indexOf(u8, label(control), needle) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, control);
            errdefer p.deinit();
            try contract.register(s.a, &s.probes, &s.kernels, p, p.kernel());
        }
    }
};
