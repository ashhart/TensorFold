//! Llama's int4 projections under the contract: the window's row kernels, and the bf16 made for the prompt GEMM.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const llama = @import("metal_llama.zig");
const c = contract.spec;
const g = contract.gates;
const dense_in = contract.dense;
const family = llama.family;
const G33 = family.nanbeige43;

/// glue.metal's ll_mv_args, as forward.zig fills it.
const MArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };

pub const group = 32;

/// A random pack-quantized [n, k]: words, bf16 scales, packed zero points, and the weights they mean as float64.
const Int4 = struct {
    q: []u32,
    s: []u16,
    z: []u32,

    fn deinit(m: Int4, a: std.mem.Allocator) void {
        a.free(m.q);
        a.free(m.s);
        a.free(m.z);
    }

    /// Random codes, zeros and scales; gate 4's integers as q - z (z 8, scale 1); non-finite scales when planted.
    fn init(a: std.mem.Allocator, n: usize, k: usize, fill: g.Fill, seed: u64, w: []f64) !Int4 {
        const groups = k / group;
        const m = Int4{ .q = try a.alloc(u32, n * k / 8), .s = try a.alloc(u16, n * groups), .z = try a.alloc(u32, (n + 7) / 8 * groups) };
        @memset(m.q, 0);
        @memset(m.z, 0);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x4444);
        const rng = prng.random();
        const h = k / 2;
        for (0..n) |j| for (0..groups) |gi| {
            const z: u32 = if (fill == .stress) 8 else rng.uintLessThan(u32, 16);
            m.z[(j / 8) * groups + gi] |= z << @intCast(4 * (j % 8));
            const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            const scale: f32 = if (fill == .stress) 1 else std.math.ldexp(1 + rng.float(f32), -@as(i32, @intCast(rng.uintLessThan(u32, 6))));
            // the gate's non-finite inputs: rows 0-2's first scales NaN, +Inf and -Inf
            m.s[j * groups + gi] = c.toBf16(if (fill == .nonfinite and gi == 0 and j < planted.len) planted[j] else scale);
            const s = c.fromBf16(m.s[j * groups + gi]);
            for (gi * group..(gi + 1) * group) |i| {
                const q: u32 = if (fill != .stress) rng.uintLessThan(u32, 16) else if (i < h) 8 + @as(u32, @intCast(1 + (i * 7 + j) % 6)) else 8 - @as(u32, @intCast(1 + ((i - h) * 7 + j) % 6));
                m.q[(j * k + i) / 8] |= q << @intCast(4 * (i % 8));
                // the decompressor's weight: bf16((q - z) * s), exact before that one rounding
                w[j * k + i] = c.fromBf16(c.toBf16(@floatCast((@as(f64, @floatFromInt(q)) - @as(f64, @floatFromInt(z))) * s)));
            }
        };
        return m;
    }
};

/// A group's 8 float4 dots summed, the lane's groups added, then the simd tree.
fn matvecDepth(s: c.Shape) usize {
    return s.k / 1024 + 24;
}

/// The int4 row projection ll_q4_matvec_ROWS: Y[r, j] = bf16(sum X[r, i] W[j, i]) in fp32 over W as dequantized.
pub const Matvec = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    per: u32,
    name: []const u8,
    served: [2]c.Shape,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    d: ?dense_in.Dense = null,
    m: ?Int4 = null,
    x: rig_.Ref = undefined,
    in: [3]rig_.Ref = undefined,
    y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, per: u32) !Matvec {
        const lib = try family.glueLibrary(device, a);
        errdefer lib.deinit();
        const name = try std.fmt.allocPrint(a, "llama glue ll_q4_matvec_{d}", .{per});
        errdefer a.free(name);
        const fun = name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..];
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .lib = lib, .pipe = try llama.pipelineOf(lib, device, fun), .per = per, .name = name, .served = .{ .{ .rows = per, .n = 1024, .k = G33.intermediate }, .{ .rows = per, .n = G33.hidden, .k = G33.hidden } } };
    }

    fn drop(p: *Matvec) void {
        if (p.d) |d| d.deinit();
        p.d = null;
        if (p.m) |m| m.deinit(p.a);
        p.m = null;
        p.rig.reset();
    }

    pub fn deinit(p: *Matvec) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
        p.lib.deinit();
        p.a.free(p.name);
    }

    const vt = g.hooks(Matvec);

    pub fn kernel(p: *Matvec) g.Kernel {
        return .{ .spec = .{ .name = p.name, .accum = .f32, .depth = matvecDepth, .out = .bf16, .tiles = .{ .{ .size = p.per, .step = 1, .min = 1 }, .{ .size = 8, .step = 1, .min = 1 }, .{ .size = 1024, .step = group, .min = group } }, .served = &p.served }, .ctx = p, .vt = &vt };
    }

    pub fn prepare(p: *Matvec, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.n == 0 or s.k == 0 or s.k % group != 0) return false;
        p.drop();
        p.shape = s;
        const d = try dense_in.Dense.init(p.a, s, 1);
        p.d = d;
        d.fillX(fill, seed, 2);
        const m = try Int4.init(p.a, s.n, s.k, fill, seed, d.w);
        p.m = m;
        p.x = try p.rig.input(std.mem.sliceAsBytes(d.x));
        p.in = .{ try p.rig.input(std.mem.sliceAsBytes(m.q)), try p.rig.input(std.mem.sliceAsBytes(m.s)), try p.rig.input(std.mem.sliceAsBytes(m.z)) };
        p.y = try p.rig.output(d.outputs() * 2);
        return true;
    }

    pub fn count(p: *Matvec) usize {
        return p.d.?.outputs();
    }

    pub fn run(p: *Matvec) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        var at: usize = 0;
        while (at < s.rows) : (at += p.per) {
            const here: u32 = @intCast(@min(p.per, s.rows - at));
            e.setBuffer(p.x.buf, p.x.off + at * s.k * 2, 0);
            for (p.in, 1..) |r, slot| e.setBuffer(r.buf, r.off, slot);
            e.setValue(MArgs{ .rows = here, .n = @intCast(s.n), .k = @intCast(s.k), .ldy = @intCast(s.n) }, 4);
            e.setBuffer(p.y.buf, p.y.off + at * s.n * 2, 5);
            e.dispatchGroups(mtl.Size.of((s.n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        }
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Matvec, want: []f64, mag: []f64) !void {
        return p.d.?.reference(want, mag);
    }

    pub fn simulate(p: *Matvec, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.d.?.simulate(acc, order, out);
    }
};

fn none(_: c.Shape) usize {
    return 0;
}

/// ll_q4_dequant: W[j, i] = bf16((q - z) * s) for the prompt GEMM; the shape's rows are W's, k their width.
pub const Dequant = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    w: []f64 = &.{},
    m: ?Int4 = null,
    in: [3]rig_.Ref = undefined,
    y: rig_.Ref = undefined,

    // a slice of the gate and down projections' rows: every row is the same work
    const served = [_]c.Shape{ .{ .rows = 512, .n = G33.hidden, .k = G33.hidden }, .{ .rows = 512, .n = G33.intermediate, .k = G33.intermediate } };

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Dequant {
        const lib = try family.glueLibrary(device, a);
        errdefer lib.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .lib = lib, .pipe = try llama.pipelineOf(lib, device, "ll_q4_dequant") };
    }

    fn drop(p: *Dequant) void {
        p.a.free(p.w);
        p.w = &.{};
        if (p.m) |m| m.deinit(p.a);
        p.m = null;
        p.rig.reset();
    }

    pub fn deinit(p: *Dequant) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
        p.lib.deinit();
    }

    const vt = g.hooks(Dequant);

    pub fn kernel(p: *Dequant) g.Kernel {
        const no_sums: c.Sums = .{ .f16 = 0, .bf16 = 0, .why = "no sums: one product of exact factors, rounded once" };
        return .{ .spec = .{ .name = "llama glue ll_q4_dequant", .accum = .none, .depth = none, .out = .bf16, .tiles = .{ .{ .size = 8, .step = 1, .min = 1 }, .{}, .{ .size = 256, .step = group, .min = group } }, .served = &served, .sums = no_sums }, .ctx = p, .vt = &vt };
    }

    pub fn prepare(p: *Dequant, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k == 0 or s.k % group != 0) return false;
        p.drop();
        p.shape = s;
        p.w = try p.a.alloc(f64, s.rows * s.k);
        const m = try Int4.init(p.a, s.rows, s.k, fill, seed, p.w);
        p.m = m;
        p.in = .{ try p.rig.input(std.mem.sliceAsBytes(m.q)), try p.rig.input(std.mem.sliceAsBytes(m.s)), try p.rig.input(std.mem.sliceAsBytes(m.z)) };
        p.y = try p.rig.output(s.rows * s.k * 2);
        return true;
    }

    pub fn count(p: *Dequant) usize {
        return p.shape.rows * p.shape.k;
    }

    pub fn run(p: *Dequant) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.in, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
        e.setValue(MArgs{ .rows = 0, .n = @intCast(s.rows), .k = @intCast(s.k), .ldy = 0 }, 3);
        e.setBuffer(p.y.buf, p.y.off, 4);
        e.dispatchThreads(mtl.Size.of(s.k / 8, s.rows, 1), mtl.Size.of(128, 2, 1));
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Dequant, want: []f64, mag: []f64) !void {
        @memcpy(want, p.w);
        for (mag, p.w) |*x, v| x.* = @abs(v);
    }

    pub fn simulate(p: *Dequant, _: c.Accum, _: g.Order, out: []f64) !void {
        @memcpy(out, p.w);
    }
};
