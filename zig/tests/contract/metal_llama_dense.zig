//! Llama's bf16 projections under the contract: the row kernels of a window and the tensor-unit GEMM of a prompt.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const llama = @import("metal_llama.zig");
const controls = @import("metal_llama_controls.zig");
const c = contract.spec;
const g = contract.gates;
const dense_in = contract.dense;
const family = llama.family;
const G33 = family.nanbeige43;
const G3 = family.nanbeige41;

/// glue.metal's ll_mv_args, as forward.zig fills it.
const MArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };

/// A lane's dots over its 8-wide slices of the row, then the simd tree.
fn matvecDepth(s: c.Shape) usize {
    return s.k / 32 + 8;
}

/// The row projection ll_matvec_ROWS: Y[r, j] = bf16(sum X[r, i] W[j, i]) in fp32, ROWS rows a launch.
pub const Matvec = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    per: u32,
    mutation: controls.Mutation,
    name: []const u8,
    served: [2]c.Shape,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    d: ?dense_in.Dense = null,
    w16: []u16 = &.{},
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, per: u32, mutation: controls.Mutation) !Matvec {
        const lib = try controls.glueLibrary(device, a, mutation);
        errdefer lib.deinit();
        const name = try std.fmt.allocPrint(a, "llama glue ll_matvec_{d}", .{per});
        errdefer a.free(name);
        const fun = name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..];
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .lib = lib, .pipe = try llama.pipelineOf(lib, device, fun), .per = per, .mutation = mutation, .name = name, .served = .{ .{ .rows = per, .n = 1024, .k = G33.intermediate }, .{ .rows = per, .n = G33.hidden, .k = G3.hidden } } };
    }

    fn drop(p: *Matvec) void {
        if (p.d) |d| d.deinit();
        p.d = null;
        p.a.free(p.w16);
        p.w16 = &.{};
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
        return controls.mark(.{ .spec = .{ .name = p.name, .accum = .f32, .depth = matvecDepth, .out = .bf16, .tiles = .{ .{ .size = p.per, .step = 1, .min = 1 }, .{ .size = 8, .step = 1, .min = 1 }, .{ .size = 256, .step = 8, .min = 8 } }, .served = &p.served }, .ctx = p, .vt = &vt }, p.mutation);
    }

    pub fn prepare(p: *Matvec, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.n == 0 or s.k == 0 or s.k % 8 != 0) return false;
        p.drop();
        p.shape = s;
        const d = try dense_in.Dense.init(p.a, s, 1);
        p.d = d;
        d.fillX(fill, seed, 2);
        p.w16 = try p.a.alloc(u16, s.n * s.k);
        try denseWeights(p.w16, d, s, fill, seed);
        p.x = try p.rig.input(std.mem.sliceAsBytes(d.x));
        p.w = try p.rig.input(std.mem.sliceAsBytes(p.w16));
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
            e.setBuffer(p.w.buf, p.w.off, 1);
            e.setValue(MArgs{ .rows = here, .n = @intCast(s.n), .k = @intCast(s.k), .ldy = @intCast(s.n) }, 2);
            e.setBuffer(p.y.buf, p.y.off + at * s.n * 2, 3);
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

/// Bf16 weights for a dense family: random in [-1, 1), or gate 4's integers, mirrored and negated over the halves.
pub fn denseWeights(w16: []u16, d: dense_in.Dense, s: c.Shape, fill: g.Fill, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed ^ 0x5151);
    const rng = prng.random();
    const h = s.k / 2;
    for (0..s.n) |j| for (0..s.k) |i| {
        const v: f32 = if (fill != .stress) rng.float(f32) * 2 - 1 else if (i < h) @floatFromInt(1 + (i * 7 + j) % 6) else -@as(f32, @floatFromInt(1 + ((i - h) * 7 + j) % 6));
        w16[j * s.k + i] = c.toBf16(v);
        d.w[j * s.k + i] = c.fromBf16(w16[j * s.k + i]);
    };
}

/// The tensor-op accumulation over the K tiles, then the output rounding.
fn gemmDepth(s: c.Shape) usize {
    return s.k / 16 + 16;
}

/// The prompt GEMM ll_gemm_AB: A whole 64-row tiles of M, B whole 128-column tiles of N; a probe pads to its form.
pub const Gemm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    am: bool,
    an: bool,
    mutation: controls.Mutation,
    name: []const u8,
    served: [2]c.Shape,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    d: ?dense_in.Dense = null,
    w16: []u16 = &.{},
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, am: bool, an: bool, mutation: controls.Mutation) !Gemm {
        const plain = try family.gemmSource(device, a);
        defer a.free(plain);
        const src = try controls.apply(a, plain, mutation);
        defer a.free(src);
        const name = try std.fmt.allocPrint(a, "llama gemm ll_gemm_{d}{d}", .{ @intFromBool(am), @intFromBool(an) });
        errdefer a.free(name);
        const fun = name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..];
        const rows: usize = if (am) 256 else 100;
        const n: usize = if (an) 1024 else 1088;
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, src, fun), .am = am, .an = an, .mutation = mutation, .name = name, .served = .{ .{ .rows = rows, .n = n, .k = G33.hidden }, .{ .rows = if (am) 64 else 33, .n = if (an) G3.hidden else G3.hidden + 64, .k = G3.hidden } } };
    }

    fn drop(p: *Gemm) void {
        if (p.d) |d| d.deinit();
        p.d = null;
        p.a.free(p.w16);
        p.w16 = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Gemm) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
        p.a.free(p.name);
    }

    const vt = g.hooks(Gemm);

    pub fn kernel(p: *Gemm) g.Kernel {
        return controls.mark(.{ .spec = .{ .name = p.name, .accum = .f32, .depth = gemmDepth, .out = .bf16, .tiles = .{ .{ .size = 64, .step = 1, .min = 1 }, .{ .size = 128, .step = 1, .min = 1 }, .{ .size = 256, .step = 256, .min = 256 } }, .served = &p.served }, .ctx = p, .vt = &vt }, p.mutation);
    }

    /// The shape the kernel runs at: whole tiles where its form assumes them (the padded rows are the probe's too).
    fn padded(p: *const Gemm, s: c.Shape) c.Shape {
        return .{ .rows = if (p.am) std.mem.alignForward(usize, s.rows, 64) else s.rows, .n = if (p.an) std.mem.alignForward(usize, s.n, 128) else s.n, .k = s.k };
    }

    pub fn prepare(p: *Gemm, s0: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s0.rows == 0 or s0.n == 0 or s0.k == 0 or s0.k % 256 != 0) return false;
        const s = p.padded(s0);
        if (s.rows > 2048) return false;
        p.drop();
        p.shape = s;
        const d = try dense_in.Dense.init(p.a, s, 1);
        p.d = d;
        d.fillX(fill, seed, 2);
        p.w16 = try p.a.alloc(u16, s.n * s.k);
        try denseWeights(p.w16, d, s, fill, seed);
        p.x = try p.rig.input(std.mem.sliceAsBytes(d.x));
        p.w = try p.rig.input(std.mem.sliceAsBytes(p.w16));
        p.y = try p.rig.output(d.outputs() * 2);
        return true;
    }

    pub fn count(p: *Gemm) usize {
        return p.d.?.outputs();
    }

    pub fn run(p: *Gemm) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        e.setBuffer(p.x.buf, p.x.off, 0);
        e.setBuffer(p.w.buf, p.w.off, 1);
        const tn = (s.n + 127) / 128;
        const tm = (s.rows + 63) / 64;
        var prm: [16]i32 = @splat(0);
        const vals = [_]usize{ s.rows, s.n, s.k, s.k, s.k, s.n, tn, tm, 2, s.k / 256 };
        for (vals, 0..) |v, i| prm[i] = @intCast(v);
        e.setBytes(std.mem.asBytes(&prm), 2);
        e.setBuffer(p.y.buf, p.y.off, 3);
        e.dispatchThreads(mtl.Size.of((tn << 2) * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Gemm, want: []f64, mag: []f64) !void {
        return p.d.?.reference(want, mag);
    }

    pub fn simulate(p: *Gemm, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.d.?.simulate(acc, order, out);
    }
};
