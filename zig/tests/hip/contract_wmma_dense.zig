//! The gfx11 matrix-core dense tile of a prompt under the contract on HIP: affine_wmma_gemm's bytes at each K.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const tiles = hip.launches.affine;
pub const Arg = tiles.Arg;

/// Flash Next's trunk products: 4-bit codes in groups of 32, bf16 x and tables.
pub const bits = 4;
pub const group = 32;

/// A K the tile compiles, its output (bf16 at out16 as tf_cast rounds, else fp32) and the engine's (rows, n) at it.
pub const Case = struct { k: usize, half: bool, name: []const u8, served: []const c.Shape };

const ks = [_]struct { usize, []const c.Shape }{
    // hc up: the 4 streams' 2,560 columns from 320
    .{ 320, &.{.{ .rows = 383, .n = 10240, .k = 320 }} },
    .{ 2560, &.{.{ .rows = 300, .n = 640, .k = 2560 }} },
    .{ 6144, &.{.{ .rows = 257, .n = 2560, .k = 6144 }} },
    // hc down: 320 columns, and 324 with the 4 stream injections
    .{ 10240, &.{ .{ .rows = 200, .n = 324, .k = 10240 }, .{ .rows = 200, .n = 320, .k = 10240 } } },
};

pub const cases = blk: {
    var out: [2 * ks.len]Case = undefined;
    for (ks, 0..) |e, i| {
        for ([_]bool{ false, true }, 0..) |half, h| {
            const say = "hip affine_wmma_dense<{d}> (4-bit, group 32, {s} out)";
            out[2 * i + h] = .{ .k = e[0], .half = half, .served = e[1], .name = std.fmt.comptimePrint(say, .{ e[0], if (half) "bf16" else "fp32" }) };
        }
    }
    break :blk out;
};

/// A group's 16 dot2 steps from zero (2 u each: an ulp of the step's largest addend), then its scale and bias fmas.
fn depth(s: c.Shape) usize {
    return 2 * 16 + 2 * (s.k / group) + 8;
}

pub fn spec(cfg: Case) c.Spec {
    return .{
        .name = cfg.name,
        .accum = .f32,
        .depth = depth,
        .out = if (cfg.half) .bf16 else .f32,
        .replaces = if (cfg.half) "affine_wmma_gemm then tf_cast" else "affine_wmma_gemm",
        // the registry takes it from 33 rows (the K-parallel tiles below) and one full column tile
        .tiles = .{ .{ .size = 128, .min = 33 }, .{ .size = 128, .min = 128 }, .{} },
        .served = cfg.served,
    };
}

/// x, the packed codes and the bf16 tables of `f` uploaded as the tiles read them; the caller binds the output.
pub fn upload(r: *rig_.Rig, a: std.mem.Allocator, f: Affine) !Arg {
    const w = try a.alloc(u32, f.shape.n * f.shape.k * bits / 32);
    f.words(w);
    return .{
        .x = try r.input(std.mem.sliceAsBytes(f.x)),
        .words = try r.input(std.mem.sliceAsBytes(w)),
        .scale = .{ .p = try r.input(std.mem.sliceAsBytes(f.scales)), .kind = 1 },
        .bias = .{ .p = try r.input(std.mem.sliceAsBytes(f.biases)), .kind = 1 },
        .out = 0,
        .m = @intCast(f.shape.rows),
        .n = @intCast(f.shape.n),
        .k = @intCast(f.shape.k),
        .bits = bits,
        .group = group,
        .fp16 = 0,
    };
}

/// Gate 1's table plants beside the x plants: a NaN scale, an infinite bias and a -inf scale, each in its own column.
pub fn plantTables(f: Affine) void {
    const groups = f.shape.k / f.group;
    f.scales[3 * groups] = c.toBf16(std.math.nan(f32));
    f.biases[5 * groups + 1] = c.toBf16(std.math.inf(f32));
    f.scales[7 * groups + 2] = c.toBf16(-std.math.inf(f32));
}

/// x [rows][K] bf16 times the MLX 4-bit matrix [n][K]: y [rows][n], fp32 or bf16.
pub const Dense = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    arena: std.heap.ArenaAllocator,
    index: usize,
    f: ?Affine = null,
    arg: Arg = undefined,
    sum: u64 = 0,
    base: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, i: usize) Dense {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .arena = .init(a), .index = i };
    }

    pub fn deinit(self: *Dense) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    /// The chip has the tile: gfx11's matrix cores.
    pub fn ready(self: *const Dense) bool {
        return self.l.affine.dense[tiles.Kernels.denseIndex(@intCast(cases[self.index].k)).?] != null;
    }

    pub fn kernel(self: *Dense) g.Kernel {
        return .{ .spec = spec(cases[self.index]), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Dense);

    pub fn prepare(self: *Dense, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const cfg = cases[self.index];
        if (shape.rows == 0 or shape.n < 8 or shape.k != cfg.k) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const f = try Affine.init(a, shape, bits, group);
        try f.fill(fill, seed);
        if (fill == .nonfinite) plantTables(f);
        self.f = f;
        self.arg = try upload(&self.rig, a, f);
        const n = f.outputs();
        if (cfg.half) self.arg.out16 = try self.rig.output(n * 2) else self.arg.out = try self.rig.output(n * 4);
        self.sum = try self.rig.scratch(n * 4);
        self.base = if (cfg.half) try self.rig.scratch(n * 2) else 0;
        return true;
    }

    pub fn run(self: *Dense) !void {
        try self.l.affine.denseGo(self.l.r, self.arg, self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    /// The matrix tile the registry took before (fp32 sums), then tf_cast's rounding for a bf16 output.
    pub fn replaced(self: *Dense) ![]const u8 {
        var m = self.arg;
        m.out = self.sum;
        m.out16 = 0;
        const st = self.rig.s.handle;
        try self.l.affine.blockGo(self.l.r, m, st, 1, .matrix);
        const n = self.f.?.outputs();
        if (!cases[self.index].half) return self.rig.bytes(self.sum, n * 4);
        const src = kit.ptr(?*const anyopaque, self.sum);
        try self.l.tf_cast(src, @backingInt(kit.Kind.f32), kit.ptr(?*anyopaque, self.base), @backingInt(kit.Kind.bf16), @intCast(n), st);
        return self.rig.bytes(self.base, n * 2);
    }

    pub fn count(self: *Dense) usize {
        return self.f.?.outputs();
    }

    /// MLX's factored formula in float64: per group s * sum(q x) + b * sum(x).
    pub fn reference(self: *Dense, want: []f64, mag: []f64) !void {
        try self.f.?.reference(want, mag);
    }

    pub fn simulate(self: *Dense, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.f.?.simulate(acc, order, out);
    }
};
