//! dense_gemm_block under the contract on HIP: BF16 x times raw BF16 weights on the dot2 GEMM tile, fp32 and BF16 out.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;

/// The kernel's own chain (launches/dense.zig): a group's dot2s, then one add a group into the accumulator.
fn depth(s: c.Shape) usize {
    return @FieldType(hip.Launcher, "dense").chain(s.k);
}

/// Qwen3-8B's projections as float64 runs them: 2,048-row passes, 12,288-wide gate/up, K to 12,288, both tile edges.
const served = [_]c.Shape{
    .{ .rows = 1, .n = 4096, .k = 4096 },
    .{ .rows = 7, .n = 1024, .k = 12288 },
    .{ .rows = 16, .n = 12288, .k = 4096 },
    .{ .rows = 129, .n = 130, .k = 4096 },
    .{ .rows = 300, .n = 256, .k = 4096 },
    .{ .rows = 2048, .n = 128, .k = 4096 },
};

pub fn spec(wide: bool) c.Spec {
    return .{
        .name = if (wide) "hip dense_gemm_block (bf16 weights, fp32 out)" else "hip dense_gemm_block (bf16 weights, bf16 out)",
        .accum = .f32,
        .depth = depth,
        .out = if (wide) .f32 else .bf16,
        .served = &served,
        .tiles = .{ .{ .size = 128 }, .{ .size = 128 }, .{ .size = 12288, .step = 128, .min = 128 } },
    };
}

pub const DenseGemm = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    /// fp32 out, or BF16 rounded once in the epilogue.
    wide: bool,
    d: ?Dense = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, wide: bool) DenseGemm {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .wide = wide };
    }

    pub fn deinit(self: *DenseGemm) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *DenseGemm) g.Kernel {
        return .{ .spec = spec(self.wide), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(DenseGemm);

    /// x from the dense family; W as BF16 in [-1, 1), or the stress fill's mirrored integers; FP16 builds serve none.
    pub fn prepare(self: *DenseGemm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 128 != 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.a;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 1);
        const bits = try a.alloc(u16, d.w.len);
        defer a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed ^ 0xD3);
        const rng = prng.random();
        const k = shape.k;
        for (bits) |*b| b.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .stress) for (0..shape.n) |j| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            bits[j * k + i] = c.toBf16(v);
            bits[j * k + k / 2 + i] = c.toBf16(-v);
        };
        if (fill == .nonfinite and shape.n > 5) {
            bits[1 * k + k / 4] = c.toBf16(std.math.nan(f32));
            bits[5 * k] = c.toBf16(std.math.inf(f32));
        }
        for (d.w, bits) |*w, b| w.* = c.fromBf16(b);
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(bits));
        self.out = try self.rig.output(d.outputs() * @as(usize, if (self.wide) 4 else 2));
        return true;
    }

    pub fn run(self: *DenseGemm) !void {
        const s = self.d.?.shape;
        const out32: u64 = if (self.wide) self.out else 0;
        const out16: u64 = if (self.wide) 0 else self.out;
        try self.l.dense.gemm(self.l.r, self.x, self.w, out32, out16, @intCast(s.rows), @intCast(s.n), @intCast(s.k), self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *DenseGemm) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *DenseGemm, want: []f64, mag: []f64) !void {
        try self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *DenseGemm, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(acc, order, out);
    }
};
