//! MXFP8 projections through the common non-finite, repeat, shape and accumulator gates.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const Rig = @import("contract_rig.zig").Rig;

fn depth(shape: c.Shape) usize {
    return shape.k / 16 + 24;
}

const served = [_]c.Shape{
    .{ .rows = 1, .n = 70, .k = 1024 },
    .{ .rows = 17, .n = 70, .k = 1024 },
    .{ .rows = 33, .n = 70, .k = 1024 },
    .{ .rows = 256, .n = 70, .k = 1024 },
    .{ .rows = 16, .n = 1280, .k = 2560 },
    .{ .rows = 16, .n = 2560, .k = 640 },
    .{ .rows = 7, .n = 13952, .k = 2560 },
};

pub const Product = struct {
    rig: Rig,
    lane: *const cuda.qmmf.Lane,
    clusters: bool,
    d: ?Dense = null,
    weight: cuda.qmmf.Weight = undefined,
    x: u64 = 0,
    out: u64 = 0,
    part: u64 = 0,

    pub fn deinit(self: *Product) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Product) g.Kernel {
        return .{ .spec = .{
            .name = if (self.clusters) "cuda MXFP8 lane clustered slices" else "cuda MXFP8 lane explicit slice reduction",
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .tiles = .{ .{ .size = 16 }, .{ .size = 64 }, .{ .size = 512, .step = 64, .min = 64 } },
            .served = &served,
        }, .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Product);

    pub fn prepare(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        self.part = 0;
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 16);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x8f);
        const random = prng.random();
        const n = shape.n;
        const k = shape.k;
        const codes = try a.alloc(u8, n * k);
        defer a.free(codes);
        const scales = try a.alloc(u8, n * k / 32);
        defer a.free(scales);
        for (scales) |*s| s.* = if (fill == .stress) 127 else 123 + random.uintLessThan(u8, 8);
        for (0..n) |j| for (0..k) |i| {
            codes[j * k + i] = if (fill == .stress) 0x38 + random.uintLessThan(u8, 24) else random.uintLessThan(u8, 0x7f);
            if (fill != .stress and random.boolean()) codes[j * k + i] |= 0x80;
        };
        if (fill == .stress) for (0..n) |j| for (k / 2..k) |i| {
            codes[j * k + i] = codes[j * k + i - k / 2] | 0x80;
        };
        for (0..n) |j| for (0..k) |i| {
            const scale = std.math.ldexp(@as(f64, 1), @as(i32, scales[j * (k / 32) + i / 32]) - 127);
            d.w[j * k + i] = core.quant.fp8.e4m3(codes[j * k + i]) * scale;
        };
        const npad = cuda.mxfp8.padded(n);
        const tiled_codes = try a.alloc(u8, npad * k);
        defer a.free(tiled_codes);
        const packed_scales = try a.alloc(u8, npad * k / 32);
        defer a.free(packed_scales);
        cuda.mxfp8.packCodes(tiled_codes, codes, n, k);
        cuda.mxfp8.packScales(packed_scales, scales, n, k);
        self.weight = cuda.mxfp8.weight(try self.rig.input(tiled_codes), try self.rig.input(packed_scales), n, k);
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        const bytes = self.lane.partBytes(shape.rows, self.weight);
        if (bytes > 0) self.part = try self.rig.scratch(bytes);
        self.out = try self.rig.output(shape.rows * n * 2);
        return true;
    }

    pub fn run(self: *Product) !void {
        const d = self.d.?;
        try self.lane.matmul(self.rig.s, self.x, d.shape.k, d.shape.rows, self.weight, self.out, self.part);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Product) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Product, want: []f64, magnitude: []f64) !void {
        try self.d.?.reference(want, magnitude);
    }

    pub fn simulate(self: *Product, accumulator: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(accumulator, order, out);
    }
};
