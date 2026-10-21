//! The checkpoint math's dense NVFP4 projections under the contract on CUDA: ck_dense4.cu's warp and prompt kernels.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Ck = cuda.ck.Ck;
const Dense4 = cuda.ck_dense.Dense4;

pub const Kind = enum { warp, prompt };

/// A 64-input mma step a link, after the products inside it (as the NVFP4 experts declare it).
fn stepDepth(s: c.Shape) usize {
    return s.k / 64 + 64;
}

/// The ModelOpt Qwen3.8-27B's NVFP4 layers: gate and up (5120 inputs), down (17408), the head (5120), columns cut.
pub fn spec(kind: Kind) c.Spec {
    const tiles = [3]c.Tile{ .{ .size = 16 }, .{ .size = 32, .step = 64, .min = 64 }, .{ .size = 256, .step = 64, .min = 64 } };
    const prompt_tiles = [3]c.Tile{ .{ .size = 128 }, .{ .size = 128, .step = 64, .min = 64 }, .{ .size = 256, .step = 64, .min = 64 } };
    return switch (kind) {
        .warp => .{ .name = "cuda ck dense NVFP4 tf_ck4_dense", .accum = .f32, .depth = stepDepth, .ops = 1, .out = .bf16, .tiles = tiles, .served = &.{ .{ .rows = 1, .n = 512, .k = 5120 }, .{ .rows = 16, .n = 1024, .k = 5120 }, .{ .rows = 33, .n = 256, .k = 17408 }, .{ .rows = 64, .n = 512, .k = 5120 } } },
        .prompt => .{ .name = "cuda ck dense NVFP4 tf_ck4_dense_prompt", .accum = .f32, .depth = stepDepth, .ops = 1, .out = .bf16, .replaces = "tf_ck4_dense", .tiles = prompt_tiles, .served = &.{ .{ .rows = 65, .n = 512, .k = 5120 }, .{ .rows = 128, .n = 1024, .k = 5120 }, .{ .rows = 300, .n = 256, .k = 17408 } } },
    };
}

pub const Dense = struct {
    rig: rig_.Rig,
    kind: Kind,
    ck: *const Ck,
    dense: *const Dense4,
    shape: c.Shape = undefined,
    x: []u8 = &.{}, // NVFP4 rows: codes, then block scales
    w: []u8 = &.{}, // [n][k/2] codes
    ws: []u8 = &.{}, // [n][k/16] block scales
    alpha: f32 = 1,
    xd: u64 = 0,
    weight: cuda.qmmf.Weight = undefined,
    out: u64 = 0,
    base: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Dense) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Dense) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.w);
        a.free(self.ws);
        self.x = &.{};
        self.w = &.{};
        self.ws = &.{};
    }

    pub fn kernel(self: *Dense) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Dense);

    /// Random codes and scales; non-finite: a NaN block scale on row 1 and on column 3; stress: mirrored halves.
    pub fn prepare(self: *Dense, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 64 != 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const n = shape.n;
        const k = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try a.alloc(u8, cuda.ck.rowsBytes(shape.rows, k));
        self.w = try a.alloc(u8, n * k / 2);
        self.ws = try a.alloc(u8, n * k / 16);
        const xs = self.x[shape.rows * k / 2 ..];
        rng.bytes(self.x[0 .. shape.rows * k / 2]);
        rng.bytes(self.w);
        for (xs) |*s| s.* = 0x20 + rng.uintLessThan(u8, 0x28);
        for (self.ws) |*s| s.* = 0x20 + rng.uintLessThan(u8, 0x28);
        self.alpha = 0.005 + rng.float(f32) * 0.03;
        switch (fill) {
            .random => {},
            .nonfinite => {
                if (shape.rows > 1) xs[1 * (k / 16)] = 0x7f;
                self.ws[3 * (k / 16)] = 0x7f;
            },
            .stress => self.stress(rng),
        }
        self.xd = try self.rig.input(self.x);
        const bytes = Ck.packedBytes(1, n, k);
        const zeros = try a.alloc(u8, bytes[0]);
        defer a.free(zeros);
        @memset(zeros, 0);
        const wd = [2]u64{ try self.rig.input(zeros[0..bytes[0]]), try self.rig.input(zeros[0..bytes[1]]) };
        const src = [2]u64{ try self.rig.scratch(self.w.len), try self.rig.scratch(self.ws.len) };
        for (src, [2][]const u8{ self.w, self.ws }) |p, h| try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = p, .len = h.len }).upload(0, h);
        try self.ck.pack(self.rig.s, src[0], src[1], 1, n, k, 0, k, wd[0], wd[1]);
        self.weight = cuda.ck_dense.weight(wd[0], wd[1], 1, self.alpha, n, k);
        try self.rig.s.synchronize();
        self.out = try self.rig.output(shape.rows * n * 2);
        self.base = try self.rig.scratch(shape.rows * n * 2);
        return true;
    }

    /// Gate 4: block scales 16 (x) and 8 (w), values 4 and 6 mirrored across halves (8 of x's a step lower), alpha 1.
    fn stress(self: *Dense, rng: std.Random) void {
        const k = self.shape.k;
        @memset(self.x[self.shape.rows * k / 2 ..], 0x58);
        @memset(self.ws, 0x50);
        self.alpha = 1;
        for (0..self.shape.rows) |r| for (0..k / 2) |i| {
            const v: u8 = 6 + rng.uintLessThan(u8, 2);
            put(self.x[r * k / 2 ..], i, v);
            put(self.x[r * k / 2 ..], k / 2 + i, if (i < 8) v - 1 else v);
        };
        for (0..self.shape.n) |j| for (0..k / 2) |i| {
            const v: u8 = 6 + rng.uintLessThan(u8, 2);
            put(self.w[j * k / 2 ..], i, v);
            put(self.w[j * k / 2 ..], k / 2 + i, v | 8);
        };
    }

    fn put(codes: []u8, i: usize, v: u8) void {
        const b = &codes[i / 2];
        b.* = if (i % 2 == 0) (b.* & 0xf0) | v else (b.* & 0x0f) | v << 4;
    }

    fn value(codes: []const u8, scales: []const u8, r: usize, k: usize, i: usize) f64 {
        const b = codes[r * k / 2 + i / 2];
        const nib = if (i % 2 == 0) b & 0xf else b >> 4;
        return core.quant.nvfp4.e2m1(nib) * core.quant.fp8.e4m3(scales[r * k / 16 + i / 16]);
    }

    /// Row r's products with column j, in input order.
    fn terms(self: *const Dense, r: usize, j: usize, t: []f64) void {
        const k = self.shape.k;
        const xs = self.x[self.shape.rows * k / 2 ..];
        for (t, 0..) |*o, i| o.* = value(self.x, xs, r, k, i) * value(self.w, self.ws, j, k, i);
    }

    pub fn run(self: *Dense) !void {
        try self.dense.matmulForm(self.rig.s, self.xd, self.shape.rows, self.weight, self.out, @backingInt(self.kind));
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Dense) ![]const u8 {
        if (self.kind != .prompt) return error.NotAReplacement;
        try self.dense.matmulForm(self.rig.s, self.xd, self.shape.rows, self.weight, self.base, 0);
        return self.rig.bytes(self.base, self.shape.rows * self.shape.n * 2);
    }

    pub fn count(self: *Dense) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *Dense, want: []f64, mag: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        for (0..self.shape.rows) |r| for (0..self.shape.n) |j| {
            self.terms(r, j, t);
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (t) |v| {
                sum += v;
                abs += @abs(v);
            }
            want[r * self.shape.n + j] = sum * self.alpha;
            mag[r * self.shape.n + j] = abs * @abs(self.alpha);
        };
    }

    pub fn simulate(self: *Dense, acc: c.Accum, order: g.Order, out: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        for (0..self.shape.rows) |r| for (0..self.shape.n) |j| {
            self.terms(r, j, t);
            out[r * self.shape.n + j] = contract.rows.sumIn(acc, order, t) * self.alpha;
        };
    }
};
