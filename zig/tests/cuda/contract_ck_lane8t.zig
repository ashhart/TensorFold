//! A compressed-tensors export's FP8 math under the contract on CUDA: ck_lane8t.cu's 16-, 64- and 128-row tiles.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const L = cuda.ck_lane8t;
const e4m3 = core.quant.fp8.e4m3;

pub const Kind = enum { t16, t64, t128 };

/// 32-input MMA steps over the whole of K, the products inside one.
fn depth(s: c.Shape) usize {
    return s.k / 32 + 32;
}

/// The unsloth 27B's FP8 layers (mixers at 5120 and 6144 inputs, MLPs 56-63 at 17408), columns cut.
pub fn spec(kind: Kind) c.Spec {
    const tiles = [3]c.Tile{ .{ .size = 16 }, .{ .size = 64, .step = 64, .min = 64 }, .{} };
    return switch (kind) {
        .t16 => .{ .name = "cuda ck_lane8t tf_ck8t_16 (rows a token's scale, a scale a column)", .accum = .f32, .depth = depth, .ops = 2, .out = .bf16, .tiles = tiles, .served = &.{ .{ .rows = 1, .n = 512, .k = 5120 }, .{ .rows = 16, .n = 256, .k = 17408 }, .{ .rows = 3, .n = 640, .k = 6144 } } },
        .t64 => .{ .name = "cuda ck_lane8t tf_ck8t_64", .accum = .f32, .depth = depth, .ops = 2, .out = .bf16, .replaces = "tf_ck8t_16", .tiles = tiles, .served = &.{ .{ .rows = 17, .n = 512, .k = 5120 }, .{ .rows = 64, .n = 256, .k = 17408 } } },
        .t128 => .{ .name = "cuda ck_lane8t tf_ck8t_128", .accum = .f32, .depth = depth, .ops = 2, .out = .bf16, .replaces = "tf_ck8t_16", .tiles = tiles, .served = &.{ .{ .rows = 65, .n = 512, .k = 5120 }, .{ .rows = 300, .n = 256, .k = 6144 } } },
    };
}

fn rows(kind: Kind) usize {
    return switch (kind) {
        .t16 => 16,
        .t64 => 64,
        .t128 => 128,
    };
}

/// bf16 rows quantized by tf_ck_quant8t (its codes and scales downloaded for the reference) times e4m3 weights.
pub const Lane8t = struct {
    rig: rig_.Rig,
    kind: Kind,
    l: *const L.Lane8t,
    packer: *const cuda.modelopt.Packer,
    shape: c.Shape = undefined,
    xq: []u8 = &.{}, // the rows' codes in input order
    rs: []f32 = &.{},
    w: []u8 = &.{},
    cs: []f32 = &.{},
    codes: u64 = 0,
    rsd: u64 = 0,
    weight: L.Weight = undefined,
    out: u64 = 0,
    base: u64 = 0,

    pub fn deinit(self: *Lane8t) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Lane8t) void {
        const a = self.rig.gpu.gpa;
        for ([_][]u8{ self.xq, self.w }) |b| a.free(b);
        for ([_][]f32{ self.rs, self.cs }) |b| a.free(b);
        self.xq = &.{};
        self.w = &.{};
        self.rs = &.{};
        self.cs = &.{};
    }

    pub fn kernel(self: *Lane8t) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = if (self.kind == .t16) &vt16 else &vt };
    }

    const vt = g.hooks(Lane8t);
    /// The 16-row tile replaces nothing: float64's classes and bound.
    const vt16 = blk: {
        var v = g.hooks(Lane8t);
        v.replaced = null;
        break :blk v;
    };

    /// Random rows; non-finite: NaN in row 1 and column 5; stress: integers, 448 heads (scale one), negated halves.
    pub fn prepare(self: *Lane8t, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 64 != 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const n = shape.n;
        const k = shape.k;
        const m = shape.rows;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const xs = try a.alloc(u16, m * k);
        defer a.free(xs);
        self.w = try a.alloc(u8, n * k);
        const ints = [_]u8{ 0x38, 0x40, 0x44, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58 };
        for (0..m) |r| {
            const span: f32 = std.math.ldexp(@as(f32, 1), @as(i32, rng.uintLessThan(u8, 9)) - 6);
            for (0..k) |i| {
                const v: f32 = switch (fill) {
                    .stress => if (i == 0 or i == k / 2) 448 else @floatCast(e4m3(ints[if (i < k / 2) rng.uintLessThan(usize, 16) else 0])),
                    else => (rng.float(f32) * 2 - 1) * span,
                };
                xs[r * k + i] = @truncate(@as(u32, @bitCast(v)) >> 16);
            }
            if (fill == .stress) for (1..k / 2) |i| { // the halves equal but 8 inputs a step lower; weights negated
                const v: f32 = @bitCast(@as(u32, xs[r * k + i]) << 16);
                const lower: f32 = if (i <= 8 and v > 1) v - 1 else v;
                xs[r * k + k / 2 + i] = @truncate(@as(u32, @bitCast(lower)) >> 16);
            };
        }
        for (0..n) |j| for (0..k) |i| {
            self.w[j * k + i] = switch (fill) {
                .stress => if (i < k / 2) ints[rng.uintLessThan(usize, 16)] else self.w[j * k + i - k / 2] | 0x80,
                else => rng.int(u8) & 0x77,
            };
        };
        if (fill == .nonfinite) {
            if (m > 1) xs[k + 3] = 0x7fc0;
            if (n > 5) self.w[5 * k + k / 2] = 0x7f;
        }
        self.cs = try a.alloc(f32, n);
        for (self.cs) |*v| v.* = if (fill == .stress) 1 else 0.002 + rng.float(f32) * 0.01;
        const xd = try self.rig.scratch(xs.len * 2);
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = xd, .len = xs.len * 2 }).upload(0, std.mem.sliceAsBytes(xs));
        const zeros = try a.alloc(u8, cuda.modelopt.padded(n) * k);
        defer a.free(zeros);
        @memset(zeros, 0);
        self.codes = try self.rig.input(zeros[0 .. m * k]);
        self.rsd = try self.rig.input(zeros[0 .. m * 4]);
        try self.l.quantRows(self.rig.s, xd, k, m, k, self.codes, self.rsd);
        const wsrc = try self.rig.scratch(self.w.len);
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = wsrc, .len = self.w.len }).upload(0, self.w);
        const wd = try self.rig.input(zeros);
        try self.packer.fp8Lane(self.rig.s, wsrc, wd, n, k);
        self.weight = .{ .codes = wd, .cs = try self.rig.input(std.mem.sliceAsBytes(self.cs)), .n = n, .k = k };
        try self.rig.s.synchronize();
        // the reference reads what the quantizer made: its codes in input order, its scales
        const got = try self.rig.bytes(self.codes, m * k);
        self.xq = try a.alloc(u8, m * k);
        for (0..m) |r| for (0..k / 16) |grp| for (0..16) |slot| {
            const q = slot / 4;
            const j = slot % 4;
            self.xq[r * k + grp * 16 + 2 * q + (j % 2) + 8 * (j / 2)] = got[r * k + grp * 16 + slot];
        };
        self.rs = try a.alloc(f32, m);
        @memcpy(std.mem.sliceAsBytes(self.rs), try self.rig.bytes(self.rsd, m * 4));
        self.out = try self.rig.output(m * n * 2);
        self.base = try self.rig.scratch(m * n * 2);
        return true;
    }

    pub fn run(self: *Lane8t) !void {
        try self.l.matmulTile(self.rig.s, self.codes, self.rsd, self.shape.rows, self.weight, self.out, rows(self.kind));
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Lane8t) ![]const u8 {
        try self.l.matmulTile(self.rig.s, self.codes, self.rsd, self.shape.rows, self.weight, self.base, 16);
        return self.rig.bytes(self.base, self.count() * 2);
    }

    pub fn count(self: *Lane8t) usize {
        return self.shape.rows * self.shape.n;
    }

    fn terms(self: *const Lane8t, r: usize, j: usize, t: []f64) void {
        const k = self.shape.k;
        for (t, 0..) |*o, i| o.* = e4m3(self.xq[r * k + i]) * e4m3(self.w[j * k + i]);
    }

    pub fn reference(self: *Lane8t, want: []f64, mag: []f64) !void {
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
            const f = @as(f64, self.rs[r]) * self.cs[j];
            want[r * self.shape.n + j] = sum * f;
            mag[r * self.shape.n + j] = abs * @abs(f);
        };
    }

    pub fn simulate(self: *Lane8t, acc: c.Accum, order: g.Order, out: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        for (0..self.shape.rows) |r| for (0..self.shape.n) |j| {
            self.terms(r, j, t);
            out[r * self.shape.n + j] = contract.rows.sumIn(acc, order, t) * self.rs[r] * self.cs[j];
        };
    }
};
