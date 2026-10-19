//! The checkpoint math's FP8 prompt tiles under the contract on CUDA: ck_lane.cu's 128-row tiles, either slice meeting.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Ck = cuda.ck.Ck;
const e4m3 = core.quant.fp8.e4m3;

/// Slices meeting in a cluster (tf_ck8_128c), or partials summed by tf_ck8_reduce (tf_ck8_128, GPUs without clusters).
pub const Kind = enum { cluster, reduce };

const rule = "as the 64-row tile it replaces gives them: a NaN input and a NaN weight code";

/// 32-input MMA steps over a slice, the products inside one, then up to 8 slices added.
fn laneDepth(s: c.Shape) usize {
    return s.k / 32 + 32 + 8;
}

/// Super's FP8 projections at prompt rows: the Mamba out projection, the latent projection and back.
pub fn spec(kind: Kind) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 129, .n = 4096, .k = 8192 }, .{ .rows = 256, .n = 1024, .k = 4096 }, .{ .rows = 300, .n = 4096, .k = 1024 } };
    const tiles = [3]c.Tile{ .{ .size = 128 }, .{ .size = 128, .step = 64, .min = 64 }, .{} };
    return switch (kind) {
        .cluster => .{ .name = "cuda ck FP8 prompt tile tf_ck8_128c (slices in a cluster)", .accum = .f32, .depth = laneDepth, .ops = 1, .out = .bf16, .replaces = "tf_ck8_64c and tf_ck8_64sc", .nan = .{ .stated = rule }, .tiles = tiles, .served = served },
        .reduce => .{ .name = "cuda ck FP8 prompt tile tf_ck8_128 (slices by tf_ck8_reduce)", .accum = .f32, .depth = laneDepth, .ops = 1, .out = .bf16, .replaces = "tf_ck8_64 and tf_ck8_64s", .nan = .{ .stated = rule }, .tiles = tiles, .served = served },
    };
}

/// shape.rows e4m3 rows (quant8's, from bf16 values it keeps exactly) times n e4m3 weight rows, row-major on the host.
pub const Lane = struct {
    rig: rig_.Rig,
    kind: Kind,
    ck: *const Ck,
    packer: *const cuda.modelopt.Packer,
    shape: c.Shape = undefined,
    x: []u8 = &.{}, // e4m3 codes [rows][k]
    w: []u8 = &.{}, // e4m3 codes [n][k]
    scale: f32 = 1,
    xd: u64 = 0,
    wd: u64 = 0,
    part: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Lane) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Lane) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.w);
        self.x = &.{};
        self.w = &.{};
    }

    pub fn kernel(self: *Lane) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Lane);

    /// A finite e4m3 code: every byte but the two NaNs.
    fn code(rng: std.Random) u8 {
        const b = rng.int(u8);
        return if (b & 0x7f == 0x7f) b ^ 1 else b;
    }

    /// Random finite codes; non-finite: a NaN input in row 1, a NaN weight code in column 5; stress: mirrored halves.
    pub fn prepare(self: *Lane, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 64 != 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const n = shape.n;
        const k = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try a.alloc(u8, shape.rows * k);
        self.w = try a.alloc(u8, n * k);
        for (self.x) |*v| v.* = code(rng);
        for (self.w) |*v| v.* = code(rng);
        self.scale = 0.005 + rng.float(f32) * 0.03;
        switch (fill) {
            .random => {},
            .nonfinite => {
                if (shape.rows > 1) self.x[k + 3] = 0x7f;
                if (n > 5) self.w[5 * k + k / 2] = 0xff;
            },
            .stress => self.stress(rng),
        }
        // quant8 at input scale 1 keeps every finite e4m3 value, so x8 holds self.x
        const xs = try a.alloc(u16, self.x.len);
        defer a.free(xs);
        for (xs, self.x) |*h, v| h.* = c.toBf16(@floatCast(e4m3(v)));
        const src = try self.rig.scratch(xs.len * 2);
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = src, .len = xs.len * 2 }).upload(0, std.mem.sliceAsBytes(xs));
        const wbytes = cuda.modelopt.padded(n) * k;
        const zeros = try a.alloc(u8, @max(self.x.len, wbytes));
        defer a.free(zeros);
        @memset(zeros, 0);
        self.xd = try self.rig.input(zeros[0..self.x.len]);
        try self.ck.quant8(self.rig.s, src, k, shape.rows, k, 1, self.xd);
        const wsrc = try self.rig.scratch(self.w.len);
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = wsrc, .len = self.w.len }).upload(0, self.w);
        self.wd = try self.rig.input(zeros[0..wbytes]);
        try self.packer.fp8Lane(self.rig.s, wsrc, self.wd, n, k);
        try self.rig.s.synchronize();
        self.part = try self.rig.scratch(cuda.qmmf.splitK(n, k) * shape.rows * n * 4);
        self.out = try self.rig.output(shape.rows * n * 2);
        self.base = try self.rig.scratch(shape.rows * n * 2);
        return true;
    }

    /// Gate 4: integers 1 to 16, x's second half its first but 8 inputs a step lower, w's second half negated; scale 1.
    fn stress(self: *Lane, rng: std.Random) void {
        const k = self.shape.k;
        const ints = [_]u8{ 0x38, 0x40, 0x44, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58 };
        self.scale = 1;
        for (0..self.shape.rows) |r| for (0..k / 2) |i| {
            const v = 1 + rng.uintLessThan(usize, 15);
            self.x[r * k + i] = ints[v];
            self.x[r * k + k / 2 + i] = ints[if (i < 8) v - 1 else v];
        };
        for (0..self.shape.n) |j| for (0..k / 2) |i| {
            const v = ints[rng.uintLessThan(usize, 16)];
            self.w[j * k + i] = v;
            self.w[j * k + k / 2 + i] = v | 0x80;
        };
    }

    fn weight(self: *const Lane) cuda.qmmf.Weight {
        return cuda.modelopt.fp8Weight(self.wd, self.scale, self.shape.n, self.shape.k);
    }

    pub fn run(self: *Lane) !void {
        try self.ck.lane8Tile(self.rig.s, 4, self.xd, self.shape.rows, self.weight(), 1, self.out, self.part);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Lane) ![]const u8 {
        const tile: usize = if (self.shape.rows % 64 == 0) 2 else 3;
        try self.ck.lane8Tile(self.rig.s, tile, self.xd, self.shape.rows, self.weight(), 1, self.base, self.part);
        return self.rig.bytes(self.base, self.count() * 2);
    }

    /// Gate 1: the replaced tile's bytes, with some output non-finite to prove the planted codes reach it.
    pub fn check(self: *Lane, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from {s}'s", .{ at, got.len, spec(self.kind).replaces.? });
        for (0..self.count()) |i| if (c.classOf(c.read(.bf16, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&self.msg, "no output is non-finite: the planted codes prove nothing", .{});
    }

    pub fn count(self: *Lane) usize {
        return self.shape.rows * self.shape.n;
    }

    fn terms(self: *const Lane, r: usize, j: usize, t: []f64) void {
        const k = self.shape.k;
        for (t, 0..) |*o, i| o.* = e4m3(self.x[r * k + i]) * e4m3(self.w[j * k + i]);
    }

    pub fn reference(self: *Lane, want: []f64, mag: []f64) !void {
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
            want[r * self.shape.n + j] = sum * self.scale;
            mag[r * self.shape.n + j] = abs * self.scale;
        };
    }

    pub fn simulate(self: *Lane, acc: c.Accum, order: g.Order, out: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        for (0..self.shape.rows) |r| for (0..self.shape.n) |j| {
            self.terms(r, j, t);
            out[r * self.shape.n + j] = contract.rows.sumIn(acc, order, t) * self.scale;
        };
    }
};
