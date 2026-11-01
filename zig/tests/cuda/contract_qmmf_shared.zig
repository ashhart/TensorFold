//! Strided shared-down stores keep each BF16-weight product's FP32 bytes and leave every other slot unchanged.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const Dense = core.contract.dense.Dense;
const q = cuda.qmmf;

fn depth(s: c.Shape) usize {
    return s.k / 16 + 24;
}

pub const Product = struct {
    rig: Rig,
    lane: *const q.Lane,
    index: usize,
    control: bool = false,
    d: ?Dense = null,
    w: q.Weight = undefined,
    x: u64 = 0,
    slots: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,

    pub fn deinit(self: *Product) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }
    const vt = g.hooks(Product);

    pub fn kernel(self: *Product) g.Kernel {
        const names = [_][]const u8{ "cuda qmmf shared stride half16", "cuda qmmf shared stride 16", "cuda qmmf shared stride 32", "cuda qmmf shared stride 64" };
        const bm: usize = if (self.index < 2) 16 else if (self.index == 2) 32 else 64;
        return .{ .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null, .spec = .{
            .name = if (self.control) "control: shared down scaled 1.5x" else names[self.index],
            .accum = .f32,
            .depth = depth,
            .ops = 1,
            .out = .f32,
            .replaces = "contiguous qmmf BF16-weight FP32 down before shared scatter",
            .tiles = .{ .{ .size = bm }, .{ .size = if (self.index == 0) 32 else 64, .step = 32, .min = 32 }, .{ .size = 640, .step = 64, .min = 64 } },
            .served = switch (self.index) {
                0, 1 => &.{ .{ .rows = 1, .n = 2560, .k = 640 }, .{ .rows = 16, .n = 2560, .k = 640 } },
                2 => &.{.{ .rows = 32, .n = 2560, .k = 640 }},
                else => &.{ .{ .rows = 65, .n = 2560, .k = 640 }, .{ .rows = 1536, .n = 2560, .k = 640 }, .{ .rows = 4218, .n = 2560, .k = 640 } },
            },
        } };
    }
    pub fn prepare(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 256);
        const weights = try a.alloc(u16, shape.n * shape.k);
        defer a.free(weights);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x5107);
        const rng = prng.random();
        for (weights) |*v| v.* = c.toBf16(if (fill == .stress) @floatFromInt(1 + rng.uintLessThan(u32, 7)) else rng.float(f32) - 0.5);
        if (fill == .stress) for (0..shape.n) |j| for (shape.k / 2..shape.k) |i| {
            weights[j * shape.k + i] = weights[j * shape.k + i - shape.k / 2] | 0x8000;
        };
        for (d.w, weights) |*v, w| v.* = c.fromBf16(w);
        const npad = cuda.modelopt.padded(shape.n);
        const tiled = try a.alloc(u16, npad * shape.k);
        defer a.free(tiled);
        q.packBf16(tiled, weights, shape.n, shape.k);
        self.w = cuda.modelopt.bf16Weight(try self.rig.input(std.mem.sliceAsBytes(tiled)), shape.n, shape.k);
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.slots = try self.rig.scratch(self.count() * 11 * 4);
        self.out = try self.rig.output(self.count() * 4);
        self.base = try self.rig.scratch(self.count() * 4);
        return true;
    }
    pub fn run(self: *Product) !void {
        const s = self.d.?.shape;
        const bm: usize = if (self.index < 2) 16 else if (self.index == 2) 32 else 64;
        const bn: usize = if (self.index == 0) 32 else 64;
        const band = @max(1, @min((s.rows + bm - 1) / bm, (12 << 20) / (bm * s.k * 2)));
        try self.rig.zero(self.slots, self.count() * 11 * 4);
        var args: cuda.Args = .{};
        args.add(self.x);
        args.add(self.w.codes);
        args.add(if (self.control) @as(f32, 1.5) else self.w.scale);
        args.add(self.slots + 10 * s.n * 4);
        for ([_]usize{ s.rows, s.n, s.k, self.w.npad, s.k, band, 11 * s.n }) |v| args.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(self.lane.shared_down[self.index], .{
            .grid = .{ .x = @intCast(((s.rows + bm - 1) / bm) * ((s.n + bn - 1) / bn)) },
            .block = .{ .x = 128 },
            .shared = if (self.index == 0) q.halfShared(.bf16) else q.sharedBytes(.bf16, @intCast(bm)),
        }, self.rig.s, &args);
        const bytes = try self.rig.bytes(self.slots, self.count() * 11 * 4);
        for (0..s.rows) |row| for (bytes[row * 11 * s.n * 4 ..][0 .. 10 * s.n * 4]) |v| if (v != 0) return error.SharedDownTouchedRoutedSlot;
        for (0..s.rows) |row| try self.rig.gpu.d.check(self.rig.gpu.d.api.cuMemcpyDtoDAsync_v2(
            self.out + row * s.n * 4,
            self.slots + (row * 11 + 10) * s.n * 4,
            s.n * 4,
            self.rig.s.handle,
        ), "cuMemcpyDtoDAsync");
        try self.rig.s.synchronize();
    }
    pub fn replaced(self: *Product) ![]const u8 {
        const lane = self.lane.*;
        const s = self.d.?.shape;
        try lane.matmulF32(self.rig.s, self.x, s.k, s.rows, self.w, self.base, 0);
        return self.rig.bytes(self.base, self.count() * 4);
    }
    pub fn count(self: *Product) usize {
        return self.d.?.outputs();
    }
    pub fn reference(self: *Product, want: []f64, mag: []f64) !void {
        try self.d.?.reference(want, mag);
    }
    pub fn simulate(self: *Product, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(acc, order, out);
    }
};

/// Owned registration storage keeps kernel contexts valid throughout the full contract run.
pub const Registrations = struct {
    products: [5]Product,

    pub fn create(gpu: check.Gpu, stream: cuda.Stream, lane: *const q.Lane, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !*Registrations {
        const self = try gpu.gpa.create(Registrations);
        errdefer gpu.gpa.destroy(self);
        for (&self.products, 0..) |*p, i| p.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .lane = lane, .index = i % 4, .control = i == 4 };
        errdefer for (&self.products) |*p| p.deinit();
        for (&self.products) |*p| try kernels.append(a, p.kernel());
        return self;
    }

    pub fn deinit(self: *Registrations) void {
        const a = self.products[0].rig.gpu.gpa;
        for (&self.products) |*p| p.deinit();
        a.destroy(self);
    }
};

/// Complete gates 1-4 and planted control for the strided shared-down operator.
pub fn run(gpu: check.Gpu) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var lane = try q.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    const regs = try Registrations.create(gpu, stream, &lane, gpu.gpa, &kernels);
    defer regs.deinit();
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x5107, &tally);
    std.debug.print("SUMMARY shared down gates 1-4: {d} pass, {d} fail; {d} controls red, {d} missed\n", .{ tally.passed, tally.failed, tally.caught, tally.missed });
    try check.expect(tally.problems() == 0, "shared down contract failed", .{});
}
