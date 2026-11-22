//! MXFP8 shared-down direct stores replace the same contiguous product and shared scatter, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const dims = 2560;
const width = 640;
const slots = 11;
const shapes = blk: {
    var out: [16]c.Shape = undefined;
    for (&out, 0..) |*s, i| s.* = .{ .rows = i + 1, .n = dims, .k = width };
    break :blk out;
};
fn depth(_: c.Shape) usize {
    return 0;
}

pub const Product = struct {
    rig: Rig,
    lane: cuda.qmmf.Lane,
    glue: *const flash.Glue,
    half: bool,
    control: bool = false,
    rows: usize = 0,
    w: cuda.qmmf.Weight = undefined,
    x: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    contiguous: u64 = 0,
    pub fn deinit(self: *Product) void {
        self.rig.deinit();
    }
    const vt = g.hooks(Product);
    pub fn kernel(self: *Product) g.Kernel {
        return .{ .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null, .spec = .{
            .name = if (self.control) "control: MXFP8 direct shared down scaled 1.5x" else if (self.half) "cuda qmmf MXFP8 direct shared half16" else "cuda qmmf MXFP8 direct shared16",
            .accum = .none,
            .depth = depth,
            .out = .raw,
            .nan = .{ .stated = "all slot bytes equal the identical contiguous MXFP8 matmul then sharedScatter, including nonfinites" },
            .replaces = "contiguous qmmf MXFP8 FP32 shared down then sharedScatter",
            .served = &shapes,
        } };
    }
    pub fn prepare(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.rows < 1 or shape.rows > 16 or shape.n != dims or shape.k != width) return false;
        self.rows = shape.rows;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const x = try a.alloc(u16, self.rows * width);
        defer a.free(x);
        for (x) |*v| v.* = c.toBf16((rng.float(f32) - 0.5) * 4);
        if (fill == .nonfinite) for (0..self.rows) |row| {
            x[row * width + row] = switch (row % 4) {
                0 => 0x7fc1,
                1 => 0x7f80,
                2 => 0xff80,
                else => 0x8000,
            };
        };
        if (fill == .stress) for (0..self.rows) |row| for (0..width) |k| {
            x[row * width + k] = c.toBf16(if (k % 2 == 0) @as(f32, 256) else -256);
        };
        const codes = try a.alloc(u8, dims * width);
        defer a.free(codes);
        for (codes) |*v| v.* = @as(u8, @intCast(rng.uintLessThan(u32, 0x60))) | (if (rng.boolean()) @as(u8, 0x80) else 0);
        const scales = try a.alloc(u8, dims * (width / 32));
        defer a.free(scales);
        for (scales) |*v| v.* = @intCast(125 + rng.uintLessThan(u32, 5));
        const packed_codes = try a.alloc(u8, codes.len);
        defer a.free(packed_codes);
        const packed_scales = try a.alloc(u8, scales.len);
        defer a.free(packed_scales);
        cuda.mxfp8.packCodes(packed_codes, codes, dims, width);
        cuda.mxfp8.packScales(packed_scales, scales, dims, width);
        self.w = cuda.mxfp8.weight(try self.rig.input(packed_codes), try self.rig.input(packed_scales), dims, width);
        self.w.scale = 0.75;
        self.x = try self.rig.input(std.mem.sliceAsBytes(x));
        self.out = try self.rig.output(self.count());
        self.base = try self.rig.scratch(self.count());
        self.contiguous = try self.rig.scratch(self.rows * dims * 4);
        return true;
    }
    pub fn run(self: *Product) !void {
        var w = self.w;
        if (self.control) w.scale *= 1.5;
        try cuda.qmmf_shared.matmulMxfp8(self.lane, self.rig.s, self.x, width, self.rows, w, self.out + 10 * dims * 4, slots * dims);
        try self.rig.s.synchronize();
    }
    pub fn replaced(self: *Product) ![]const u8 {
        try self.rig.gpu.d.check(self.rig.gpu.d.api.cuMemsetD8Async(self.base, 0xff, self.count(), self.rig.s.handle), "cuMemsetD8Async");
        try self.lane.matmulF32(self.rig.s, self.x, width, self.rows, self.w, self.contiguous, 0);
        try self.glue.sharedScatter(self.rig.s, self.contiguous, self.base, self.rows, dims, slots);
        return self.rig.bytes(self.base, self.count());
    }
    pub fn check(self: *Product, _: g.Gate) !?[]const u8 {
        const got = try self.rig.outputBytes();
        const want = try self.replaced();
        return if (std.mem.eql(u8, got, want)) null else "direct shared down differs from contiguous plus scatter";
    }
    pub fn count(self: *Product) usize {
        return self.rows * slots * dims * 4;
    }
    pub fn reference(_: *Product, _: []f64, _: []f64) !void {
        return error.NoFloat64Reference;
    }
};

pub const Set = struct {
    glue: flash.Glue,
    products: [3]Product,
    pub fn create(gpu: check.Gpu, s: cuda.Stream, original: cuda.qmmf.Lane, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const self = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(self);
        self.glue = try flash.Glue.load(gpu.d);
        errdefer self.glue.deinit();
        var lane = original;
        try cuda.qmmf_shared.loadMxfp8(&lane);
        for (&self.products, 0..) |*p, i| {
            var own = lane;
            own.half_below = if (i == 1) 0 else std.math.maxInt(usize);
            p.* = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = own, .glue = &self.glue, .half = i != 1, .control = i == 2 };
        }
        errdefer for (&self.products) |*p| p.deinit();
        for (&self.products) |*p| try list.append(a, p.kernel());
        return self;
    }
    pub fn deinit(self: *Set) void {
        const a = self.products[0].rig.gpu.gpa;
        for (&self.products) |*p| p.deinit();
        self.glue.deinit();
        a.destroy(self);
    }
};
