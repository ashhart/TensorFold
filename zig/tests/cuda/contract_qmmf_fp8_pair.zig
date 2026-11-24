//! The paired Qwen GDN projection is a byte replacement of two scalar FP8 lane launches.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check_gpu = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const q = cuda.qmmf;

const rows = 16;
const inputs = 5120;
const widths = [_]usize{ 10240, 6144 };
const byte_sizes = [_]usize{ rows * widths[0] * 2, rows * widths[1] * 2 };
const served: c.Shape = .{ .rows = rows, .n = widths[0] + widths[1], .k = inputs };
const name = "cuda qmmf FP8 QKV and Z pair";
const rule = "each projection has the bytes of its separate FP8 lane launch";

fn depth(_: c.Shape) usize {
    return 0;
}

const Product = struct {
    rig: Rig,
    lane: *const q.Lane,
    pair: q.Fp8Pair,
    control: bool,
    x: u64 = 0,
    weights: [2]q.Weight = undefined,
    outputs: [2]u64 = undefined,
    baseline: u64 = 0,

    const vt = g.hooks(Product);

    fn kernel(p: *Product) g.Kernel {
        return .{
            .ctx = p,
            .vt = &vt,
            .control = if (p.control) .shapes else null,
            .spec = .{
                .name = if (p.control) "control: qmmf FP8 QKV and Z pair changes the Z scale" else name,
                .out = .raw,
                .accum = .none,
                .depth = depth,
                .replaces = "two separate scalar FP8 lane launches",
                .nan = .{ .stated = rule },
                .served = &.{served},
            },
        };
    }

    pub fn prepare(p: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.rig.reset();
        if (shape.n != served.n or shape.k != inputs) return false;
        // The non-finite gate clamps its requested rows to eight. The paired entry still runs all 16.
        if (shape.rows != rows and fill != .nonfinite) return false;
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        const a = p.rig.gpu.gpa;
        const x = try a.alloc(u16, rows * inputs);
        defer a.free(x);
        for (x) |*v| v.* = c.toBf16(random.float(f32) - 0.5);
        if (fill == .nonfinite) {
            x[0] = 0x7fc0;
            x[inputs] = 0x7f80;
            x[2 * inputs] = 0xff80;
        }
        p.x = try p.rig.input(std.mem.sliceAsBytes(x));
        for (widths, byte_sizes, [_]f32{ 0.625, 0.875 }, 0..) |n, bytes, scale, i| {
            const codes = try a.alloc(u8, n * inputs);
            defer a.free(codes);
            random.bytes(codes);
            for (codes) |*v| v.* = 0x30 | (v.* & 0x8f);
            const tiled = try a.alloc(u8, cuda.modelopt.padded(n) * inputs);
            defer a.free(tiled);
            cuda.fp8.packCodes(tiled, codes, n, inputs);
            p.weights[i] = cuda.modelopt.fp8Weight(try p.rig.input(tiled), scale, n, inputs);
            p.outputs[i] = try p.rig.output(bytes);
        }
        p.baseline = try p.rig.scratch(byte_sizes[0] + byte_sizes[1]);
        return true;
    }

    pub fn run(p: *Product) !void {
        var z = p.weights[1];
        if (p.control) z.scale *= 1.5;
        if (!try p.pair.project(p.lane, p.rig.s, p.x, rows, p.weights[0], z, p.outputs[0], p.outputs[1])) return error.RefusedPairFixture;
        try p.rig.s.synchronize();
    }

    pub fn replaced(p: *Product) ![]const u8 {
        var offset: usize = 0;
        for (p.weights, byte_sizes) |weight, bytes| {
            try p.lane.matmul(p.rig.s, p.x, inputs, rows, weight, p.baseline + offset, 0);
            offset += bytes;
        }
        return p.rig.bytes(p.baseline, offset);
    }

    pub fn count(_: *Product) usize {
        return rows * served.n;
    }

    pub fn reference(_: *Product, _: []f64, _: []f64) !void {
        return error.RequiresReplacementBytes;
    }

    pub fn check(p: *Product, _: c.Gate) !?[]const u8 {
        // The candidate and baseline downloads use different host buffers in Rig.
        const actual = try p.rig.outputBytes();
        const expected = try p.replaced();
        return if (std.mem.eql(u8, actual, expected)) null else rule;
    }
};

pub const Registrations = struct {
    products: [2]Product,

    pub fn create(gpu: check_gpu.Gpu, stream: cuda.Stream, lane: *const q.Lane, cap: u32, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !?*Registrations {
        if (cap != 120) return null;
        const pair = try q.Fp8Pair.load(lane);
        const self = try gpu.gpa.create(Registrations);
        errdefer gpu.gpa.destroy(self);
        for (&self.products, 0..) |*p, i| p.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .lane = lane, .pair = pair, .control = i != 0 };
        for (&self.products) |*p| try kernels.append(a, p.kernel());
        return self;
    }

    pub fn deinit(self: *Registrations) void {
        const a = self.products[0].rig.gpu.gpa;
        for (&self.products) |*p| p.rig.deinit();
        a.destroy(self);
    }
};

pub fn run(gpu: check_gpu.Gpu) !void {
    const cap = try gpu.ctx.capability();
    try check_gpu.expect(cap == 120, "FP8 QKV and Z pair contract requires sm_120", .{});
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var lane = try q.Lane.load(gpu.d, cap / 10);
    defer lane.unload();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    const regs = (try Registrations.create(gpu, stream, &lane, cap, gpu.gpa, &kernels)) orelse return error.UnsupportedDevice;
    defer regs.deinit();
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x517a, &tally);
    std.debug.print("SUMMARY FP8 QKV and Z pair gates 1-4: {d} pass, {d} fail; {d} controls red, {d} missed\n", .{ tally.passed, tally.failed, tally.caught, tally.missed });
    try check_gpu.expect(tally.problems() == 0, "FP8 QKV and Z pair contract failed", .{});
}
