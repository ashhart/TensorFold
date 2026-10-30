//! Qwen3.8 GB10 large affine-4 prompts on the shipped 2x2-warp tile, byte-equal to the default tile.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const gpu_check = @import("check.zig");
const many = @import("contract_qmm_prefill_many.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const qlinear = cuda.qlinear;

fn noSums(_: c.Shape) usize {
    return 0;
}

const served = [_]c.Shape{
    .{ .rows = 2048, .n = 12288, .k = 5120 },
    .{ .rows = 2048, .n = 10240, .k = 5120 },
    .{ .rows = 2048, .n = 17408, .k = 5120 },
    .{ .rows = 4096, .n = 12288, .k = 5120 },
    .{ .rows = 4096, .n = 10240, .k = 5120 },
    .{ .rows = 4096, .n = 17408, .k = 5120 },
};

const rule = "every output byte equals qmm_prefill's default 2x4-warp tile on the same BF16 input and affine-4 weights";

pub const Registrations = struct {
    good: many.Product,
    swapped: many.Product,

    pub fn init(gpu: gpu_check.Gpu, s: cuda.Stream, affine: *qlinear.Affine4Kernels) Registrations {
        return .{
            .good = .{ .rig = .{ .gpu = gpu, .s = s }, .affine = affine },
            .swapped = .{ .rig = .{ .gpu = gpu, .s = s }, .affine = affine },
        };
    }

    pub fn deinit(self: *Registrations) void {
        self.good.deinit();
        self.swapped.deinit();
    }

    pub fn append(self: *Registrations, cap: u32, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        if (cap != 121) return;
        try kernels.append(a, kernel(&self.good));
        try kernels.append(a, swappedKernel(&self.swapped));
    }
};

pub fn run(gpu: gpu_check.Gpu) !void {
    const cap = try gpu.ctx.capability();
    try gpu_check.expect(cap == 121, "qmm_prefill 2x2 tile contract needs sm_121, got sm_{d}", .{cap});
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const kk = cuda.kernels;
    var mods: [4]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*m| m.unload();
    for ([_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack }, 0..) |image, i| {
        mods[i] = try cuda.Module.load(gpu.d, image);
        loaded += 1;
    }
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var affine = try qlinear.Affine4Kernels.resolve(gpu.d, mods[0], mods[1], mods[2], mods[3], sms, true);
    defer affine.deinit();
    var registrations = Registrations.init(gpu, s, &affine);
    defer registrations.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try registrations.append(cap, gpu.gpa, &kernels);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x221398, &tally);
    std.debug.print("SUMMARY qmm_prefill_tile22: {d} pass, {d} fail; {d} controls red, {d} missed\n", .{
        tally.passed, tally.failed, tally.caught, tally.missed,
    });
    try gpu_check.expect(tally.problems() == 0, "qmm_prefill 2x2 tile contract: {d} failures", .{tally.problems()});
}

pub fn kernel(product: *many.Product) g.Kernel {
    return .{
        .spec = .{
            .name = "cuda qmm_prefill 2x2 bit-exact BF16 GB10 large tile",
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .replaces = "qmm_prefill's default 2x4-warp, three-stage tile",
            .nan = .{ .stated = rule },
            .served = &served,
        },
        .ctx = product,
        .vt = &tile_vt,
    };
}

const tile_vt: g.VTable = blk: {
    var vt = g.hooks(many.Product);
    vt.run = runHalf;
    break :blk vt;
};

fn runHalf(ctx: *anyopaque) anyerror!void {
    const product: *many.Product = @ptrCast(@alignCast(ctx));
    return product.runHalf();
}

fn swappedKernel(product: *many.Product) g.Kernel {
    var k = kernel(product);
    k.spec.name = "control: qmm_prefill 2x2 swaps two equal-width output bindings";
    // Product.check snapshots the swapped output before the replacement readback reuses the rig buffer.
    k.spec.replaces = null;
    k.vt = &swapped_vt;
    k.control = .shapes;
    return k;
}

const swapped_vt: g.VTable = blk: {
    var vt = g.hooks(many.Product);
    vt.run = runSwapped;
    break :blk vt;
};

fn runSwapped(ctx: *anyopaque) anyerror!void {
    const product: *many.Product = @ptrCast(@alignCast(ctx));
    return product.runHalfSwapped();
}
