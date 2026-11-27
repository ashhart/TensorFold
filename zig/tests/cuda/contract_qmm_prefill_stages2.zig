//! Two-stage versus three-stage bytes at both GB10 prompt shapes, including complete row tile edges.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const many = @import("contract_qmm_prefill_many.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const A = cuda.qlinear.Affine4Kernels;

fn noSums(_: c.Shape) usize {
    return 0;
}

const served = blk: {
    var shapes: [18]c.Shape = undefined;
    var at: usize = 0;
    for ([_]usize{ 6144, 17408 }) |k| for ([_]usize{ 17, 127, 128, 129, 2048, 4095, 4096, 8191, 8192 }) |rows| {
        shapes[at] = .{ .rows = rows, .n = 5120, .k = k };
        at += 1;
    };
    break :blk shapes;
};

pub const Registrations = struct {
    affine: A,
    module: cuda.Module,
    good: many.Product,
    shifted: many.Product,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, affine: *A, module: cuda.Module) Registrations {
        return .{
            .affine = affine.*,
            .module = module,
            .good = .{ .rig = .{ .gpu = gpu, .s = s }, .affine = affine, .member_widths = &.{5120} },
            .shifted = .{ .rig = .{ .gpu = gpu, .s = s }, .affine = affine, .member_widths = &.{5120} },
        };
    }

    pub fn append(self: *Registrations, cap: u32, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        if (cap != 121) return;
        try self.affine.enablePromptStages2(self.module);
        self.good.affine = &self.affine;
        self.shifted.affine = &self.affine;
        try kernels.append(a, kernel(&self.good));
        var control = kernel(&self.shifted);
        control.spec.name = "control: qmm_prefill stages2 shifts its output by two BF16 elements";
        control.spec.replaces = null;
        control.vt = &shifted_vt;
        control.control = .shapes;
        try kernels.append(a, control);
    }

    pub fn deinit(self: *Registrations) void {
        self.good.deinit();
        self.shifted.deinit();
    }
};

fn kernel(product: *many.Product) g.Kernel {
    return .{
        .spec = .{
            .name = "cuda qmm_prefill stages2 exact Qwen prompt",
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .replaces = "same 2x4-warp tile at three stages",
            .nan = .{ .stated = "all output bytes equal the three-stage tile, nonfinite payloads included" },
            .served = &served,
        },
        .ctx = product,
        .vt = &vt,
    };
}

const vt: g.VTable = blk: {
    var value = g.hooks(many.Product);
    value.run = runStages2;
    value.replaced = replaced;
    value.check = checkStages2;
    break :blk value;
};
const shifted_vt: g.VTable = blk: {
    var value = vt;
    value.run = runShifted;
    break :blk value;
};
fn runStages2(ctx: *anyopaque) anyerror!void {
    return @as(*many.Product, @ptrCast(@alignCast(ctx))).runStages2();
}
fn runShifted(ctx: *anyopaque) anyerror!void {
    return @as(*many.Product, @ptrCast(@alignCast(ctx))).runStages2Shifted();
}
fn replaced(ctx: *anyopaque) anyerror![]const u8 {
    return @as(*many.Product, @ptrCast(@alignCast(ctx))).replacedThreeStage();
}
fn checkStages2(ctx: *anyopaque, gate: c.Gate) anyerror!?[]const u8 {
    return @as(*many.Product, @ptrCast(@alignCast(ctx))).checkStages2(gate);
}

pub fn run(gpu: check.Gpu) !void {
    const cap = try gpu.ctx.capability();
    try check.expect(cap == 121, "two-stage prompt contract requires GB10 sm_121; got {d}", .{cap});
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var mods: [4]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*m| m.unload();
    for ([_][]const u8{ cuda.kernels.qmm_group, cuda.kernels.qmm_prefill, cuda.kernels.lane_gemv, cuda.kernels.affine4_pack }) |image| {
        mods[loaded] = try cuda.Module.load(gpu.d, image);
        loaded += 1;
    }
    var affine = try A.resolve(gpu.d, mods[0], mods[1], mods[2], mods[3], 0, true);
    defer affine.deinit();
    var registrations = Registrations.init(gpu, s, &affine, mods[1]);
    defer registrations.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try registrations.append(cap, gpu.gpa, &kernels);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x246464, &tally);
    std.debug.print("SUMMARY qmm_prefill_stages2: {d} pass, {d} fail; {d} controls red, {d} missed\n", .{
        tally.passed, tally.failed, tally.caught, tally.missed,
    });
    try check.expect(tally.problems() == 0, "two-stage prompt contract: {d} problems", .{tally.problems()});
}
