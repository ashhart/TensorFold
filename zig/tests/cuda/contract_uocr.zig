//! Unlimited-OCR-2's CUDA kernels under the contract (uocr-contract), one file per group (contract_uocr_*.zig).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
pub const kernels = @import("uocr_cuda_kernels");

const g = core.contract.gates;
const groups = .{ @import("contract_uocr_rows.zig"), @import("contract_uocr_moves.zig"), @import("contract_uocr_route.zig"), @import("contract_uocr_attention.zig"), @import("contract_uocr_matvec.zig") };

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

/// What every group's registrations share: the GPU, one stream, the family's kernels and the planted controls.
pub const Ctx = struct {
    gpu: check.Gpu,
    s: cuda.Stream,
    k: *const kernels.Kernels,
    ctl: cuda.Module,

    /// A fresh rig on the shared stream (each registration owns one).
    pub fn rig(ctx: Ctx) rig_.Rig {
        return .{ .gpu = ctx.gpu, .s = ctx.s };
    }

    /// The family's kernels with `name` swapped for the planted control `control`.
    pub fn with(ctx: Ctx, comptime name: []const u8, control: [:0]const u8) !kernels.Kernels {
        var k = ctx.k.*;
        try k.swap(name, try ctx.ctl.function(control));
        return k;
    }
};

/// The served widths (the release's config).
pub const hidden = 1280;
pub const heads = 10;
pub const head_dim = 128;
pub const router_dim = 128;
pub const experts = 64;
pub const topk = 6;
pub const expert_ffn = 896;
pub const shared_ffn = 2 * expert_ffn;
pub const dense_ffn = 6848;
pub const vocab = 129280;
pub const sam_width = 768;
pub const clip_width = 1024;

/// The Unlimited-OCR-2 kernels' gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var k = try kernels.Kernels.load(gpu.d);
    defer k.unload();
    var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
    defer ctl.unload();
    const ctx: Ctx = .{ .gpu = gpu, .s = s, .k = &k, .ctl = ctl };
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const rows = try groups[0].Group.init(ctx, &list);
    defer rows.deinit();
    const moves = try groups[1].Group.init(ctx, &list);
    defer moves.deinit();
    const route = try groups[2].Group.init(ctx, &list);
    defer route.deinit();
    const attention = try groups[3].Group.init(ctx, &list);
    defer attention.deinit();
    const matvec = try groups[4].Group.init(ctx, &list);
    defer matvec.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x756f6372, &t);
    std.debug.print("SUMMARY uocr: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "uocr-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
