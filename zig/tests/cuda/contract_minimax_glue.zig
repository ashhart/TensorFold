//! MiniMax-M3.1-Flash's glue kernels (minimax_m3.cu) under the contract: every registration and its controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const mm3 = @import("minimax_m3_cuda");
const check = @import("check.zig");
const ref = @import("contract_minimax_glue_ref.zig");
const norm = @import("contract_minimax_glue_norm.zig");
const moe = @import("contract_minimax_glue_moe.zig");
const pick = @import("contract_minimax_glue_pick.zig");
const copy = @import("contract_minimax_glue_copy.zig");

const g = core.contract.gates;
const Kernels = mm3.kernels.Kernels;

/// The glue's registrations, held where the gates point at them (the caller's heap set).
pub const Set = struct {
    norm: ref.Gated(norm.Norm) = undefined,
    adds: [2]ref.Gated(norm.AddNorm) = undefined,
    preps: [4]ref.Gated(norm.Prep) = undefined,
    swiglus: [2]ref.Gated(moe.Swiglu) = undefined,
    router: ref.Gated(moe.Router) = undefined,
    routes: [2]ref.Checked(moe.Route) = undefined,
    partials: [2]ref.Gated(moe.Partial) = undefined,
    top1s: [2]ref.Checked(pick.Top1) = undefined,
    markovs: [2]ref.Gated(pick.Markov) = undefined,
    merge: ref.Checked(pick.Merge) = undefined,
    copies: [3]ref.Gated(copy.Copy) = undefined,
    words: ref.Checked(copy.PutWords) = undefined,
    rope: ref.Gated(copy.VitRope) = undefined,
    patch: ref.Gated(copy.PatchEmbed) = undefined,

    /// Every registration over `k`, appended to `list` (each control after the kernel it breaks).
    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, k: *const Kernels, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        const rig: @import("contract_rig.zig").Rig = .{ .gpu = gpu, .s = s };
        set.norm = .{ .rig = rig, .k = .{ .kern = k } };
        for (&set.adds, [_]norm.View{ .resid, .normed }) |*x, v| x.* = .{ .rig = rig, .k = .{ .kern = k, .view = v } };
        for (&set.preps, [_]norm.Form{ .full, .draft, .absorb, .full }, 0..) |*x, f, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .form = f, .late = i == 3 } };
        for (&set.swiglus, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .minus = i == 1 } };
        set.router = .{ .rig = rig, .k = .{ .kern = k } };
        for (&set.routes, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .reversed = i == 1 } };
        for (&set.partials, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .shifted = i == 1 } };
        for (&set.top1s, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .f32_in = i == 1 } };
        for (&set.markovs, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .late = i == 1 } };
        set.merge = .{ .rig = rig, .k = .{ .kern = k } };
        for (&set.copies, std.enums.values(copy.Move)) |*x, m| x.* = .{ .rig = rig, .k = .{ .kern = k, .move = m } };
        set.words = .{ .rig = rig, .k = .{ .kern = k } };
        set.rope = .{ .rig = rig, .k = .{ .kern = k } };
        set.patch = .{ .rig = rig, .k = .{ .kern = k } };
        for (&set.copies) |*x| try list.append(a, x.kernel());
        try list.append(a, set.norm.kernel());
        for (&set.adds) |*x| try list.append(a, x.kernel());
        for (&set.preps) |*x| try list.append(a, x.kernel());
        for (&set.swiglus) |*x| try list.append(a, x.kernel());
        try list.append(a, set.router.kernel());
        for (&set.routes) |*x| try list.append(a, x.kernel());
        for (&set.partials) |*x| try list.append(a, x.kernel());
        for (&set.top1s) |*x| try list.append(a, x.kernel());
        for (&set.markovs) |*x| try list.append(a, x.kernel());
        try list.append(a, set.merge.kernel());
        try list.append(a, set.words.kernel());
        try list.append(a, set.rope.kernel());
        try list.append(a, set.patch.kernel());
    }

    pub fn deinit(set: *Set) void {
        set.norm.deinit();
        for (&set.adds) |*x| x.deinit();
        for (&set.preps) |*x| x.deinit();
        for (&set.swiglus) |*x| x.deinit();
        set.router.deinit();
        for (&set.routes) |*x| x.deinit();
        for (&set.partials) |*x| x.deinit();
        for (&set.top1s) |*x| x.deinit();
        for (&set.markovs) |*x| x.deinit();
        set.merge.deinit();
        for (&set.copies) |*x| x.deinit();
        set.words.deinit();
        set.rope.deinit();
        set.patch.deinit();
    }
};
