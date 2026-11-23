//! Inkling 2's DSpark drafter glue (inkling2/dspark.cu) under the contract, with controls.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const dk = @import("inkling2").dspark_ops;
const ref = @import("contract_minimax_glue_ref.zig");
const norm = @import("contract_inkling2_dspark_norm.zig");
const attn = @import("contract_inkling2_dspark_attn.zig");
const head = @import("contract_inkling2_dspark_head.zig");
const glue = @import("contract_inkling2_dspark_glue.zig");
const pick = @import("contract_inkling2_ops_pick.zig");
const opsref = @import("contract_inkling2_ops_ref.zig");

/// The drafter's registrations, each control after the kernel it breaks (the caller's heap set holds them).
pub const Set = struct {
    k: dk.Kernels = undefined,
    mixes: [2]ref.Gated(norm.Mix) = undefined,
    adds: [3]ref.Gated(norm.AddNorm) = undefined,
    qks: [3]ref.Gated(norm.Qk) = undefined,
    attends: [2]attn.Attend = undefined,
    merges: [2]ref.Gated(attn.Merge) = undefined,
    silus: [2]ref.Gated(glue.Silu) = undefined,
    ctxs: [2]ref.Gated(head.Ctx) = undefined,
    rnns: [2]ref.Gated(head.Rnn) = undefined,
    vocabs: [4]head.Banded(head.Vocab) = undefined,
    picks: [4]ref.Checked(glue.Pick) = undefined,
    // Mini's batched pass (dspark_many.cuh), each with the control of the kernel it copies
    qkms: [3]ref.Gated(norm.Qk) = undefined,
    attms: [2]attn.Attend = undefined,
    ctxms: [2]ref.Gated(head.Ctx) = undefined,
    rnnms: [2]ref.Gated(head.Rnn) = undefined,
    top1ms: [2]opsref.Checked(pick.Top1) = undefined,
    racems: [2]opsref.Checked(pick.Race) = undefined,

    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        set.k = try dk.Kernels.load(gpu.d);
        errdefer set.k.unload();
        const k = &set.k;
        const rig: @import("contract_rig.zig").Rig = .{ .gpu = gpu, .s = s };
        for (&set.mixes, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .drop = i == 1 } };
        for (&set.adds, [_]norm.View{ .resid, .normed, .normed }, 0..) |*x, v, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .view = v, .short = i == 2 } };
        for (&set.qks, [_]norm.Form{ .draft, .absorb, .draft }, 0..) |*x, f, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .form = f, .late = i == 2 } };
        for (&set.attends, 0..) |*x, i| x.* = .{ .rig = rig, .k = k, .late = i == 1 };
        for (&set.merges, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .short = i == 1 } };
        for (&set.silus, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .swapped = i == 1 } };
        for (&set.ctxs, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .early = i == 1 } };
        for (&set.rnns, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .late = i == 1 } };
        for (&set.vocabs, 0..) |*x, i| x.* = .{ .inner = .{ .rig = rig, .k = .{ .kern = k, .short = i % 2 == 1, .rows = i >= 2 } } };
        for (&set.picks, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .alone = i % 2 == 1, .rows = i >= 2 } };
        for (&set.qkms, [_]norm.Form{ .draft, .absorb, .draft }, 0..) |*x, f, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .form = f, .late = i == 2, .many = true } };
        for (&set.attms, 0..) |*x, i| x.* = .{ .rig = rig, .k = k, .late = i == 1, .many = true };
        for (&set.ctxms, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .early = i == 1, .many = true } };
        for (&set.rnnms, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .late = i == 1, .many = true } };
        for (&set.top1ms, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = undefined, .short = i == 1, .many = k } };
        for (&set.racems, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = undefined, .late = i == 1, .many = k } };
        for (&set.mixes) |*x| try list.append(a, x.kernel());
        for (&set.adds) |*x| try list.append(a, x.kernel());
        for (&set.qks) |*x| try list.append(a, x.kernel());
        for (&set.attends) |*x| try list.append(a, x.kernel());
        for (&set.merges) |*x| try list.append(a, x.kernel());
        for (&set.silus) |*x| try list.append(a, x.kernel());
        for (&set.ctxs) |*x| try list.append(a, x.kernel());
        for (&set.rnns) |*x| try list.append(a, x.kernel());
        for (&set.vocabs) |*x| try list.append(a, x.kernel());
        for (&set.picks) |*x| try list.append(a, x.kernel());
        inline for (.{ "qkms", "attms", "ctxms", "rnnms", "top1ms", "racems" }) |f| for (&@field(set, f)) |*x| try list.append(a, x.kernel());
    }

    pub fn deinit(set: *Set) void {
        for (&set.mixes) |*x| x.deinit();
        for (&set.adds) |*x| x.deinit();
        for (&set.qks) |*x| x.deinit();
        for (&set.attends) |*x| x.deinit();
        for (&set.merges) |*x| x.deinit();
        for (&set.silus) |*x| x.deinit();
        for (&set.ctxs) |*x| x.deinit();
        for (&set.rnns) |*x| x.deinit();
        for (&set.vocabs) |*x| x.deinit();
        for (&set.picks) |*x| x.deinit();
        inline for (.{ "qkms", "attms", "ctxms", "rnnms", "top1ms", "racems" }) |f| for (&@field(set, f)) |*x| x.deinit();
        set.k.unload();
    }
};
