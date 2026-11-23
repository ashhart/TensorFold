//! Inkling 2's glue (inkling2/ops.cu, rows.cu) under the contract, with controls.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const ref = @import("contract_inkling2_ops_ref.zig");
const mix = @import("contract_inkling2_ops_mix.zig");
const rows = @import("contract_inkling2_ops_rows.zig");
const route = @import("contract_inkling2_ops_route.zig");
const conv = @import("contract_inkling2_ops_conv.zig");
const pick = @import("contract_inkling2_ops_pick.zig");

const G = ref.Gated;
const K = ref.Checked;
const groups = .{ "embeds", "mixes", "dots", "topks", "gateups", "convs", "rewinds", "caps", "tails", "top1s", "races", "tobfs", "words" };

pub const Set = struct {
    k: ref.Kernels = undefined,
    loaded: bool = false,
    mixes: [7]G(mix.Mix) = undefined,
    embeds: [2]G(rows.Embed) = undefined,
    dots: [6]G(rows.Rowdot) = undefined,
    caps: [2]G(rows.Softcap) = undefined,
    topks: [2]K(route.Topk) = undefined,
    gateups: [2]G(route.Gateup) = undefined,
    tails: [3]G(route.Tail) = undefined,
    convs: [8]G(conv.Conv) = undefined,
    rewinds: [2]K(conv.Rewind) = undefined,
    top1s: [2]K(pick.Top1) = undefined,
    races: [2]K(pick.Race) = undefined,
    tobfs: [2]K(pick.Tobf) = undefined,
    words: [2]K(pick.Words) = undefined,

    /// Every registration, each control after the kernel it breaks, appended to `list`.
    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        set.k = try ref.Kernels.load(gpu.d);
        set.loaded = true;
        const k = &set.k;
        const rig: @import("contract_rig.zig").Rig = .{ .gpu = gpu, .s = s };
        const forms = [_]mix.Form{ .attn, .mlp, .first, .attn, .mlp, .first, .mlp };
        for (&set.mixes, forms, 0..) |*x, f, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .form = f, .narrow = i >= 3 and i < 6, .fault = if (i == 6) .short else .none } };
        for (&set.embeds, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .shifted = i == 1 } };
        const dots = [_]rows.Dot{ .router, .sink, .taps, .router, .taps, .router };
        for (&set.dots, dots, 0..) |*x, d, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .dot = d, .tiles = i == 3 or i == 4, .short = i == 5 } };
        for (&set.caps, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .wrong = i == 1 } };
        for (&set.topks, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .unbiased = i == 1 } };
        for (&set.gateups, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .swapped = i == 1 } };
        for (&set.tails, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .f32_out = i == 1, .shifted = i == 2 } };
        const cf = [_]conv.Form{ .kv_static, .kv_dynamic, .attn, .attn_boundary, .mlp, .kv_static, .attn, .kv_static };
        for (&set.convs, cf, 0..) |*x, f, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .form = f, .view = if (i == 5 or i == 6) .state else .out, .fault = if (i == 7) .one_slot else .none } };
        for (&set.rewinds, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .long = i == 1 } };
        for (&set.top1s, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .short = i == 1 } };
        for (&set.races, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .late = i == 1 } };
        for (&set.tobfs, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .unpadded = i == 1 } };
        for (&set.words, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .short = i == 1 } };
        inline for (groups) |f| for (&@field(set, f)) |*x| try list.append(a, x.kernel());
    }

    pub fn deinit(set: *Set) void {
        if (!set.loaded) return;
        inline for (groups) |f| for (&@field(set, f)) |*x| x.deinit();
        set.k.unload();
        set.loaded = false;
    }
};
