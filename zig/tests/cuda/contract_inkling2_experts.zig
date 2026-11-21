//! Inkling 2's prompt-chunk experts on core's wide kernel and its vision tower (vision.cu), with controls.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");
const grouped = @import("contract_inkling2_experts_grouped.zig");
const vision = @import("contract_inkling2_experts_vision.zig");
const modelopt = @import("contract_inkling2_modelopt.zig");
const Attention = @import("contract_inkling2_experts_attn.zig").Attention;

/// Every registration, each control after the kernel it breaks; the gates point at these fields.
pub const Set = struct {
    live: bool = false,
    ek: ink.experts.Kernels = undefined,
    vk: ink.vision.Kernels = undefined,
    mk: ink.modelopt.Kernels = undefined,
    fa: cuda.vit.Attention = undefined,
    projections: [2]grouped.Grouped = undefined,
    moe: [13]modelopt.ModelOpt = undefined,
    embeds: [2]ref.Gated(vision.Embed) = undefined,
    ropes: [2]ref.Gated(vision.Rope) = undefined,
    adds: [2]ref.Gated(vision.ScaleAdd) = undefined,
    attn: Attention = undefined,

    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        set.ek = try ink.experts.Kernels.load(gpu.d);
        errdefer set.ek.unload();
        set.vk = try ink.vision.Kernels.load(gpu.d);
        errdefer set.vk.unload();
        const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
        set.mk = try ink.modelopt.Kernels.load(gpu.d, sms);
        errdefer set.mk.unload();
        set.fa = try cuda.vit.Attention.load(gpu.d);
        errdefer set.fa.unload();
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        for (&set.projections, 0..) |*x, i| x.* = .{ .rig = rig, .k = &set.ek, .fault = if (i == 1) .swapped else .none };
        const prompt = [13]bool{ false, true, false, true, false, true, false, false, false, true, false, false, false };
        for (&set.moe, 0..) |*x, i| x.* = .{
            .rig = rig,
            .k = &set.mk,
            .format = if (i < 4 or i == 11) .nvfp4 else .mxfp8,
            .phase = if (prompt[i]) .prompt else .decode,
            .fault = switch (i) {
                0, 1, 4, 5, 11, 12 => .none,
                2, 6 => .wrong_combine,
                3, 9 => .wrong_prompt_up,
                7 => .wrong_plan,
                8 => .wrong_up,
                10 => .wrong_act,
                else => unreachable,
            },
            .tail = i >= 11, // K = 32, the scales' last row ending its allocation
        };
        for (&set.embeds, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.vk, .late = i == 1 } };
        for (&set.ropes, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.vk, .swapped = i == 1 } };
        for (&set.adds, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.vk, .late = i == 1 } };
        set.attn = .{ .rig = rig, .att = &set.fa };
        set.live = true;
        for (&set.projections) |*x| try list.append(a, x.kernel());
        for (&set.moe) |*x| try list.append(a, x.kernel());
        for (&set.embeds) |*x| try list.append(a, x.kernel());
        for (&set.ropes) |*x| try list.append(a, x.kernel());
        for (&set.adds) |*x| try list.append(a, x.kernel());
        try list.append(a, set.attn.kernel());
    }

    pub fn deinit(set: *Set) void {
        if (!set.live) return;
        set.attn.deinit();
        for (&set.adds) |*x| x.deinit();
        for (&set.ropes) |*x| x.deinit();
        for (&set.embeds) |*x| x.deinit();
        for (&set.moe) |*x| x.deinit();
        for (&set.projections) |*x| x.deinit();
        set.fa.unload();
        set.mk.unload();
        set.vk.unload();
        set.ek.unload();
        set.live = false;
    }
};

test {
    _ = @import("contract_inkling2_experts_ref.zig");
}
