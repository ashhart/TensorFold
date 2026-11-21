//! Inkling 2's grouped experts and vision tower (inkling2/experts.cu, vision.cu) under the contract, with controls.
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
    groups: [2]grouped.Group = undefined,
    projections: [4]grouped.Grouped = undefined,
    moe: [11]modelopt.ModelOpt = undefined,
    embeds: [2]ref.Gated(vision.Embed) = undefined,
    ropes: [2]ref.Gated(vision.Rope) = undefined,
    adds: [2]ref.Gated(vision.ScaleAdd) = undefined,
    attns: [2]Attention = undefined,

    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        set.ek = try ink.experts.Kernels.load(gpu.d);
        errdefer set.ek.unload();
        set.vk = try ink.vision.Kernels.load(gpu.d);
        errdefer set.vk.unload();
        const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
        set.mk = try ink.modelopt.Kernels.load(gpu.d, sms);
        errdefer set.mk.unload();
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        for (&set.groups, 0..) |*x, i| x.* = .{ .rig = rig, .k = &set.ek, .wrong_run = i == 1 };
        const forms = [4]@import("contract_inkling2_experts_ref.zig").Form{ .rows, .rows, .wide, .wide };
        for (&set.projections, forms, 0..) |*x, f, i| x.* = .{ .rig = rig, .k = &set.ek, .form = f, .fault = if (i == 1) .short else if (i == 3) .swapped else .none };
        const prompt = [11]bool{ false, true, false, true, false, true, false, false, false, true, false };
        for (&set.moe, 0..) |*x, i| x.* = .{
            .rig = rig,
            .k = &set.mk,
            .format = if (i < 4) .nvfp4 else .mxfp8,
            .phase = if (prompt[i]) .prompt else .decode,
            .fault = switch (i) {
                0, 1, 4, 5 => .none,
                2, 6 => .wrong_combine,
                3, 9 => .wrong_prompt_up,
                7 => .wrong_plan,
                8 => .wrong_up,
                10 => .wrong_act,
                else => unreachable,
            },
        };
        for (&set.embeds, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.vk, .late = i == 1 } };
        for (&set.ropes, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.vk, .swapped = i == 1 } };
        for (&set.adds, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = &set.vk, .late = i == 1 } };
        for (&set.attns, 0..) |*x, i| x.* = .{ .rig = rig, .k = &set.vk, .wide = i == 1 };
        set.live = true;
        for (&set.groups) |*x| try list.append(a, x.kernel());
        for (&set.projections) |*x| try list.append(a, x.kernel());
        for (&set.moe) |*x| try list.append(a, x.kernel());
        for (&set.embeds) |*x| try list.append(a, x.kernel());
        for (&set.ropes) |*x| try list.append(a, x.kernel());
        for (&set.adds) |*x| try list.append(a, x.kernel());
        for (&set.attns) |*x| try list.append(a, x.kernel());
    }

    pub fn deinit(set: *Set) void {
        if (!set.live) return;
        for (&set.attns) |*x| x.deinit();
        for (&set.adds) |*x| x.deinit();
        for (&set.ropes) |*x| x.deinit();
        for (&set.embeds) |*x| x.deinit();
        for (&set.moe) |*x| x.deinit();
        for (&set.projections) |*x| x.deinit();
        for (&set.groups) |*x| x.deinit();
        set.mk.unload();
        set.vk.unload();
        set.ek.unload();
        set.live = false;
    }
};

test {
    _ = @import("contract_inkling2_experts_ref.zig");
}
