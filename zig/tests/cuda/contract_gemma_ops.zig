//! Owned Gemma CUDA operator registrations beside the current shared contract runner.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const softcap = @import("contract_softcap.zig");
const norm = @import("contract_layer_norm.zig");

pub const Group = struct {
    softcaps: [softcap.entries.len]softcap.SoftCap = undefined,
    norms: [norm.entries.len]norm.LayerNorm = undefined,
    soft_made: usize = 0,
    norm_made: usize = 0,

    pub fn init(group: *Group, gpu: check.Gpu, s: cuda.Stream, act: *const cuda.act_rows.ActRows, vit: *const cuda.vit.Kernels, controls: cuda.Module) !void {
        errdefer group.deinit();
        for (softcap.entries, &group.softcaps) |entry, *item| {
            const ctl = if (entry.keep == null) null else try controls.function("tf_ctl_softcap");
            item.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = entry, .act = act, .ctl = ctl };
            group.soft_made += 1;
        }
        for (norm.entries, &group.norms) |entry, *item| {
            const ctl = if (entry.keep == null) null else try controls.function("tf_ctl_vit_norm");
            item.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = entry, .vit = vit, .ctl = ctl };
            group.norm_made += 1;
        }
    }

    pub fn append(group: *Group, a: std.mem.Allocator, kernels: *std.ArrayList(core.contract.gates.Kernel)) !void {
        for (group.softcaps[0..group.soft_made]) |*item| try kernels.append(a, item.kernel());
        for (group.norms[0..group.norm_made]) |*item| try kernels.append(a, item.kernel());
    }

    pub fn deinit(group: *Group) void {
        for (group.norms[0..group.norm_made]) |*item| item.deinit();
        group.norm_made = 0;
        for (group.softcaps[0..group.soft_made]) |*item| item.deinit();
        group.soft_made = 0;
    }
};
