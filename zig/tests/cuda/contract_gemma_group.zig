//! Owned registrations for the Gemma 4 tower, appended after their final stack address is stable.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const tower = @import("contract_vit_gemma.zig");

pub const Group = struct {
    entries: [tower.entries.len]tower.Glue = undefined,
    made: usize = 0,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, vit: *const cuda.vit.Kernels, controls: cuda.Module) !Group {
        var group: Group = .{};
        errdefer group.deinit();
        for (tower.entries, &group.entries) |entry, *item| {
            const ctl = if (entry.keep == null) null else try controls.function(switch (entry.kind) {
                .qkv => "tf_ctl_vit_g4_qkv",
                .gelu => "tf_ctl_vit_g4_gelu_mul",
                .pool => "tf_ctl_vit_g4_pool",
            });
            item.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = entry, .vit = vit, .ctl = ctl };
            group.made += 1;
        }
        return group;
    }

    pub fn append(group: *Group, a: std.mem.Allocator, kernels: *std.ArrayList(core.contract.gates.Kernel)) !void {
        for (group.entries[0..group.made]) |*item| try kernels.append(a, item.kernel());
    }

    pub fn deinit(group: *Group) void {
        for (group.entries[0..group.made]) |*item| item.deinit();
        group.made = 0;
    }
};
