//! GLM-5.3-Flash's served Metal launches under the contract: gates 1-4 against their predecessors, then gate 5.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const bits = @import("metal_glm_bits.zig");
const Launch = @import("metal_limits.zig").Launch;

const groups = .{ @import("metal_glm_select.zig"), @import("metal_glm_kda.zig"), @import("metal_glm_attn.zig"), @import("metal_glm_hc.zig"), @import("metal_glm_experts_pick.zig"), @import("metal_glm_experts.zig"), @import("metal_glm_route.zig"), @import("metal_glm_route_select.zig"), @import("metal_glm_mla_norms.zig"), @import("metal_glm_prompt_mm.zig"), @import("metal_glm_index.zig") };
/// Gate 5's text owners by registration prefix (a longer prefix before its own start): the groups and the softmax.
const launches = groups ++ .{@import("metal_glm_attn.zig").softmax};

pub const Suite = struct {
    inner: bits.Suite,
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(contract.gates.Kernel)) Suite {
        return .{ .inner = .{ .a = a, .device = device, .queue = queue, .kernels = kernels } };
    }
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        inline for (groups) |grp| try grp.collect(&s.inner, only);
    }
    pub fn deinit(s: *Suite) void {
        s.inner.deinit();
    }
};

/// Gate 5 for a `glm launch ` registration: the library the loader builds for it here, and its pre-M5 form.
pub fn launchOf(a: std.mem.Allocator, device: mtl.Device, d: contract.limits.MetalLaunch, fun: []const u8, comptime assemble: bool) !?Launch {
    if (!std.mem.startsWith(u8, d.name, "glm launch ")) return null;
    inline for (launches) |grp| if (std.mem.startsWith(u8, d.name, grp.prefix)) {
        if (!assemble) return @as(Launch, undefined);
        const texts = try grp.limitsText(a, device, d.name);
        return .{ .decl = d, .function = fun, .here = texts.here, .pre_m5 = texts.pre_m5 };
    };
    return error.UnknownLaunch;
}
