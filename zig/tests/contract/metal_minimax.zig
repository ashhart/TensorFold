//! MiniMax-M3.1-Flash's Metal kernels under the contract: glue, routing, the tensor-unit kernels and the drafters.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");

const g = contract.gates;
const groups = .{ @import("metal_minimax_glue.zig"), @import("metal_minimax_route.zig"), @import("metal_minimax_nax.zig"), @import("metal_minimax_draft.zig") };

/// A launch's two texts: this GPU's and the pre-M5 one (equal unless the source includes nax.h).
pub const Texts = struct { here: []const u8, pre_m5: []const u8 };

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    glue: groups[0].Suite = undefined,
    route: groups[1].Suite = undefined,
    nax: groups[2].Suite = undefined,
    draft: groups[3].Suite = undefined,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        inline for (.{ "glue", "route", "nax", "draft" }) |f| {
            @field(s, f) = .{ .a = s.a, .device = s.device, .queue = s.queue, .kernels = s.kernels };
            try @field(s, f).collect(only);
        }
    }
    pub fn deinit(s: *Suite) void {
        inline for (.{ "glue", "route", "nax", "draft" }) |f| @field(s, f).deinit();
    }
};

/// Gate 5's texts for a `minimax_m3 ` registration: the group whose launches it names compiles it.
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !Texts {
    inline for (groups) |grp| if (try grp.limitsText(a, device, name)) |t| return t;
    return error.UnknownLaunch;
}
