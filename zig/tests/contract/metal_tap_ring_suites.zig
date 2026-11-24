//! The Qwen3.8 and GLM DFlash tap-ring suites, registered in that order behind one create.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const g = contract.gates;
const Q27 = @import("metal_q27_tap_ring.zig").Suite;
const Glm = @import("metal_glm_tap_ring.zig").Suite;

pub const Suites = struct {
    q27: *Q27,
    glm: *Glm,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !Suites {
        const q27 = try Q27.create(a, device, queue, kernels, only);
        errdefer q27.destroy();
        return .{ .q27 = q27, .glm = try Glm.create(a, device, queue, kernels, only) };
    }

    pub fn destroy(s: Suites) void {
        s.glm.destroy();
        s.q27.destroy();
    }
};
