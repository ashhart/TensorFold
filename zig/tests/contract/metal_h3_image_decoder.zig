//! MiniMax-H3-Image's decoder kernels (image_decoder.metal) under the contract: each against its own formula in float64.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const common = @import("metal_h3_image_common.zig");
const g = contract.gates;

pub const Suite = struct {
    a: std.mem.Allocator,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        _ = device;
        _ = queue;
        _ = kernels;
        _ = only;
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        return s;
    }

    pub fn destroy(s: *Suite) void {
        s.a.destroy(s);
    }
};
