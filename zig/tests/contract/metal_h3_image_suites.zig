//! MiniMax-H3-Image's suites in the full Metal kernel-contract runner.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const g = contract.gates;
const encoder = @import("metal_h3_image_encoder.zig");
const prompt = @import("metal_h3_image_prompt.zig");
const decoder = @import("metal_h3_image_decoder.zig");

/// All three H3 Image suites, registered in the same order as the source-only runner.
pub const Suites = struct {
    encoder: *encoder.Suite,
    prompt: *prompt.Suite,
    decoder: *decoder.Suite,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !Suites {
        const e = try encoder.Suite.create(a, device, queue, kernels, only);
        errdefer e.destroy();
        const p = try prompt.Suite.create(a, device, queue, kernels, only);
        errdefer p.destroy();
        const d = try decoder.Suite.create(a, device, queue, kernels, only);
        return .{ .encoder = e, .prompt = p, .decoder = d };
    }

    pub fn destroy(s: *Suites) void {
        s.decoder.destroy();
        s.prompt.destroy();
        s.encoder.destroy();
    }
};
