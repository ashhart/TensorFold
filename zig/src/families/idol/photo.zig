//! A raw photo to IdolPipeline.preprocess's 1024 x 1024 picture on Metal: photo_host.zig's steps, BiRefNet on the GPU.
const std = @import("std");
const mtl = @import("metal");
const host = @import("photo_host.zig");
const bn = @import("birefnet.zig");

pub const Timing = host.Timing;
pub const isPhoto = host.isPhoto;
pub const png = host.png;

/// The picture [1024 * 1024 * 3] of the photo at `path`; BiRefNet is loaded from `weights` only if needed.
pub fn preprocess(a: std.mem.Allocator, io: std.Io, device: mtl.Device, weights: [:0]const u8, path: []const u8, crop_margin: f64, out: []u8, t: *Timing) !void {
    return host.preprocess(Gpu{ .device = device }, a, io, weights, path, crop_margin, out, t);
}

const Gpu = struct {
    device: mtl.Device,

    pub fn seconds(_: Gpu) f64 {
        return mtl.clock.seconds();
    }

    pub fn load(g: Gpu, weights: [:0]const u8) !*bn.Net {
        return bn.Net.load(std.heap.page_allocator, g.device, weights);
    }

    pub fn allocated(g: Gpu) u64 {
        return g.device.allocated();
    }
};

test {
    _ = host;
}
