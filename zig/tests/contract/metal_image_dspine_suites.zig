//! The Moss image, Moss image decoder and DSpine suites, registered in that order behind one create.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const g = contract.gates;
const MossImage = @import("metal_moss_image.zig").Suite;
const MossDecoder = @import("metal_moss_image_decoder.zig").Suite;
const DSpine = @import("metal_qwen27_dspine.zig").Suite;

pub const Suites = struct {
    moss_image: *MossImage,
    moss_decoder: *MossDecoder,
    dspine: *DSpine,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !Suites {
        const moss_image = try MossImage.create(a, device, queue, kernels, only);
        errdefer moss_image.destroy();
        const moss_decoder = try MossDecoder.create(a, device, queue, kernels, only);
        errdefer moss_decoder.destroy();
        return .{ .moss_image = moss_image, .moss_decoder = moss_decoder, .dspine = try DSpine.create(a, device, queue, kernels, only) };
    }

    pub fn destroy(s: Suites) void {
        s.dspine.destroy();
        s.moss_decoder.destroy();
        s.moss_image.destroy();
    }
};
