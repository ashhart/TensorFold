//! The K3 test programs' GPU: device, queue, our kimi metallib and its pipelines, residency for address-read buffers.
const std = @import("std");
const mtl = @import("metal");
const k3 = @import("kimi_k3");
const metallib = @import("kimi_metallib");

pub const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,
    k: k3.kernels.Kernels,

    pub fn init() !Gpu {
        const device = try mtl.Device.init();
        const lib = try mtl.Library.fromBytes(device, metallib.bytes);
        return .{ .device = device, .queue = try device.queue(), .lib = lib, .k = try k3.kernels.Kernels.init(device, lib) };
    }

    pub fn deinit(g: *Gpu) void {
        g.k.deinit();
        g.lib.deinit();
        g.queue.deinit();
        g.device.deinit();
    }
};

/// Every buffer a test reads through GPU addresses (expert tables, state arenas) resident for the queue.
pub fn resident(g: *const Gpu, sets: []const []const mtl.Buffer) !mtl.ResidencySet {
    var n: usize = 0;
    for (sets) |s| n += s.len;
    const set = try g.device.residencySet(n);
    for (sets) |s| for (s) |b| set.add(b);
    set.commit();
    g.queue.addResidencySet(set);
    return set;
}
