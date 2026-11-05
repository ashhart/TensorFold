//! The resident K3 adapter's GPU resources, released after its lane host and graph stop.
const mtl = @import("metal");
const family = @import("kimi_k3");
pub const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    library: mtl.Library,
    k: family.kernels.Kernels,

    pub fn init(dsa: bool) !Gpu {
        const device = try mtl.Device.init();
        errdefer device.deinit();
        const queue = try device.queue();
        errdefer queue.deinit();
        const library = try mtl.Library.fromBytes(device, @import("kimi_metallib").bytes);
        errdefer library.deinit();
        return .{ .device = device, .queue = queue, .library = library, .k = try family.kernels.Kernels.init(device, library, dsa) };
    }
    pub fn deinit(g: *Gpu) void {
        g.k.deinit();
        g.library.deinit();
        g.queue.deinit();
        g.device.deinit();
    }
};
