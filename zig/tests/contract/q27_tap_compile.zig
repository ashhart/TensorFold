//! Compile all new Metal fixture hooks without linking a Metal framework or executing a GPU.
const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;
export fn q27TapContractCompile(a: *std.mem.Allocator, device: *mtl.Device, queue: *mtl.Queue) usize {
    var kernels: std.ArrayList(g.Kernel) = .empty;
    const suite = @import("metal_q27_tap_ring.zig").Suite.create(a.*, device.*, queue.*, &kernels, null) catch return 0;
    defer suite.destroy();
    return kernels.items.len;
}
