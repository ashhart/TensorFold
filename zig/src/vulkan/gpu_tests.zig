//! Real Vulkan device checks, run only by `zig build vulkan-gpu-test -Dvulkan=true`: staged round trips and refusals.
const std = @import("std");
const vulkan = @import("vulkan");

const timeout_ns = 10_000_000_000;

test "staged upload and readback through a device-local buffer return the same bytes" {
    var rt = try vulkan.Runtime.open();
    defer rt.close();
    var dev = try vulkan.Device.open(&rt, .{});
    defer dev.close();
    const p = dev.properties;
    std.debug.print("vulkan device: {s} (api {d}.{d}), driver {s} {s}\n", .{
        std.mem.sliceTo(&p.deviceName, 0),       p.apiVersion >> 22 & 0x7f,               p.apiVersion >> 12 & 0x3ff,
        std.mem.sliceTo(&dev.v12.driverName, 0), std.mem.sliceTo(&dev.v12.driverInfo, 0),
    });
    var cmds = try vulkan.Commands.init(&dev);
    defer cmds.deinit();
    var gpu = try vulkan.Buffer.init(&dev, .device, 1 << 17);
    defer gpu.deinit();
    // smaller than the data, so the transfers take several chunks and end on a partial one
    var staging = try vulkan.Buffer.init(&dev, .staging, 4096 + 4);
    defer staging.deinit();
    var data: [65573]u8 = undefined;
    for (&data, 0..) |*b, i| b.* = @truncate(i * 31 ^ i >> 8);
    try vulkan.orExit(vulkan.upload(&cmds, gpu, 12, staging, &data, timeout_ns));
    var back: [data.len]u8 = @splat(0xa5);
    try vulkan.orExit(vulkan.download(&cmds, gpu, 12, staging, &back, timeout_ns));
    try std.testing.expectEqualSlices(u8, &data, &back);
    try std.testing.expectError(error.OutOfRange, vulkan.upload(&cmds, gpu, gpu.len - 3, staging, data[0..4], timeout_ns));
    try std.testing.expectError(error.Invalid, vulkan.upload(&cmds, staging, 0, staging, data[0..4], timeout_ns));
    try std.testing.expectError(error.Invalid, vulkan.download(&cmds, staging, 0, gpu, back[0..4], timeout_ns));
}

test "buffers and descriptors past the device's limits are refused" {
    var rt = try vulkan.Runtime.open();
    defer rt.close();
    var dev = try vulkan.Device.open(&rt, .{});
    defer dev.close();
    try std.testing.expectError(error.OutOfRange, vulkan.Buffer.init(&dev, .device, 0));
    if (dev.v13.maxBufferSize < std.math.maxInt(u64))
        try std.testing.expectError(error.TooLarge, vulkan.Buffer.init(&dev, .device, dev.v13.maxBufferSize + 1));
    var gpu = try vulkan.Buffer.init(&dev, .device, 1 << 16);
    defer gpu.deinit();
    const limits = dev.properties.limits;
    const info = try gpu.descriptor(limits.minStorageBufferOffsetAlignment, 256);
    try std.testing.expectEqual(limits.minStorageBufferOffsetAlignment, info.offset);
    try std.testing.expectError(error.OutOfRange, gpu.descriptor(0, (1 << 16) + 4));
    try std.testing.expectError(error.NotMapped, gpu.bytes());
    if (limits.minStorageBufferOffsetAlignment > 1)
        try std.testing.expectError(error.Misaligned, gpu.descriptor(limits.minStorageBufferOffsetAlignment - 1, 4));
}
