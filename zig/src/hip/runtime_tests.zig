//! Model-free real-GPU qualification; failures never fall back to a mock.
const std = @import("std");
const Runtime = @import("runtime.zig").Runtime;
const Context = @import("context.zig").Context;
const Stream = @import("stream.zig").Stream;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const HostBuffer = @import("memory.zig").HostBuffer;
const Module = @import("module.zig").Module;
const launch = @import("launch.zig");
const codeobject = @import("hip_probe");

test "HIP copies fills and mixed-width kernel arguments on real GPU" {
    var r = try Runtime.open();
    defer r.close();
    var ctx = try Context.init(&r, 0);
    defer ctx.deinit();
    try ctx.synchronize();
    var stream = try Stream.init(&r);
    defer stream.deinit();
    const n = 1025;
    var b = try DeviceBuffer.alloc(&r, n * @sizeOf(u32));
    defer b.free();
    var got: [n]u32 = undefined;
    const pattern: [n]u32 = @splat(0x12345678);
    try b.upload(0, std.mem.asBytes(&pattern));
    try b.download(0, std.mem.asBytes(&got));
    try std.testing.expectEqualSlices(u32, &pattern, &got);
    var pinned_src = try HostBuffer.alloc(&r, b.len);
    defer pinned_src.free();
    var pinned_dst = try HostBuffer.alloc(&r, b.len);
    defer pinned_dst.free();
    // Different from device contents: a no-op async upload must fail this check.
    @memset(pinned_src.bytes, 0x3c);
    @memset(pinned_dst.bytes, 0);
    try b.uploadAsync(0, pinned_src, stream);
    try b.downloadAsync(0, pinned_dst, stream);
    try stream.synchronize();
    try std.testing.expectEqualSlices(u8, pinned_src.bytes, pinned_dst.bytes);
    try b.fill8(0x5a);
    try b.download(0, std.mem.asBytes(&got));
    for (got) |v| try std.testing.expectEqual(@as(u32, 0x5a5a5a5a), v);
    try std.testing.expectError(error.Invalid, b.upload(b.len, &.{1}));
    var arch_buffer: [256]u8 = undefined;
    const arch = try @import("device_arch.zig").query(&r, 0, &arch_buffer);
    try std.testing.expect(@import("caps.zig").Caps.of(arch) != null);
    const images = [_]@import("code_object.zig").Image{.{ .arch = codeobject.arch, .bytes = &codeobject.bytes }};
    try std.testing.expectError(error.UnsupportedArchitecture, Module.loadForArchitecture(&r, &images, "gfx9999"));
    var m = try Module.loadForArchitecture(&r, &images, arch);
    defer m.unload();
    // No intervening host sync: the kernel must read the same-stream fill.
    const read_fill = try m.function("tf_hip_read_fill");
    var fill_args: launch.Args = .{};
    try fill_args.add(b.ptr);
    try fill_args.add(@as(u32, n));
    for ([_]u8{ 0xa5, 0x3c }) |value| {
        try b.fill8Async(value, stream);
        try launch.launch(read_fill, .{ .grid = .{ .x = 5 }, .block = .{ .x = 256 } }, stream, &fill_args);
        try stream.synchronize();
        try b.download(0, std.mem.asBytes(&got));
        const byte: u32 = value ^ 1;
        for (got) |v| try std.testing.expectEqual(byte * 0x01010101, v);
    }
    const f = try m.function("tf_hip_probe");
    var args: launch.Args = .{};
    try args.add(b.ptr);
    try args.add(@as(u32, n));
    try args.add(@as(u32, 17));
    try args.add(@as(u64, 0x0000000300000000));
    try launch.launch(f, .{ .grid = .{ .x = 5 }, .block = .{ .x = 256 } }, stream, &args);
    try stream.synchronize();
    try b.download(0, std.mem.asBytes(&got));
    for (got, 0..) |v, i| try std.testing.expectEqual(@as(u32, @intCast(i + 20)), v);
}
