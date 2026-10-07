//! Model-free real-GPU qualification; failures never fall back to a mock.
const std = @import("std");
const Runtime = @import("runtime.zig").Runtime;
const Context = @import("context.zig").Context;
const Stream = @import("stream.zig").Stream;
const Event = @import("stream.zig").Event;
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

test "an event orders a second stream after the first and times it; queries see unfinished work" {
    var r = try Runtime.open();
    defer r.close();
    var ctx = try Context.init(&r, 0);
    defer ctx.deinit();
    var arch_buffer: [256]u8 = undefined;
    const arch = try @import("device_arch.zig").query(&r, 0, &arch_buffer);
    const images = [_]@import("code_object.zig").Image{.{ .arch = codeobject.arch, .bytes = &codeobject.bytes }};
    var m = try Module.loadForArchitecture(&r, &images, arch);
    defer m.unload();
    const delay = try m.function("tf_hip_delay_write");
    var writer = try Stream.init(&r);
    defer writer.deinit();
    var reader = try Stream.init(&r);
    defer reader.deinit();
    var b = try DeviceBuffer.alloc(&r, @sizeOf(u32));
    defer b.free();
    try b.fill8(0);
    try ctx.synchronize();
    var host = try HostBuffer.alloc(&r, @sizeOf(u32));
    defer host.free();
    @memset(host.bytes, 0xff);
    var start = try Event.init(&r, true);
    defer start.deinit();
    var end = try Event.init(&r, true);
    defer end.deinit();
    try start.record(writer);
    var args: launch.Args = .{};
    try args.add(b.ptr);
    try args.add(@as(u32, 0xc0ffee));
    try args.add(@as(u32, 8000));
    try launch.launch(delay, .{ .grid = .{ .x = 1 }, .block = .{ .x = 64 } }, writer, &args);
    try end.record(writer);
    // Enough GPU work is queued that neither query may claim it finished.
    try std.testing.expect(!try end.done());
    try std.testing.expect(!try writer.done());
    try reader.wait(end);
    try b.downloadAsync(0, host, reader);
    try reader.synchronize();
    // Without the wait the reader's copy can run first and return the 0 fill.
    try std.testing.expectEqual(@as(u32, 0xc0ffee), std.mem.bytesToValue(u32, host.bytes[0..4]));
    try std.testing.expect(try end.done());
    try std.testing.expect(try writer.done());
    try std.testing.expect(try Event.elapsedMs(start, end) > 1.0);
}

test "memory info reports the device" {
    var r = try Runtime.open();
    defer r.close();
    var ctx = try Context.init(&r, 0);
    defer ctx.deinit();
    const first = try ctx.memInfo();
    try std.testing.expect(first.total >= 1 << 30);
    try std.testing.expectEqual(first.total, (try ctx.memInfo()).total);
}

test "memory info sees a new allocation" {
    var r = try Runtime.open();
    defer r.close();
    var ctx = try Context.init(&r, 0);
    defer ctx.deinit();
    const before = try ctx.memInfo();
    const len = 64 << 20;
    // A device another process has filled cannot show the drop; report a skip, never a pass.
    if (before.free < 2 * len) return error.SkipZigTest;
    var b = try DeviceBuffer.alloc(&r, len);
    defer b.free();
    try b.fill8(1);
    try ctx.synchronize();
    const after = try ctx.memInfo();
    try std.testing.expectEqual(before.total, after.total);
    try std.testing.expect(after.free + len <= before.free);
}
