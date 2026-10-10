//! The runtime pieces an engine uses beyond the model-free tests: device count, free memory, addresses, raw copies.
const std = @import("std");
const Runtime = @import("runtime.zig").Runtime;
const Context = @import("context.zig").Context;
const Stream = @import("stream.zig").Stream;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const HostBuffer = @import("memory.zig").HostBuffer;
const raw = @import("raw.zig");
const counting = @import("launches/util.zig").counting;

test "device count and free memory, buffer addresses, pinned views and raw-stream copies and fills" {
    var r = try Runtime.open();
    defer r.close();
    try std.testing.expect(try r.deviceCount() >= 1);
    var ctx = try Context.init(&r, 0);
    defer ctx.deinit();
    const mem = try ctx.memInfo();
    try std.testing.expect(mem.total > 0 and mem.free <= mem.total);
    var stream = try Stream.init(&r);
    defer stream.deinit();
    const n = 4096;
    var a = try DeviceBuffer.alloc(&r, n);
    defer a.free();
    var b = try DeviceBuffer.alloc(&r, n);
    defer b.free();
    try std.testing.expectEqual(@intFromPtr(a.ptr), a.base());
    var host = try HostBuffer.alloc(&r, 2 * n);
    defer host.free();
    for (host.slice(u32), 0..) |*w, i| w.* = @intCast(i);
    try std.testing.expectEqual(@as(usize, 2 * n / 4), host.slice(u32).len);
    // a view of the second half goes up through the #463 API, comes back through a raw-stream download
    try a.uploadAsync(0, host.view(n, 2 * n), stream);
    try raw.copy(b, 16, a.base() + 32, n - 32, stream.handle);
    try raw.fill8(a, 0xab, stream.handle);
    try raw.download(b, 0, host.bytes[0..n], stream.handle);
    try stream.synchronize();
    const words = host.slice(u32);
    for (words[4 .. n / 4 - 4], n / 4 + 8..) |w, want| try std.testing.expectEqual(@as(u32, @intCast(want)), w);
    var back: [n]u8 = undefined;
    try a.download(0, &back);
    for (back) |x| try std.testing.expectEqual(@as(u8, 0xab), x);
    try raw.fill32(a, 0x01020304, stream.handle);
    try raw.upload(b, 0, host.bytes[n .. n + 8], stream.handle);
    // the counting stream queues nothing
    try raw.fill8(a, 0, counting);
    try raw.copy(a, 0, b.base(), n, counting);
    try stream.synchronize();
    try a.download(0, &back);
    for (std.mem.bytesAsSlice(u32, &back)) |w| try std.testing.expectEqual(@as(u32, 0x01020304), w);
    try std.testing.expectError(error.Invalid, raw.copy(a, n - 4, b.base(), 8, stream.handle));
    try std.testing.expectError(error.Invalid, raw.upload(a, n, host.bytes[0..1], stream.handle));
}
