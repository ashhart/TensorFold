const std = @import("std");
const png = @import("core").picture.png;
const Allocator = std.mem.Allocator;

fn chunk(a: Allocator, out: *std.ArrayList(u8), kind: *const [4]u8, body: []const u8) !void {
    var len: [4]u8 = undefined;
    std.mem.writeInt(u32, &len, @intCast(body.len), .big);
    try out.appendSlice(a, &len);
    try out.appendSlice(a, kind);
    try out.appendSlice(a, body);
    var crc = std.hash.Crc32.init();
    crc.update(kind);
    crc.update(body);
    var sum: [4]u8 = undefined;
    std.mem.writeInt(u32, &sum, crc.final(), .big);
    try out.appendSlice(a, &sum);
}

fn leadingTextPng(a: Allocator, width: u32) ![]u8 {
    var compressed: std.Io.Writer.Allocating = .init(a);
    defer compressed.deinit();
    try compressed.ensureUnusedCapacity(16);
    var window: [std.compress.flate.max_window_len]u8 = undefined;
    var z = try std.compress.flate.Compress.init(&compressed.writer, &window, .zlib, .default);
    try z.writer.splatByteAll('x', 1 << 20);
    try z.finish();

    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, png.signature);
    var head: [13]u8 = undefined;
    std.mem.writeInt(u32, head[0..4], width, .big);
    std.mem.writeInt(u32, head[4..8], 1, .big);
    head[8..13].* = .{ 8, 2, 0, 0, 0 };
    try chunk(a, &out, "IHDR", &head);
    const body = try std.mem.concat(a, u8, &.{ "k\x00\x00", compressed.written() });
    defer a.free(body);
    try chunk(a, &out, "zTXt", body);
    try chunk(a, &out, "IDAT", "");
    return out.toOwnedSlice(a);
}
test "HTTP limits: PNG refuses an oversized IHDR before inflating a leading text chunk" {
    const oversized = try leadingTextPng(std.testing.allocator, 8193);
    defer std.testing.allocator.free(oversized);
    var storage: [1]u8 = .{0};
    var bounded: std.heap.FixedBufferAllocator = .init(&storage);
    try std.testing.expectError(error.Oversized, png.decode(bounded.allocator(), oversized));

    // The same bounded allocator at a valid width proves the fixture otherwise reaches inflation.
    const valid = try leadingTextPng(std.testing.allocator, 8192);
    defer std.testing.allocator.free(valid);
    var control_storage: [1]u8 = .{0};
    var control: std.heap.FixedBufferAllocator = .init(&control_storage);
    try std.testing.expectError(error.OutOfMemory, png.decode(control.allocator(), valid));
}
