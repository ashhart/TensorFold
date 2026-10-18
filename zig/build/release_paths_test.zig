//! Planted source paths must fail even inside compressed CUDA payloads.
const std = @import("std");
const paths = @import("release_paths.zig");
const cubin = @import("contract").cubin;
const a = std.testing.allocator;
const root = "/opt/private-build-root";

test "release paths: build roots and absolute source paths refuse without rejecting runtime paths or URLs" {
    for ([_][]const u8{ "/opt/private-build-root/data.bin", "/home/sample/repo/zig/kernels/cuda/part", "/some/location/source.cu", "D:\\work\\source.cuh", "/ci/zig-cache/o/hash/object" }) |text|
        try std.testing.expectError(error.PrivateBuildPath, paths.check(a, text, root));
    for ([_][]const u8{ "/dev/urandom", "/proc/self/maps", "/v1/chat/completions", "https://example.org/repo/zig/src/main.zig", "zig/kernels/cuda/part.cu", "\x00T:\\B\xa9\xe0\x00" }) |text|
        try paths.check(a, text, root);
}

test "release paths: UTF16 roots and foreign home paths refuse at either byte alignment" {
    for ([_][]const u8{ "/opt/例/private/data.bin", "/home/sample/source.cu" }) |text| {
        const wide = try std.unicode.utf8ToUtf16LeAlloc(a, text);
        defer a.free(wide);
        for (0..2) |_| {
            var bytes: std.ArrayList(u8) = .empty;
            defer bytes.deinit(a);
            try bytes.append(a, 0xff);
            try bytes.appendSlice(a, std.mem.sliceAsBytes(wide));
            try std.testing.expectError(error.PrivateBuildPath, paths.check(a, bytes.items, "/opt/例/private"));
            for (wide) |*unit| unit.* = @byteSwap(unit.*);
        }
    }
}
fn elf(text: []const u8) ![]u8 {
    const bytes = try a.alloc(u8, 64 + text.len);
    @memset(bytes, 0);
    @memcpy(bytes[0..4], "\x7fELF");
    bytes[4] = 2;
    bytes[5] = 1;
    @memcpy(bytes[64..], text);
    return bytes;
}
fn compressedFatbin(text: []const u8) ![]u8 {
    const plain = try elf(text);
    defer a.free(plain);
    var frame: std.ArrayList(u8) = .empty;
    defer frame.deinit(a);
    try frame.appendSlice(a, "\x28\xb5\x2f\xfd\xa0");
    var length: [4]u8 = undefined;
    std.mem.writeInt(u32, &length, @intCast(plain.len), .little);
    try frame.appendSlice(a, &length);
    for (plain, 0..) |byte, i| {
        var head: [3]u8 = undefined;
        std.mem.writeInt(u24, &head, 8 | @as(u24, @intFromBool(i + 1 == plain.len)), .little);
        try frame.appendSlice(a, &head);
        try frame.append(a, byte);
    }
    const bytes = try a.alloc(u8, 80 + frame.items.len);
    @memset(bytes, 0);
    std.mem.writeInt(u32, bytes[0..4], 0xba55ed50, .little);
    std.mem.writeInt(u16, bytes[4..6], 1, .little);
    std.mem.writeInt(u16, bytes[6..8], 16, .little);
    std.mem.writeInt(u64, bytes[8..16], bytes.len - 16, .little);
    std.mem.writeInt(u16, bytes[16..18], 2, .little);
    std.mem.writeInt(u32, bytes[20..24], 64, .little);
    std.mem.writeInt(u64, bytes[24..32], frame.items.len, .little);
    std.mem.writeInt(u32, bytes[44..48], 121, .little);
    @memcpy(bytes[80..], frame.items);
    return bytes;
}

test "release paths: compressed CUDA images are inspected standalone and inside the packaged executable" {
    const bad = try compressedFatbin("/opt/private-build-root/zig/kernels/cuda/unit.cu");
    defer a.free(bad);
    try std.testing.expect(std.mem.indexOf(u8, bad, root) == null);
    const refusal = if (@hasDecl(cubin, "freeImages")) error.PrivateBuildPath else error.UnsupportedImageEncoding;
    try std.testing.expectError(refusal, paths.check(a, bad, root));
    var binary: std.ArrayList(u8) = .empty;
    defer binary.deinit(a);
    const head = try elf("");
    defer a.free(head);
    try binary.appendSlice(a, head);
    try binary.appendSlice(a, bad);
    try std.testing.expectError(refusal, paths.check(a, binary.items, root));
    const clean = try compressedFatbin("zig/kernels/cuda/unit.cu");
    defer a.free(clean);
    if (@hasDecl(cubin, "freeImages")) try paths.check(a, clean, root) else try std.testing.expectError(error.UnsupportedImageEncoding, paths.check(a, clean, root));
}

test "release paths: malformed containers and unauditable compressed ELF sections fail closed" {
    var bad: [16]u8 = @splat(0);
    @memcpy(bad[0..4], "\x50\xed\x55\xba");
    try std.testing.expectError(error.BadFatbin, paths.check(a, &bad, root));
    const object = try elf(&@as([64]u8, @splat(0)));
    defer a.free(object);
    std.mem.writeInt(u64, object[40..48], 64, .little);
    std.mem.writeInt(u16, object[58..60], 64, .little);
    std.mem.writeInt(u16, object[60..62], 1, .little);
    std.mem.writeInt(u64, object[72..80], 0x800, .little);
    try std.testing.expectError(error.UnsupportedCompressedSection, paths.check(a, object, root));
}
