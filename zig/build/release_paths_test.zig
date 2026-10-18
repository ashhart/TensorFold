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

fn stringTableElf(text: []const u8) ![]u8 {
    const names = "\x00.shstrtab\x00.strtab\x00";
    const names_at = 64;
    const text_at = names_at + names.len;
    const table_at = std.mem.alignForward(usize, text_at + text.len + 2, 8);
    const bytes = try a.alloc(u8, table_at + 3 * 64);
    @memset(bytes, 0);
    @memcpy(bytes[0..4], "\x7fELF");
    bytes[4] = 2;
    bytes[5] = 1;
    bytes[6] = 1;
    std.mem.writeInt(u16, bytes[16..18], 1, .little);
    std.mem.writeInt(u16, bytes[18..20], 190, .little);
    std.mem.writeInt(u32, bytes[20..24], 1, .little);
    std.mem.writeInt(u16, bytes[52..54], 64, .little);
    std.mem.writeInt(u64, bytes[40..48], table_at, .little);
    std.mem.writeInt(u16, bytes[58..60], 64, .little);
    std.mem.writeInt(u16, bytes[60..62], 3, .little);
    std.mem.writeInt(u16, bytes[62..64], 1, .little);
    @memcpy(bytes[names_at..][0..names.len], names);
    @memcpy(bytes[text_at + 1 ..][0..text.len], text);
    for ([_]usize{ 1, 2 }, [_]usize{ 1, 11 }, [_]usize{ names_at, text_at }, [_]usize{ names.len, text.len + 2 }) |index, name, offset, size| {
        const h = table_at + index * 64;
        std.mem.writeInt(u32, bytes[h..][0..4], @intCast(name), .little);
        std.mem.writeInt(u32, bytes[h + 4 ..][0..4], 3, .little);
        std.mem.writeInt(u64, bytes[h + 24 ..][0..8], offset, .little);
        std.mem.writeInt(u64, bytes[h + 32 ..][0..8], size, .little);
    }
    return bytes;
}

test "release paths: short drive-like machine-code bytes are not a text path" {
    const noise = "\x01r:\\&\x02";
    try paths.check(a, noise, root);
    const container = try compressedFatbin(noise);
    defer a.free(container);
    if (@hasDecl(cubin, "freeImages")) try paths.check(a, container, root) else try std.testing.expectError(error.UnsupportedImageEncoding, paths.check(a, container, root));
}

test "release paths: Windows source paths in ELF string tables still refuse" {
    const clean = try stringTableElf("relative/source.cu");
    defer a.free(clean);
    try paths.check(a, clean, root);
    const object = try stringTableElf("C:\\Users\\x\\build\\y.cu");
    defer a.free(object);
    try std.testing.expectError(error.PrivateBuildPath, paths.check(a, object, root));
    for ([_][]const u8{ "C:\\Users\\x", "C:/ci/workspace/unit.cubin" }) |text|
        try std.testing.expectError(error.PrivateBuildPath, paths.check(a, text, root));
}

test "release paths: single-component drive source paths fall through in raw ELF and UTF16 text" {
    for ([_][]const u8{ "C:\\main.cu", "C:/main.cu", "D:\\private.h", "E:/source.zig" }) |text| {
        try std.testing.expectError(error.PrivateBuildPath, paths.check(a, text, root));
        const object = try stringTableElf(text);
        defer a.free(object);
        try std.testing.expectError(error.PrivateBuildPath, paths.check(a, object, root));
        const wide = try std.unicode.utf8ToUtf16LeAlloc(a, text);
        defer a.free(wide);
        for (0..2) |_| {
            for (0..2) |alignment| {
                var bytes: std.ArrayList(u8) = .empty;
                defer bytes.deinit(a);
                if (alignment == 1) try bytes.append(a, 0xff);
                try bytes.appendSlice(a, std.mem.sliceAsBytes(wide));
                try std.testing.expectError(error.PrivateBuildPath, paths.check(a, bytes.items, root));
            }
            for (wide) |*unit| unit.* = @byteSwap(unit.*);
        }
    }
}

test "release paths: relative root text cannot classify instruction bytes as a private path" {
    const noise = "\x01./.\x02";
    try paths.check(a, noise, "./.");
    const container = try compressedFatbin(noise);
    defer a.free(container);
    try paths.check(a, container, "./.");
    const wide = try std.unicode.utf8ToUtf16LeAlloc(a, "./.");
    defer a.free(wide);
    for (0..2) |_| {
        for (0..2) |alignment| {
            var bytes: std.ArrayList(u8) = .empty;
            defer bytes.deinit(a);
            if (alignment == 1) try bytes.append(a, 0xff);
            try bytes.appendSlice(a, std.mem.sliceAsBytes(wide));
            try paths.check(a, bytes.items, "./.");
        }
        for (wide) |*unit| unit.* = @byteSwap(unit.*);
    }
}

test "release paths: a named absolute build root still refuses in an ELF string table" {
    const object = try stringTableElf(root ++ "/assets/data.bin");
    defer a.free(object);
    try std.testing.expectError(error.PrivateBuildPath, paths.check(a, object, root));
}

test "release paths: a bare root has no named build directory text" {
    for ([_][]const u8{ "//", "C:\\", "/." }, [_][]const u8{ "\x01//\x02", "\x01C:\\&\x02", "\x01/.\x02" }) |base, noise|
        try paths.check(a, noise, base);
}
