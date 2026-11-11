//! Release payloads may not disclose the build root or absolute source/cache paths.
const std = @import("std");
const cubin = @import("contract").cubin;
const magic = "\x50\xed\x55\xba";

fn separator(c: u8) bool {
    return c == '/' or c == '\\';
}
fn boundary(c: u8) bool {
    return c <= ' ' or c == '"' or c == '\'' or c == '=' or c == '(' or c == '[';
}
fn equalPath(a: u8, b: u8) bool {
    return std.ascii.toLower(if (separator(a)) '/' else a) == std.ascii.toLower(b);
}
fn starts(path: []const u8, text: []const u8) bool {
    if (path.len < text.len) return false;
    for (path[0..text.len], text) |a, b| if (!equalPath(a, b)) return false;
    return true;
}
fn contains(path: []const u8, text: []const u8) bool {
    if (path.len < text.len) return false;
    for (0..path.len - text.len + 1) |i| if (starts(path[i..], text)) return true;
    return false;
}
fn absoluteAt(bytes: []const u8, i: usize) bool {
    if (i > 0 and !boundary(bytes[i - 1])) return false;
    if (separator(bytes[i])) return true;
    return i + 2 < bytes.len and std.ascii.isAlphabetic(bytes[i]) and bytes[i + 1] == ':' and separator(bytes[i + 2]);
}
// Isolated drive-like instruction bytes need two named components before they qualify as a text path.
fn driveTail(path: []const u8) bool {
    var named = false;
    var separated = false;
    for (path) |c| {
        if (c < ' ' or c == 127) break;
        if (separator(c)) {
            if (named) separated = true;
        } else if (c > ' ') {
            if (separated) return true;
            named = true;
        }
    }
    return false;
}
fn privatePath(path: []const u8) bool {
    if (path.len >= 3 and std.ascii.isAlphabetic(path[0]) and path[1] == ':' and separator(path[2]) and driveTail(path[3..]))
        return true;
    for ([_][]const u8{ "/users/", "/home/", "/root/", "/build/", "/workspace/", "/workspaces/", "/__w/", "/agent/_work/" }) |prefix| {
        if (starts(path, prefix) and path.len > prefix.len and !boundary(path[prefix.len])) return true;
    }
    for ([_][]const u8{ "/zig/src/", "/zig/build/", "/zig/kernels/", "/zig-cache/", "/.zig-cache/" }) |part| {
        if (contains(path, part)) return true;
    }
    for ([_][]const u8{ ".zig", ".cu", ".cuh", ".c", ".cpp", ".h", ".hpp", ".metal", ".ptx", ".cubin", ".fatbin", ".o", ".obj" }) |ext| {
        if (path.len < ext.len) continue;
        for (0..path.len - ext.len + 1) |i| {
            if (i == 0 or separator(path[i - 1]) or !starts(path[i..], ext)) continue;
            const end = i + ext.len;
            if (end == path.len or boundary(path[end]) or path[end] == ':') return true;
        }
    }
    return false;
}
pub fn validRoot(root: []const u8) bool {
    if (root.len <= 1) return false;
    const drive = root.len >= 3 and std.ascii.isAlphabetic(root[0]) and root[1] == ':' and separator(root[2]);
    const unc = std.mem.startsWith(u8, root, "\\\\");
    if (!std.fs.path.isAbsolutePosix(root) and !drive and !unc) return false;
    var parts = std.mem.tokenizeAny(u8, if (drive) root[3..] else root, "/\\");
    var names: usize = 0;
    while (parts.next()) |part| {
        if (!std.mem.eql(u8, part, ".") and !std.mem.eql(u8, part, "..")) names += 1;
    }
    return names >= @as(usize, if (unc) 2 else 1);
}
fn textLeak(bytes: []const u8, root: []const u8) bool {
    if (validRoot(root) and std.mem.indexOf(u8, bytes, root) != null) return true;
    for (0..bytes.len) |i| {
        if (!absoluteAt(bytes, i)) continue;
        const quoted = i == 0 or bytes[i - 1] == 0 or bytes[i - 1] == '"' or bytes[i - 1] == '\'';
        var end = i;
        while (end < bytes.len and end - i < 32768 and bytes[end] >= ' ' and bytes[end] != 127 and bytes[end] != '"' and bytes[end] != '\'') : (end += 1) {
            if (!quoted and bytes[end] == ' ') break;
        }
        const path = bytes[i..end];
        if (std.unicode.utf8ValidateSlice(path) and privatePath(path)) return true;
    }
    return false;
}
fn plainLeak(a: std.mem.Allocator, bytes: []const u8, root: []const u8) !bool {
    if (textLeak(bytes, root)) return true;
    if (validRoot(root)) {
        const wide = try std.unicode.utf8ToUtf16LeAlloc(a, root);
        defer a.free(wide);
        if (std.mem.indexOf(u8, bytes, std.mem.sliceAsBytes(wide)) != null) return true;
        for (wide) |*unit| unit.* = @byteSwap(unit.*);
        if (std.mem.indexOf(u8, bytes, std.mem.sliceAsBytes(wide)) != null) return true;
    }
    for ([_]usize{ 0, 1 }) |offset| {
        for ([_]bool{ false, true }) |big| {
            var text: std.ArrayList(u8) = .empty;
            defer text.deinit(a);
            var i = offset;
            while (i + 1 < bytes.len) : (i += 2) {
                const low = bytes[i + @intFromBool(big)];
                const high = bytes[i + @intFromBool(!big)];
                try text.append(a, if (high == 0) low else '?');
            }
            if (textLeak(text.items, root)) return true;
        }
    }
    return false;
}
const DecodeState = struct { bytes: usize = 0, frames: usize = 0 };
const max_decoded = 8 << 30;
const EncodedRange = struct { start: usize, end: usize };

fn addRange(a: std.mem.Allocator, ranges: *std.ArrayList(EncodedRange), range: EncodedRange) !void {
    for (ranges.items) |old| {
        if (old.start == range.start and old.end == range.end) return;
        if (old.start < range.end and range.start < old.end) return error.OverlappingEncodedRanges;
    }
    try ranges.append(a, range);
}

fn encodedRanges(a: std.mem.Allocator, bytes: []const u8, base: usize, root: []const u8, state: *DecodeState, depth: usize, ranges: *std.ArrayList(EncodedRange)) !void {
    var cursor: usize = 0;
    while (std.mem.indexOfPos(u8, bytes, cursor, "\x28\xb5\x2f\xfd")) |start| {
        if (depth >= 4 or state.frames >= 100000) return error.ReleaseDecodeLimit;
        const decoded = (try cubin.zstdPrefix(a, bytes[start..], @min(1 << 28, max_decoded - state.bytes))) orelse return error.BadZstdFatbin;
        defer a.free(decoded.data);
        state.bytes += decoded.data.len;
        state.frames += 1;
        try checkInner(a, decoded.data, root, state, depth + 1);
        cursor = start + decoded.consumed;
        try addRange(a, ranges, .{ .start = base + start, .end = base + cursor });
    }
}

fn peDataFrames(a: std.mem.Allocator, bytes: []const u8, root: []const u8, state: *DecodeState, depth: usize, ranges: *std.ArrayList(EncodedRange)) !void {
    if (bytes.len < 64) return error.BadPe;
    const pe: usize = read(u32, bytes, 60);
    if (pe < 64 or pe > bytes.len or bytes.len - pe < 24 or !std.mem.eql(u8, bytes[pe..][0..4], "PE\x00\x00")) return error.BadPe;
    const count: usize = read(u16, bytes, pe + 6);
    const table = pe + 24 + read(u16, bytes, pe + 20);
    if (count == 0 or count > 96 or table > bytes.len or count * 40 > bytes.len - table) return error.BadPe;
    for (0..count) |index| {
        const section = table + index * 40;
        const size: usize = read(u32, bytes, section + 16);
        const offset: usize = read(u32, bytes, section + 20);
        if (offset > bytes.len or size > bytes.len - offset or (size != 0 and offset < table + count * 40)) return error.BadPe;
        for (0..index) |previous| {
            const old = table + previous * 40;
            const old_size: usize = read(u32, bytes, old + 16);
            const old_offset: usize = read(u32, bytes, old + 20);
            if (size != 0 and old_size != 0 and offset < old_offset + old_size and old_offset < offset + size) return error.BadPe;
        }
        if (std.mem.eql(u8, bytes[section..][0..8], ".data\x00\x00\x00"))
            try encodedRanges(a, bytes[offset..][0..size], offset, root, state, depth, ranges);
    }
}

fn rawLeak(a: std.mem.Allocator, bytes: []const u8, root: []const u8, ranges: []const EncodedRange) !bool {
    // Keep exact build-root checks over the full bytes, including encoded payloads.
    if (validRoot(root)) {
        if (std.mem.indexOf(u8, bytes, root) != null) return true;
        const wide = try std.unicode.utf8ToUtf16LeAlloc(a, root);
        defer a.free(wide);
        if (std.mem.indexOf(u8, bytes, std.mem.sliceAsBytes(wide)) != null) return true;
        for (wide) |*unit| unit.* = @byteSwap(unit.*);
        if (std.mem.indexOf(u8, bytes, std.mem.sliceAsBytes(wide)) != null) return true;
    }
    var cursor: usize = 0;
    for (ranges) |range| {
        if (try plainLeak(a, bytes[cursor..range.start], root)) return true;
        cursor = range.end;
    }
    return plainLeak(a, bytes[cursor..], root);
}
fn read(comptime T: type, bytes: []const u8, at: usize) T {
    return std.mem.readInt(T, bytes[at..][0..@sizeOf(T)], .little);
}
fn uncompressedElf(bytes: []const u8) !void {
    if (bytes.len < 4 or !std.mem.eql(u8, bytes[0..4], "\x7fELF")) return;
    if (bytes.len < 64 or bytes[4] != 2 or bytes[5] != 1) return error.UnsupportedElf;
    const at = std.math.cast(usize, read(u64, bytes, 40)) orelse return error.BadElf;
    const size: usize = read(u16, bytes, 58);
    const count: usize = read(u16, bytes, 60);
    if (count == 0) {
        if (at != 0) return error.UnsupportedExtendedElf;
        return;
    }
    const end = std.math.add(usize, at, count * size) catch return error.BadElf;
    if (size < 64 or end > bytes.len) return error.BadElf;
    const strings: usize = read(u16, bytes, 62);
    if (strings >= count) return error.BadElf;
    var names: []const u8 = &.{};
    if (strings != 0) {
        const off = std.math.cast(usize, read(u64, bytes, at + strings * size + 24)) orelse return error.BadElf;
        const len = std.math.cast(usize, read(u64, bytes, at + strings * size + 32)) orelse return error.BadElf;
        const tail = std.math.add(usize, off, len) catch return error.BadElf;
        if (tail > bytes.len) return error.BadElf;
        names = bytes[off..tail];
    }
    for (0..count) |i| {
        const h = at + i * size;
        if (read(u64, bytes, h + 8) & 0x800 != 0) return error.UnsupportedCompressedSection;
        if (names.len > 0) {
            const n: usize = read(u32, bytes, h);
            if (n >= names.len) return error.BadElf;
            if (std.mem.startsWith(u8, std.mem.sliceTo(names[n..], 0), ".zdebug_")) return error.UnsupportedCompressedSection;
        }
    }
}
fn inspectContainer(a: std.mem.Allocator, bytes: []const u8, base: usize, root: []const u8, state: *DecodeState, depth: usize, ranges: *std.ArrayList(EncodedRange)) !void {
    const head: usize = read(u16, bytes, 6);
    var at = head;
    while (at + 48 <= bytes.len) {
        const h: usize = read(u32, bytes, at + 4);
        const size = std.math.cast(usize, read(u64, bytes, at + 8)) orelse return error.BadFatbin;
        const start = std.math.add(usize, at, h) catch return error.BadFatbin;
        const end = std.math.add(usize, start, size) catch return error.BadFatbin;
        if (h < 48 or end > bytes.len) return error.BadFatbin;
        const payload = bytes[start..end];
        const kind = read(u16, bytes, at);
        if (kind != 1 and kind != 2) return error.UnsupportedImageEncoding;
        if (std.mem.startsWith(u8, payload, "\x28\xb5\x2f\xfd")) {
            if (depth >= 4 or state.frames >= 100000) return error.ReleaseDecodeLimit;
            const decoded = (try cubin.zstdPrefix(a, payload, @min(1 << 28, max_decoded - state.bytes))).?;
            defer a.free(decoded.data);
            if (payload.len - decoded.consumed > 7 or !std.mem.allEqual(u8, payload[decoded.consumed..], 0)) return error.BadZstdFatbin;
            if (kind == 2 and !std.mem.startsWith(u8, decoded.data, "\x7fELF")) return error.UnsupportedImageEncoding;
            state.bytes += decoded.data.len;
            state.frames += 1;
            try checkInner(a, decoded.data, root, state, depth + 1);
            try addRange(a, ranges, .{ .start = base + start, .end = base + start + decoded.consumed });
        } else {
            if (read(u64, bytes, at + 40) & 0x2000 != 0) return error.UnsupportedImageEncoding;
            if (kind == 2 and !std.mem.startsWith(u8, payload, "\x7fELF")) return error.UnsupportedImageEncoding;
            try checkInner(a, payload, root, state, depth + 1);
        }
        at = end;
    }
    if (bytes.len - at > 7 or !std.mem.allEqual(u8, bytes[at..], 0)) return error.BadFatbin;
}
/// A thin Mach-O's code signature (LC_CODE_SIGNATURE) holds the linker's hashes, which can spell a path by chance.
fn signature(bytes: []const u8) ?[2]usize {
    if (bytes.len < 32 or read(u32, bytes, 0) != 0xfeedfacf) return null;
    const count: usize = read(u32, bytes, 16);
    var at: usize = 32;
    for (0..count) |_| {
        if (at + 8 > bytes.len) return null;
        const size: usize = read(u32, bytes, at + 4);
        if (size < 8) return null;
        if (read(u32, bytes, at) == 0x1d) {
            if (at + 16 > bytes.len) return null;
            const off: usize = read(u32, bytes, at + 8);
            const len: usize = read(u32, bytes, at + 12);
            return if (off <= bytes.len and len <= bytes.len - off) .{ off, off + len } else null;
        }
        at += size;
    }
    return null;
}
pub fn check(a: std.mem.Allocator, bytes: []const u8, root: []const u8) !void {
    var state: DecodeState = .{};
    try checkInner(a, bytes, root, &state, 0);
}

fn checkInner(a: std.mem.Allocator, bytes: []const u8, root: []const u8, state: *DecodeState, depth: usize) anyerror!void {
    if (depth > 4) return error.ReleaseDecodeLimit;
    if (signature(bytes)) |s| {
        if (s[0] != s[1]) {
            if (validRoot(root) and std.mem.indexOf(u8, bytes[s[0]..s[1]], root) != null) return error.PrivateBuildPath;
            try checkInner(a, bytes[0..s[0]], root, state, depth + 1);
            try checkInner(a, bytes[s[1]..], root, state, depth + 1);
            return;
        }
    }
    var ranges: std.ArrayList(EncodedRange) = .empty;
    defer ranges.deinit(a);
    if (std.mem.startsWith(u8, bytes, "MZ")) {
        try peDataFrames(a, bytes, root, state, depth, &ranges);
    } else if (!std.mem.startsWith(u8, bytes, "\x7fELF") and !std.mem.startsWith(u8, bytes, magic)) {
        try encodedRanges(a, bytes, 0, root, state, depth, &ranges);
    }
    try uncompressedElf(bytes);
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, bytes, at, magic)) |start| {
        at = start + 4;
        var encoded = false;
        for (ranges.items) |range| {
            if (range.start <= start and start < range.end) {
                at = range.end;
                encoded = true;
                break;
            }
        }
        if (encoded) continue;
        if (bytes.len - start < 16) {
            if (start == 0) return error.BadFatbin;
            continue;
        }
        const h: usize = read(u16, bytes, start + 6);
        const size = std.math.cast(usize, read(u64, bytes, start + 8)) orelse return error.BadFatbin;
        const body = std.math.add(usize, start, h) catch return error.BadFatbin;
        const end = std.math.add(usize, body, size) catch return error.BadFatbin;
        if (h < 16 or end > bytes.len) {
            if (start == 0) return error.BadFatbin;
            continue;
        }
        try inspectContainer(a, bytes[start..end], start, root, state, depth, &ranges);
        at = end;
    }
    std.mem.sort(EncodedRange, ranges.items, {}, struct {
        fn less(_: void, left: EncodedRange, right: EncodedRange) bool {
            return left.start < right.start;
        }
    }.less);
    if (try rawLeak(a, bytes, root, ranges.items)) return error.PrivateBuildPath;
}
