//! Timed costs are measured once per engine build, chip and model shape, then read back by later loads.
const std = @import("std");
const Allocator = std.mem.Allocator;

const magic = "TFWC2\n";

/// What the costs depend on: the running executable's bytes (the layout that wrote them) and the caller's parts.
pub fn key(a: Allocator, io: std.Io, parts: []const []const u8) ![32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    const exe = try std.process.executablePathAlloc(io, a);
    defer a.free(exe);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, exe, a, .limited(1 << 30));
    defer a.free(bytes);
    hashPart(&h, bytes);
    for (parts) |p| hashPart(&h, p);
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

fn hashPart(h: *std.crypto.hash.sha2.Sha256, part: []const u8) void {
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, part.len, .little);
    h.update(&len);
    h.update(part);
}

fn path(a: Allocator, k: [32]u8, suffix: []const u8) !?[]u8 {
    const home = std.c.getenv("HOME") orelse return null;
    return try std.fmt.allocPrint(a, "{s}/.cache/tensorfold/window-costs/{s}{s}", .{ std.mem.span(home), std.fmt.bytesToHex(k, .lower), suffix });
}

/// The value kept for `k`, or null when none is kept or the file is not one of ours.
pub fn load(comptime T: type, a: Allocator, io: std.Io, k: [32]u8) ?T {
    const p = (path(a, k, ".bin") catch return null) orelse return null;
    defer a.free(p);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, p, a, .limited(1 << 20)) catch return null;
    defer a.free(bytes);
    return parse(T, bytes);
}

fn parse(comptime T: type, bytes: []const u8) ?T {
    if (bytes.len != magic.len + @sizeOf(T) or !std.mem.eql(u8, bytes[0..magic.len], magic)) return null;
    var value: T = undefined;
    @memcpy(std.mem.asBytes(&value), bytes[magic.len..]);
    return value;
}

/// Keep `value` for `k`, written whole and renamed into place; a failure only costs the next load a measurement.
pub fn save(comptime T: type, a: Allocator, io: std.Io, k: [32]u8, value: T) void {
    const p = (path(a, k, ".bin") catch return) orelse return;
    defer a.free(p);
    std.Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(p) orelse return) catch return;
    const temp = std.fmt.allocPrint(a, "{s}.{d}.tmp", .{ p, std.c.getpid() }) catch return;
    defer a.free(temp);
    const bytes = std.mem.concat(a, u8, &.{ magic, std.mem.asBytes(&value) }) catch return;
    defer a.free(bytes);
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = temp, .data = bytes }) catch return;
    std.Io.Dir.renameAbsolute(temp, p, io) catch std.Io.Dir.cwd().deleteFile(io, temp) catch {};
}

test "a kept value reads back whole and any other length or tag is refused" {
    const V = struct { n: u32, ms: [3]f64 };
    const v: V = .{ .n = 3, .ms = .{ 33.56, 34.1, 37.3 } };
    const bytes = try std.mem.concat(std.testing.allocator, u8, &.{ magic, std.mem.asBytes(&v) });
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqualDeep(v, parse(V, bytes).?);
    try std.testing.expectEqual(@as(?V, null), parse(V, bytes[0 .. bytes.len - 1]));
    bytes[0] = 'X';
    try std.testing.expectEqual(@as(?V, null), parse(V, bytes));
}
