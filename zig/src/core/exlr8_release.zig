//! An EXLR8 release folder's published hashes: hashes/X.sha256 holds "<sha256> <size>" for experts/X and dense/X.

const std = @import("std");
const Io = std.Io;

/// Where the release publishes the hash of a file given relative to the root (experts/... or dense/...).
pub fn hashPath(buf: []u8, rel: []const u8) ![]const u8 {
    const base = std.fs.path.basename(rel);
    if (std.mem.startsWith(u8, rel, "experts/")) return std.fmt.bufPrint(buf, "hashes/{s}.sha256", .{base});
    if (std.mem.startsWith(u8, rel, "dense/")) return std.fmt.bufPrint(buf, "hashes/dense/{s}.sha256", .{base});
    return error.NotAnExlr8File;
}

pub const Published = struct { sha256: [32]u8, size: u64 };

pub fn parsePublished(text: []const u8) !Published {
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n");
    const hex = it.next() orelse return error.BadExlr8Hash;
    const size = it.next() orelse return error.BadExlr8Hash;
    var p: Published = .{ .sha256 = undefined, .size = std.fmt.parseInt(u64, size, 10) catch return error.BadExlr8Hash };
    if (hex.len != 64) return error.BadExlr8Hash;
    _ = std.fmt.hexToBytes(&p.sha256, hex) catch return error.BadExlr8Hash;
    return p;
}

/// Hashes the whole file and compares it with its published sha256 and size: true when both match.
pub fn checkFile(gpa: std.mem.Allocator, io: Io, root: []const u8, rel: []const u8) !bool {
    var buf: [512]u8 = undefined;
    const hp = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ root, try hashPath(&buf, rel) });
    defer gpa.free(hp);
    const text = try Io.Dir.cwd().readFileAlloc(io, hp, gpa, .limited(4096));
    defer gpa.free(text);
    const want = try parsePublished(text);
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ root, rel });
    defer gpa.free(path);
    var file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const len = try file.length(io);
    if (len != want.size) return false;
    if (len == 0) return std.mem.eql(u8, &want.sha256, &emptySha());
    var map = try Io.File.MemoryMap.create(io, file, .{ .len = @intCast(len), .protection = .{ .read = true, .write = false }, .populate = false });
    defer map.destroy(io);
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(map.memory, &d, .{});
    return std.mem.eql(u8, &want.sha256, &d);
}

fn emptySha() [32]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("", &d, .{});
    return d;
}

test "published hash lines and their paths parse; anything else is refused" {
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings("hashes/exlr8-layer-040.k3.safetensors.sha256", try hashPath(&buf, "experts/exlr8-layer-040.k3.safetensors"));
    try std.testing.expectEqualStrings("hashes/dense/exlr8-vocab.safetensors.sha256", try hashPath(&buf, "dense/exlr8-vocab.safetensors"));
    try std.testing.expectError(error.NotAnExlr8File, hashPath(&buf, "manifests/exlr8-manifest-home.json"));
    const p = try parsePublished("4326bf6b023d5ae27e981ff0e62f7f3f7d3aa6bbc1ad54bac6754f838d118a1f 13303812\n");
    try std.testing.expectEqual(@as(u64, 13303812), p.size);
    try std.testing.expectEqual(@as(u8, 0x43), p.sha256[0]);
    try std.testing.expectError(error.BadExlr8Hash, parsePublished("4326bf 13303812"));
    try std.testing.expectError(error.BadExlr8Hash, parsePublished(""));
}
