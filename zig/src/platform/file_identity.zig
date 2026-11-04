//! Bounded fingerprints of serving code and external kernel files.
const std = @import("std");

pub fn hash(io: std.Io, path: []const u8, maximum: u64) ![32]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const before = try file.stat(io);
    if (before.size > maximum) return error.IdentityFileTooLarge;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [32768]u8 = undefined;
    var at: u64 = 0;
    while (at < before.size) {
        const n = try file.readPositionalAll(io, buf[0..@intCast(@min(buf.len, before.size - at))], at);
        if (n == 0) return error.IdentityFileChanged;
        h.update(buf[0..n]);
        at += n;
    }
    const after = try file.stat(io);
    if (before.size != after.size or before.inode != after.inode or
        !std.meta.eql(before.mtime, after.mtime) or !std.meta.eql(before.ctime, after.ctime))
        return error.IdentityFileChanged;
    return h.finalResult();
}

pub fn bind(h: *std.hash.Wyhash, digest: [32]u8) void {
    h.update("serving-binary-v1");
    h.update(&digest);
}

/// The running binary includes compiled host code, embedded GPU images, compiler options and snapshot layouts.
pub fn current(h: *std.hash.Wyhash, a: std.mem.Allocator, io: std.Io) !void {
    const path = try std.process.executablePathAlloc(io, a);
    defer a.free(path);
    bind(h, try hash(io, path, 1 << 31));
}
