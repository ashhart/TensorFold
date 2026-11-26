//! Rank fingerprint hashes, kept separate from settings and agreement.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

/// sha256 of a file's bytes (read whole: config.json, the executable).
pub fn hashFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![32]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    return hashBytes(bytes);
}

pub fn hashBytes(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    Sha256.hash(bytes, &out, .{});
    return out;
}
