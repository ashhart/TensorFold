//! Stream the executable once and retain its SHA256 state for byte-identical window-cost keys.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const buffer_bytes = 64 * 1024;
pub const max_executable_bytes = 1 << 30;

pub const Seed = struct {
    prefix: Sha256,
    executable_sha256: ?[32]u8,

    pub fn key(s: Seed, parts: []const []const u8) [32]u8 {
        var h = s.prefix;
        for (parts) |part| hashPart(&h, part);
        return h.finalResult();
    }
};

pub fn hashPart(h: *Sha256, part: []const u8) void {
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, part.len, .little);
    h.update(&len);
    h.update(part);
}

const FileReader = struct {
    file: std.Io.File,
    io: std.Io,

    pub fn read(r: @This(), bytes: []u8, offset: u64) !usize {
        return r.file.readPositionalAll(r.io, bytes, offset);
    }

    pub fn stat(r: @This()) !std.Io.File.Stat {
        return r.file.stat(r.io);
    }
};

pub fn fromFile(file: std.Io.File, io: std.Io, trace: bool) !Seed {
    return fromSource(FileReader{ .file = file, .io = io }, trace);
}

pub fn fromSource(source: anytype, trace: bool) !Seed {
    const before = try source.stat();
    if (before.kind != .file) return error.ExecutableNotRegular;
    const seed = try fromReader(source, before.size, trace);
    const after = try source.stat();
    if (!sameFile(before, after)) return error.ExecutableChanged;
    return seed;
}

fn sameFile(a: std.Io.File.Stat, b: std.Io.File.Stat) bool {
    return a.inode == b.inode and a.kind == b.kind and a.size == b.size and
        a.mtime.toNanoseconds() == b.mtime.toNanoseconds() and a.ctime.toNanoseconds() == b.ctime.toNanoseconds();
}

pub fn fromReader(reader: anytype, size: u64, trace: bool) !Seed {
    if (size >= max_executable_bytes) return error.StreamTooLong;
    var h = Sha256.init(.{});
    var raw = Sha256.init(.{});
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, size, .little);
    h.update(&len);
    var buffer: [buffer_bytes]u8 = undefined;
    var offset: u64 = 0;
    while (offset < size) {
        const wanted: usize = @intCast(@min(buffer.len, size - offset));
        const n = try reader.read(buffer[0..wanted], offset);
        if (n == 0) return error.ExecutableChanged;
        if (n > wanted) return error.InvalidReadLength;
        h.update(buffer[0..n]);
        if (trace) raw.update(buffer[0..n]);
        offset += n;
    }
    return .{ .prefix = h, .executable_sha256 = if (trace) raw.finalResult() else null };
}

test {
    _ = @import("cost_identity_test.zig");
}
