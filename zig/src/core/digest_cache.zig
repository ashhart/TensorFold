//! Shared per-file SHA256 cache: bounded records, descriptor identity, locked atomic publication and integrity checks.
const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;
const descriptor = @import("digest_key.zig");
pub const Key = descriptor.Key;
pub const max_entries = 1024;
pub const record_size = 128;
pub const max_bytes = 12 + max_entries * record_size + 32;
const magic = "TFDG0001";
var mutex: if (descriptor.persistent) std.c.pthread_mutex_t else u8 = if (descriptor.persistent) .{} else 0;

pub const Value = struct { sha256: [32]u8, shared: ?[32]u8 = null };
pub const Record = struct { key: Key, value: Value };
pub const Result = struct { key: Key, value: Value, hit: bool };
pub const Cache = struct {
    root: ?[]const u8,
    limit: usize = max_entries,

    pub fn get(c: Cache, a: std.mem.Allocator, io: Io, path: []const u8, need_shared: bool, loader: anytype) !Result {
        if (c.limit == 0 or c.limit > max_entries) return error.BadDigestCacheLimit;
        const file = try Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        const before = try Key.of(file, io);
        if (comptime !descriptor.persistent) {
            const result = try compute(io, path, file, before, loader);
            if (need_shared and result.value.shared == null) return error.DigestProjectionMissing;
            return result;
        }
        if (c.root == null) {
            const result = try compute(io, path, file, before, loader);
            if (need_shared and result.value.shared == null) return error.DigestProjectionMissing;
            return result;
        }
        _ = std.c.pthread_mutex_lock(&mutex);
        defer _ = std.c.pthread_mutex_unlock(&mutex);
        try Io.Dir.cwd().createDirPath(io, c.root.?);
        var dir = try Io.Dir.cwd().openDir(io, c.root.?, .{});
        defer dir.close(io);
        const lock = dir.createFile(io, "index.lock", .{ .truncate = false }) catch |err| switch (err) {
            error.FileNotFound => try dir.openFile(io, "index.lock", .{ .mode = .read_write }),
            else => return err,
        };
        defer lock.close(io);
        try lock.lock(io, .exclusive);
        try validate(io, path, file, before);
        var records = try load(a, io, dir);
        defer records.deinit(a);
        for (records.items) |e| if (e.key.equal(before) and (!need_shared or e.value.shared != null)) {
            try validate(io, path, file, before);
            return .{ .key = before, .value = e.value, .hit = true };
        };
        const result = try compute(io, path, file, before, loader);
        if (need_shared and result.value.shared == null) return error.DigestProjectionMissing;
        var i: usize = 0;
        while (i < records.items.len) {
            const k = records.items[i].key;
            if (k.device == before.device and k.inode == before.inode) {
                _ = records.orderedRemove(i);
            } else i += 1;
        }
        while (records.items.len >= c.limit) _ = records.orderedRemove(0);
        try records.append(a, .{ .key = before, .value = result.value });
        try publish(a, io, dir, records.items);
        try validate(io, path, file, before);
        return result;
    }
};

fn compute(io: Io, path: []const u8, file: Io.File, before: Key, loader: anytype) !Result {
    const value = try loader.read(file, before.size);
    try validate(io, path, file, before);
    return .{ .key = before, .value = value, .hit = false };
}

fn validate(io: Io, path: []const u8, file: Io.File, before: Key) !void {
    if (!before.equal(try Key.of(file, io))) return error.DigestFileChanged;
    const now = try Io.Dir.cwd().openFile(io, path, .{});
    defer now.close(io);
    if (!before.equal(try Key.of(now, io))) return error.DigestFileChanged;
}

pub fn defaultRoot(a: std.mem.Allocator, home: []const u8) ![]u8 {
    if (home.len == 0 or !std.fs.path.isAbsolute(home)) return error.CacheHomeMissing;
    return std.fs.path.join(a, &.{ home, ".cache", "tensorfold", "file-digests", "v1" });
}

pub fn environmentRoot(a: std.mem.Allocator) !?[]u8 {
    if (builtin.is_test) return null;
    const home = std.c.getenv("HOME") orelse return null;
    return defaultRoot(a, std.mem.span(home)) catch |err| switch (err) {
        error.CacheHomeMissing => null,
        else => return err,
    };
}

fn load(a: std.mem.Allocator, io: Io, dir: Io.Dir) !std.ArrayList(Record) {
    const bytes = dir.readFileAlloc(io, "index.bin", a, .limited(max_bytes)) catch |err| switch (err) {
        error.FileNotFound => return .empty,
        error.StreamTooLong => return error.CorruptDigestCache,
        else => return err,
    };
    defer a.free(bytes);
    return decode(a, bytes);
}

pub fn decode(a: std.mem.Allocator, bytes: []const u8) !std.ArrayList(Record) {
    if (bytes.len < 44 or !std.mem.eql(u8, bytes[0..8], magic)) return error.CorruptDigestCache;
    const count = std.mem.readInt(u32, bytes[8..12], .little);
    if (count > max_entries or bytes.len != 12 + @as(usize, count) * record_size + 32) return error.CorruptDigestCache;
    var checksum: [32]u8 = undefined;
    Sha256.hash(bytes[0 .. bytes.len - 32], &checksum, .{});
    if (!std.mem.eql(u8, &checksum, bytes[bytes.len - 32 ..])) return error.CorruptDigestCache;
    var records: std.ArrayList(Record) = .empty;
    errdefer records.deinit(a);
    for (0..count) |i| {
        const b = bytes[12 + i * record_size ..][0..record_size];
        if (b[120] > 1 or !std.mem.allEqual(u8, b[121..], 0) or
            (b[120] == 0 and !std.mem.allEqual(u8, b[88..120], 0))) return error.CorruptDigestCache;
        const k = Key{ .device = std.mem.readInt(u64, b[0..8], .little), .inode = std.mem.readInt(u64, b[8..16], .little), .size = std.mem.readInt(u64, b[16..24], .little), .mtime = std.mem.readInt(i128, b[24..40], .little), .ctime = std.mem.readInt(i128, b[40..56], .little) };
        for (records.items) |other| if (other.key.equal(k)) return error.CorruptDigestCache;
        try records.append(a, .{ .key = k, .value = .{ .sha256 = b[56..88].*, .shared = if (b[120] == 1) b[88..120].* else null } });
    }
    return records;
}

pub fn encode(a: std.mem.Allocator, records: []const Record) ![]u8 {
    if (records.len > max_entries) return error.BadDigestCacheLimit;
    const bytes = try a.alloc(u8, 12 + records.len * record_size + 32);
    errdefer a.free(bytes);
    @memset(bytes, 0);
    @memcpy(bytes[0..8], magic);
    std.mem.writeInt(u32, bytes[8..12], @intCast(records.len), .little);
    for (records, 0..) |e, i| {
        const b = bytes[12 + i * record_size ..][0..record_size];
        std.mem.writeInt(u64, b[0..8], e.key.device, .little);
        std.mem.writeInt(u64, b[8..16], std.math.cast(u64, e.key.inode) orelse return error.DigestCacheUnsupported, .little);
        std.mem.writeInt(u64, b[16..24], e.key.size, .little);
        std.mem.writeInt(i128, b[24..40], e.key.mtime, .little);
        std.mem.writeInt(i128, b[40..56], e.key.ctime, .little);
        @memcpy(b[56..88], &e.value.sha256);
        if (e.value.shared) |shared| {
            b[120] = 1;
            @memcpy(b[88..120], &shared);
        }
    }
    var checksum: [32]u8 = undefined;
    Sha256.hash(bytes[0 .. bytes.len - 32], &checksum, .{});
    @memcpy(bytes[bytes.len - 32 ..], &checksum);
    return bytes;
}

fn publish(a: std.mem.Allocator, io: Io, dir: Io.Dir, records: []const Record) !void {
    const bytes = try encode(a, records);
    defer a.free(bytes);
    defer dir.deleteFile(io, "index.pending") catch {};
    const temp = try dir.createFile(io, "index.pending", .{});
    defer temp.close(io);
    try temp.writePositionalAll(io, bytes, 0);
    try temp.sync(io);
    try dir.rename("index.pending", dir, "index.bin", io);
}
