//! Resolved runtime facts and opened weight files extend a probe key without changing a state's image.
const std = @import("std");
const digest_cache = @import("tensorfold").digest_cache;

pub fn extend(legacy: u64, domain: []const u8, words: []const u64) u64 {
    var h = std.hash.Wyhash.init(legacy);
    h.update(domain);
    h.update(std.mem.sliceAsBytes(words));
    return h.final();
}

/// The descriptors the loader indexed, including a pack outside the checkpoint folder; order is immaterial.
pub fn files(legacy: u64, a: std.mem.Allocator, io: std.Io, descriptors: []const c_int) !u64 {
    const root = try digest_cache.environmentRoot(a);
    defer if (root) |r| a.free(r);
    return filesWithCache(legacy, a, io, descriptors, root);
}

pub fn filesWithCache(legacy: u64, a: std.mem.Allocator, io: std.Io, descriptors: []const c_int, root: ?[]const u8) !u64 {
    const generated = if (root) |r| try digest_cache.generatedRoot(a, r) else null;
    defer if (generated) |r| a.free(r);
    var known: std.ArrayList([32]u8) = .empty;
    defer known.deinit(a);
    const stamps = try a.alloc([4]i128, descriptors.len);
    defer a.free(stamps);
    for (descriptors, stamps) |fd, *stamp| {
        if (fd < 0) return error.IdentityFileUnavailable;
        const file: std.Io.File = .{ .handle = fd, .flags = .{ .nonblocking = false } };
        const stat = try file.stat(io);
        stamp.* = .{ stat.size, stat.inode, stat.mtime.nanoseconds, stat.ctime.nanoseconds };
        var cached_value = try (digest_cache.Cache{ .root = generated }).lookupFile(a, io, file);
        if (cached_value == null) cached_value = try (digest_cache.Cache{ .root = root }).lookupFile(a, io, file);
        if (cached_value) |cached| {
            if (cached.key.size != stat.size or cached.key.inode != stat.inode or
                cached.key.mtime != stat.mtime.nanoseconds or cached.key.ctime != stat.ctime.nanoseconds)
                return error.DigestFileChanged;
            var bound = std.crypto.hash.sha2.Sha256.init(.{});
            bound.update(std.mem.asBytes(&cached.key.device));
            bound.update(std.mem.sliceAsBytes(stamp));
            bound.update(&cached.value.sha256);
            try known.append(a, bound.finalResult());
        }
    }
    std.mem.sort([4]i128, stamps, {}, struct {
        fn less(_: void, x: [4]i128, y: [4]i128) bool {
            for (x, y) |left, right| if (left != right) return left < right;
            return false;
        }
    }.less);
    var h = std.hash.Wyhash.init(legacy);
    h.update("loaded-weight-stamps-v1");
    h.update(std.mem.sliceAsBytes(stamps));
    if (known.items.len > 0) {
        std.mem.sort([32]u8, known.items, {}, struct {
            fn less(_: void, x: [32]u8, y: [32]u8) bool {
                return std.mem.lessThan(u8, &x, &y);
            }
        }.less);
        h.update("loaded-weight-digests-v1");
        h.update(std.mem.sliceAsBytes(known.items));
    }
    return h.final();
}

/// Loaded metadata and shader documents are already mapped; hash only their used bytes, independent of map order.
pub fn mapped(legacy: u64, a: std.mem.Allocator, maps: anytype) !u64 {
    const digests = try a.alloc([32]u8, maps.len);
    defer a.free(digests);
    for (maps, digests) |m, *digest| std.crypto.hash.sha2.Sha256.hash(m.bytes[0..m.size], digest, .{});
    std.mem.sort([32]u8, digests, {}, struct {
        fn less(_: void, x: [32]u8, y: [32]u8) bool {
            return std.mem.lessThan(u8, &x, &y);
        }
    }.less);
    var h = std.hash.Wyhash.init(legacy);
    h.update("loaded-metadata-v1");
    h.update(std.mem.sliceAsBytes(digests));
    return h.final();
}
