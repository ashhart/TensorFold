//! Resolved runtime facts and opened weight files extend a probe key without changing a state's image.
const std = @import("std");
const digest_cache = @import("tensorfold").digest_cache;

pub fn extend(legacy: u64, domain: []const u8, words: []const u64) u64 {
    var h = std.hash.Wyhash.init(legacy);
    h.update(domain);
    h.update(std.mem.sliceAsBytes(words));
    return h.final();
}

/// Verified bytes of the actual opened inputs; a missing digest keeps optional persisted state cold.
pub fn files(legacy: u64, a: std.mem.Allocator, io: std.Io, descriptors: []const c_int) !u64 {
    const root = try digest_cache.environmentRoot(a);
    defer if (root) |r| a.free(r);
    return filesWithCache(legacy, a, io, descriptors, root);
}

pub fn filesWithCache(legacy: u64, a: std.mem.Allocator, io: std.Io, descriptors: []const c_int, root: ?[]const u8) !u64 {
    const generated = if (root) |r| try digest_cache.generatedRoot(a, r) else null;
    defer if (generated) |r| a.free(r);
    const known = try a.alloc([32]u8, descriptors.len);
    defer a.free(known);
    for (descriptors, known) |fd, *digest| {
        if (fd < 0) return error.IdentityFileUnavailable;
        const file: std.Io.File = .{ .handle = fd, .flags = .{ .nonblocking = false } };
        var value = try (digest_cache.Cache{ .root = generated }).checkedFile(a, io, file);
        if (value == null) value = try (digest_cache.Cache{ .root = root }).checkedFile(a, io, file);
        digest.* = (value orelse return error.UnverifiedPersistence).value.sha256;
    }
    std.mem.sort([32]u8, known, {}, struct {
        fn less(_: void, x: [32]u8, y: [32]u8) bool {
            return std.mem.lessThan(u8, &x, &y);
        }
    }.less);
    var h = std.hash.Wyhash.init(legacy);
    h.update("loaded-verified-weight-content-v2");
    h.update(std.mem.sliceAsBytes(known));
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
