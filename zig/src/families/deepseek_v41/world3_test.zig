//! Rank files on a synthetic pack: the even worlds pinned to main.

const std = @import("std");
const testing = std.testing;
const config = @import("config.zig");
const Config = config.Config;
const named = @import("named.zig");
const Pack = @import("pack.zig").Pack;
const plan = @import("plan.zig");
const prepared = @import("prepared.zig");
const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// tiny() with the default's eight output groups and whole blocks for every world here.
fn eightGroups() Config {
    var c = config.tiny();
    c.vocab = 640;
    c.heads = 8;
    c.head_dim = 128;
    c.o_groups = 8;
    c.expert_width = 768;
    return c;
}

/// A pack for `c` on disk; bit 14 of each 16-bit word stays clear, so every element is finite in F32, BF16 and F16.
fn writePack(a: std.mem.Allocator, io: Io, dir: Io.Dir, c: *const Config) !void {
    var n: u64 = 0;
    const json = try @import("host_test.zig").tinyPackLen(a, c, "", "", &n);
    defer a.free(json);
    const payload = try a.alloc(u8, @intCast(n));
    defer a.free(payload);
    var word: u32 = 0x9e3779b9;
    for (0..payload.len / 2) |i| {
        word = (word ^ @as(u32, @truncate(i))) *% 0x01000193;
        std.mem.writeInt(u16, payload[2 * i ..][0..2], @truncate((word >> 16) & 0xbfff), .little);
    }
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    var f = try dir.createFile(io, "model.safetensors", .{});
    defer f.close(io);
    try f.writePositionalAll(io, &len, 0);
    try f.writePositionalAll(io, json, 8);
    try f.writePositionalAll(io, payload, 8 + json.len);
}

/// Adds the named file's tag and bytes to `h`.
fn hashFile(a: std.mem.Allocator, io: Io, root: []const u8, tag: []const u8, h: *Sha256) !void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const file = try Io.Dir.cwd().openFile(io, try std.fmt.bufPrint(&buf, "{s}/{s}.bin", .{ root, tag }), .{});
    defer file.close(io);
    const size: usize = @intCast((try file.stat(io)).size);
    const bytes = try a.alloc(u8, size);
    defer a.free(bytes);
    try testing.expectEqual(size, try file.readPositionalAll(io, bytes, 0));
    h.update(tag);
    h.update(bytes);
}

/// Every rank's files as the loader writes them, a block at a time and then the model-level tensors, as one digest.
fn rankFiles(a: std.mem.Allocator, io: Io, root: []const u8, pack: *const Pack, c: *const Config, world: u32) ![64]u8 {
    var h = Sha256.init(.{});
    for (0..world) |r| {
        const rank: u32 = @intCast(r);
        var pl = try plan.build(a, c, pack, .{ .rank = rank, .world = world });
        defer pl.deinit();
        const key: prepared.Key = .{ .rank = rank, .world = world, .o_groups = c.o_groups, .blocks = &.{}, .dspark = true };
        var cache = try prepared.Cache.open(a, io, root, @splat(0), key, true, false);
        defer cache.deinit();
        for (pl.layers, 0..) |*l, i| {
            var b: named.Builder = .{ .gpa = a, .io = io, .pack = pack };
            defer b.deinit();
            var tag_buf: [16]u8 = undefined;
            const tag = try std.fmt.bufPrint(&tag_buf, "L{d}", .{i});
            try b.layer(l, c.o_groups, rank, world);
            try cache.put(&b, tag);
            try hashFile(a, io, cache.root, tag, &h);
        }
        var b: named.Builder = .{ .gpa = a, .io = io, .pack = pack };
        defer b.deinit();
        try b.top(&pl);
        try b.dspark(&pl);
        try cache.put(&b, "top");
        try hashFile(a, io, cache.root, "top", &h);
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

test "worlds one, two and four: rank files keep their main bytes" {
    const a = testing.allocator;
    const io = testing.io;
    const c = eightGroups();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writePack(a, io, tmp.dir, &c);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var pack = try Pack.open(a, io, dir);
    defer pack.deinit();
    const root = try std.fmt.allocPrint(a, "{s}/prepared", .{dir});
    defer a.free(root);
    const pins = [_]struct { world: u32, hex: []const u8 }{
        .{ .world = 1, .hex = "93c42adbc0adb4e50036c743613c41cc7be72d17b38f1061e2912ec787ac85c6" },
        .{ .world = 2, .hex = "54a3f8703840a210c409b673e96f4acfb665abba13c344aa62cb197b98376442" },
        .{ .world = 4, .hex = "272a15e624d30215336f243f6789a62b6fdb2029680e78f701b63d9cbde74bcb" },
    };
    for (pins) |pin| {
        const hex = try rankFiles(a, io, root, &pack, &c, pin.world);
        try testing.expectEqualStrings(pin.hex, &hex);
    }
}
