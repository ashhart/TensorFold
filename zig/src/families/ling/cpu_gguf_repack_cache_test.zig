//! Direct validation tests for the Ling 8-row repack cache: hit, changed source, malformed table, corrupted payload.
const std = @import("std");
const g = @import("core").gguf;
const repack8 = @import("cpu_gguf_repack8.zig");
const Cache = @import("cpu_gguf_repack_cache.zig").Cache;

const q4_bytes = 8 * 144;
const q6_bytes = 8 * 210;

fn appendInt(buf: *std.ArrayList(u8), a: std.mem.Allocator, comptime T: type, v: T) !void {
    var raw: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &raw, v, .little);
    try buf.appendSlice(a, &raw);
}

fn appendStr(buf: *std.ArrayList(u8), a: std.mem.Allocator, s: []const u8) !void {
    try appendInt(buf, a, u64, s.len);
    try buf.appendSlice(a, s);
}

fn appendTensor(buf: *std.ArrayList(u8), a: std.mem.Allocator, name: []const u8, shape: []const u64, kind: g.TensorKind, off: u64) !void {
    try appendStr(buf, a, name);
    try appendInt(buf, a, u32, @intCast(shape.len));
    for (shape) |d| try appendInt(buf, a, u64, d);
    try appendInt(buf, a, u32, @backingInt(kind));
    try appendInt(buf, a, u64, off);
}

fn fillValidQ4K(dst: []u8, seed: u8) void {
    for (0..8) |b| {
        const blk = dst[b * 144 ..][0..144];
        std.mem.writeInt(u16, blk[0..2], 0x3c00, .little);
        std.mem.writeInt(u16, blk[2..4], 0x3800, .little);
        for (blk[4..], 0..) |*byte, i| byte.* = @truncate(seed +% @as(u8, @truncate(b * 17 +% i * 7)));
    }
}

fn fillValidQ6K(dst: []u8, seed: u8) void {
    for (0..8) |b| {
        const blk = dst[b * 210 ..][0..210];
        for (blk[0..208], 0..) |*byte, i| byte.* = @truncate(seed +% @as(u8, @truncate(b * 29 +% i * 11)));
        std.mem.writeInt(u16, blk[208..210], 0x3c00, .little);
    }
}

fn buildTinyGguf(a: std.mem.Allocator, seed: u8) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(a);
    const q6_off: u64 = std.mem.alignForward(u64, q4_bytes, 32);
    try buf.appendSlice(a, "GGUF");
    try appendInt(&buf, a, u32, 3);
    try appendInt(&buf, a, u64, 2);
    try appendInt(&buf, a, u64, 0);
    try appendTensor(&buf, a, "w_q4", &.{ 256, 8 }, .q4_k, 0);
    try appendTensor(&buf, a, "w_q6", &.{ 256, 8 }, .q6_k, q6_off);
    const aligned = std.mem.alignForward(usize, buf.items.len, 32);
    try buf.appendNTimes(a, 0, aligned - buf.items.len);
    const data_start = buf.items.len;
    const data_len: usize = @intCast(q6_off + q6_bytes);
    try buf.appendNTimes(a, 0, data_len);
    fillValidQ4K(buf.items[data_start..][0..q4_bytes], seed);
    fillValidQ6K(buf.items[data_start + @as(usize, @intCast(q6_off)) ..][0..q6_bytes], seed +% 5);
    return buf.toOwnedSlice(a);
}

fn expectLookupMatches(cache: *const Cache, source: *const g.File) !void {
    const base = @intFromPtr(source.map.memory.ptr);
    const t4 = source.document.getTensor("w_q4").?;
    const r4 = cache.lookup(@intFromPtr(t4.bytes.ptr) - base, t4.bytes.len, .q4_k) orelse return error.MissingQ4Lookup;
    var want4: [1]repack8.BlockQ4Kx8 = undefined;
    repack8.repackQ4KInto(t4.bytes, 256, 8, &want4);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&want4), std.mem.sliceAsBytes(r4.q4_k));

    const t6 = source.document.getTensor("w_q6").?;
    const r6 = cache.lookup(@intFromPtr(t6.bytes.ptr) - base, t6.bytes.len, .q6_k) orelse return error.MissingQ6Lookup;
    var want6: [1]repack8.BlockQ6Kx8 = undefined;
    repack8.repackQ6KInto(t6.bytes, 256, 8, &want6);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&want6), std.mem.sliceAsBytes(r6.q6_k));
}

fn rewriteCacheFile(a: std.mem.Allocator, io: std.Io, cache_path: []const u8, mutate: *const fn ([]u8) void) !void {
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, cache_path, a, .limited(4 << 20));
    defer a.free(raw);
    mutate(raw);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = cache_path, .data = raw });
}

test "repack cache hits on unchanged source and matches repacked payloads" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const gguf_bytes = try buildTinyGguf(a, 1);
    defer a.free(gguf_bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "m.gguf", .data = gguf_bytes });
    const model_path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/m.gguf", .{tmp.sub_path});
    defer a.free(model_path);

    var src = try g.File.open(a, io, model_path, .{});
    defer src.close(io);

    var first = try Cache.openOrBuildChecked(a, io, model_path, &src);
    defer first.cache.close(io);
    try std.testing.expect(!first.hit);
    try expectLookupMatches(&first.cache, &src);

    var second = try Cache.openOrBuildChecked(a, io, model_path, &src);
    defer second.cache.close(io);
    try std.testing.expect(second.hit);
    try expectLookupMatches(&second.cache, &src);
}

test "repack cache rejects changed source even when stat identity matches header" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const v1 = try buildTinyGguf(a, 1);
    defer a.free(v1);
    const v2 = try buildTinyGguf(a, 77);
    defer a.free(v2);
    try std.testing.expectEqual(v1.len, v2.len);

    try tmp.dir.writeFile(io, .{ .sub_path = "m.gguf", .data = v1 });
    const model_path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/m.gguf", .{tmp.sub_path});
    defer a.free(model_path);
    const cache_path = try std.fmt.allocPrint(a, "{s}" ++ Cache.file_suffix, .{model_path});
    defer a.free(cache_path);

    {
        var src1 = try g.File.open(a, io, model_path, .{});
        defer src1.close(io);
        var built = try Cache.openOrBuildChecked(a, io, model_path, &src1);
        defer built.cache.close(io);
        try std.testing.expect(!built.hit);
    }

    // Overwrite source with different bytes, then patch cached stat identity so only source_hash differs.
    try tmp.dir.writeFile(io, .{ .sub_path = "m.gguf", .data = v2 });
    var src2 = try g.File.open(a, io, model_path, .{});
    defer src2.close(io);
    const stat2 = try src2.file.stat(io);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, cache_path, a, .limited(4 << 20));
    defer a.free(raw);
    const hdr: *align(1) Cache.Header = @ptrCast(raw.ptr);
    hdr.source_size = stat2.size;
    hdr.source_inode = @intCast(stat2.inode);
    hdr.source_mtime = stat2.mtime.nanoseconds;
    hdr.source_ctime = stat2.ctime.nanoseconds;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = cache_path, .data = raw });

    var reopened = try Cache.openOrBuildChecked(a, io, model_path, &src2);
    defer reopened.cache.close(io);
    try std.testing.expect(!reopened.hit);
    try expectLookupMatches(&reopened.cache, &src2);
}

test "repack cache rejects malformed entry table and rebuilds cleanly" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const gguf_bytes = try buildTinyGguf(a, 3);
    defer a.free(gguf_bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "m.gguf", .data = gguf_bytes });
    const model_path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/m.gguf", .{tmp.sub_path});
    defer a.free(model_path);
    const cache_path = try std.fmt.allocPrint(a, "{s}" ++ Cache.file_suffix, .{model_path});
    defer a.free(cache_path);

    var src = try g.File.open(a, io, model_path, .{});
    defer src.close(io);

    var built = try Cache.openOrBuildChecked(a, io, model_path, &src);
    built.cache.close(io);

    // Corrupt entry[0].kind and re-sign payload_hash so table validation itself must catch it.
    try rewriteCacheFile(a, io, cache_path, struct {
        fn badKind(raw: []u8) void {
            const hdr: *align(1) Cache.Header = @ptrCast(raw.ptr);
            const entries: [*]align(1) Cache.Entry = @ptrCast(raw.ptr + @sizeOf(Cache.Header));
            entries[0].kind = 999;
            std.crypto.hash.sha2.Sha256.hash(raw[@sizeOf(Cache.Header)..], &hdr.payload_hash, .{});
        }
    }.badKind);

    var rebuilt1 = try Cache.openOrBuildChecked(a, io, model_path, &src);
    try std.testing.expect(!rebuilt1.hit);
    try expectLookupMatches(&rebuilt1.cache, &src);
    rebuilt1.cache.close(io);

    // Corrupt data_offset so it overlaps the entry table.
    try rewriteCacheFile(a, io, cache_path, struct {
        fn badOffset(raw: []u8) void {
            const hdr: *align(1) Cache.Header = @ptrCast(raw.ptr);
            hdr.data_offset = @sizeOf(Cache.Header);
        }
    }.badOffset);

    var rebuilt2 = try Cache.openOrBuildChecked(a, io, model_path, &src);
    defer rebuilt2.cache.close(io);
    try std.testing.expect(!rebuilt2.hit);
    try expectLookupMatches(&rebuilt2.cache, &src);
}

test "repack cache rejects corrupted payload bytes and rebuilds cleanly" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const gguf_bytes = try buildTinyGguf(a, 9);
    defer a.free(gguf_bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "m.gguf", .data = gguf_bytes });
    const model_path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/m.gguf", .{tmp.sub_path});
    defer a.free(model_path);
    const cache_path = try std.fmt.allocPrint(a, "{s}" ++ Cache.file_suffix, .{model_path});
    defer a.free(cache_path);

    var src = try g.File.open(a, io, model_path, .{});
    defer src.close(io);

    var built = try Cache.openOrBuildChecked(a, io, model_path, &src);
    built.cache.close(io);

    // Flip a byte inside the repacked payload while leaving header and entry table intact.
    try rewriteCacheFile(a, io, cache_path, struct {
        fn flipPayload(raw: []u8) void {
            const hdr: *align(1) const Cache.Header = @ptrCast(raw.ptr);
            raw[@intCast(hdr.data_offset + 7)] ^= 0x5a;
        }
    }.flipPayload);

    var rebuilt = try Cache.openOrBuildChecked(a, io, model_path, &src);
    defer rebuilt.cache.close(io);
    try std.testing.expect(!rebuilt.hit);
    try expectLookupMatches(&rebuilt.cache, &src);
}
