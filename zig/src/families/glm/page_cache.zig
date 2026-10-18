//! A checkpoint's pages in the file cache, dropped before a load: F_NOCACHE reads keep out of it but evict nothing.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const paced_read = @import("../../core/paced_read.zig");

extern "c" fn msync(addr: *anyopaque, len: usize, flags: c_int) c_int;
extern "c" fn mincore(addr: *const anyopaque, len: usize, vec: [*]u8) c_int;
const MS_INVALIDATE = 2; // macOS sys/mman.h

/// What a drop found and did, in bytes; `deficit` 0: free memory held the load, no shard was looked at.
pub const Dropped = struct { deficit: u64, cached: u64 = 0, dropped: u64 = 0 };

/// Only free memory's shortfall for the load plus the paced reads' floor is dropped, shard by shard; the rest stays.
pub fn dropForLoad(gpa: std.mem.Allocator, dir: []const u8, load_bytes: u64) !Dropped {
    const need = load_bytes + paced_read.floor();
    const free = paced_read.freeBytes();
    var d: Dropped = .{ .deficit = if (need > free) need - free else 0 };
    if (d.deficit > 0) try dropCached(gpa, dir, d.deficit, &d);
    return d;
}

/// Cached shards dropped until `want` bytes are; a shard with none is skipped (`msync` would walk every page).
pub fn dropCached(gpa: std.mem.Allocator, dir: []const u8, want: u64, d: *Dropped) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const index = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors.index.json", .{dir}, 0));
    defer index.deinit();
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, index.bytes[0..index.size], .{ .allocate = .alloc_always });
    var files: std.StringArrayHashMapUnmanaged(void) = .empty;
    var it = doc.object.get("weight_map").?.object.iterator();
    while (it.next()) |kv| try files.put(a, kv.value_ptr.string, {});
    const page: usize = std.heap.pageSize();
    for (files.keys()) |name| {
        if (d.dropped >= want) break;
        const fd = platform.c.open(try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0), .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.OpenFailed;
        defer _ = platform.c.close(fd);
        const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
        if (end <= 0) continue;
        const len: usize = @intCast(end);
        const map = platform.c.mmap(null, len, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0);
        if (map == platform.c.MAP_FAILED) return error.MapFailed;
        defer _ = platform.c.munmap(@alignCast(map), len);
        const vec = try a.alloc(u8, (len + page - 1) / page);
        var cached: u64 = 0;
        if (mincore(map, len, vec.ptr) == 0) for (vec) |v| {
            cached += @as(u64, v & 1) * page;
        };
        d.cached += cached;
        if (cached == 0) continue;
        if (msync(map, len, MS_INVALIDATE) != 0) return error.InvalidateFailed;
        d.dropped += cached;
    }
}

test "a checkpoint's cached pages are dropped only as far as wanted, and a shard with none is left alone" {
    const gpa = std.testing.allocator;
    var name_buf: [64]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&name_buf, "/tmp/tf-drop-cached-{d}", .{platform.c.getpid()}, 0);
    _ = platform.c.mkdir(dir, 0o700);
    defer _ = platform.c.rmdir(dir);
    var path_buf: [128]u8 = undefined;
    const index = try std.fmt.bufPrintSentinel(&path_buf, "{s}/model.safetensors.index.json", .{dir}, 0);
    const text = "{\"weight_map\": {\"a\": \"s.safetensors\", \"b\": \"s.safetensors\"}}";
    var shard_buf: [128]u8 = undefined;
    const shard = try std.fmt.bufPrintSentinel(&shard_buf, "{s}/s.safetensors", .{dir}, 0);
    for ([_][:0]const u8{ index, shard }, [_][]const u8{ text, "" }) |file, body| {
        const fd = platform.c.open(file, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true }, @as(platform.c.mode_t, 0o600));
        try std.testing.expect(fd >= 0);
        defer _ = platform.c.close(fd);
        if (body.len > 0) {
            _ = platform.c.write(fd, body.ptr, body.len);
            continue;
        }
        const data = try gpa.alloc(u8, 4 << 20);
        defer gpa.free(data);
        @memset(data, 7);
        try std.testing.expectEqual(@as(isize, @intCast(data.len)), platform.c.write(fd, data.ptr, data.len));
        _ = platform.c.fsync(fd);
        _ = platform.c.pread(fd, data.ptr, data.len, 0); // read through the file cache
    }
    defer _ = platform.c.unlink(index);
    defer _ = platform.c.unlink(shard);
    var none: Dropped = .{ .deficit = 0 };
    try dropCached(gpa, dir, 0, &none); // nothing wanted: no shard touched, the pages stay cached
    try std.testing.expectEqual(@as(u64, 0), none.dropped);
    var all: Dropped = .{ .deficit = std.math.maxInt(u64) };
    try dropCached(gpa, dir, std.math.maxInt(u64), &all);
    try std.testing.expect(all.dropped > 0 and all.dropped == all.cached);
    var again: Dropped = .{ .deficit = std.math.maxInt(u64) };
    try dropCached(gpa, dir, std.math.maxInt(u64), &again); // nothing left cached: counted, not invalidated
    try std.testing.expectEqual(@as(u64, 0), again.cached);
    try std.testing.expectEqual(@as(u64, 0), again.dropped);
    const d = try dropForLoad(gpa, dir, 0); // this Mac's own free memory and floor: dropped only with a deficit
    try std.testing.expect(d.deficit > 0 or d.dropped == 0);
}
