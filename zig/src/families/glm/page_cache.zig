//! A checkpoint's pages in the file cache, dropped before a load: F_NOCACHE reads keep out of it but evict nothing.
const std = @import("std");
const mtl = @import("metal");

extern "c" fn msync(addr: *anyopaque, len: usize, flags: c_int) c_int;
extern "c" fn mincore(addr: *const anyopaque, len: usize, vec: [*]u8) c_int;
const MS_INVALIDATE = 2; // macOS sys/mman.h

/// Each shard's pages an earlier download or read left in the file cache, dropped; the bytes that were cached.
pub fn dropCached(gpa: std.mem.Allocator, dir: []const u8) !u64 {
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
    var cached: u64 = 0;
    for (files.keys()) |name| {
        const fd = std.c.open(try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0), .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.OpenFailed;
        defer _ = std.c.close(fd);
        const end = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (end <= 0) continue;
        const len: usize = @intCast(end);
        const map = std.c.mmap(null, len, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0);
        if (map == std.c.MAP_FAILED) return error.MapFailed;
        defer _ = std.c.munmap(@alignCast(map), len);
        const vec = try a.alloc(u8, (len + page - 1) / page);
        if (mincore(map, len, vec.ptr) == 0) for (vec) |v| {
            cached += @as(u64, v & 1) * page;
        };
        if (msync(map, len, MS_INVALIDATE) != 0) return error.InvalidateFailed;
    }
    return cached;
}

test "a checkpoint's cached pages are dropped, and counted once" {
    const gpa = std.testing.allocator;
    var name_buf: [64]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&name_buf, "/tmp/tf-drop-cached-{d}", .{std.c.getpid()}, 0);
    _ = std.c.mkdir(dir, 0o700);
    defer _ = std.c.rmdir(dir);
    var path_buf: [128]u8 = undefined;
    const index = try std.fmt.bufPrintSentinel(&path_buf, "{s}/model.safetensors.index.json", .{dir}, 0);
    const text = "{\"weight_map\": {\"a\": \"s.safetensors\", \"b\": \"s.safetensors\"}}";
    var shard_buf: [128]u8 = undefined;
    const shard = try std.fmt.bufPrintSentinel(&shard_buf, "{s}/s.safetensors", .{dir}, 0);
    for ([_][:0]const u8{ index, shard }, [_][]const u8{ text, "" }) |file, body| {
        const fd = std.c.open(file, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
        try std.testing.expect(fd >= 0);
        defer _ = std.c.close(fd);
        if (body.len > 0) {
            _ = std.c.write(fd, body.ptr, body.len);
            continue;
        }
        const data = try gpa.alloc(u8, 4 << 20);
        defer gpa.free(data);
        @memset(data, 7);
        try std.testing.expectEqual(@as(isize, @intCast(data.len)), std.c.write(fd, data.ptr, data.len));
        _ = std.c.fsync(fd);
        _ = std.c.pread(fd, data.ptr, data.len, 0); // read through the file cache
    }
    defer _ = std.c.unlink(index);
    defer _ = std.c.unlink(shard);
    try std.testing.expect(try dropCached(gpa, dir) > 0);
    try std.testing.expectEqual(@as(u64, 0), try dropCached(gpa, dir));
}
