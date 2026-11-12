//! Where a Qwen-VL tower's tensors lie in a checkpoint's shards: each header read as core reads one, no tensor byte.
const std = @import("std");
const st = @import("../safetensors.zig");

pub const prefix = "vision_tower.";

pub const Src = struct { path: [:0]const u8, off: u64, e: st.Entry };

/// Every vision tensor's file, offset and header entry, found through the shard index (or the one file).
pub fn headers(arena: std.mem.Allocator, io: std.Io, dir: []const u8) !std.StringArrayHashMapUnmanaged(Src) {
    var files: std.StringArrayHashMapUnmanaged(void) = .empty;
    const index = try std.fs.path.join(arena, &.{ dir, "model.safetensors.index.json" });
    if (std.Io.Dir.cwd().readFileAlloc(io, index, arena, .limited(64 << 20))) |text| {
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{});
        const map = (doc.object.get("weight_map") orelse return error.BadIndex).object;
        var it = map.iterator();
        while (it.next()) |kv| if (std.mem.startsWith(u8, kv.key_ptr.*, prefix)) try files.put(arena, kv.value_ptr.string, {});
    } else |_| try files.put(arena, "model.safetensors", {});
    var out: std.StringArrayHashMapUnmanaged(Src) = .empty;
    for (files.keys()) |name| {
        const path = try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ dir, name }, 0);
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.OpenFailed;
        defer _ = std.c.close(fd);
        var head: [8]u8 = undefined;
        try readAll(fd, &head, 0);
        const seek = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (seek < 0) return error.SeekFailed;
        const end: u64 = @intCast(seek);
        const hlen = try st.headerSize(std.mem.readInt(u64, &head, .little), end);
        const header = try arena.alloc(u8, hlen);
        try readAll(fd, header, 8);
        const parsed = try st.parseHeaderPrefix(arena, header, @intCast(end - 8 - hlen), prefix);
        var it = parsed.iterator();
        while (it.next()) |kv| try out.put(arena, kv.key_ptr.*, .{ .path = path, .off = 8 + hlen + kv.value_ptr.begin, .e = kv.value_ptr.* });
    }
    if (out.count() == 0) return error.NoVisionTower;
    return out;
}

pub fn readAll(fd: std.c.fd_t, dst: []u8, at: u64) !void {
    var done: usize = 0;
    while (done < dst.len) {
        const n = std.c.pread(fd, dst.ptr + done, dst.len - done, @intCast(at + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

const json =
    \\{"__metadata__":{"format":"mlx"},"language_model.w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},
    \\ "vision_tower.a":{"dtype":"BF16","shape":[2,2],"data_offsets":[4,12]},
    \\ "vision_tower.b":{"dtype":"BF16","shape":[3],"data_offsets":[12,18]}}
;

/// One shard's bytes: `json`, then each tensor's range filled with its own byte (1, 2, 3).
fn shard(a: std.mem.Allocator) ![]u8 {
    const bytes = try a.alloc(u8, 8 + json.len + 18);
    std.mem.writeInt(u64, bytes[0..8], json.len, .little);
    @memcpy(bytes[8..][0..json.len], json);
    for ([_][2]usize{ .{ 0, 4 }, .{ 4, 12 }, .{ 12, 18 } }, 1..) |r, fill| @memset(bytes[8 + json.len ..][r[0]..r[1]], @intCast(fill));
    return bytes;
}

test "Qwen-VL shards: a tower's shard gives each vision tensor's file, offset, entry and bytes, in 1 MiB" {
    const io = std.testing.io;
    var fixtures: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer fixtures.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = try shard(fixtures.allocator()) });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", fixtures.allocator());
    var limited: std.heap.FixedBufferAllocator = .init(try fixtures.allocator().alloc(u8, 1 << 20));
    const found = try headers(limited.allocator(), io, dir);
    try std.testing.expectEqual(@as(usize, 2), found.count());
    const path = try std.fmt.allocPrint(fixtures.allocator(), "{s}/model.safetensors", .{dir});
    for ([_][]const u8{ "vision_tower.a", "vision_tower.b" }, [_][]const usize{ &.{ 2, 2 }, &.{3} }, [_]u64{ 4, 12 }, 2..) |name, shape, begin, fill| {
        const s = found.get(name).?;
        try std.testing.expectEqualStrings(path, s.path);
        try std.testing.expectEqual(8 + json.len + begin, s.off);
        try std.testing.expectEqual(st.DType.bf16, s.e.dtype);
        try std.testing.expectEqualSlices(usize, shape, s.e.shape[0..s.e.rank]);
        const bytes = try fixtures.allocator().alloc(u8, s.e.end - s.e.begin);
        const fd = std.c.open(s.path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        defer _ = std.c.close(fd);
        try readAll(fd, bytes, s.off);
        for (bytes) |b| try std.testing.expectEqual(@as(u8, @intCast(fill)), b);
    }
}

test "Qwen-VL shards: a header length past 100 MB, past the file's end or cut short is refused in 1 MiB, before the read" {
    const io = std.testing.io;
    var fixtures: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer fixtures.deinit();
    const good = try shard(fixtures.allocator());
    const room = try fixtures.allocator().alloc(u8, 1 << 20);
    const Case = struct { length: ?u64 = null, keep: ?usize = null, size: ?u64 = null, blank: bool = false, want: anyerror = error.BadSafetensors };
    for ([_]Case{
        .{ .length = st.max_header_bytes + 1, .size = st.max_header_bytes + 64 }, // sparse: only the cap refuses it
        .{ .length = 2_000_000_000, .size = 3_000_000_000 },
        .{ .length = std.math.maxInt(u64) },
        .{ .length = good.len - 7 }, // one byte past the end
        .{ .keep = 8 + json.len / 2 }, // the file ends inside its header
        .{ .blank = true, .want = error.UnexpectedEndOfInput }, // the header's bytes are not JSON: core's parser says why
    }) |case| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const bytes = try fixtures.allocator().dupe(u8, good[0 .. case.keep orelse good.len]);
        if (case.length) |l| std.mem.writeInt(u64, bytes[0..8], l, .little);
        if (case.blank) @memset(bytes[8..][0..json.len], ' ');
        try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = bytes });
        if (case.size) |s| {
            const f = try tmp.dir.openFile(io, "model.safetensors", .{ .mode = .read_write });
            defer f.close(io);
            try f.setLength(io, s);
        }
        const dir = try tmp.dir.realPathFileAlloc(io, ".", fixtures.allocator());
        var limited: std.heap.FixedBufferAllocator = .init(room);
        try std.testing.expectError(case.want, headers(limited.allocator(), io, dir));
    }
}

test "Qwen-VL shards: a tensor read past a shard shortened after open stops with ShortRead" {
    const io = std.testing.io;
    var fixtures: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer fixtures.deinit();
    const good = try shard(fixtures.allocator());
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = good });
    const real = try tmp.dir.realPathFileAlloc(io, "model.safetensors", fixtures.allocator());
    const fd = std.c.open(try fixtures.allocator().dupeSentinel(u8, real, 0), .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    if (@import("builtin").os.tag == .macos) _ = std.c.fcntl(fd, 48, @as(c_int, 1)); // F_NOCACHE, as weights.zig reads it
    const f = try tmp.dir.openFile(io, "model.safetensors", .{ .mode = .read_write });
    try f.setLength(io, good.len - 3); // three bytes into vision_tower.b
    f.close(io);
    var b: [6]u8 = undefined;
    try std.testing.expectError(error.ShortRead, readAll(fd, &b, good.len - 6));
    try std.testing.expectError(error.ShortRead, readAll(fd, b[0..1], good.len - 3)); // at the new end: a zero read
    try readAll(fd, b[0..3], good.len - 6);
    try std.testing.expectEqualSlices(u8, &.{ 3, 3, 3 }, b[0..3]);
}
