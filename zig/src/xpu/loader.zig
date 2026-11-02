//! Safetensors checkpoint reader: shard headers via the index, tensor bytes streamed in chunks into device buffers.

const std = @import("std");
const rt = @import("rt.zig");
const st = @import("core").safetensors;

const platform = @import("sys");
const pread = platform.c.pread;
const close = platform.c.close;

/// `path` opened to read (the only mode the loader asks for, as flags 0).
fn open(path: [*:0]const u8, flags: c_int) c_int {
    _ = flags;
    return platform.c.open(path, .{}, @as(platform.c.mode_t, 0));
}

const chunk_bytes: usize = 64 << 20;

pub const Info = struct { fd: c_int, off: u64, len: u64, dtype: []const u8 };

pub fn readExact(fd: c_int, buf: []u8, off: u64) !void {
    var done: usize = 0;
    while (done < buf.len) {
        const n = pread(fd, buf.ptr + done, buf.len - done, @intCast(off + done));
        if (n <= 0) return error.ReadFailed;
        done += @intCast(n);
    }
}

/// The header's own spelling of a core dtype ("BF16", "F8_E4M3").
fn dtypeName(d: st.DType) []const u8 {
    return switch (d) {
        inline else => |tag| comptime blk: {
            const s = @tagName(tag);
            var b: [s.len]u8 = undefined;
            for (s, 0..) |ch, i| b[i] = std.ascii.toUpper(ch);
            const out = b;
            break :blk &out;
        },
    };
}

/// A shard's data offset and its header as core reads it; a dtype or rank core does not name is left out, not refused.
fn shardHeader(arena: std.mem.Allocator, fd: c_int) !struct { u64, st.Header } {
    const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
    if (end < 8) return error.BadSafetensors;
    const size: u64 = @intCast(end);
    var n8: [8]u8 = undefined;
    try readExact(fd, &n8, 0);
    const hlen = try st.headerSize(std.mem.readInt(u64, &n8, .little), size);
    const hdr = try arena.alloc(u8, hlen);
    try readExact(fd, hdr, 8);
    const data_len = std.math.cast(usize, size - 8 - hlen) orelse return error.BadSafetensors;
    return .{ 8 + hlen, try st.parseHeaderOptions(arena, hdr, data_len, .{ .unsupported = .omit }) };
}

fn openPath(gpa: std.mem.Allocator, dir: []const u8, name: []const u8) !c_int {
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, name }, 0);
    defer gpa.free(path);
    const fd = open(path.ptr, 0);
    if (fd < 0) return error.OpenFailed;
    return fd;
}

/// Whole small file (config.json, the index) into memory.
pub fn readFile(gpa: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    const fd = try openPath(gpa, dir, name);
    defer _ = close(fd);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var buf: [1 << 16]u8 = undefined;
    var off: u64 = 0;
    while (true) {
        const n = pread(fd, &buf, buf.len, @intCast(off));
        if (n < 0) return error.ReadFailed;
        if (n == 0) break;
        try out.appendSlice(gpa, buf[0..@intCast(n)]);
        off += @intCast(n);
    }
    return out.toOwnedSlice(gpa);
}

pub const Loader = struct {
    gpa: std.mem.Allocator,
    r: *rt.Runtime,
    map: std.StringHashMapUnmanaged(Info) = .empty,
    stage: []u8,
    /// Device bytes allocated through this loader.
    total: u64 = 0,
    /// Tensors come from a GGUF file (qwen_gguf.readHeader filled `map`).
    gguf: bool = false,

    /// A loader with a staging buffer and no tensors yet (GGUF headers are added by qwen_gguf).
    pub fn initBare(gpa: std.mem.Allocator, r: *rt.Runtime) !Loader {
        return .{ .gpa = gpa, .r = r, .stage = try gpa.alloc(u8, chunk_bytes) };
    }

    pub fn init(gpa: std.mem.Allocator, r: *rt.Runtime, dir: []const u8) !Loader {
        var l: Loader = .{ .gpa = gpa, .r = r, .stage = try gpa.alloc(u8, chunk_bytes) };
        const index = try readFile(gpa, dir, "model.safetensors.index.json");
        defer gpa.free(index);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, index, .{});
        defer parsed.deinit();
        var shards: std.StringArrayHashMapUnmanaged(void) = .empty;
        defer shards.deinit(gpa);
        var it = parsed.value.object.get("weight_map").?.object.iterator();
        while (it.next()) |e| try shards.put(gpa, e.value_ptr.string, {});
        for (shards.keys()) |shard| try l.addShard(dir, shard);
        return l;
    }

    /// The shard's tensors by name; its fd stays open for `load` once the header is read.
    pub fn addShard(l: *Loader, dir: []const u8, shard: []const u8) !void {
        const fd = try openPath(l.gpa, dir, shard);
        var arena = std.heap.ArenaAllocator.init(l.gpa);
        defer arena.deinit();
        const data, const names = shardHeader(arena.allocator(), fd) catch |e| {
            _ = close(fd);
            return e;
        };
        try l.map.ensureUnusedCapacity(l.gpa, @intCast(names.count()));
        for (names.keys(), names.values()) |name, e| {
            const t: Info = .{ .fd = fd, .off = data + e.begin, .len = e.end - e.begin, .dtype = dtypeName(e.dtype) };
            l.map.putAssumeCapacity(try l.gpa.dupe(u8, name), t);
        }
    }

    pub fn info(l: *Loader, name: []const u8) !Info {
        return l.map.get(name) orelse {
            std.log.err("tensor missing: {s}", .{name});
            return error.MissingTensor;
        };
    }

    /// Pinned host memory instead of device memory (not counted in `total`); the CPU can write it directly.
    pub fn emptyHost(l: *Loader, bytes: usize) !rt.Buffer {
        return l.r.allocHost(bytes);
    }

    pub fn empty(l: *Loader, bytes: usize) !rt.Buffer {
        const b = try l.r.alloc(bytes);
        l.total += bytes;
        return b;
    }

    pub fn zeros(l: *Loader, bytes: usize) !rt.Buffer {
        const b = try l.empty(bytes);
        @memset(l.stage[0..@min(bytes, chunk_bytes)], 0);
        var off: usize = 0;
        while (off < bytes) {
            const n = @min(bytes - off, chunk_bytes);
            try l.r.upload(.{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = n }, l.stage[0..n]);
            try l.r.sync();
            off += n;
        }
        return b;
    }

    /// Streams the tensor's bytes into a new device buffer.
    pub fn load(l: *Loader, name: []const u8) !rt.Buffer {
        const inf = try l.info(name);
        const b = try l.empty(inf.len);
        var off: u64 = 0;
        while (off < inf.len) {
            const n: usize = @intCast(@min(inf.len - off, chunk_bytes));
            try readExact(inf.fd, l.stage[0..n], inf.off + off);
            try l.r.upload(.{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = n }, l.stage[0..n]);
            try l.r.sync();
            off += n;
        }
        return b;
    }

    /// A small bf16 tensor widened to fp32 on the device.
    pub fn loadF32(l: *Loader, name: []const u8) !rt.Buffer {
        const inf = try l.info(name);
        if (!std.mem.eql(u8, inf.dtype, "BF16")) return error.UnexpectedDtype;
        const n: usize = @intCast(inf.len / 2);
        const raw = try l.gpa.alloc(u16, n);
        defer l.gpa.free(raw);
        try readExact(inf.fd, std.mem.sliceAsBytes(raw), inf.off);
        const wide = try l.gpa.alloc(f32, n);
        defer l.gpa.free(wide);
        for (raw, wide) |v, *o| o.* = @bitCast(@as(u32, v) << 16);
        const b = try l.empty(n * 4);
        try l.r.upload(b, std.mem.sliceAsBytes(wide));
        try l.r.sync();
        return b;
    }
};

/// `tmp`/s.safetensors: the length word (`len`, else the header's), the header, then `data` zero bytes.
fn writeShard(tmp: *std.testing.TmpDir, len: ?u64, header: []const u8, data: usize) !void {
    var buf: [256]u8 = @splat(0);
    std.mem.writeInt(u64, buf[0..8], len orelse header.len, .little);
    @memcpy(buf[8..][0..header.len], header);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "s.safetensors", .data = buf[0 .. 8 + header.len + data] });
}

test "a shard header is read as core reads it: malformed entries, a huge length word and offsets past the file are refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path: [std.fs.max_path_bytes]u8 = undefined;
    const dir = path[0..try tmp.dir.realPath(std.testing.io, &path)];
    var l: Loader = .{ .gpa = arena.allocator(), .r = undefined, .stage = &.{} };
    const offsets = [_][]const u8{ "[]", "[0]", "[-1,2]", "[-0,2]", "[true,2]", "[null,2]", "\"4\"", "[18446744073709551616,2]", "[7,2]" };
    for (offsets) |o| {
        try writeShard(&tmp, null, try std.fmt.allocPrint(arena.allocator(), "{{\"a\":{{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":{s}}}}}", .{o}), 2);
        try std.testing.expectError(error.BadSafetensors, l.addShard(dir, "s.safetensors"));
    }
    const entries = [_][]const u8{
        "[]",
        "{\"a\":0}",
        "{\"a\":{\"dtype\":\"U8\",\"shape\":[0]}}",
        "{\"a\":{\"shape\":[0],\"data_offsets\":[0,0]}}",
        "{\"a\":{\"dtype\":1,\"shape\":[0],\"data_offsets\":[0,0]}}",
        "{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}",
    };
    for (entries) |h| {
        try writeShard(&tmp, null, h, 2);
        try std.testing.expectError(error.BadSafetensors, l.addShard(dir, "s.safetensors"));
    }
    for ([_]u64{ std.math.maxInt(u64), 1 << 20 }) |len| {
        try writeShard(&tmp, len, "{}", 0);
        try std.testing.expectError(error.BadSafetensors, l.addShard(dir, "s.safetensors"));
    }
    try std.testing.expectEqual(0, l.map.count());
    const header = "{\"__metadata__\":{},\"a\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]},\"q\":{\"dtype\":\"Q4\",\"shape\":[1],\"data_offsets\":[4,5]}}";
    try writeShard(&tmp, null, header, 5);
    try l.addShard(dir, "s.safetensors");
    const a = l.map.get("a").?;
    defer _ = close(a.fd);
    try std.testing.expect(a.off == 8 + header.len and a.len == 4 and std.mem.eql(u8, a.dtype, "BF16"));
    try std.testing.expect(l.map.get("q") == null); // a dtype core does not name is left out
}
