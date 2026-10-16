//! Safetensors checkpoint reader: shard headers via the index, tensor bytes streamed in chunks into device buffers.

const std = @import("std");
const rt = @import("rt.zig");

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn pread(fd: c_int, buf: [*]u8, n: usize, off: i64) isize;
extern "c" fn close(fd: c_int) c_int;

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

    pub fn addShard(l: *Loader, dir: []const u8, shard: []const u8) !void {
        const fd = try openPath(l.gpa, dir, shard);
        var n8: [8]u8 = undefined;
        try readExact(fd, &n8, 0);
        const hlen = std.mem.readInt(u64, &n8, .little);
        const hdr = try l.gpa.alloc(u8, hlen);
        defer l.gpa.free(hdr);
        try readExact(fd, hdr, 8);
        const parsed = try std.json.parseFromSlice(std.json.Value, l.gpa, hdr, .{});
        defer parsed.deinit();
        var it = parsed.value.object.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
            const o = e.value_ptr.object;
            const offs = o.get("data_offsets").?.array.items;
            const lo: u64 = @intCast(offs[0].integer);
            const hi: u64 = @intCast(offs[1].integer);
            try l.map.put(l.gpa, try l.gpa.dupe(u8, e.key_ptr.*), .{
                .fd = fd,
                .off = 8 + hlen + lo,
                .len = hi - lo,
                .dtype = try l.gpa.dupe(u8, o.get("dtype").?.string),
            });
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
