//! Zero-copy mmap safetensors loader for single-file and sharded HuggingFace checkpoints.
const std = @import("std");

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, offset: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;

const prot_read: c_int = 1;
const map_shared: c_int = 1;
const seek_end: c_int = 2;

pub const Error = error{
    FileNotFound,
    MapFailed,
    InvalidCheckpoint,
    MissingTensor,
};

pub fn mapFile(a: std.mem.Allocator, path: []const u8) ![]const u8 {
    const zpath = try a.dupeSentinel(u8, path, 0);
    defer a.free(zpath);
    const fd = open(zpath.ptr, 0);
    if (fd < 0) return error.FileNotFound;
    defer _ = close(fd);
    const size = lseek(fd, 0, seek_end);
    if (size <= 0) return error.InvalidCheckpoint;
    const ptr = mmap(null, @intCast(size), prot_read, map_shared, fd, 0);
    if (ptr == null or @intFromPtr(ptr) == std.math.maxInt(usize)) return error.MapFailed;
    return @as([*]const u8, @ptrCast(ptr.?))[0..@intCast(size)];
}

pub fn unmapFile(bytes: []const u8) void {
    _ = munmap(@ptrCast(@constCast(bytes.ptr)), bytes.len);
}

pub const DType = enum { u32, bf16, f16, f32, other };

pub const Tensor = struct {
    bytes: []const u8,
    dtype: DType,
    shape: [6]usize = @splat(0),
    rank: usize = 0,

    pub fn asSlice(self: Tensor, comptime T: type) []align(1) const T {
        return std.mem.bytesAsSlice(T, self.bytes);
    }

    pub fn elemCount(self: Tensor) usize {
        if (self.rank == 0) return 0;
        var n: usize = 1;
        for (self.shape[0..self.rank]) |d| n *= d;
        return n;
    }
};

pub const Checkpoint = struct {
    a: std.mem.Allocator,
    tensors: std.StringHashMapUnmanaged(Tensor) = .empty,
    maps: std.ArrayList([]const u8) = .empty,

    /// Open either an indexed multi-shard checkpoint directory or a single model.safetensors directory.
    pub fn openDir(a: std.mem.Allocator, dir: []const u8) !Checkpoint {
        var ckpt = Checkpoint{ .a = a };
        errdefer ckpt.close();

        const index_path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
        defer a.free(index_path);

        if (mapFile(a, index_path)) |index_bytes| {
            defer unmapFile(index_bytes);
            const parsed = try std.json.parseFromSlice(std.json.Value, a, index_bytes, .{});
            defer parsed.deinit();
            const wm = parsed.value.object.get("weight_map") orelse return error.InvalidCheckpoint;
            var seen: std.StringHashMapUnmanaged(void) = .empty;
            defer seen.deinit(a);
            var it = wm.object.iterator();
            while (it.next()) |entry| {
                const fname = entry.value_ptr.string;
                if (seen.contains(fname)) continue;
                try seen.put(a, fname, {});
                const shard_path = try std.fs.path.join(a, &.{ dir, fname });
                defer a.free(shard_path);
                try ckpt.addShard(shard_path);
            }
            return ckpt;
        } else |_| {
            const single = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
            defer a.free(single);
            try ckpt.addShard(single);
            return ckpt;
        }
    }

    pub fn openFile(a: std.mem.Allocator, path: []const u8) !Checkpoint {
        var ckpt = Checkpoint{ .a = a };
        errdefer ckpt.close();
        try ckpt.addShard(path);
        return ckpt;
    }

    fn addShard(self: *Checkpoint, path: []const u8) !void {
        const file = try mapFile(self.a, path);
        try self.maps.append(self.a, file);
        if (file.len < 8) return error.InvalidCheckpoint;
        const header_len = std.mem.readInt(u64, file[0..8], .little);
        if (8 + header_len > file.len) return error.InvalidCheckpoint;

        const parsed = try std.json.parseFromSlice(std.json.Value, self.a, file[8..][0..header_len], .{});
        defer parsed.deinit();
        const data = file[8 + header_len ..];

        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
            const obj = entry.value_ptr.object;
            const offsets = obj.get("data_offsets").?.array.items;
            const start: usize = @intCast(offsets[0].integer);
            const end: usize = @intCast(offsets[1].integer);
            const dt_str = obj.get("dtype").?.string;
            const dt: DType = if (std.mem.eql(u8, dt_str, "U32"))
                .u32
            else if (std.mem.eql(u8, dt_str, "BF16"))
                .bf16
            else if (std.mem.eql(u8, dt_str, "F16"))
                .f16
            else if (std.mem.eql(u8, dt_str, "F32"))
                .f32
            else
                .other;

            var t = Tensor{ .bytes = data[start..end], .dtype = dt };
            const shape_arr = obj.get("shape").?.array.items;
            if (shape_arr.len > t.shape.len) return error.InvalidCheckpoint;
            for (shape_arr, 0..) |dim, i| t.shape[i] = @intCast(dim.integer);
            t.rank = shape_arr.len;

            const key_copy = try self.a.dupe(u8, entry.key_ptr.*);
            try self.tensors.put(self.a, key_copy, t);
        }
    }

    pub fn get(self: *const Checkpoint, name: []const u8) Error!Tensor {
        return self.tensors.get(name) orelse {
            std.debug.print("Missing tensor: {s}\n", .{name});
            return error.MissingTensor;
        };
    }

    pub fn getOptional(self: *const Checkpoint, name: []const u8) ?Tensor {
        return self.tensors.get(name);
    }

    pub fn close(self: *Checkpoint) void {
        var it = self.tensors.keyIterator();
        while (it.next()) |k| self.a.free(k.*);
        self.tensors.deinit(self.a);
        for (self.maps.items) |m| unmapFile(m);
        self.maps.deinit(self.a);
    }
};
