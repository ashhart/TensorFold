//! The shared safetensors index read into Metal buffers: one shared buffer a shard, each tensor at its data offset.
const std = @import("std");
const mtl = @import("metal");
const st = @import("safetensors.zig");

pub const DType = st.DType;

pub const Tensor = struct {
    buffer: mtl.Buffer,
    offset: usize,
    bytes: usize,
    dtype: DType,
    shape: [4]usize = .{ 1, 1, 1, 1 },
    rank: usize = 0,

    pub fn count(self: Tensor) usize {
        return self.bytes / self.dtype.size();
    }

    /// The tensor's bytes as host memory (shared storage).
    pub fn host(self: Tensor, comptime T: type) []const T {
        const base: [*]const u8 = self.buffer.contents() + self.offset;
        return @as([*]const T, @ptrCast(@alignCast(base)))[0 .. self.bytes / @sizeOf(T)];
    }
};

const Shard = struct { buffer: mtl.Buffer, bytes: usize };

/// Safetensors read into one buffer a shard from its data start: MLX's files put tensors at odd file offsets.
pub const Checkpoint = struct {
    allocator: std.mem.Allocator,
    shards: std.ArrayList(Shard) = .empty,
    tensors: std.StringHashMapUnmanaged(Tensor) = .empty,

    pub fn init(allocator: std.mem.Allocator) Checkpoint {
        return .{ .allocator = allocator };
    }

    /// Read `path`'s tensors, each named `prefix ++ name`.
    pub fn addFile(self: *Checkpoint, device: mtl.Device, path: [:0]const u8, prefix: []const u8) !void {
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) {
            std.log.err("cannot open {s}", .{path});
            return error.OpenFailed;
        }
        defer _ = std.c.close(fd);
        var head: [8]u8 = undefined;
        try readAll(fd, &head, 0);
        const header_len = std.mem.readInt(u64, &head, .little);
        const header = try self.allocator.alloc(u8, header_len);
        defer self.allocator.free(header);
        try readAll(fd, header, 8);
        const end = std.c.lseek(fd, 0, std.c.SEEK.END);
        const data_start = 8 + header_len;
        const data_len: usize = @as(usize, @intCast(end)) - data_start;

        const buffer = try device.buffer(@max(data_len, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        {
            errdefer buffer.deinit();
            try readParallel(fd, buffer.contents()[0..data_len], data_start);
            try self.shards.append(self.allocator, .{ .buffer = buffer, .bytes = data_len });
        }
        try self.index(buffer, header, data_len, prefix);
    }

    /// Name the tensors a safetensors `header` places in `buffer`, its data region of `data_len` bytes.
    pub fn index(self: *Checkpoint, buffer: mtl.Buffer, header: []const u8, data_len: usize, prefix: []const u8) !void {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const entries = try st.parseHeader(arena.allocator(), header, data_len);
        var it = entries.iterator();
        while (it.next()) |entry| {
            const e = entry.value_ptr.*;
            var t = Tensor{ .buffer = buffer, .offset = e.begin, .bytes = e.end - e.begin, .dtype = e.dtype, .rank = e.rank };
            for (0..e.rank) |i| t.shape[i] = e.shape[i];
            const name = try std.mem.concat(self.allocator, u8, &.{ prefix, entry.key_ptr.* });
            try self.tensors.put(self.allocator, name, t);
        }
    }

    pub fn get(self: *const Checkpoint, name: []const u8) !Tensor {
        return self.tensors.get(name) orelse {
            std.log.err("checkpoint has no tensor {s}", .{name});
            return error.MissingTensor;
        };
    }

    pub fn has(self: *const Checkpoint, name: []const u8) bool {
        return self.tensors.contains(name);
    }

    /// Bytes held in shard buffers.
    pub fn residentBytes(self: *const Checkpoint) usize {
        var total: usize = 0;
        for (self.shards.items) |s| total += s.bytes;
        return total;
    }

    pub fn deinit(self: *Checkpoint) void {
        var it = self.tensors.keyIterator();
        while (it.next()) |k| self.allocator.free(k.*);
        self.tensors.deinit(self.allocator);
        for (self.shards.items) |s| s.buffer.deinit();
        self.shards.deinit(self.allocator);
    }
};

fn readAll(fd: std.c.fd_t, dest: []u8, at: usize) !void {
    var done: usize = 0;
    while (done < dest.len) {
        const n = std.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

const Part = struct { fd: std.c.fd_t, dest: []u8, at: usize, failed: bool = false };

fn readPart(part: *Part) void {
    readAll(part.fd, part.dest, part.at) catch {
        part.failed = true;
    };
}

/// pread `dest.len` bytes from `at` on 8 threads (page cache or disk, the copy is the load's main cost).
fn readParallel(fd: std.c.fd_t, dest: []u8, at: usize) !void {
    const n_threads = 8;
    var parts: [n_threads]Part = undefined;
    var threads: [n_threads]?std.Thread = @splat(null);
    const chunk = (dest.len + n_threads - 1) / n_threads;
    for (0..n_threads) |i| {
        const lo = @min(i * chunk, dest.len);
        const hi = @min(lo + chunk, dest.len);
        parts[i] = .{ .fd = fd, .dest = dest[lo..hi], .at = at + lo };
        threads[i] = std.Thread.spawn(.{}, readPart, .{&parts[i]}) catch null;
        if (threads[i] == null) readPart(&parts[i]);
    }
    for (threads) |t| if (t) |th| th.join();
    for (parts) |p| if (p.failed) return error.ShortRead;
}
