//! The shared safetensors index read into Metal buffers: one shared buffer a shard, each tensor at its data offset.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const st = @import("safetensors.zig");

pub const DType = st.DType;

pub const Tensor = struct {
    buffer: mtl.Buffer,
    offset: usize,
    bytes: usize,
    dtype: DType,
    shape: [st.max_rank]usize = @splat(1),
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

/// A tensor left in its shard file at load: `read` copies its bytes out.
pub const Deferred = struct { file: usize, at: usize, bytes: usize, dtype: DType, shape: [st.max_rank]usize = @splat(1), rank: usize = 0 };

/// Names (of tensors of `rank`) that stay in their files at load: a family reads them once, into layouts it owns.
pub const Later = *const fn (name: []const u8, rank: usize) bool;

/// A shard file that deferred tensors stay in, and its address-only views' reservation once one is asked for.
const File = struct { fd: platform.c.fd_t, end: usize = 0, views: ?[]align(std.heap.page_size_min) u8 = null };

/// Kept tensors pack at this alignment; reads this large or larger run on 8 threads.
pub const alignment = 256;
const parallel_bytes = 4 << 20;

/// Safetensors read into one buffer a shard from its data start: MLX's files put tensors at odd file offsets.
pub const Checkpoint = struct {
    allocator: std.mem.Allocator,
    shards: std.ArrayList(Shard) = .empty,
    tensors: std.StringHashMapUnmanaged(Tensor) = .empty,
    deferred: std.StringHashMapUnmanaged(Deferred) = .empty,
    files: std.ArrayList(File) = .empty,

    pub fn init(allocator: std.mem.Allocator) Checkpoint {
        return .{ .allocator = allocator };
    }

    /// Read `path`'s tensors, each named `prefix ++ name`.
    pub fn addFile(self: *Checkpoint, device: mtl.Device, path: [:0]const u8, prefix: []const u8) !void {
        return self.addFileSplit(device, path, prefix, null, null);
    }

    /// Load only `selected`'s tensor namespace and its contiguous data region; names retain the caller's prefix.
    pub fn addFileSelected(self: *Checkpoint, device: mtl.Device, path: [:0]const u8, prefix: []const u8, selected: ?[]const u8) !void {
        return self.addFileSplit(device, path, prefix, selected, null);
    }

    /// As addFileSelected; with `later`, the tensors it names stay in the file for `read` and the rest pack into one buffer.
    pub fn addFileSplit(self: *Checkpoint, device: mtl.Device, path: [:0]const u8, prefix: []const u8, selected: ?[]const u8, later: ?Later) !void {
        const fd = platform.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) {
            std.log.err("cannot open {s}", .{path});
            return error.OpenFailed;
        }
        var held = false;
        defer if (!held) {
            _ = platform.c.close(fd);
        };
        var head: [8]u8 = undefined;
        try readAll(fd, &head, 0);
        const header_len = std.mem.readInt(u64, &head, .little);
        const header = try self.allocator.alloc(u8, header_len);
        defer self.allocator.free(header);
        try readAll(fd, header, 8);
        const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
        const data_start = 8 + header_len;
        const data_len: usize = @as(usize, @intCast(end)) - data_start;

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const entries = try st.parseHeaderPrefix(arena.allocator(), header, data_len, selected);
        if (entries.count() == 0) return;
        if (later) |pick| return self.split(device, fd, &held, data_start, entries, prefix, pick);
        var begin: usize = if (selected == null) 0 else data_len;
        var limit: usize = if (selected == null) data_len else 0;
        for (entries.values()) |e| {
            begin = @min(begin, e.begin);
            limit = @max(limit, e.end);
        }
        const bytes = limit - begin;
        const buffer = try device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        var transferred = false;
        errdefer if (!transferred) buffer.deinit();
        try readParallel(fd, buffer.contents()[0..bytes], data_start + begin);
        try self.shards.append(self.allocator, .{ .buffer = buffer, .bytes = bytes });
        transferred = true;
        var it = entries.iterator();
        while (it.next()) |entry| {
            const e = entry.value_ptr.*;
            var t = Tensor{ .buffer = buffer, .offset = e.begin - begin, .bytes = e.end - e.begin, .dtype = e.dtype, .rank = e.rank };
            for (0..e.rank) |i| t.shape[i] = e.shape[i];
            const name = try std.mem.concat(self.allocator, u8, &.{ prefix, entry.key_ptr.* });
            errdefer self.allocator.free(name);
            const slot = try self.tensors.getOrPut(self.allocator, name);
            if (slot.found_existing) return error.DuplicateTensor;
            slot.value_ptr.* = t;
        }
    }

    /// The kept entries packed into one buffer, the `later` ones recorded for `read`; `held` once the file stays open.
    fn split(self: *Checkpoint, device: mtl.Device, fd: platform.c.fd_t, held: *bool, data_start: usize, entries: st.Header, prefix: []const u8, later: Later) !void {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const names = try arena.allocator().alloc([]const u8, entries.count());
        const deferred = try arena.allocator().alloc(bool, entries.count());
        var bytes: usize = 0;
        var any = false;
        for (entries.keys(), entries.values(), names, deferred) |key, e, *name, *d| {
            name.* = try std.mem.concat(arena.allocator(), u8, &.{ prefix, key });
            if (self.tensors.contains(name.*) or self.deferred.contains(name.*)) return error.DuplicateTensor;
            d.* = later(name.*, e.rank);
            any = any or d.*;
            if (!d.*) bytes = std.mem.alignForward(usize, bytes, alignment) + (e.end - e.begin);
        }
        var buffer: ?mtl.Buffer = null;
        if (bytes > 0) {
            const b = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            self.shards.append(self.allocator, .{ .buffer = b, .bytes = bytes }) catch |err| {
                b.deinit();
                return err;
            };
            buffer = b;
        }
        if (any) {
            try self.files.append(self.allocator, .{ .fd = fd });
            held.* = true;
        }
        var at: usize = 0;
        for (entries.values(), names, deferred) |e, name, d| {
            var shape: [st.max_rank]usize = @splat(1);
            for (0..e.rank) |i| shape[i] = e.shape[i];
            const n = e.end - e.begin;
            if (!d) {
                at = std.mem.alignForward(usize, at, alignment);
                try readInto(fd, buffer.?.contents()[at..][0..n], data_start + e.begin);
            }
            const key = try self.allocator.dupe(u8, name);
            errdefer self.allocator.free(key);
            if (d) {
                const file = &self.files.items[self.files.items.len - 1];
                file.end = @max(file.end, data_start + e.end);
                try self.deferred.put(self.allocator, key, .{ .file = self.files.items.len - 1, .at = data_start + e.begin, .bytes = n, .dtype = e.dtype, .shape = shape, .rank = e.rank });
            } else {
                try self.tensors.put(self.allocator, key, .{ .buffer = buffer.?, .offset = at, .bytes = n, .dtype = e.dtype, .shape = shape, .rank = e.rank });
                at += n;
            }
        }
    }

    /// Bytes [from, from + dest.len) of the deferred tensor `name`, from its file.
    pub fn read(self: *const Checkpoint, name: []const u8, from: usize, dest: []u8) !void {
        const d = self.deferred.get(name) orelse return error.MissingTensor;
        if (from > d.bytes or dest.len > d.bytes - from) return error.ShortRead;
        const fd = self.files.items[d.file].fd;
        if (fd < 0) return error.ShardClosed;
        try readInto(fd, dest, d.at + from);
    }

    /// Deferred `name`'s address-only view: its length, unique while the checkpoint lives, unreadable (no memory behind it).
    pub fn view(self: *Checkpoint, name: []const u8) ![]const u8 {
        const d = self.deferred.get(name) orelse return error.MissingTensor;
        const file = &self.files.items[d.file];
        const span = file.views orelse try std.posix.mmap(null, file.end, .{}, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        file.views = span;
        return span[d.at..][0..d.bytes];
    }

    /// Close the files the deferred tensors live in, once load has read them.
    pub fn closeFiles(self: *Checkpoint) void {
        for (self.files.items) |*file| if (file.fd >= 0) {
            _ = platform.c.close(file.fd);
            file.fd = -1;
        };
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
            if (self.deferred.contains(name)) {
                std.log.warn("checkpoint tensor {s} stays in its file: read it through a Stage", .{name});
                return error.DeferredTensor;
            }
            std.log.err("checkpoint has no tensor {s}", .{name});
            return error.MissingTensor;
        };
    }

    pub fn has(self: *const Checkpoint, name: []const u8) bool {
        return self.tensors.contains(name) or self.deferred.contains(name);
    }

    /// Hold a buffer made after loading as a shard: the checkpoint frees it.
    pub fn own(self: *Checkpoint, buffer: mtl.Buffer, bytes: usize) !void {
        try self.shards.append(self.allocator, .{ .buffer = buffer, .bytes = bytes });
    }

    /// Name `t` (in a shard or an owned buffer), in place of any tensor already of that name.
    pub fn set(self: *Checkpoint, name: []const u8, t: Tensor) !void {
        if (self.tensors.getPtr(name)) |old| {
            old.* = t;
            return;
        }
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);
        try self.tensors.put(self.allocator, key, t);
    }

    /// Bytes held in shard buffers.
    pub fn residentBytes(self: *const Checkpoint) usize {
        var total: usize = 0;
        for (self.shards.items) |s| total += s.bytes;
        return total;
    }

    pub fn deinit(self: *Checkpoint) void {
        self.closeFiles();
        for (self.files.items) |file| if (file.views) |span| std.posix.munmap(span);
        self.files.deinit(self.allocator);
        var dk = self.deferred.keyIterator();
        while (dk.next()) |k| self.allocator.free(k.*);
        self.deferred.deinit(self.allocator);
        var it = self.tensors.keyIterator();
        while (it.next()) |k| self.allocator.free(k.*);
        self.tensors.deinit(self.allocator);
        for (self.shards.items) |s| s.buffer.deinit();
        self.shards.deinit(self.allocator);
    }
};

/// A repack's source: deferred tensors read into one reused shared buffer, each view valid until `reset`.
pub const Stage = struct {
    device: mtl.Device,
    buffer: ?mtl.Buffer = null,
    used: usize = 0,

    /// Room for the deferred tensors among `names` (a kept one is read where it is), reserved while the stage is empty.
    pub fn reserve(s: *Stage, ck: *const Checkpoint, names: []const []const u8) !void {
        if (s.used != 0) return error.StageInUse;
        var bytes: usize = 0;
        for (names) |name| {
            if (ck.deferred.get(name)) |d| bytes = std.mem.alignForward(usize, bytes, alignment) + d.bytes else if (!ck.tensors.contains(name)) return error.MissingTensor;
        }
        if (s.buffer) |b| {
            if (b.length() >= bytes) return;
            b.deinit();
            s.buffer = null;
        }
        s.buffer = try s.device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    }

    /// `name`: a kept tensor where it is, a deferred one read into the stage.
    pub fn read(s: *Stage, ck: *const Checkpoint, name: []const u8) !Tensor {
        const d = ck.deferred.get(name) orelse return ck.get(name);
        const buffer = s.buffer orelse return error.StageFull;
        const at = std.mem.alignForward(usize, s.used, alignment);
        if (at + d.bytes > buffer.length()) return error.StageFull;
        try ck.read(name, 0, buffer.contents()[at..][0..d.bytes]);
        s.used = at + d.bytes;
        return .{ .buffer = buffer, .offset = at, .bytes = d.bytes, .dtype = d.dtype, .shape = d.shape, .rank = d.rank };
    }

    pub fn reset(s: *Stage) void {
        s.used = 0;
    }

    pub fn deinit(s: *Stage) void {
        if (s.buffer) |b| b.deinit();
        s.buffer = null;
        s.used = 0;
    }
};

fn readAll(fd: platform.c.fd_t, dest: []u8, at: usize) !void {
    var done: usize = 0;
    while (done < dest.len) {
        const n = platform.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

const Part = struct { fd: platform.c.fd_t, dest: []u8, at: usize, failed: bool = false };

fn readPart(part: *Part) void {
    readAll(part.fd, part.dest, part.at) catch {
        part.failed = true;
    };
}

/// pread `dest.len` bytes from `at` on 8 threads (page cache or disk, the copy is the load's main cost).
fn readParallel(fd: platform.c.fd_t, dest: []u8, at: usize) !void {
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

/// One tensor's bytes: small ones on this thread, large ones on 8.
fn readInto(fd: platform.c.fd_t, dest: []u8, at: usize) !void {
    return if (dest.len >= parallel_bytes) readParallel(fd, dest, at) else readAll(fd, dest, at);
}

/// A three-tensor safetensors file in `tmp`, each data byte its own offset; returns its path.
fn fixture(gpa: std.mem.Allocator, tmp: *std.testing.TmpDir) ![:0]u8 {
    const header =
        \\{"padding":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},
        \\ "text.weight":{"dtype":"U32","shape":[2,3],"data_offsets":[4,28]},
        \\ "text.scales":{"dtype":"BF16","shape":[1],"data_offsets":[28,30]}}
    ;
    var data: [8 + header.len + 30]u8 = undefined;
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    for (data[8 + header.len ..], 0..) |*b, i| b.* = @intCast(i);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "fixture.safetensors", .data = &data });
    return std.fmt.allocPrintSentinel(gpa, ".zig-cache/tmp/{s}/fixture.safetensors", .{tmp.sub_path}, 0);
}

test "full and selected checkpoint views retain data, ownership and relative offsets" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try fixture(gpa, &tmp);
    defer gpa.free(path);
    var full = Checkpoint.init(gpa);
    defer full.deinit();
    try full.addFile(device, path, "");
    var selected = Checkpoint.init(gpa);
    defer selected.deinit();
    try selected.addFileSelected(device, path, "", "text.");
    try std.testing.expectEqual(@as(usize, 30), full.residentBytes());
    try std.testing.expectEqual(@as(usize, 26), selected.residentBytes());
    const a = try full.get("text.weight");
    const b = try selected.get("text.weight");
    try std.testing.expectEqual(@as(usize, 4), a.offset);
    try std.testing.expectEqual(@as(usize, 0), b.offset);
    try std.testing.expectEqualSlices(u32, a.host(u32), b.host(u32));
    try std.testing.expectEqualSlices(u16, (try full.get("text.scales")).host(u16), (try selected.get("text.scales")).host(u16));
    try std.testing.expect(!selected.has("padding"));
    try std.testing.expectError(error.DuplicateTensor, selected.addFileSelected(device, path, "", "text."));
    try selected.addFileSelected(device, path, "", "absent.");
}

test "a split checkpoint keeps later tensors in the file, packs the rest, and stages the same bytes" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try fixture(gpa, &tmp);
    defer gpa.free(path);
    var full = Checkpoint.init(gpa);
    defer full.deinit();
    try full.addFile(device, path, "");
    const Weights = struct {
        fn later(name: []const u8, rank: usize) bool {
            return rank == 2 and std.mem.endsWith(u8, name, ".weight");
        }
    };
    var split = Checkpoint.init(gpa);
    defer split.deinit();
    try split.addFileSplit(device, path, "", "text.", Weights.later);
    try std.testing.expectEqual(@as(usize, 2), split.residentBytes());
    try std.testing.expect(split.has("text.weight"));
    try std.testing.expectError(error.DeferredTensor, split.get("text.weight"));
    try std.testing.expectEqualSlices(u16, (try full.get("text.scales")).host(u16), (try split.get("text.scales")).host(u16));
    var stage = Stage{ .device = device };
    defer stage.deinit();
    try stage.reserve(&split, &.{ "text.weight", "text.scales" });
    const w = try stage.read(&split, "text.weight");
    try std.testing.expectEqualSlices(u32, (try full.get("text.weight")).host(u32), w.host(u32));
    try std.testing.expectEqual((try split.get("text.scales")).buffer.id, (try stage.read(&split, "text.scales")).buffer.id);
    try std.testing.expectError(error.StageInUse, stage.reserve(&split, &.{"text.weight"}));
    var part: [8]u8 = undefined;
    try split.read("text.weight", 4, &part);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes((try full.get("text.weight")).host(u32))[4..12], &part);
    try std.testing.expectError(error.ShortRead, split.read("text.weight", 20, &part));
    const v = try split.view("text.weight");
    try std.testing.expectEqual(@as(usize, 24), v.len);
    try std.testing.expect(v.ptr == (try split.view("text.weight")).ptr);
    split.closeFiles();
    try std.testing.expectError(error.ShardClosed, split.read("text.weight", 0, &part));
}
