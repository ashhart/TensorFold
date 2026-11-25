//! One safetensors file mapped read-only, its tensors found by name in core's index of the header.
const std = @import("std");
const report = @import("report.zig");
const st = @import("core").safetensors;

const c = struct {
    extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
    extern "c" fn close(fd: c_int) c_int;
    extern "c" fn lseek(fd: c_int, offset: i64, whence: c_int) i64;
    extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque;
    extern "c" fn munmap(addr: ?*anyopaque, len: usize) c_int;
    extern "c" fn madvise(addr: ?*anyopaque, len: usize, advice: c_int) c_int;
    extern "c" fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) c_int;
};

/// Where a tensor sits in the data that follows the header.
pub const Span = struct { rows: usize, cols: usize, start: usize, end: usize };

/// The bf16 tensor `name` in core's index of a header: a vector counts as one row.
pub fn locate(names: st.Header, name: []const u8) !Span {
    const e = names.get(name) orelse return error.MissingTensor;
    if (e.dtype != .bf16) return error.WrongDType;
    if (e.rank == 0 or e.rank > 2) return error.BadHeader;
    const rows = if (e.rank == 2) e.shape[0] else 1;
    const cols = e.shape[e.rank - 1];
    if (rows == 0 or cols == 0) return error.BadHeader;
    return .{ .rows = rows, .cols = cols, .start = e.begin, .end = e.end };
}

pub const Checkpoint = struct {
    bytes: []const u8,
    /// Open until `deinit`, so `evict` can drop the file's pages from the page cache.
    fd: c_int,
    header: []const u8,
    data: usize,
    names: st.Header,
    arena: std.heap.ArenaAllocator,

    pub const Entry = struct { rows: usize, cols: usize, bytes: []const u8 };

    pub fn open(path: [*:0]const u8) !Checkpoint {
        return openWith(path, c);
    }

    fn openWith(path: [*:0]const u8, comptime Ops: type) !Checkpoint {
        const fd = Ops.open(path, 0);
        if (fd < 0) return error.CannotOpen;
        errdefer _ = Ops.close(fd);
        const size = Ops.lseek(fd, 0, 2);
        if (size < 16) return error.CannotOpen;
        const len = std.math.cast(usize, size) orelse return error.CheckpointTooLarge;
        const mapped = Ops.mmap(null, len, 1, 2, fd, 0) orelse return error.CannotMap;
        if (@intFromPtr(mapped) == std.math.maxInt(usize)) return error.CannotMap;
        errdefer _ = Ops.munmap(mapped, len);
        const bytes = @as([*]const u8, @ptrCast(mapped))[0..len];
        const n = st.headerSize(std.mem.readInt(u64, bytes[0..8], .little), len) catch return error.BadHeader;
        var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
        errdefer arena.deinit();
        const names = st.parseHeader(arena.allocator(), bytes[8..][0..n], len - 8 - n) catch return error.BadHeader;
        return .{ .bytes = bytes, .fd = fd, .header = bytes[8 .. 8 + n], .data = 8 + n, .names = names, .arena = arena };
    }

    pub fn deinit(self: *Checkpoint) void {
        self.deinitWith(c);
    }

    fn deinitWith(self: *Checkpoint, comptime Ops: type) void {
        self.arena.deinit();
        _ = Ops.munmap(@ptrCast(@constCast(self.bytes.ptr)), self.bytes.len);
        _ = Ops.close(self.fd);
    }

    /// Discards the pages from the mapping and page cache; faulted back from disk on a later read. Linux only.
    pub fn evict(self: *const Checkpoint) void {
        if (@import("builtin").os.tag != .linux) return;
        _ = c.madvise(@ptrCast(@constCast(self.bytes.ptr)), self.bytes.len, 4); // MADV_DONTNEED
        _ = c.posix_fadvise(self.fd, 0, 0, 4); // POSIX_FADV_DONTNEED: a cached copy would crowd out the prompt's
    }

    /// The tensor `name` of `dtype` and `count` values: its bytes as stored.
    pub fn at(self: *const Checkpoint, name: []const u8, dtype: st.DType, count: usize) ![]const u8 {
        const e = self.names.get(name) orelse {
            report.say("no tensor {s}", .{name});
            return error.MissingTensor;
        };
        const size = std.math.mul(usize, count, dtype.size()) catch return error.BadTensor;
        if (e.dtype != dtype or e.end - e.begin != size) {
            report.say("tensor {s}: unexpected type or size", .{name});
            return error.BadTensor;
        }
        if (self.data > self.bytes.len or e.end > self.bytes.len - self.data) return error.BadHeader;
        return self.bytes[self.data + e.begin ..][0 .. e.end - e.begin];
    }

    /// The tensor `name` (float32 or bf16) as float32 values on the host.
    pub fn floats(self: *const Checkpoint, a: std.mem.Allocator, name: []const u8) ![]f32 {
        const e = self.names.get(name) orelse {
            report.say("no tensor {s}", .{name});
            return error.MissingTensor;
        };
        if (self.data > self.bytes.len or e.end > self.bytes.len - self.data) return error.BadHeader;
        const from = self.bytes[self.data + e.begin ..];
        const out = try a.alloc(f32, (e.end - e.begin) / e.dtype.size());
        for (out, 0..) |*o, i| o.* = switch (e.dtype) {
            .f32 => @bitCast(std.mem.readInt(u32, from[4 * i ..][0..4], .little)),
            .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, from[2 * i ..][0..2], .little)) << 16),
            else => return error.BadTensor,
        };
        return out;
    }

    /// A one-file view through the sharded interface; closed by this checkpoint's `deinit`.
    pub fn shards(self: *Checkpoint) Shards {
        return .{ .files = self[0..1], .owned = false };
    }

    pub fn bf16(self: *const Checkpoint, name: []const u8) !Entry {
        const span = locate(self.names, name) catch |e| {
            report.say("tensor {s}: {t}", .{ name, e });
            return e;
        };
        if (self.data > self.bytes.len or span.end > self.bytes.len - self.data) return error.BadHeader;
        return .{ .rows = span.rows, .cols = span.cols, .bytes = self.bytes[self.data + span.start .. self.data + span.end] };
    }
};

/// The block count advertised in one header: null when it names no block, an error when it names them badly.
pub fn blockCount(a: std.mem.Allocator, header: []const u8) !?usize {
    const parsed = try std.json.parseFromSlice(std.json.Value, a, header, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadBlocks;
    const obj = parsed.value.object;
    var count: ?usize = null;
    for (obj.keys()) |key| {
        if (!std.mem.startsWith(u8, key, "blocks.")) continue;
        const rest = key[7..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.BadBlocks;
        const id = std.fmt.parseInt(usize, rest[0..dot], 10) catch return error.BadBlocks;
        if (dot == 0 or (dot > 1 and rest[0] == '0')) return error.BadBlocks;
        count = @max(count orelse 0, std.math.add(usize, id, 1) catch return error.BadBlocks);
    }
    return count;
}

/// `fmt` of `args` as a C path in `buf`, its terminator inside the buffer.
fn cPath(buf: *[1024]u8, comptime fmt: []const u8, args: anytype) ![*:0]const u8 {
    const text = std.fmt.bufPrint(buf[0 .. buf.len - 1], fmt, args) catch return error.NameTooLong;
    buf[text.len] = 0;
    return @ptrCast(text.ptr);
}

/// A checkpoint in shards, each mapped once; a tensor is found in whichever holds it.
pub const Shards = struct {
    files: []Checkpoint,
    /// `open` allocated the list; a single file's `shards()` view borrows it.
    owned: bool = true,

    /// `dir`'s `count` shards named `{stem}-{i:05}-of-{count:05}.safetensors`, both released and diffusers stems.
    pub fn open(gpa: std.mem.Allocator, dir: []const u8, count: usize) !Shards {
        if (count == 0 or dir.len == 0) return error.BadShardIndex;
        const files = try gpa.alloc(Checkpoint, count);
        var made: usize = 0;
        errdefer {
            for (files[0..made]) |*file| file.deinit();
            gpa.free(files);
        }
        const shard = "{s}/{s}-{d:0>5}-of-{d:0>5}.safetensors";
        var path: [1024]u8 = undefined;
        var stem: []const u8 = "model";
        for (files, 1..) |*file, i| {
            file.* = Checkpoint.open(try cPath(&path, shard, .{ dir, stem, i, count })) catch |err| first: {
                if (i > 1 or std.mem.eql(u8, stem, "diffusion_pytorch_model")) return err;
                stem = "diffusion_pytorch_model";
                break :first try Checkpoint.open(try cPath(&path, shard, .{ dir, stem, i, count }));
            };
            made += 1;
        }
        return .{ .files = files };
    }

    /// Unmaps every shard and frees the list when `open` allocated it.
    pub fn deinit(self: *Shards, gpa: std.mem.Allocator) void {
        for (self.files) |*file| file.deinit();
        if (self.owned) gpa.free(self.files);
        self.files = &.{};
    }

    /// Every shard's `evict`: once the weights sit on the device, a resident process pins no host copy of them.
    pub fn evict(self: *const Shards) void {
        for (self.files) |*file| file.evict();
    }

    pub fn entry(self: *const Shards, name: []const u8) ?st.Entry {
        for (self.files) |*file| if (file.names.get(name)) |e| return e;
        return null;
    }

    /// The bf16 tensor `name` in whichever shard holds it.
    pub fn bf16(self: *const Shards, name: []const u8) !Checkpoint.Entry {
        for (self.files) |*file| if (file.names.get(name) != null) return file.bf16(name);
        report.say("tensor {s}: {t}", .{ name, error.MissingTensor });
        return error.MissingTensor;
    }

    /// The tensor `name` of `dtype` and `count` values: its bytes as stored, in whichever shard holds it.
    pub fn at(self: *const Shards, name: []const u8, dtype: st.DType, count: usize) ![]const u8 {
        for (self.files) |*file| if (file.names.get(name) != null) return file.at(name, dtype, count);
        report.say("no tensor {s}", .{name});
        return error.MissingTensor;
    }

    /// The tensor `name` (float32 or bf16) as float32 values, in whichever shard holds it.
    pub fn floats(self: *const Shards, a: std.mem.Allocator, name: []const u8) ![]f32 {
        for (self.files) |*file| if (file.names.get(name) != null) return file.floats(a, name);
        report.say("no tensor {s}", .{name});
        return error.MissingTensor;
    }

    /// The deepest block index across every shard's header; a shard may hold no block tensors.
    pub fn layers(self: *const Shards, a: std.mem.Allocator) !usize {
        var last: ?usize = null;
        for (self.files) |*file| {
            if (try blockCount(a, file.header)) |n| last = @max(last orelse 0, n);
        }
        return last orelse error.BadBlocks;
    }
};

/// A checkpoint folder whose shards are mapped on first use, each once, until `deinit`.
pub const Folder = FolderWith(c);

fn FolderWith(comptime Ops: type) type {
    return struct {
        const Self = @This();

        dir: []const u8,
        /// Shard file name to its mapping; the names outlive the folder.
        files: std.StringArrayHashMapUnmanaged(Checkpoint) = .empty,

        /// The shard `file`, mapped on its first call; a failed open leaves nothing behind.
        pub fn shard(self: *Self, gpa: std.mem.Allocator, file: []const u8) !*Checkpoint {
            const gop = try self.files.getOrPut(gpa, file);
            if (gop.found_existing) return gop.value_ptr;
            errdefer self.files.swapRemoveAt(gop.index);
            var path: [1024]u8 = undefined;
            gop.value_ptr.* = try Checkpoint.openWith(try cPath(&path, "{s}/{s}", .{ self.dir, file }), Ops);
            return gop.value_ptr;
        }

        pub fn deinit(self: *Self, gpa: std.mem.Allocator) void {
            for (self.files.values()) |*file| file.deinitWith(Ops);
            self.files.deinit(gpa);
        }
    };
}

/// core's index of `header`, for the tests.
fn indexOf(a: std.mem.Allocator, header: []const u8, data_len: usize) !st.Header {
    return st.parseHeader(a, header, data_len);
}

test "a matrix and a vector are found by name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const header = try indexOf(arena.allocator(), "{\"a.weight\":{\"dtype\":\"BF16\",\"shape\":[3,4],\"data_offsets\":[8,32]},\"a.norm\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,8]}}", 32);
    const w = try locate(header, "a.weight");
    try std.testing.expectEqual(Span{ .rows = 3, .cols = 4, .start = 8, .end = 32 }, w);
    const n = try locate(header, "a.norm");
    try std.testing.expectEqual(Span{ .rows = 1, .cols = 4, .start = 0, .end = 8 }, n);
}

test "a missing name, another dtype and a size that does not match are refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const header = try indexOf(arena.allocator(), "{\"f.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"s.weight\":{\"dtype\":\"BF16\",\"shape\":[2,2,2],\"data_offsets\":[16,32]}}", 32);
    try std.testing.expectError(error.MissingTensor, locate(header, "x.weight"));
    try std.testing.expectError(error.WrongDType, locate(header, "f.weight"));
    try std.testing.expectError(error.BadHeader, locate(header, "s.weight"));
    try std.testing.expectError(error.BadSafetensors, indexOf(arena.allocator(), "{\"b.weight\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,6]}}", 16));
}

test "h3 ownership: mapped checkpoint is unmapped after header rejection" {
    const Map = struct {
        var bytes: [24]u8 = @splat(0);
        var unmapped: usize = 0;
        fn open(_: [*:0]const u8, _: c_int) c_int {
            return 1;
        }
        fn close(_: c_int) c_int {
            return 0;
        }
        fn lseek(_: c_int, _: i64, _: c_int) i64 {
            return bytes.len;
        }
        fn mmap(_: ?*anyopaque, _: usize, _: c_int, _: c_int, _: c_int, _: i64) ?*anyopaque {
            return &bytes;
        }
        fn munmap(_: ?*anyopaque, _: usize) c_int {
            unmapped += 1;
            return 0;
        }
    };
    Map.unmapped = 0;
    std.mem.writeInt(u64, Map.bytes[0..8], 64, .little);
    try std.testing.expectError(error.BadHeader, Checkpoint.openWith("fake", Map));
    try std.testing.expectEqual(@as(usize, 1), Map.unmapped);
}

test "h3 arithmetic: tensor sizes and offsets reject overflow" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.BadSafetensors, indexOf(arena.allocator(), "{\"a\":{\"dtype\":\"BF16\",\"shape\":[18446744073709551615,2],\"data_offsets\":[0,0]}}", 0));
    var bytes: [16]u8 = @splat(0);
    const header = "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    const ck: Checkpoint = .{ .bytes = &bytes, .fd = -1, .data = std.math.maxInt(usize), .header = header, .names = try indexOf(arena.allocator(), header, 2), .arena = .init(std.testing.allocator) };
    try std.testing.expectError(error.BadHeader, ck.bf16("a"));
}

test "h3 arithmetic: header length refuses overflow" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var bytes: [24]u8 = @splat(0);
    std.mem.writeInt(u64, bytes[0..8], std.math.maxInt(u64), .little);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "bad.safetensors", .data = &bytes });
    const path = try tmp.dir.realPathFileAlloc(std.testing.io, "bad.safetensors", std.testing.allocator);
    defer std.testing.allocator.free(path);
    const z = try std.testing.allocator.dupeSentinel(u8, path, 0);
    defer std.testing.allocator.free(z);
    try std.testing.expectError(error.BadHeader, Checkpoint.open(z.ptr));
}

test "a tensor is found by core's index: not a dtype in another field, not a look-alike in metadata, spaced JSON read" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const typed = try indexOf(a, "{\"x\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2],\"n\":\"BF16\"}}", 2);
    try std.testing.expectError(error.WrongDType, locate(typed, "x"));
    const meta = try indexOf(a, "{\"__metadata__\":{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}},\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[2,4]}}", 4);
    try std.testing.expectEqual(Span{ .rows = 1, .cols = 1, .start = 2, .end = 4 }, try locate(meta, "a"));
    const spaced = try indexOf(a, "{\"a\": {\"dtype\": \"BF16\", \"shape\": [1], \"data_offsets\": [0, 2]}}", 2);
    try std.testing.expectEqual(Span{ .rows = 1, .cols = 1, .start = 0, .end = 2 }, try locate(spaced, "a"));
    for ([_][]const u8{ "{\"w\":{\"dtype\":\"BF16\",", "{\"w\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}x", "{\"w\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},\"w\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}" }) |bad|
        if (indexOf(a, bad, 2)) |_| return error.TestUnexpectedResult else |_| {};
}
fn safetensors(a: std.mem.Allocator, header: []const u8, data: []const u8) ![]u8 {
    const out = try a.alloc(u8, 8 + header.len + data.len);
    std.mem.writeInt(u64, out[0..8], header.len, .little);
    @memcpy(out[8..][0..header.len], header);
    @memcpy(out[8 + header.len ..], data);
    return out;
}

test "a sharded checkpoint finds a tensor in whichever shard holds it and counts blocks across the headers" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    // shard 1: norm1 only; shard 2: a bf16 matrix and an f32 vector
    const one = try safetensors(a, "{\"blocks.0.norm1.weight\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,8]}}", &.{ 0, 60, 0, 62, 0, 64, 0, 66 });
    const two = try safetensors(a, "{\"blocks.2.norm1.weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]},\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[4,12]}}", &.{ 0, 66, 0, 68, 0, 0, 128, 63, 0, 0, 0, 64 });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model-00001-of-00002.safetensors", .data = one });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model-00002-of-00002.safetensors", .data = two });
    var shards = try Shards.open(a, dir, 2);
    defer shards.deinit(a);
    try std.testing.expectEqual(@as(usize, 3), try shards.layers(a));
    const e = try shards.bf16("blocks.2.norm1.weight");
    try std.testing.expectEqual(@as(usize, 1), e.rows);
    try std.testing.expectEqualSlices(u8, &.{ 0, 66, 0, 68 }, e.bytes);
    try std.testing.expectError(error.MissingTensor, shards.bf16("none"));
    const f = try shards.floats(a, "w");
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0 }, f);
    const g = try shards.at("blocks.0.norm1.weight", .bf16, 4);
    try std.testing.expectEqualSlices(u8, &.{ 0, 60, 0, 62, 0, 64, 0, 66 }, g);
}

test "shard headers with no blocks, a gapped id and a bad id" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqual(@as(?usize, null), try blockCount(a, "{\"time_embedder.proj_in.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}"));
    try std.testing.expectEqual(@as(?usize, 2), try blockCount(a, "{\"blocks.0.a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},\"blocks.1.b\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}"));
    try std.testing.expectEqual(@as(?usize, 10), try blockCount(a, "{\"blocks.9.a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}"));
    try std.testing.expectError(error.BadBlocks, blockCount(a, "{\"blocks.01.a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},\"blocks.1.a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}"));
    try std.testing.expectError(error.BadShardIndex, Shards.open(a, "x", 0));
}

test "a count-checked fetch refuses a tensor shorter than asked, another dtype and a size that overflows" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var bytes: [8]u8 = @splat(0);
    const header = "{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
    const ck: Checkpoint = .{ .bytes = &bytes, .fd = -1, .data = 0, .header = header, .names = try indexOf(arena.allocator(), header, 8), .arena = .init(std.testing.allocator) };
    try std.testing.expectEqual(@as(usize, 8), (try ck.at("w", .f32, 2)).len);
    try std.testing.expectError(error.BadTensor, ck.at("w", .f32, 3));
    try std.testing.expectError(error.BadTensor, ck.at("w", .bf16, 4));
    try std.testing.expectError(error.BadTensor, ck.at("w", .f32, std.math.maxInt(usize) / 2 + 3));
    try std.testing.expectError(error.BadBlocks, blockCount(arena.allocator(), "{\"blocks.18446744073709551615.a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}"));
}

test "h3 ownership: a folder maps a shard once, retries a failed open and unmaps what it mapped" {
    const Count = struct {
        var maps: usize = 0;
        var unmaps: usize = 0;
        const open = c.open;
        const close = c.close;
        const lseek = c.lseek;
        fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque {
            maps += 1;
            return c.mmap(addr, len, prot, flags, fd, offset);
        }
        fn munmap(addr: ?*anyopaque, len: usize) c_int {
            unmaps += 1;
            return c.munmap(addr, len);
        }
    };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", std.testing.allocator);
    defer std.testing.allocator.free(dir);
    var folder: FolderWith(Count) = .{ .dir = dir };
    try std.testing.expectError(error.CannotOpen, folder.shard(std.testing.allocator, "a.safetensors"));
    try std.testing.expectEqual(@as(usize, 0), folder.files.count());
    const file = try safetensors(std.testing.allocator, "{\"w\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}", &.{ 0, 63 });
    defer std.testing.allocator.free(file);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "a.safetensors", .data = file });
    const first = try folder.shard(std.testing.allocator, "a.safetensors");
    try std.testing.expectEqual(first, try folder.shard(std.testing.allocator, "a.safetensors"));
    try std.testing.expectEqual(@as(usize, 1), Count.maps);
    folder.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), Count.unmaps);
}
