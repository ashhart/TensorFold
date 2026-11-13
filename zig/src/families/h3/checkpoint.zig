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
        defer _ = Ops.close(fd);
        const size = Ops.lseek(fd, 0, 2);
        if (size < 16) return error.CannotOpen;
        const len = std.math.cast(usize, size) orelse return error.CheckpointTooLarge;
        const at = Ops.mmap(null, len, 1, 2, fd, 0) orelse return error.CannotMap;
        if (@intFromPtr(at) == std.math.maxInt(usize)) return error.CannotMap;
        errdefer _ = Ops.munmap(at, len);
        const bytes = @as([*]const u8, @ptrCast(at))[0..len];
        const n = st.headerSize(std.mem.readInt(u64, bytes[0..8], .little), len) catch return error.BadHeader;
        var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
        errdefer arena.deinit();
        const names = st.parseHeader(arena.allocator(), bytes[8..][0..n], len - 8 - n) catch return error.BadHeader;
        return .{ .bytes = bytes, .header = bytes[8 .. 8 + n], .data = 8 + n, .names = names, .arena = arena };
    }

    pub fn deinit(self: *Checkpoint) void {
        self.arena.deinit();
        _ = c.munmap(@ptrCast(@constCast(self.bytes.ptr)), self.bytes.len);
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
    const ck: Checkpoint = .{ .bytes = &bytes, .data = std.math.maxInt(usize), .header = header, .names = try indexOf(arena.allocator(), header, 2), .arena = .init(std.testing.allocator) };
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
