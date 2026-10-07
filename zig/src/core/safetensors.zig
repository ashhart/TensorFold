//! safetensors: the JSON header's names, dtypes, shapes and byte ranges (every backend's index), and read-only maps.

const std = @import("std");
const Io = std.Io;

pub const DType = enum {
    bool,
    u8,
    i8,
    u16,
    i16,
    f16,
    bf16,
    u32,
    i32,
    f32,
    u64,
    i64,
    f64,

    pub fn size(self: DType) usize {
        return switch (self) {
            .bool, .u8, .i8 => 1,
            .u16, .i16, .f16, .bf16 => 2,
            .u32, .i32, .f32 => 4,
            .u64, .i64, .f64 => 8,
        };
    }

    pub fn parse(text: []const u8) ?DType {
        const names = .{ .{ "BOOL", .bool }, .{ "U8", .u8 }, .{ "I8", .i8 }, .{ "U16", .u16 }, .{ "I16", .i16 }, .{ "F16", .f16 }, .{ "BF16", .bf16 }, .{ "U32", .u32 }, .{ "I32", .i32 }, .{ "F32", .f32 }, .{ "U64", .u64 }, .{ "I64", .i64 }, .{ "F64", .f64 } };
        inline for (names) |n| if (std.mem.eql(u8, text, n[0])) return n[1];
        return null;
    }
};

pub const max_rank = 4;

/// A tensor's header entry: `begin` and `end` count from the data region's start (8 + the header's length).
pub const Entry = struct {
    dtype: DType,
    rank: u8,
    shape: [max_rank]usize,
    begin: usize,
    end: usize,

    pub fn dim(self: Entry, i: usize) usize {
        return if (i < self.rank) self.shape[i] else 1;
    }
};

pub const Header = std.StringArrayHashMapUnmanaged(Entry);

/// The header's entries (names live in `arena`); an entry that breaks the format, runs past `data_len` bytes or has the wrong size is refused.
pub fn parseHeader(arena: std.mem.Allocator, json: []const u8, data_len: usize) !Header {
    return parseHeaderPrefix(arena, json, data_len, null);
}

/// Select one tensor namespace before interpreting shapes, so a text engine need not admit a vision tower's layouts.
pub fn parseHeaderPrefix(arena: std.mem.Allocator, json: []const u8, data_len: usize, prefix: ?[]const u8) !Header {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{});
    if (parsed != .object) return error.BadSafetensors;
    var out: Header = .empty;
    var it = parsed.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
        if (prefix) |p| if (!std.mem.startsWith(u8, kv.key_ptr.*, p)) continue;
        if (kv.value_ptr.* != .object) return error.BadSafetensors;
        const o = kv.value_ptr.object;
        const dtype = DType.parse(str(o.get("dtype")) orelse return error.BadSafetensors) orelse return error.UnsupportedDType;
        const shape = list(o.get("shape")) orelse return error.BadSafetensors;
        if (shape.len > max_rank) return error.RankTooHigh;
        var e: Entry = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .begin = 0, .end = 0 };
        var n: usize = dtype.size();
        for (shape, 0..) |d, i| {
            e.shape[i] = uint(d) orelse return error.BadSafetensors;
            n = std.math.mul(usize, n, e.shape[i]) catch return error.BadSafetensors;
        }
        const offs = list(o.get("data_offsets")) orelse return error.BadSafetensors;
        if (offs.len != 2) return error.BadSafetensors;
        e.begin = uint(offs[0]) orelse return error.BadSafetensors;
        e.end = uint(offs[1]) orelse return error.BadSafetensors;
        if (e.end < e.begin or e.end - e.begin != n or e.end > data_len) return error.BadSafetensors;
        try out.put(arena, kv.key_ptr.*, e);
    }
    return out;
}

fn str(v: ?std.json.Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

fn list(v: ?std.json.Value) ?[]const std.json.Value {
    const x = v orelse return null;
    return if (x == .array) x.array.items else null;
}

/// A non-negative JSON integer, as `zig/src/cluster/checkpoint.zig` reads one; one past i64 arrives as a number string.
fn uint(v: std.json.Value) ?usize {
    return switch (v) {
        .integer => |i| std.math.cast(usize, i),
        .number_string => |s| std.fmt.parseInt(usize, s, 10) catch null,
        else => null,
    };
}

/// One tensor's bytes in a mapped file (they live as long as the file).
pub const Tensor = struct {
    dtype: DType,
    rank: u8,
    shape: [max_rank]usize,
    bytes: []const u8,

    pub fn dim(self: Tensor, i: usize) usize {
        return if (i < self.rank) self.shape[i] else 1;
    }

    pub fn numel(self: Tensor) usize {
        var n: usize = 1;
        for (self.shape[0..self.rank]) |d| n *= d;
        return n;
    }

    pub fn is(self: Tensor, dtype: DType, shape: []const usize) bool {
        return self.dtype == dtype and self.rank == shape.len and std.mem.eql(usize, self.shape[0..self.rank], shape);
    }
};

/// A file mapped read-only with its header indexed.
pub const File = struct {
    file: Io.File,
    map: Io.File.MemoryMap,
    data: usize,
    names: Header,
    arena: std.heap.ArenaAllocator,

    pub fn open(gpa: std.mem.Allocator, io: Io, path: []const u8) !File {
        var file = try Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const len: usize = @intCast(try file.length(io));
        if (len < 8) return error.BadSafetensors;
        var map = try Io.File.MemoryMap.create(io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
        errdefer map.destroy(io);
        const header_len: usize = @intCast(std.mem.readInt(u64, map.memory[0..8], .little));
        if (header_len > len - 8) return error.BadSafetensors;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const names = try parseHeader(arena.allocator(), map.memory[8..][0..header_len], len - 8 - header_len);
        return .{ .file = file, .map = map, .data = 8 + header_len, .names = names, .arena = arena };
    }

    pub fn close(self: *File, io: Io) void {
        self.map.destroy(io);
        self.file.close(io);
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn get(self: *const File, name: []const u8) ?Tensor {
        const e = self.names.get(name) orelse return null;
        return .{ .dtype = e.dtype, .rank = e.rank, .shape = e.shape, .bytes = self.map.memory[self.data + e.begin .. self.data + e.end] };
    }
};

test "header entries" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"__metadata__": {"format": "mlx"}, "a.weight": {"dtype": "U32", "shape": [2, 3], "data_offsets": [0, 24]},
        \\ "a.scales": {"dtype": "BF16", "shape": [2], "data_offsets": [24, 28]}}
    ;
    const h = try parseHeader(arena.allocator(), json, 28);
    try std.testing.expectEqual(@as(usize, 2), h.count());
    try std.testing.expectEqual(DType.bf16, h.get("a.scales").?.dtype);
    try std.testing.expectError(error.BadSafetensors, parseHeader(arena.allocator(), json, 20));
}

test "namespace selection admits text without interpreting vision layouts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"vision.weight":{"dtype":"BF16","shape":[1,1,1,1,1],"data_offsets":[0,2]},
        \\ "text.weight":{"dtype":"U32","shape":[2,3],"data_offsets":[2,26]}}
    ;
    try std.testing.expectError(error.RankTooHigh, parseHeader(arena.allocator(), json, 26));
    const h = try parseHeaderPrefix(arena.allocator(), json, 26, "text.");
    try std.testing.expectEqual(@as(usize, 1), h.count());
    try std.testing.expectEqual(@as(usize, 2), h.get("text.weight").?.begin);
    try std.testing.expectError(error.BadSafetensors, parseHeaderPrefix(arena.allocator(), json, 25, "text."));
    const empty = try parseHeaderPrefix(arena.allocator(), json, 26, "missing.");
    try std.testing.expectEqual(@as(usize, 0), empty.count());
}

test "a header that breaks the format, or whose byte count overflows, is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const bad = [_][]const u8{
        // 2^62 rows of 4 bytes is 2^64 bytes, which wraps to 0 in a usize: the size of the empty range [0, 0]
        "{\"t\": {\"dtype\": \"U8\", \"shape\": [4611686018427387904, 4], \"data_offsets\": [0, 0]}}",
        "[1]",
        "{\"t\": 5}",
        "{\"t\": {\"shape\": [1], \"data_offsets\": [0, 1]}}",
        "{\"t\": {\"dtype\": 5, \"shape\": [1], \"data_offsets\": [0, 1]}}",
        "{\"t\": {\"dtype\": \"U8\", \"shape\": [-1], \"data_offsets\": [0, 1]}}",
        "{\"t\": {\"dtype\": \"U8\", \"shape\": [1.0], \"data_offsets\": [0, 1]}}",
        "{\"t\": {\"dtype\": \"U8\", \"shape\": [\"1\"], \"data_offsets\": [0, 1]}}",
        "{\"t\": {\"dtype\": \"U8\", \"shape\": [1]}}",
        "{\"t\": {\"dtype\": \"U8\", \"shape\": [1], \"data_offsets\": [0]}}",
        "{\"t\": {\"dtype\": \"U8\", \"shape\": [1], \"data_offsets\": [0, 1, 2]}}",
        "{\"t\": {\"dtype\": \"U8\", \"shape\": [1], \"data_offsets\": [\"0\", \"1\"]}}",
    };
    for (bad) |json| try std.testing.expectError(error.BadSafetensors, parseHeader(arena.allocator(), json, 16));
}
