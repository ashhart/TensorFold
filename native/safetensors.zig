//! Bounded, validated safetensors reads. Tensor payloads stay on disk until requested.
const std = @import("std");
pub const DType = enum {
    BOOL,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    BF16,
    F32,
    F64,
    pub fn bytes(t: DType) u64 {
        return switch (t) {
            .BOOL, .U8, .I8 => 1,
            .U16, .I16, .F16, .BF16 => 2,
            .U32, .I32, .F32 => 4,
            .U64, .I64, .F64 => 8,
        };
    }
};
pub const Tensor = struct {
    dtype: DType,
    dims: [8]i32 = @splat(0),
    rank: usize,
    offset: u64,
    len: u64,
    pub fn shape(t: *const Tensor) []const i32 {
        return t.dims[0..t.rank];
    }
    pub fn rowBytes(t: Tensor) !usize {
        if (t.rank != 2 or t.dims[0] <= 0) return error.InvalidTensorShape;
        return @intCast(@as(u64, @intCast(t.dims[1])) * t.dtype.bytes());
    }
};
pub const Header = struct {
    parsed: std.json.Parsed(std.json.Value),
    tensors: std.StringHashMap(Tensor),
    pub fn deinit(h: *Header) void {
        h.tensors.deinit();
        h.parsed.deinit();
    }
    pub fn parse(a: std.mem.Allocator, json: []const u8, data_bytes: u64) !Header {
        var h = Header{
            .parsed = std.json.parseFromSlice(std.json.Value, a, json, .{ .allocate = .alloc_always }) catch |err| return if (err == error.OutOfMemory) err else error.InvalidSafetensorHeader,
            .tensors = std.StringHashMap(Tensor).init(a),
        };
        errdefer h.deinit();
        if (h.parsed.value != .object) return error.InvalidSafetensorHeader;
        var ranges: std.ArrayList(Tensor) = .empty;
        defer ranges.deinit(a);
        var it = h.parsed.value.object.iterator();
        while (it.next()) |entry| {
            const v = entry.value_ptr.*;
            if (v != .object) return error.InvalidSafetensorHeader;
            if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) {
                for (v.object.values()) |value| if (value != .string) return error.InvalidSafetensorHeader;
                continue;
            }
            const dt = v.object.get("dtype") orelse return error.InvalidSafetensorHeader;
            const dims = v.object.get("shape") orelse return error.InvalidSafetensorHeader;
            const offsets = v.object.get("data_offsets") orelse return error.InvalidSafetensorHeader;
            if (dt != .string or dims != .array or offsets != .array or offsets.array.items.len != 2 or dims.array.items.len > 8) return error.InvalidSafetensorHeader;
            var tensor = Tensor{ .dtype = std.meta.stringToEnum(DType, dt.string) orelse return error.InvalidTensorDType, .rank = dims.array.items.len, .offset = 0, .len = 0 };
            var size: u64 = tensor.dtype.bytes();
            for (dims.array.items, 0..) |d, i| {
                if (d != .integer or d.integer < 0 or d.integer > std.math.maxInt(i32)) return error.InvalidTensorShape;
                tensor.dims[i] = @intCast(d.integer);
                size = std.math.mul(u64, size, @intCast(d.integer)) catch return error.InvalidTensorShape;
            }
            const from = offsets.array.items[0];
            const to = offsets.array.items[1];
            if (from != .integer or to != .integer or from.integer < 0 or to.integer < from.integer) return error.InvalidTensorOffsets;
            tensor.offset = @intCast(from.integer);
            tensor.len = @intCast(to.integer - from.integer);
            if (tensor.len != size or @as(u64, @intCast(to.integer)) > data_bytes) return error.InvalidTensorOffsets;
            try h.tensors.put(entry.key_ptr.*, tensor);
            try ranges.append(a, tensor);
        }
        std.mem.sort(Tensor, ranges.items, {}, struct {
            fn less(_: void, x: Tensor, y: Tensor) bool {
                return if (x.offset == y.offset) x.len < y.len else x.offset < y.offset;
            }
        }.less);
        var end: u64 = 0;
        for (ranges.items) |tensor| {
            if (tensor.offset != end) return error.InvalidTensorOffsets;
            end += tensor.len;
        }
        if (end != data_bytes) return error.InvalidTensorOffsets;
        return h;
    }
};
pub const File = struct {
    file: std.Io.File,
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    header: Header,
    data_offset: u64,
    pub fn open(a: std.mem.Allocator, io: std.Io, path: []const u8) !File {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const size = (try file.stat(io)).size;
        if (size < 8) return error.TruncatedSafetensors;
        var prefix: [8]u8 = undefined;
        if (try file.readPositionalAll(io, &prefix, 0) != 8) return error.TruncatedSafetensors;
        const header_len = std.mem.readInt(u64, &prefix, .little);
        if (header_len < 2 or header_len > 100 * 1024 * 1024 or header_len > size - 8) return error.InvalidSafetensorHeader;
        const json = try a.alloc(u8, @intCast(header_len));
        defer a.free(json);
        if (try file.readPositionalAll(io, json, 8) != json.len) return error.TruncatedSafetensors;
        const owned = try a.dupe(u8, path);
        errdefer a.free(owned);
        return .{ .file = file, .io = io, .allocator = a, .path = owned, .header = try Header.parse(a, json, size - 8 - header_len), .data_offset = 8 + header_len };
    }
    pub fn deinit(f: *File) void {
        f.header.deinit();
        f.file.close(f.io);
        f.allocator.free(f.path);
    }
    pub fn readRow(f: *const File, tensor: Tensor, row: usize, buffer: []u8) !void {
        const width = try tensor.rowBytes();
        if (row >= tensor.dims[0] or buffer.len != width) return error.InvalidTensorRow;
        const offset = f.data_offset + tensor.offset + @as(u64, @intCast(row)) * width;
        if (try f.file.readPositionalAll(f.io, buffer, offset) != width) return error.TruncatedSafetensors;
    }
};
pub fn shardName(name: []const u8) !void {
    if (name.len == 0 or std.mem.indexOfAny(u8, name, "/\\") != null or !std.mem.endsWith(u8, name, ".safetensors")) return error.InvalidShardName;
}
pub fn validateFile(io: std.Io, path: []const u8) !void {
    var file = try File.open(std.heap.c_allocator, io, path);
    defer file.deinit();
}

fn fixture(io: std.Io, path: []const u8, prefix: u64, json: []const u8, payload: []const u8) !void {
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var size: [8]u8 = undefined;
    std.mem.writeInt(u64, &size, prefix, .little);
    try file.writeStreamingAll(io, &size);
    try file.writeStreamingAll(io, json);
    try file.writeStreamingAll(io, payload);
}
fn openAllocated(a: std.mem.Allocator, io: std.Io, path: []const u8) !void {
    var file = try File.open(a, io, path);
    defer file.deinit();
}
pub fn checkFiles(io: std.Io, dir: []const u8) !void {
    try std.Io.Dir.cwd().createDirPath(io, dir);
    var buffer: [4096]u8 = undefined;
    const path = try std.fmt.bufPrint(&buffer, "{s}/checkpoint.safetensors", .{dir});
    const a = std.heap.c_allocator;
    const payload = "0123456789abcdefghij";
    try fixture(io, path, valid.len, valid, payload);
    try std.testing.checkAllAllocationFailures(a, openAllocated, .{ io, path });
    {
        var file = try File.open(a, io, path);
        defer file.deinit();
        const tensor = file.header.tensors.get("x").?;
        var row: [8]u8 = undefined;
        try file.readRow(tensor, 0, &row);
        try std.testing.expectEqualSlices(u8, payload[0..8], &row);
        try file.readRow(tensor, 1, &row);
        try std.testing.expectEqualSlices(u8, payload[8..16], &row);
        try std.testing.expectError(error.InvalidTensorRow, file.readRow(tensor, 2, &row));
        try std.testing.expectError(error.InvalidTensorRow, file.readRow(tensor, 0, row[0..7]));
        // Revalidate the actual read even if a file is truncated after its header was checked.
        const writer = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = false });
        defer writer.close(io);
        try writer.setLength(io, file.data_offset + 9);
        try std.testing.expectError(error.TruncatedSafetensors, file.readRow(tensor, 1, &row));
    }
    try std.testing.expectError(error.InvalidTensorOffsets, File.open(a, io, path));
    try fixture(io, path, 100 * 1024 * 1024 + 1, "{}", "");
    try std.testing.expectError(error.InvalidSafetensorHeader, File.open(a, io, path));
    try fixture(io, path, 100, "{}", "");
    try std.testing.expectError(error.InvalidSafetensorHeader, File.open(a, io, path));
    try fixture(io, path, 2, "{}", "trailing payload");
    try std.testing.expectError(error.InvalidTensorOffsets, File.open(a, io, path));
    const short = try std.Io.Dir.cwd().createFile(io, path, .{});
    short.close(io);
    try std.testing.expectError(error.TruncatedSafetensors, File.open(a, io, path));
    const missing = try std.fmt.bufPrint(&buffer, "{s}/missing.safetensors", .{dir});
    try std.testing.expectError(error.FileNotFound, File.open(a, io, missing));
    std.debug.print("PASS: safetensors positional rows, missing/truncated files, oversized headers, invalid row sizes, and every open allocation failure\n", .{});
}

const valid = "{\"x\":{\"dtype\":\"U32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"y\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[16,20]}}";
fn parseAllocated(a: std.mem.Allocator) !void {
    var h = try Header.parse(a, valid, 20);
    defer h.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 2, 2 }, h.tensors.get("x").?.shape());
}
test "safetensors parser cleans up at every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, parseAllocated, .{});
}
test "safetensors reject malformed, overlapping, truncated and overflowing tensors" {
    const cases = .{
        .{ "null", 0, error.InvalidSafetensorHeader },
        .{ "{", 0, error.InvalidSafetensorHeader },
        .{ "{\"__metadata__\":{\"x\":1}}", 0, error.InvalidSafetensorHeader },
        .{ "{\"x\":{\"shape\":[1],\"data_offsets\":[0,4]}}", 4, error.InvalidSafetensorHeader },
        .{ "{\"x\":{\"dtype\":12,\"shape\":[1],\"data_offsets\":[0,4]}}", 4, error.InvalidSafetensorHeader },
        .{ "{\"x\":{\"dtype\":\"U32\",\"shape\":[1],\"data_offsets\":[0]}}", 4, error.InvalidSafetensorHeader },
        .{ "{\"x\":{\"dtype\":\"U32\",\"shape\":[1,1,1,1,1,1,1,1,1],\"data_offsets\":[0,4]}}", 4, error.InvalidSafetensorHeader },
        .{ "{\"x\":{},\"x\":{}}", 0, error.InvalidSafetensorHeader },
        .{ "{\"x\":{\"dtype\":\"F8\",\"shape\":[1],\"data_offsets\":[0,1]}}", 1, error.InvalidTensorDType },
        .{ "{\"x\":{\"dtype\":\"U32\",\"shape\":[-1],\"data_offsets\":[0,4]}}", 4, error.InvalidTensorShape },
        .{ "{\"x\":{\"dtype\":\"U32\",\"shape\":[2147483647,2147483647,2147483647],\"data_offsets\":[0,4]}}", 4, error.InvalidTensorShape },
        .{ "{\"x\":{\"dtype\":\"U32\",\"shape\":[1],\"data_offsets\":[4,8]}}", 8, error.InvalidTensorOffsets },
        .{ "{\"x\":{\"dtype\":\"U32\",\"shape\":[1],\"data_offsets\":[0,3]}}", 3, error.InvalidTensorOffsets },
        .{ "{\"x\":{\"dtype\":\"U32\",\"shape\":[1],\"data_offsets\":[4,0]}}", 4, error.InvalidTensorOffsets },
        .{ "{\"x\":{\"dtype\":\"U32\",\"shape\":[1],\"data_offsets\":[0,4]},\"y\":{\"dtype\":\"U32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 4, error.InvalidTensorOffsets },
        .{ valid, 19, error.InvalidTensorOffsets },
        .{ valid, 21, error.InvalidTensorOffsets },
    };
    inline for (cases) |case| try std.testing.expectError(case[2], Header.parse(std.testing.allocator, case[0], case[1]));
    for ([_][]const u8{ "", "../model.safetensors", "/model.safetensors", "dir/model.safetensors", "model.bin", "dir\\model.safetensors" }) |name| try std.testing.expectError(error.InvalidShardName, shardName(name));
    try shardName("model-00001-of-00022.safetensors");
    var scalar = try Header.parse(std.testing.allocator, "{\"x\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4]},\"empty\":{\"dtype\":\"U32\",\"shape\":[0,2],\"data_offsets\":[4,4]},\"__metadata__\":{\"format\":\"mlx\"}}", 4);
    defer scalar.deinit();
    try std.testing.expectEqual(@as(usize, 0), scalar.tensors.get("x").?.rank);
    try std.testing.expectEqual(@as(u64, 0), scalar.tensors.get("empty").?.len);
}
