//! NVIDIA tensor-map descriptors for native FP4 prompt products over the existing decode weight packs.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const abi = @import("abi.zig");

pub const Map = extern struct { words: [16]u64 align(64) };
pub const Single = extern struct { activation: Map, activation_scales: Map, weights: [1]Map, weight_scales: [1]Map };
pub const Pair = extern struct { activation: Map, activation_scales: Map, weights: [2]Map, weight_scales: [2]Map };

pub const Encode = *const fn (*Map, c_int, u32, ?*anyopaque, [*]const u64, [*]const u64, [*]const u32, [*]const u32, c_int, c_int, c_int, c_int) callconv(.c) abi.Result;

pub const Api = struct {
    d: *const Driver,
    encode: Encode,

    pub fn load(d: *const Driver) !Api {
        return .{ .d = d, .encode = @constCast(&d.lib).lookup(Encode, "cuTensorMapEncodeTiled") orelse
            return error.TensorMapUnavailable };
    }

    pub fn make(self: Api, out: *Map, dtype: c_int, address: u64, dims: []const u64, strides: []const u64, box: []const u32, swizzle: c_int) !void {
        if (address == 0 or address % 16 != 0 or dims.len == 0 or dims.len > 5 or
            strides.len + 1 != dims.len or box.len != dims.len) return error.InvalidTensorMap;
        if (swizzle != 0 and address % 128 != 0) return error.InvalidTensorMap;
        for (dims, box) |dim, extent| if (dim == 0 or dim > 0x80000000 or extent == 0 or extent > 256)
            return error.InvalidTensorMap;
        for (strides) |stride| if (stride == 0 or stride % 16 != 0 or stride >= 1 << 40)
            return error.InvalidTensorMap;
        const elements: [5]u32 = @splat(1);
        try self.d.check(self.encode(out, dtype, @intCast(dims.len), @ptrFromInt(address), dims.ptr, strides.ptr, box.ptr, &elements, 0, swizzle, 0, 0), "cuTensorMapEncodeTiled");
    }

    pub fn activation(self: Api, address: u64, rows: usize, k: usize, block_rows: usize) !Map {
        try shape(rows, k, block_rows);
        var out: Map = undefined;
        try self.make(&out, 0, address, &.{ k / 2, rows }, &.{k / 2}, &.{ 64, @intCast(block_rows) }, 2);
        return out;
    }

    pub fn activationScales(self: Api, address: u64, rows: usize, k: usize, block_rows: usize) !Map {
        try shape(rows, k, block_rows);
        var out: Map = undefined;
        try self.make(&out, 2, address, &.{ k / 64, rows }, &.{k / 16}, &.{ 4, @intCast(block_rows) }, 0);
        return out;
    }

    pub fn weights(self: Api, address: u64, padded: usize, k: usize, block_columns: usize) !Map {
        try weightShape(padded, k, block_columns);
        var out: Map = undefined;
        try self.make(&out, 4, address, &.{ 32, 8, k / 64, padded / 64 }, &.{ 256, 2048, (k / 64) * 2048 }, &.{ 32, 8, 2, @intCast(block_columns / 64) }, 0);
        return out;
    }

    pub fn weightScales(self: Api, address: u64, padded: usize, k: usize, block_columns: usize) !Map {
        try weightShape(padded, k, block_columns);
        var out: Map = undefined;
        try self.make(&out, 2, address, &.{ 64, k / 64, padded / 64 }, &.{ 256, (k / 64) * 256 }, &.{ 64, 2, @intCast(block_columns / 64) }, 0);
        return out;
    }

    pub fn single(self: Api, x: u64, xs: u64, w: u64, ws: u64, rows: usize, padded: usize, k: usize, block_rows: usize) !Single {
        return .{ .activation = try self.activation(x, rows, k, block_rows), .activation_scales = try self.activationScales(xs, rows, k, block_rows), .weights = .{try self.weights(w, padded, k, 128)}, .weight_scales = .{try self.weightScales(ws, padded, k, 128)} };
    }

    pub fn pair(self: Api, x: u64, xs: u64, w: [2]u64, ws: [2]u64, rows: usize, padded: usize, k: usize, block_rows: usize) !Pair {
        return .{ .activation = try self.activation(x, rows, k, block_rows), .activation_scales = try self.activationScales(xs, rows, k, block_rows), .weights = .{ try self.weights(w[0], padded, k, 64), try self.weights(w[1], padded, k, 64) }, .weight_scales = .{ try self.weightScales(ws[0], padded, k, 64), try self.weightScales(ws[1], padded, k, 64) } };
    }
};

fn shape(rows: usize, k: usize, block_rows: usize) !void {
    if (rows == 0 or rows > 0x7fffffff or k == 0 or k > 0x7fffffff or k % 256 != 0 or
        (block_rows != 128 and block_rows != 256)) return error.InvalidTensorMap;
}

fn weightShape(padded: usize, k: usize, block_columns: usize) !void {
    try shape(1, k, 128);
    if (padded == 0 or padded > 0x7fffffff or padded % 64 != 0 or
        (block_columns != 64 and block_columns != 128)) return error.InvalidTensorMap;
}

pub fn sharedBytes(block_rows: usize) usize {
    std.debug.assert(block_rows == 128 or block_rows == 256);
    return 3 * (block_rows * 80 + 128 * 72) + 48;
}

test "tensor map packets agree with the CUDA parameter ABI and shared layouts" {
    try std.testing.expectEqual(@as(usize, 128), @sizeOf(Map));
    try std.testing.expectEqual(@as(usize, 64), @alignOf(Map));
    try std.testing.expectEqual(@as(usize, 512), @sizeOf(Single));
    try std.testing.expectEqual(@as(usize, 768), @sizeOf(Pair));
    try std.testing.expectEqual(@as(usize, 58416), sharedBytes(128));
    try std.testing.expectEqual(@as(usize, 89136), sharedBytes(256));
}

test "tensor map geometry rejects unsupported short K and invalid tiles" {
    try shape(17, 5120, 128);
    try shape(4096, 17408, 256);
    try weightShape(17408, 5120, 64);
    try std.testing.expectError(error.InvalidTensorMap, shape(0, 5120, 128));
    try std.testing.expectError(error.InvalidTensorMap, shape(17, 192, 128));
    try std.testing.expectError(error.InvalidTensorMap, shape(17, 5120, 64));
    try std.testing.expectError(error.InvalidTensorMap, weightShape(129, 5120, 64));
}

test "tensor map encoder receives the existing decode-pack strides and exact tile dimensions" {
    const Fake = struct {
        var rank: u32 = 0;
        var dims: [5]u64 = @splat(0);
        var strides: [4]u64 = @splat(0);
        var box: [5]u32 = @splat(0);
        var dtype: c_int = 0;
        var swizzle: c_int = 0;
        fn encode(out: *Map, kind: c_int, count: u32, _: ?*anyopaque, ds: [*]const u64, ss: [*]const u64, bs: [*]const u32, _: [*]const u32, _: c_int, sw: c_int, _: c_int, _: c_int) callconv(.c) abi.Result {
            @memset(&out.words, 0);
            rank = count;
            dtype = kind;
            swizzle = sw;
            @memcpy(dims[0..count], ds[0..count]);
            @memcpy(strides[0 .. count - 1], ss[0 .. count - 1]);
            @memcpy(box[0..count], bs[0..count]);
            return abi.success;
        }
    };
    var driver: Driver = undefined;
    const api: Api = .{ .d = &driver, .encode = Fake.encode };
    _ = try api.weights(0x10000, 17408, 5120, 128);
    try std.testing.expectEqual(@as(u32, 4), Fake.rank);
    try std.testing.expectEqual(@as(c_int, 4), Fake.dtype);
    try std.testing.expectEqualSlices(u64, &.{ 32, 8, 80, 272 }, Fake.dims[0..4]);
    try std.testing.expectEqualSlices(u64, &.{ 256, 2048, 163840 }, Fake.strides[0..3]);
    try std.testing.expectEqualSlices(u32, &.{ 32, 8, 2, 2 }, Fake.box[0..4]);
    _ = try api.activation(0x20000, 17, 5120, 128);
    try std.testing.expectEqualSlices(u64, &.{ 2560, 17 }, Fake.dims[0..2]);
    try std.testing.expectEqual(@as(c_int, 2), Fake.swizzle);
    _ = try api.activationScales(0x30000, 17, 5120, 128);
    try std.testing.expectEqualSlices(u64, &.{ 80, 17 }, Fake.dims[0..2]);
    try std.testing.expectEqual(@as(u64, 320), Fake.strides[0]);
    try std.testing.expectEqualSlices(u32, &.{ 4, 128 }, Fake.box[0..2]);
    try std.testing.expectError(error.InvalidTensorMap, api.activation(0x20010, 17, 5120, 128));
}
