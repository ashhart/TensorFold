//! Small source views and MLA reconstruction; only the six split MLA tensors may become cached F32 matrices.
const std = @import("std");
const g = @import("core").gguf;
const q = @import("cpu_gguf_rows.zig");
const views = @import("cpu_gguf_views.zig");

pub fn small(a: std.mem.Allocator, v: views.View, shape: []const usize) ![]f32 {
    if (v.tensor.kind != .f32) return error.UnsupportedLingGgufValueView;
    var expected: [3]usize = @splat(1);
    var rank: usize = v.name.rank;
    switch (v.name.role) {
        .vector => expected[0] = @intCast(v.name.dims[0]),
        .positive_exp_decay => {
            expected[0] = 16;
            rank = 1;
        },
        .convolution => for (0..3) |i| {
            expected[i] = @intCast(v.name.dims[2 - i]);
        },
        else => return error.UnsupportedLingGgufValueView,
    }
    if (shape.len != rank or !std.mem.eql(usize, shape, expected[0..rank])) return error.BadLingGgufTensorShape;
    const count = v.tensor.bytes.len / 4;
    const out = try a.alloc(f32, count);
    errdefer a.free(out);
    try q.decodeTensor(v.tensor, out);
    return out;
}

pub fn joinMla(a: std.mem.Allocator, key: g.Tensor, value: g.Tensor) ![]f32 {
    if (key.rank != 3 or value.rank != 3 or !std.mem.eql(u64, key.shape[0..3], &.{ 128, 512, 16 }) or !std.mem.eql(u64, value.shape[0..3], &.{ 512, 128, 16 })) return error.BadLingGgufTensorShape;
    var scratch = std.heap.ArenaAllocator.init(a);
    defer scratch.deinit();
    const k = try scratch.allocator().alloc(f32, 16 * 512 * 128);
    const v = try scratch.allocator().alloc(f32, k.len);
    try q.decodeTensor(key, k);
    try q.decodeTensor(value, v);
    const result = try a.alloc(f32, 16 * 256 * 512);
    for (0..16) |h| for (0..128) |n| for (0..512) |j| {
        result[(h * 256 + n) * 512 + j] = k[(h * 512 + j) * 128 + n];
        result[(h * 256 + 128 + n) * 512 + j] = v[(h * 128 + n) * 512 + j];
    };
    return result;
}

pub fn embedding(d: *const g.Document, ids: []const u32, output: []f32) !void {
    const t = d.getTensor("token_embd.weight") orelse return error.MissingLingGgufTensor;
    if (output.len != try std.math.mul(usize, ids.len, 1536)) return error.BadLingGgufTensorShape;
    for (ids) |id| if (id >= 157184) return error.InvalidToken;
    for (ids, 0..) |id, i| {
        const one = try views.row(t, id);
        try q.decodeTensor(one, output[i * 1536 ..][0..1536]);
    }
}

test "split MLA views reconstruct each head without mixing key and value coordinates" {
    const a = std.testing.allocator;
    const count = 16 * 512 * 128;
    const k = try a.alloc(f32, count);
    defer a.free(k);
    const v = try a.alloc(f32, count);
    defer a.free(v);
    for (k, v, 0..) |*key, *value, i| {
        key.* = @floatFromInt(i);
        value.* = -@as(f32, @floatFromInt(i));
    }
    const key = g.Tensor{ .name = "key", .kind = .f32, .rank = 3, .shape = .{ 128, 512, 16, 1 }, .offset = 0, .len = count * 4, .bytes = std.mem.sliceAsBytes(k) };
    const value = g.Tensor{ .name = "value", .kind = .f32, .rank = 3, .shape = .{ 512, 128, 16, 1 }, .offset = 0, .len = count * 4, .bytes = std.mem.sliceAsBytes(v) };
    const joined = try joinMla(a, key, value);
    defer a.free(joined);
    for (0..16) |h| for (0..128) |n| for (0..512) |j| {
        try std.testing.expectEqual(k[(h * 512 + j) * 128 + n], joined[(h * 256 + n) * 512 + j]);
        try std.testing.expectEqual(v[(h * 128 + n) * 512 + j], joined[(h * 256 + 128 + n) * 512 + j]);
    };
    var bad = key;
    bad.shape[0] = 127;
    try std.testing.expectError(error.BadLingGgufTensorShape, joinMla(a, bad, value));
}
