//! Weights kept in float [N, K]: fp32 (the unquantized projections a conversion leaves, an MTP head's fc), or BF16 /
//! FP16 as stored when the loader asks (`readStored`: a checkpoint published in 16-bit floats).

const std = @import("std");
const types = @import("types.zig");
const slice = @import("slice.zig");
const convert = @import("convert.zig");

const Tensor = types.Tensor;
const Allocator = std.mem.Allocator;
const join = types.join;

pub const id: types.Format = .dense;
pub const decoder: types.Decoder = .dense;

/// A float weight is not named by the config: nothing here declares it.
pub const Config = void;

pub fn detect(a: Allocator, src: types.Sources) Allocator.Error!?Config {
    _ = a;
    _ = src;
    return null;
}

pub const Host = struct { weight: Tensor };

/// Whether `key` holds a float matrix with no table of scales beside it.
pub fn matches(t: anytype, key: []const u8) bool {
    var buf: [256]u8 = undefined;
    const w = t.get(join(&buf, &.{ key, ".weight" })) catch return false;
    return convert.isFloat(w.dtype) and w.rank == 2 and !t.has(join(&buf, &.{ key, ".scales" }));
}

/// The float weight widened to fp32.
pub fn read(a: Allocator, t: anytype, key: []const u8) !Host {
    var buf: [256]u8 = undefined;
    return .{ .weight = try convert.float32(a, try t.get(join(&buf, &.{ key, ".weight" }))) };
}

/// The float weight in its stored dtype when that is BF16 or FP16 (shared, not copied), else widened as `read` does.
pub fn readStored(a: Allocator, t: anytype, key: []const u8) !Host {
    var buf: [256]u8 = undefined;
    const w = try t.get(join(&buf, &.{ key, ".weight" }));
    return .{ .weight = if (w.dtype == .bf16 or w.dtype == .f16) w else try convert.float32(a, w) };
}

pub fn rows(h: Host) usize {
    return h.weight.shape[0];
}

pub fn bytes(h: Host) usize {
    return h.weight.bytes.len;
}

/// Both halves along K, copied.
pub fn halves(a: Allocator, p: Host) ![2]Host {
    const half = p.weight.shape[1] / 2;
    return .{
        .{ .weight = try slice.takeCols(a, p.weight, 0, half) },
        .{ .weight = try slice.takeCols(a, p.weight, half, p.weight.shape[1]) },
    };
}

pub fn sliceRows(a: Allocator, p: Host, spans: []const slice.Span) slice.Error!Host {
    return .{ .weight = try slice.takeRows(a, p.weight, spans) };
}

pub fn sliceCols(a: Allocator, p: Host, r: slice.Rank, what: []const u8) slice.Error!Host {
    const width = try slice.even(p.weight.shape[1], r.world, what);
    return .{ .weight = try slice.takeCols(a, p.weight, r.rank * width, (r.rank + 1) * width) };
}

pub fn sliceStack(a: Allocator, p: Host, r: slice.Rank, part: usize) slice.Error!Host {
    return .{ .weight = try slice.takeExperts(a, p.weight, r, part) };
}

pub const Device = struct { weight: types.Buf };

pub fn upload(u: types.Uploader, h: Host) !Device {
    return .{ .weight = try u.tensor(h.weight) };
}

/// What the kernels read: the weight's address and its element type (fp32, or BF16 / FP16 from `readStored`).
pub const Matrix = struct { weight: u64, dtype: types.DType = .f32 };

pub const View = struct { n: u32, k: u32, matrix: Matrix };

pub fn view(d: Device, stacked: bool) error{UnsupportedTables}!View {
    const lead: usize = @intFromBool(stacked);
    return .{ .n = @intCast(d.weight.dim(lead)), .k = @intCast(d.weight.dim(lead + 1)), .matrix = .{ .weight = d.weight.ptr, .dtype = d.weight.dtype } };
}

pub fn elements(h: Host) usize {
    return h.weight.numel();
}

/// The weight as fp64, row by row, read in its own dtype.
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    const dtype = h.weight.dtype;
    if (dtype != .f32 and dtype != .bf16 and dtype != .f16) return error.UnexpectedTensor;
    if (out.len != h.weight.numel() or h.weight.bytes.len != out.len * dtype.size()) return error.UnexpectedTensor;
    for (out, 0..) |*o, i| o.* = convert.load(dtype, h.weight.bytes, i);
}

test "a float weight halves along K and cuts its rows" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const h: Host = .{ .weight = .{ .dtype = .f32, .rank = 2, .shape = .{ 2, 4, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&data) } };
    const pair = try halves(a, h);
    try std.testing.expectEqualSlices(f32, &.{ 3, 4, 7, 8 }, @alignCast(std.mem.bytesAsSlice(f32, pair[1].weight.bytes)));
    const low = try sliceRows(a, h, &.{.{ .from = 1, .to = 2 }});
    var out: [4]f64 = undefined;
    try reference(low, &out);
    try std.testing.expectEqualSlices(f64, &.{ 5, 6, 7, 8 }, &out);
}

/// A one-tensor table for the tests: `get` and `has` over a single `<key>.weight`.
const OneTable = struct {
    weight: Tensor,
    pub fn get(t: *const OneTable, name: []const u8) !Tensor {
        if (!std.mem.eql(u8, name, "p.weight")) return error.MissingTensor;
        return t.weight;
    }
    pub fn has(_: *const OneTable, name: []const u8) bool {
        return std.mem.eql(u8, name, "p.weight");
    }
    /// What the affine reader asks of a table; this one declares no quantization.
    pub fn width(_: *const OneTable, _: []const u8) error{UnsupportedQuantization}!@import("mlx.zig").Width {
        return error.UnsupportedQuantization;
    }
};

test "readStored keeps a BF16 or FP16 weight as stored; read still widens it to fp32" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // 1.5, -2, 0.25, 3 in BF16 and in FP16
    const bf = [_]u16{ 0x3FC0, 0xC000, 0x3E80, 0x4040 };
    const hf = [_]u16{ 0x3E00, 0xC000, 0x3400, 0x4200 };
    for ([_]struct { dtype: types.DType, bits: []const u16 }{ .{ .dtype = .bf16, .bits = &bf }, .{ .dtype = .f16, .bits = &hf } }) |c| {
        const t: OneTable = .{ .weight = .{ .dtype = c.dtype, .rank = 2, .shape = .{ 2, 2, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(c.bits) } };
        const stored = try readStored(a, &t, "p");
        try std.testing.expectEqual(c.dtype, stored.weight.dtype);
        try std.testing.expectEqual(@as(usize, 8), stored.weight.bytes.len);
        var got: [4]f64 = undefined;
        try reference(stored, &got);
        try std.testing.expectEqualSlices(f64, &.{ 1.5, -2, 0.25, 3 }, &got);
        const wide = try read(a, &t, "p");
        try std.testing.expectEqual(types.DType.f32, wide.weight.dtype);
        try reference(wide, &got);
        try std.testing.expectEqualSlices(f64, &.{ 1.5, -2, 0.25, 3 }, &got);
        // a BF16 host halves along K and cuts rows by its own element size
        const pair = try halves(a, stored);
        try std.testing.expectEqual(@as(usize, 4), pair[1].weight.bytes.len);
        var right: [2]f64 = undefined;
        try reference(pair[1], &right);
        try std.testing.expectEqualSlices(f64, &.{ -2, 3 }, &right);
    }
}

test "the view of an uploaded 16-bit weight reports its dtype" {
    var b: types.Buf = undefined;
    b.ptr = 0x1000;
    b.dtype = .bf16;
    b.rank = 2;
    b.shape = .{ 3, 128, 1, 1, 1 };
    b.len = 3 * 128 * 2;
    const v = try view(.{ .weight = b }, false);
    try std.testing.expectEqual(types.DType.bf16, v.matrix.dtype);
    try std.testing.expectEqual(@as(u32, 3), v.n);
    try std.testing.expectEqual(@as(u32, 128), v.k);
}

test "a reference refuses bytes that do not match the dtype's size" {
    const bits = [_]u16{ 0x3FC0, 0xC000 };
    const h: Host = .{ .weight = .{ .dtype = .f32, .rank = 2, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&bits)[0..2] } };
    var out: [1]f64 = undefined;
    try std.testing.expectError(error.UnexpectedTensor, reference(h, &out));
}

test "quant.readStored dispatches a float tensor to the dense format in its stored dtype; quant.read widens it" {
    const quant = @import("quant.zig");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bf = [_]u16{ 0x3FC0, 0xC000, 0x3E80, 0x4040 };
    const t: OneTable = .{ .weight = .{ .dtype = .bf16, .rank = 2, .shape = .{ 2, 2, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&bf) } };
    const stored = try quant.readStored(a, &t, "p");
    try std.testing.expectEqual(types.DType.bf16, stored.dense.weight.dtype);
    const wide = try quant.read(a, &t, "p");
    try std.testing.expectEqual(types.DType.f32, wide.dense.weight.dtype);
    const f32s = [_]f32{ 1, 2 };
    const t32: OneTable = .{ .weight = .{ .dtype = .f32, .rank = 2, .shape = .{ 1, 2, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&f32s) } };
    try std.testing.expectEqual(types.DType.f32, (try quant.readStored(a, &t32, "p")).dense.weight.dtype);
}
