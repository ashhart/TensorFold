const std = @import("std");
const st = @import("safetensors");

pub const Column = struct { maximum_absolute: f64 = 0, maximum_scaled: f64 = 0, worst_row: usize = 0 };
pub const Difference = struct {
    values: usize,
    unequal_values: usize = 0,
    sign_changes: usize = 0,
    nonfinite_values: usize = 0,
    exact_bytes: bool,
    mean_absolute_finite: f64 = 0,
    rms_finite: f64 = 0,
    maximum_absolute: f64 = 0,
    worst_column: usize = 0,
    columns: []Column,

    pub fn passed(self: Difference) bool {
        return self.nonfinite_values == 0 and self.exact_bytes;
    }
};

pub fn analyze(a: std.mem.Allocator, actual: st.Tensor, expected: st.Tensor) !Difference {
    if (actual.rank == 0 or actual.rank != expected.rank or !std.mem.eql(usize, actual.shape[0..actual.rank], expected.shape[0..expected.rank])) return error.TensorShapeMismatch;
    for ([_]st.DType{ actual.dtype, expected.dtype }) |dtype| switch (dtype) {
        .f64, .f32, .f16, .bf16 => {},
        else => return error.ExpectedFloatingPointTensor,
    };
    const count = actual.numel();
    const width = actual.dim(actual.rank - 1);
    if (count == 0 or width == 0) return error.EmptyTensorComparison;
    const columns = try a.alloc(Column, width);
    errdefer a.free(columns);
    @memset(columns, .{});
    var result = Difference{
        .values = count,
        .columns = columns,
        .exact_bytes = actual.dtype == expected.dtype and std.mem.eql(u8, actual.bytes, expected.bytes),
    };
    for (0..count) |i| {
        const x = read(actual, i);
        const y = read(expected, i);
        if (!std.math.isFinite(x) or !std.math.isFinite(y)) {
            result.nonfinite_values += 1;
            continue;
        }
        result.unequal_values += @intFromBool(x != y);
        result.sign_changes += @intFromBool((x > 0) != (y > 0));
        const error_ = @abs(x - y);
        if (!std.math.isFinite(error_)) return error.TensorDifferenceOverflow;
        result.mean_absolute_finite += error_;
        result.rms_finite += error_ * error_;
        const column = &columns[i % width];
        if (error_ > column.maximum_absolute) {
            column.maximum_absolute = error_;
            column.worst_row = i / width;
        }
        column.maximum_scaled = @max(column.maximum_scaled, error_ / @max(1, @abs(y)));
        if (error_ > result.maximum_absolute) {
            result.maximum_absolute = error_;
            result.worst_column = i % width;
        }
    }
    const finite: f64 = @floatFromInt(@max(1, count - result.nonfinite_values));
    result.mean_absolute_finite /= finite;
    result.rms_finite = @sqrt(result.rms_finite / finite);
    if (!std.math.isFinite(result.mean_absolute_finite) or !std.math.isFinite(result.rms_finite)) return error.TensorDifferenceOverflow;
    return result;
}

/// The first `like.dim(0)` rows of `taller`, which must otherwise have `like`'s rank, trailing shape and dtype.
pub fn leadingRows(taller: st.Tensor, like: st.Tensor) !st.Tensor {
    if (taller.rank == 0 or taller.rank != like.rank or taller.dtype != like.dtype or like.dim(0) > taller.dim(0) or
        !std.mem.eql(usize, taller.shape[1..taller.rank], like.shape[1..like.rank])) return error.TensorShapeMismatch;
    var out = taller;
    out.shape[0] = like.dim(0);
    out.bytes = taller.bytes[0 .. like.numel() * taller.dtype.size()];
    return out;
}

/// A copy of `tensor` at `index` of `axis`, with that axis removed.
pub fn alongAxis(a: std.mem.Allocator, tensor: st.Tensor, axis: usize, index: usize) !st.Tensor {
    if (axis >= tensor.rank or tensor.rank < 2 or index >= tensor.dim(axis)) return error.TensorShapeMismatch;
    var outer: usize = 1;
    for (tensor.shape[0..axis]) |dim| outer *= dim;
    var inner: usize = tensor.dtype.size();
    for (tensor.shape[axis + 1 .. tensor.rank]) |dim| inner *= dim;
    const bytes = try a.alloc(u8, outer * inner);
    for (0..outer) |o| @memcpy(bytes[o * inner ..][0..inner], tensor.bytes[(o * tensor.dim(axis) + index) * inner ..][0..inner]);
    var out = tensor;
    out.rank -= 1;
    for (axis..out.rank) |d| out.shape[d] = tensor.shape[d + 1];
    out.bytes = bytes;
    return out;
}

/// `tensor` read in `like`'s shape; the element counts and dtypes must agree.
pub fn reshapeLike(tensor: st.Tensor, like: st.Tensor) !st.Tensor {
    if (tensor.numel() != like.numel() or tensor.dtype != like.dtype) return error.TensorShapeMismatch;
    var out = like;
    out.bytes = tensor.bytes;
    return out;
}

test "one index along an axis removes that axis and keeps the order of the others" {
    const values = [_]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    const picked = try alongAxis(std.testing.allocator, f32Tensor(&values, &.{ 2, 3, 2 }), 1, 2);
    defer std.testing.allocator.free(picked.bytes);
    try std.testing.expectEqual(@as(usize, 2), picked.rank);
    try std.testing.expectEqualSlices(f32, &.{ 4, 5, 10, 11 }, std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(picked.bytes))));
    const flat = try reshapeLike(picked, f32Tensor(values[0..4], &.{4}));
    try std.testing.expectEqual(@as(usize, 1), flat.rank);
}

test "leading rows keep the trailing shape and cut the payload" {
    const values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const tall = f32Tensor(&values, &.{ 3, 2 });
    const cut = try leadingRows(tall, f32Tensor(values[0..2], &.{ 1, 2 }));
    try std.testing.expectEqual(@as(usize, 1), cut.dim(0));
    try std.testing.expectEqual(@as(usize, 8), cut.bytes.len);
    try std.testing.expectError(error.TensorShapeMismatch, leadingRows(tall, f32Tensor(&values, &.{ 2, 3 })));
}

pub const Summary = struct { minimum: f64 = std.math.inf(f64), maximum: f64 = -std.math.inf(f64), positive: usize = 0, zero: usize = 0, negative: usize = 0, interior: usize = 0 };

/// Element `index` of a floating-point or integer tensor as f64.
pub fn element(tensor: st.Tensor, index: usize) f64 {
    return switch (tensor.dtype) {
        .f64, .f32, .f16, .bf16 => read(tensor, index),
        .i64 => @floatFromInt(@as(i64, @bitCast(std.mem.readInt(u64, tensor.bytes[index * 8 ..][0..8], .little)))),
        .u64 => @floatFromInt(std.mem.readInt(u64, tensor.bytes[index * 8 ..][0..8], .little)),
        .u8, .bool => @floatFromInt(tensor.bytes[index]),
        else => std.math.nan(f64),
    };
}

pub fn summarize(tensor: st.Tensor) Summary {
    var out = Summary{};
    for (0..tensor.numel()) |i| {
        const v = element(tensor, i);
        out.minimum = @min(out.minimum, v);
        out.maximum = @max(out.maximum, v);
        if (v > 0) out.positive += 1 else if (v == 0) out.zero += 1 else if (v < 0) out.negative += 1;
    }
    for (0..tensor.numel()) |i| {
        const v = element(tensor, i);
        out.interior += @intFromBool(v != 0 and v > out.minimum and v < out.maximum);
    }
    return out;
}

fn read(tensor: st.Tensor, index: usize) f64 {
    return switch (tensor.dtype) {
        .f64 => @bitCast(std.mem.readInt(u64, tensor.bytes[index * 8 ..][0..8], .little)),
        .f32 => @as(f32, @bitCast(std.mem.readInt(u32, tensor.bytes[index * 4 ..][0..4], .little))),
        .f16 => @as(f16, @bitCast(std.mem.readInt(u16, tensor.bytes[index * 2 ..][0..2], .little))),
        .bf16 => @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, tensor.bytes[index * 2 ..][0..2], .little)) << 16)),
        else => unreachable,
    };
}

fn f32Tensor(values: []const f32, shape: []const usize) st.Tensor {
    var out: st.Tensor = .{ .dtype = .f32, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = std.mem.sliceAsBytes(values) };
    @memcpy(out.shape[0..shape.len], shape);
    return out;
}

test "tensor diagnostics retain the worst column even when the average is small" {
    const a = std.testing.allocator;
    var actual: [1024]f32 = @splat(0);
    const expected: [1024]f32 = @splat(0);
    actual[777] = 1;
    const difference = try analyze(a, f32Tensor(&actual, &.{ 2, 512 }), f32Tensor(&expected, &.{ 2, 512 }));
    defer a.free(difference.columns);
    try std.testing.expect(!difference.passed());
    try std.testing.expect(difference.mean_absolute_finite < 0.001);
    try std.testing.expectEqual(@as(usize, 265), difference.worst_column);
    try std.testing.expectEqual(@as(f64, 1), difference.columns[265].maximum_absolute);
    try std.testing.expectEqual(@as(usize, 1), difference.columns[265].worst_row);
}

test "tensor exactness rejects signed-zero changes, nonfinite values and changed geometry" {
    const a = std.testing.allocator;
    const zero = f32Tensor(&.{0}, &.{ 1, 1 });
    for ([_]f32{ -0.0, std.math.nan(f32), std.math.inf(f32) }) |value| {
        const difference = try analyze(a, f32Tensor(&.{value}, &.{ 1, 1 }), zero);
        defer a.free(difference.columns);
        try std.testing.expect(!difference.passed());
    }
    try std.testing.expectError(error.TensorShapeMismatch, analyze(a, f32Tensor(&.{0}, &.{1}), zero));
    var half = zero;
    half.dtype = .f16;
    half.bytes = &.{ 0, 0 };
    const changed_dtype = try analyze(a, half, zero);
    defer a.free(changed_dtype.columns);
    try std.testing.expectEqual(@as(usize, 0), changed_dtype.unequal_values);
    try std.testing.expect(!changed_dtype.passed());
}

test "tensor diagnostics reject overflowing differences and finite error statistics" {
    for ([_][2]f64{ .{ std.math.floatMax(f64), -std.math.floatMax(f64) }, .{ 1e200, 0 } }) |pair| {
        var actual: st.Tensor = .{ .dtype = .f64, .rank = 1, .shape = @splat(1), .bytes = std.mem.sliceAsBytes(pair[0..1]) };
        var expected = actual;
        expected.bytes = std.mem.sliceAsBytes(pair[1..2]);
        try std.testing.expectError(error.TensorDifferenceOverflow, analyze(std.testing.allocator, actual, expected));
        actual.bytes = expected.bytes;
        const equal = try analyze(std.testing.allocator, actual, expected);
        defer std.testing.allocator.free(equal.columns);
        try std.testing.expect(equal.passed());
    }
}
