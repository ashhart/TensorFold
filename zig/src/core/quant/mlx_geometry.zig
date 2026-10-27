//! Checked affine storage geometry shared by host readers and backend views; no numeric conversion.
const std = @import("std");
const types = @import("types.zig");
const convert = @import("convert.zig");

pub const Error = error{UnexpectedTensor};
pub const Shape = struct { n: u32, k: u32, elements: usize };

pub fn product(x: usize, y: usize) Error!usize {
    return std.math.mul(usize, x, y) catch error.UnexpectedTensor;
}

fn length(t: anytype) usize {
    return if (@hasField(@TypeOf(t), "bytes")) t.bytes.len else t.len;
}

/// Host tensors are exact slices; uploaded buffers may have an allocation larger than the declared tensor.
fn storage(t: anytype, rank: usize) Error!void {
    if (t.rank != rank) return error.UnexpectedTensor;
    var bytes = t.dtype.size();
    for (t.shape[0..rank]) |dim| {
        if (dim == 0) return error.UnexpectedTensor;
        bytes = try product(bytes, dim);
    }
    if (length(t) < bytes) return error.UnexpectedTensor;
    if (@hasField(@TypeOf(t), "bytes") and length(t) != bytes) return error.UnexpectedTensor;
}

/// A projection [N,K] or expert stack [E,N,K]; the backend addresses each projection with u32 dimensions.
pub fn shape(words: anytype, scales: anytype, biases: anytype, bits: usize, group: usize, stacked: bool) Error!Shape {
    if (std.mem.indexOfScalar(usize, &.{ 2, 3, 4, 5, 6, 8 }, bits) == null or
        std.mem.indexOfScalar(usize, &.{ 32, 64, 128 }, group) == null) return error.UnexpectedTensor;
    return scalarShape(words, scales, biases, bits, group, stacked);
}

/// The scalar oracle also decodes 7-bit codes and small groups; these are not admitted device formats.
pub fn scalarShape(words: anytype, scales: anytype, biases: anytype, bits: usize, group: usize, stacked: bool) Error!Shape {
    if (bits < 2 or bits > 8 or group == 0) return error.UnexpectedTensor;
    const lead: usize = @intFromBool(stacked);
    const rank = lead + 2;
    if (words.dtype != .i32 and words.dtype != .u32) return error.UnexpectedTensor;
    if (!convert.isFloat(scales.dtype) or !convert.isFloat(biases.dtype)) return error.UnexpectedTensor;
    try storage(words, rank);
    try storage(scales, rank);
    try storage(biases, rank);
    if (!std.mem.eql(usize, scales.shape[0..rank], biases.shape[0..rank]) or
        !std.mem.eql(usize, words.shape[0 .. lead + 1], scales.shape[0 .. lead + 1]))
        return error.UnexpectedTensor;
    const k = try product(scales.shape[lead + 1], group);
    const packed_bits = try product(k, bits);
    if (packed_bits % 32 != 0 or words.shape[lead + 1] != packed_bits / 32) return error.UnexpectedTensor;
    const n = std.math.cast(u32, words.shape[lead]) orelse return error.UnexpectedTensor;
    const columns = std.math.cast(u32, k) orelse return error.UnexpectedTensor;
    const rows = try product(if (stacked) words.shape[0] else 1, n);
    return .{ .n = n, .k = columns, .elements = try product(rows, k) };
}

/// No wrapping address range may be handed to a device kernel.
pub fn address(buffer: types.Buf) Error!void {
    if (buffer.ptr == 0) return error.UnexpectedTensor;
    _ = std.math.add(u64, buffer.ptr, buffer.len) catch return error.UnexpectedTensor;
}
