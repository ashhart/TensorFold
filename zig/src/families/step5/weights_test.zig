//! Real affine loader metadata validation without mapping or touching a Metal buffer.
const std = @import("std");
const production = @import("weights.zig");
const Tensor = production.Tensor;
fn tensor(dtype: @TypeOf(@as(Tensor, undefined).dtype), dims: []const usize) Tensor {
    var t: Tensor = .{ .buffer = undefined, .offset = 0, .bytes = dtype.size(), .dtype = dtype, .rank = dims.len };
    for (dims, 0..) |n, i| {
        t.shape[i] = n;
        t.bytes *= n;
    }
    return t;
}
fn accept(w: Tensor, s: Tensor, b: Tensor, stack: usize, n: usize, k: usize) !void {
    _ = try production.checkedAffine(w, s, b, stack, n, k);
}

test "step5 host: affine expert axes cannot be replaced by equal element counts" {
    const w = tensor(.u32, &.{ 4, 16, 8 });
    const s = tensor(.bf16, &.{ 4, 16, 1 });
    try accept(w, s, s, 4, 16, 128);
    try std.testing.expectError(error.UnexpectedStep5Tensor, accept(tensor(.u32, &.{ 2, 32, 8 }), s, s, 4, 16, 128));
    try std.testing.expectError(error.UnexpectedStep5Tensor, accept(w, tensor(.bf16, &.{ 2, 32, 1 }), s, 4, 16, 128));
    try std.testing.expectError(error.UnexpectedStep5Tensor, accept(w, s, tensor(.bf16, &.{ 2, 32, 1 }), 4, 16, 128));
    try std.testing.expectError(error.UnexpectedStep5Tensor, accept(w, tensor(.bf16, &.{ 64, 1 }), s, 4, 16, 128));
}

test "step5 host: affine invalid rank bytes and packed widths return named errors without traps" {
    var w = tensor(.u32, &.{ 4, 16, 8 });
    const s = tensor(.bf16, &.{ 4, 16, 1 });
    for ([_]usize{ 0, 1, 2, 4, std.math.maxInt(usize) }) |rank| {
        w.rank = rank;
        try std.testing.expectError(error.UnexpectedStep5Tensor, accept(w, s, s, 4, 16, 128));
    }
    w = tensor(.u32, &.{ 4, 16, 8 });
    w.bytes += 1;
    try std.testing.expectError(error.UnexpectedStep5Tensor, accept(w, s, s, 4, 16, 128));
    try std.testing.expectError(error.UnexpectedStep5Tensor, accept(tensor(.u32, &.{ 4, 16, 4096 }), s, s, 4, 16, 128));
    try std.testing.expectError(error.UnexpectedStep5Tensor, accept(tensor(.u32, &.{ 1, 8, 16 }), tensor(.bf16, &.{ 8, 1 }), tensor(.bf16, &.{ 8, 1 }), 1, 8, 64));
    const dense = try production.checkedAffine(tensor(.u32, &.{ 8, 16 }), tensor(.bf16, &.{ 8, 1 }), tensor(.bf16, &.{ 8, 1 }), 1, 8, 64);
    try std.testing.expectEqual(@as(u8, 8), dense.bits);
    try std.testing.expectEqual(@as(usize, 64), dense.group);
}
