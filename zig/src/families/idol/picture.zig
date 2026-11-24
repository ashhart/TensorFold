//! The picture tf-idol-generate encodes: float32 `pixels` [3, 1024, 1024] (DINOv3's CHW) or `image` [1024, 1024, 3].
const std = @import("std");
const safetensors = @import("../../core/safetensors.zig");

/// The picture's side, the 1,024 x 1,024 IdolPipeline.preprocess returns.
pub const side = 1024;
pub const chw = [_]usize{ 3, side, side };
pub const hwc = [_]usize{ side, side, 3 };

pub const Error = error{ BadPixels, BadImage, MissingPicture, OutOfMemory };

/// The picture as [0] CHW and [1] HWC; each name must hold exactly its own layout (an HWC `pixels` is refused).
pub fn layouts(a: std.mem.Allocator, pixels: ?safetensors.Tensor, image: ?safetensors.Tensor) Error![2][]f32 {
    const n = 3 * side * side;
    if (pixels) |t| if (!t.is(.f32, &chw)) return error.BadPixels;
    if (pixels == null) if (image) |t| if (!t.is(.f32, &hwc)) return error.BadImage;
    const given = pixels orelse image orelse return error.MissingPicture;
    const planar = try a.alloc(f32, n);
    errdefer a.free(planar);
    const packed_ = try a.alloc(f32, n);
    errdefer a.free(packed_);
    const src = try a.alloc(f32, n);
    defer a.free(src);
    @memcpy(std.mem.sliceAsBytes(src), given.bytes);
    if (pixels != null) {
        @memcpy(planar, src);
        for (0..side * side) |i| for (0..3) |c| {
            packed_[i * 3 + c] = src[c * side * side + i];
        };
    } else {
        @memcpy(packed_, src);
        for (0..side * side) |i| for (0..3) |c| {
            planar[c * side * side + i] = src[i * 3 + c];
        };
    }
    return .{ planar, packed_ };
}

/// What `layouts` refused, for the user.
pub fn explain(e: Error) []const u8 {
    return switch (e) {
        error.BadPixels => "`pixels` must be float32 [3, 1024, 1024] (channels first)",
        error.BadImage => "`image` must be float32 [1024, 1024, 3] (channels last)",
        error.MissingPicture => "holds neither `pixels` [3, 1024, 1024] nor `image` [1024, 1024, 3]",
        error.OutOfMemory => "out of memory",
    };
}

fn tensor(bytes: []const u8, dtype: safetensors.DType, shape: []const usize) safetensors.Tensor {
    var t: safetensors.Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(0), .bytes = bytes };
    @memcpy(t.shape[0..shape.len], shape);
    return t;
}

test "a picture is read in its own layout, and the other layout's shape is refused under either name" {
    const a = std.testing.allocator;
    const values = try a.alloc(f32, 3 * side * side);
    defer a.free(values);
    for (values, 0..) |*v, i| v.* = @floatFromInt(i % 4093);
    const bytes = std.mem.sliceAsBytes(values);

    // pixels as CHW: [0] is the tensor itself, and [1] is its transpose
    const ok = try layouts(a, tensor(bytes, .f32, &chw), null);
    defer for (ok) |x| a.free(x);
    try std.testing.expectEqualSlices(f32, values, ok[0]);
    const at = 5 * side + 7;
    for (0..3) |c| try std.testing.expectEqual(values[c * side * side + at], ok[1][at * 3 + c]);

    // the same values as image (HWC) give the same pair back the other way
    const back = try layouts(a, null, tensor(std.mem.sliceAsBytes(ok[1]), .f32, &hwc));
    defer for (back) |x| a.free(x);
    try std.testing.expectEqualSlices(f32, ok[0], back[0]);
    try std.testing.expectEqualSlices(f32, ok[1], back[1]);

    // 3,145,728 values in the wrong layout, or the wrong dtype, are refused rather than read
    try std.testing.expectError(error.BadPixels, layouts(a, tensor(bytes, .f32, &hwc), null));
    try std.testing.expectError(error.BadPixels, layouts(a, tensor(bytes, .f32, &.{3 * side * side}), null));
    try std.testing.expectError(error.BadPixels, layouts(a, tensor(bytes[0 .. bytes.len / 2], .bf16, &chw), null));
    try std.testing.expectError(error.BadImage, layouts(a, null, tensor(bytes, .f32, &chw)));
    // a malformed pixels is not passed over for a well-formed image
    try std.testing.expectError(error.BadPixels, layouts(a, tensor(bytes, .f32, &hwc), tensor(bytes, .f32, &hwc)));
    try std.testing.expectError(error.MissingPicture, layouts(a, null, null));
}
