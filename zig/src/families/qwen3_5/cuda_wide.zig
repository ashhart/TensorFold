//! A widened dense MLP: preserve the base codes and read the exact saved pocket. Credit @ZackSample #911.
const std = @import("std");
const affine = @import("living").affine;

/// Preserve each base row's byte prefix, including separate affine groups in the down projection.
pub fn base(a: std.mem.Allocator, bytes: []const u8, rows: usize, stride: usize, kept: usize) ![]u8 {
    if (stride == 0 or kept > stride or rows == 0 or bytes.len % stride != 0 or bytes.len / stride < rows)
        return error.UnexpectedTensor;
    const out = try a.alloc(u8, rows * kept);
    for (0..rows) |row| @memcpy(out[row * kept ..][0..kept], bytes[row * stride ..][0..kept]);
    return out;
}
fn value(ts: anytype, width: usize, row: usize, col: usize) f32 {
    const w = std.mem.readInt(u32, ts[0].bytes[(row * width + col) / 8 * 4 ..][0..4], .little);
    const i = (row * width + col) / 64;
    const scale = affine.f32of(std.mem.readInt(u16, ts[1].bytes[i * 2 ..][0..2], .little));
    const bias = affine.f32of(std.mem.readInt(u16, ts[2].bytes[i * 2 ..][0..2], .little));
    return scale * @as(f32, @floatFromInt((w >> @intCast(4 * (col % 8))) & 15)) + bias;
}
pub fn read(ck: anytype, prefix: []const u8, name: []const u8, n: usize, k: usize, extra: []f32) !void {
    var buf: [192]u8 = undefined;
    var ts: [3]@TypeOf(try ck.get("")) = undefined;
    for ([_][]const u8{ "weight", "scales", "biases" }, &ts) |suffix, *t|
        t.* = try ck.get(try std.fmt.bufPrint(&buf, "{s}{s}.{s}", .{ prefix, name, suffix }));
    const na = ts[0].dim(0);
    const ka = ts[0].dim(1) * 8;
    if (!ts[0].is(.u32, &.{ na, ka / 8 }) or !ts[1].is(.bf16, &.{ na, ka / 64 }) or
        !ts[2].is(.bf16, &.{ na, ka / 64 }) or n > na or k > ka or ka % 64 != 0) return error.UnexpectedTensor;
    if (na > n and ka == k and extra.len == (na - n) * k) {
        for (n..na) |row| for (0..k) |col| {
            extra[(row - n) * k + col] = value(ts, ka, row, col);
        };
    } else if (na == n and ka > k and extra.len == (ka - k) * n) {
        for (0..n) |row| for (k..ka) |col| {
            extra[(col - k) * n + row] = value(ts, ka, row, col);
        };
    } else return error.UnexpectedTensor;
}

test "widened down projection keeps every base group byte for byte across row strides" {
    const bytes = [_]u8{ 1, 2, 3, 4, 90, 91, 5, 6, 7, 8, 92, 93 };
    const out = try base(std.testing.allocator, &bytes, 2, 6, 4);
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, out);
}
