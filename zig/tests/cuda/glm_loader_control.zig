//! A loader control changes only an upload already equal to its independent expected bytes.
const std = @import("std");

/// Refuse existing corruption or empty uploads before flipping one byte, so only the planted fault can count.
pub fn plant(got: []u8, want: []const u8) !void {
    if (!std.mem.eql(u8, got, want)) return error.UploadAlreadyDiffers;
    if (got.len == 0) return error.EmptyUpload;
    got[got.len / 2] ^= 1;
}

test "loader tamper changes a matching upload and cannot bless existing corruption" {
    const want = [_]u8{ 1, 2, 3, 4, 5 };
    var got = want;
    try plant(&got, &want);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 2, 4, 5 }, &got);
    try std.testing.expectError(error.UploadAlreadyDiffers, plant(&got, &want));
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 2, 4, 5 }, &got);
    try std.testing.expectError(error.UploadAlreadyDiffers, plant(got[0..4], &want));
    var empty: [0]u8 = .{};
    try std.testing.expectError(error.EmptyUpload, plant(&empty, &.{}));
}
