//! A token list as Python's json.dumps writes it, "[1, 2, 3]": the text every engine's token SHA is taken over.
const std = @import("std");

/// The caller owns the text.
pub fn write(gpa: std.mem.Allocator, ids: []const u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.append(gpa, '[');
    for (ids, 0..) |t, i| {
        if (i > 0) try out.appendSlice(gpa, ", ");
        try out.print(gpa, "{d}", .{t});
    }
    try out.append(gpa, ']');
    return out.toOwnedSlice(gpa);
}

test "matches json.dumps" {
    const text = try write(std.testing.allocator, &.{ 1, 22, 333 });
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings("[1, 22, 333]", text);
}
