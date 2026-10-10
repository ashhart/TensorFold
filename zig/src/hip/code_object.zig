//! Exact architecture selection; never substitute a nearby GPU generation.
const std = @import("std");

pub const Image = struct { arch: []const u8, bytes: []align(8) const u8 };
pub const Error = error{ UnsupportedArchitecture, InvalidImage, DuplicateArchitecture };

pub fn select(images: []const Image, arch: []const u8) Error![]align(8) const u8 {
    if (arch.len == 0) return error.UnsupportedArchitecture;
    var selected: ?[]align(8) const u8 = null;
    for (images) |image| {
        if (!std.mem.eql(u8, image.arch, arch)) continue;
        if (selected != null) return error.DuplicateArchitecture;
        if (image.bytes.len == 0) return error.InvalidImage;
        selected = image.bytes;
    }
    return selected orelse error.UnsupportedArchitecture;
}

test "code objects require an exact architecture match" {
    const a align(8) = [_]u8{1};
    const b align(8) = [_]u8{2};
    const images = [_]Image{ .{ .arch = "gfx1150", .bytes = &a }, .{ .arch = "gfx1151", .bytes = &b } };
    try std.testing.expectEqualSlices(u8, &b, try select(&images, "gfx1151"));
    try std.testing.expectError(error.UnsupportedArchitecture, select(&images, "gfx1201"));
    try std.testing.expectError(error.UnsupportedArchitecture, select(&images, ""));
    try std.testing.expectError(error.UnsupportedArchitecture, select(&images, "gfx1151:xnack-"));
}

test "ambiguous or empty code objects fail closed" {
    const a align(8) = [_]u8{1};
    const duplicates = [_]Image{ .{ .arch = "gfx1151", .bytes = &a }, .{ .arch = "gfx1151", .bytes = &a } };
    try std.testing.expectError(error.DuplicateArchitecture, select(&duplicates, "gfx1151"));
    const empty align(8) = [_]u8{};
    const images = [_]Image{.{ .arch = "gfx1151", .bytes = &empty }};
    try std.testing.expectError(error.InvalidImage, select(&images, "gfx1151"));
}
