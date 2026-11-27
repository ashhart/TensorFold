//! Role dispatch controls applied to the production shaders.
const std = @import("std");
const anchor = "const uint4 xv = x[j + i];\n      dot = ink_loop_pair(dot, wv[i].x, xv.x);";
pub fn source(a: std.mem.Allocator, text: []const u8) ![]u8 {
    if (std.mem.count(u8, text, anchor) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, text, anchor, "const uint4 xv = x[j + i];");
}
test "role source control deletes one serial term and refuses missing or repeated anchors" {
    const a = std.testing.allocator;
    const text = @import("inkling2_metal").loop;
    const changed = try source(a, text);
    defer a.free(changed);
    try std.testing.expect(!std.mem.eql(u8, text, changed));
    try std.testing.expectError(error.ControlEditMissing, source(a, changed));
    const twice = try std.mem.concat(a, u8, &.{ text, text });
    defer a.free(twice);
    try std.testing.expectError(error.ControlEditMissing, source(a, twice));
}
