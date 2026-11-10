//! Runtime recurrent variants are checked edits of the maintained recorded source.
const std = @import("std");
pub fn last(a: std.mem.Allocator, text: []const u8) ![]u8 {
    const from = "SO[((size_t(r) * NV + hv)";
    if (std.mem.count(u8, text, from) != 1) return error.GdnPatch;
    return std.mem.replaceOwned(u8, a, text, from, "if (r == R - 1) SO[((size_t(0) * NV + hv)");
}
pub fn peer(a: std.mem.Allocator, text: []const u8) ![]u8 {
    const from = "const int hv = int(threadgroup_position_in_grid.x);";
    if (std.mem.count(u8, text, from) != 1) return error.GdnPatch;
    return std.mem.replaceOwned(u8, a, text, from, "const int hv = int(threadgroup_position_in_grid.x) + 24;");
}
