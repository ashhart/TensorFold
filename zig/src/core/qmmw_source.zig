//! Width specialization shared by the prompt-weight loaders and source compiler gate.
const std = @import("std");
pub fn source(a: std.mem.Allocator, body: []const u8, bits: u8, group: u16) ![]u8 {
    return std.fmt.allocPrint(a, "#define WB {d}\n#define WG {d}\n{s}", .{ bits, group, body });
}
