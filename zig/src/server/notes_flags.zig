//! Notes CLI flags so later slices can add language and lookup without growing cli.zig.
const std = @import("std");

/// True when name is a notes flag this slice knows; later slices add arms here.
pub fn parse(out: anytype, name: []const u8, v: []const u8) bool {
    if (std.mem.eql(u8, name, "--notes")) {
        out.notes = v;
    } else return false;
    return true;
}
