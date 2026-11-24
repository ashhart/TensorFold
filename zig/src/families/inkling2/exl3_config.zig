//! Checkpoint EXL3 metadata shared by Inkling 2 format admission and packed weights.
const std = @import("std");

pub fn admit(q: std.json.ObjectMap, version_optional: bool) !void {
    const book = q.get("codebook") orelse return error.UnsupportedExl3;
    if (book != .string or !std.mem.eql(u8, book.string, "mul1")) return error.UnsupportedExl3;
    const version = q.get("version") orelse return if (version_optional) {} else error.UnsupportedExl3;
    if (version != .string or !std.mem.startsWith(u8, version.string, "1.")) return error.UnsupportedExl3;
}
