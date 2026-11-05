//! Count-checked comptime edits of a kernel's source text, for paged twins derived from main's kernels.
const std = @import("std");

/// Every match of `from` in `text` becomes `to`, and there must be `count` of them, or the build stops naming it.
pub fn edit(comptime text: []const u8, comptime from: []const u8, comptime to: []const u8, comptime count: usize) []const u8 {
    @setEvalBranchQuota(20_000_000);
    if (std.mem.count(u8, text, from) != count) @compileError("source edit: expected " ++ std.fmt.comptimePrint("{d}", .{count}) ++ " of: " ++ from);
    var out: []const u8 = "";
    var rest = text;
    while (std.mem.indexOf(u8, rest, from)) |i| {
        out = out ++ rest[0..i] ++ to;
        rest = rest[i + from.len ..];
    }
    return out ++ rest;
}

/// The text from `start` up to (not including) the next `end`.
pub fn between(comptime text: []const u8, comptime start: []const u8, comptime end: []const u8) []const u8 {
    @setEvalBranchQuota(20_000_000);
    const a = std.mem.indexOf(u8, text, start) orelse @compileError("source edit: no " ++ start);
    const b = std.mem.indexOfPos(u8, text, a, end) orelse @compileError("source edit: no " ++ end);
    return text[a..b];
}

/// The build stops when `text` no longer holds `what`.
pub fn present(comptime text: []const u8, comptime what: []const u8) void {
    @setEvalBranchQuota(20_000_000);
    if (std.mem.indexOf(u8, text, what) == null) @compileError("source edit: the source no longer holds " ++ what);
}
