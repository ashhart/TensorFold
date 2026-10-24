//! Notes CLI flags so later slices add settings here instead of growing cli.zig.
const std = @import("std");

/// Snapshot path and the n-gram index; later slices add language and lookup here.
pub const Options = struct {
    path: ?[]const u8 = null,
    index: []const u8 = "sa", // suffix array, or LLWZ posting lists ("postings")
};

/// True when name is a notes flag this slice knows; later slices add arms here.
pub fn parse(out: anytype, name: []const u8, v: []const u8) bool {
    if (std.mem.eql(u8, name, "--notes")) {
        out.notes.path = v;
    } else if (std.mem.eql(u8, name, "--notes-index")) {
        out.notes.index = v;
    } else return false;
    return true;
}

/// Native `--notes` flags for cli.zig's table; later slices grow this array only.
pub fn native(comptime Flag: type) [2]Flag {
    return .{
        .{ .name = "--notes", .native = true },
        .{ .name = "--notes-index", .choices = &.{ "postings", "sa" }, .native = true },
    };
}

pub const usage = " [--notes SNAPSHOT] [--notes-index {postings,sa}]";

pub const help =
    "  --notes SNAPSHOT  look up an LLWZ note snapshot for each chat request that sends \"notes\": true; a snapshot naming a tokenizer.json SHA-256 starts only when the model's tokenizer.json is read and matches it, and one naming none is checked by vocabulary size only (vocabulary-only compatibility); /v1/notes adds, changes, takes back, erases and lists notes, each journaled beside the snapshot before its reply\n" ++
    "  --notes-index {postings,sa}  the notes' n-gram index: a suffix array over the notes' folded tokens with a side buffer of the notes written since (the default: the same lookups, a smaller store), or LLWZ's posting lists\n";
