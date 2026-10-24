//! Notes CLI flags so later slices add settings here instead of growing cli.zig.
const std = @import("std");

/// Snapshot path, the n-gram index, the past-cue language and the default lookup.
pub const Options = struct {
    path: ?[]const u8 = null,
    index: []const u8 = "sa", // suffix array, or LLWZ posting lists ("postings")
    language: []const u8 = "en", // past-cue language when the request omits notes_language
    lookup: []const u8 = "who", // a chat request's lookup when it sends no notes_lookup: who first, or "plain"
};

/// True when name is a notes flag this slice knows; later slices add arms here.
pub fn parse(out: anytype, name: []const u8, v: []const u8) bool {
    if (std.mem.eql(u8, name, "--notes")) {
        out.notes.path = v;
    } else if (std.mem.eql(u8, name, "--notes-index")) {
        out.notes.index = v;
    } else if (std.mem.eql(u8, name, "--notes-language")) {
        out.notes.language = v;
    } else if (std.mem.eql(u8, name, "--notes-lookup")) {
        out.notes.lookup = v;
    } else return false;
    return true;
}

/// Native `--notes` flags for cli.zig's table; later slices grow this array only.
pub fn native(comptime Flag: type) [4]Flag {
    return .{
        .{ .name = "--notes", .native = true },
        .{ .name = "--notes-index", .choices = &.{ "postings", "sa" }, .native = true },
        .{ .name = "--notes-language", .choices = &.{ "en", "de", "fi" }, .native = true },
        .{ .name = "--notes-lookup", .choices = &.{ "who", "plain" }, .native = true },
    };
}

pub const usage = " [--notes SNAPSHOT] [--notes-index {postings,sa}] [--notes-language {en,de,fi}] [--notes-lookup {who,plain}]";

pub const help =
    "  --notes SNAPSHOT  look up an LLWZ note snapshot for each chat request that sends \"notes\": true; a snapshot naming a tokenizer.json SHA-256 starts only when the model's tokenizer.json is read and matches it, and one naming none is checked by vocabulary size only (vocabulary-only compatibility); /v1/notes adds, changes, takes back, erases and lists notes, each journaled beside the snapshot before its reply\n" ++
    "  --notes-index {postings,sa}  the notes' n-gram index, rebuilt whole on every write and erase: a suffix array over the notes' folded tokens (the default: the same lookups, a smaller store), or LLWZ's posting lists\n" ++
    "  --notes-language {en,de,fi}  the language of the past cues read from a chat request's question when the request sends no notes_language (default: en)\n" ++
    "  --notes-lookup {who,plain}  a chat request's lookup when it sends no notes_lookup: who first (the default), or the earlier plain lookup\n";
