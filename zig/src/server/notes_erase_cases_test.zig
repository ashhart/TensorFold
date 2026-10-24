//! Fictional note strings covering JSON escapes, Unicode and later note-state changes.
pub const Case = struct {
    sibling: bool = false,
    name: []const u8,
    json: []const u8, // the note's text as a JSON string, for POST /v1/notes
    raw: []const u8, // its text
    rendered: []const u8, // its text as the block holds it: JSON-quoted, non-ASCII kept
    question: []const u8 = "What is my favourite colour?",
    aged: enum { none, superseded, renumbered } = .none,
};

pub const cases = [_]Case{
    .{ .name = "plain", .json = "\"My favourite colour is green.\"", .raw = "My favourite colour is green.", .rendered = "\"My favourite colour is green.\"" },
    .{ .name = "quotes", .json = "\"My favourite \\\"colour\\\" is green.\"", .raw = "My favourite \"colour\" is green.", .rendered = "\"My favourite \\\"colour\\\" is green.\"" },
    .{ .name = "backslashes", .json = "\"My favourite colour is C:\\\\paint\\\\green.\"", .raw = "My favourite colour is C:\\paint\\green.", .rendered = "\"My favourite colour is C:\\\\paint\\\\green.\"" },
    .{ .name = "newline", .json = "\"My favourite colour is green\\nand blue.\"", .raw = "My favourite colour is green\nand blue.", .rendered = "\"My favourite colour is green\\nand blue.\"" },
    .{ .name = "finnish", .json = "\"My favourite colour is vihreä ja sininen öisin.\"", .raw = "My favourite colour is vihreä ja sininen öisin.", .rendered = "\"My favourite colour is vihreä ja sininen öisin.\"" },
    .{ .name = "superseded after it was cached", .json = "\"My favourite colour is green.\"", .raw = "My favourite colour is green.", .rendered = "\"My favourite colour is green.\"", .aged = .superseded },
    .{ .name = "renumbered after it was cached", .json = "\"My favourite colour is green.\"", .raw = "My favourite colour is green.", .rendered = "\"My favourite colour is green.\"", .aged = .renumbered },
};
