//! The deterministic part of LLWZ's intake check (zengine/intake.py): a sentence's normal form, and the verdicts.
const std = @import("std");
const question = @import("question.zig");
const Allocator = std.mem.Allocator;

/// intake.VERDICTS. A write here gives IMPLIED only on replay, for a take-back of a note that was erased.
pub const Verdict = enum {
    NEW,
    REPLACES,
    ALREADY_KNOWN,
    IMPLIED,

    /// intake.LEARN: whether the verdict writes a note.
    pub fn learn(v: Verdict) bool {
        return v == .NEW or v == .REPLACES;
    }
};

/// Python's str.isspace() for one code point: what str.split() with no argument splits on.
fn isSpace(cp: u21) bool {
    return switch (cp) {
        '\t', '\n', 0x0b, 0x0c, '\r', 0x1c...0x1f, ' ', 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

/// lwlogic.APOS, the translation norm() applies first: curly quotes straight, the two dashes ", ".
fn apos(cp: u21) ?[]const u8 {
    return switch (cp) {
        0x2019, 0x2018 => "'",
        0x201c, 0x201d => "\"",
        0x2014, 0x2013 => ", ",
        else => null,
    };
}

/// intake.norm(text) with ASCII letters lowered: one line, straight quotes, single spaces, no closing ".!? ".
pub fn norm(a: Allocator, text: []const u8) Allocator.Error![]u8 {
    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(a);
    var it = std.unicode.Utf8View.initUnchecked(text).iterator();
    while (it.nextCodepointSlice()) |s| {
        const cp = std.unicode.utf8Decode(s) catch {
            try t.appendSlice(a, s);
            continue;
        };
        if (apos(cp)) |r| try t.appendSlice(a, r) else try t.appendSlice(a, s);
    }
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var words = std.unicode.Utf8View.initUnchecked(t.items).iterator();
    var gap = false;
    while (words.nextCodepointSlice()) |s| {
        const cp = std.unicode.utf8Decode(s) catch 0xfffd;
        if (isSpace(cp)) {
            gap = out.items.len > 0;
            continue;
        }
        if (gap) try out.append(a, ' ');
        gap = false;
        for (s) |b| try out.append(a, std.ascii.toLower(b));
    }
    var end = out.items.len;
    while (end > 0 and std.mem.indexOfScalar(u8, ".!? ", out.items[end - 1]) != null) end -= 1;
    out.shrinkRetainingCapacity(end);
    return out.toOwnedSlice(a);
}

/// The most a chat message written as a note may hold: one sentence, not a pasted document (the server's own rule).
pub const statement_limit: usize = 400;

/// A user's chat message as the intake takes it: its text without the space around it, or why it is not a note.
pub const Statement = union(enum) { text: []const u8, skip: []const u8 };

pub fn statement(text: []const u8) Statement {
    var start: ?usize = null;
    var end: usize = 0;
    var words = false;
    var it = std.unicode.Utf8View.initUnchecked(text).iterator();
    while (it.nextCodepointSlice()) |s| {
        const cp = std.unicode.utf8Decode(s) catch 0xfffd;
        if (isSpace(cp)) continue;
        if (start == null) start = it.i - s.len;
        end = it.i;
        if (cp >= 0x80 and !question.isDecoration(cp) or cp < 0x80 and std.ascii.isAlphanumeric(@intCast(cp))) words = true;
    }
    const t = text[start orelse 0..end];
    // intake.readings' rule ("My eyes are green?" asks; it does not tell), widened for chat: question.zig
    if (question.reason(t)) |why| return .{ .skip = why };
    if (!words) return .{ .skip = "no words" };
    if (std.mem.indexOfAny(u8, t, "\n\r") != null) return .{ .skip = "more than one line: a note is one sentence" };
    if (t.len > statement_limit) return .{ .skip = "over 400 bytes: a note is one sentence" };
    return .{ .text = t };
}

test "a statement is written; a question, an empty message, several lines or a long text are not" {
    try std.testing.expectEqualStrings("My favourite colour is green.", statement(" My favourite colour is green.\n").text);
    try std.testing.expectEqualStrings("Yes, that's right.", statement("Yes, that's right.").text);
    try std.testing.expectEqualStrings("a question: it ends in a question mark", statement("What colour do I like?").skip);
    try std.testing.expectEqualStrings("a question: it ends in a question mark", statement("My eyes are green? \u{3000}\n").skip);
    try std.testing.expectEqualStrings("a question: it opens as one", statement("What colour do I like").skip);
    try std.testing.expectEqualStrings("no words", statement(" ... ").skip);
    try std.testing.expectEqualStrings("no words", statement("\u{1F44D}\u{1F3FD} \u{2764}\u{FE0F}").skip);
    try std.testing.expectEqualStrings("more than one line: a note is one sentence", statement("a\nb").skip);
    const long: [401]u8 = @splat('x');
    try std.testing.expect(statement(long[0..400]) == .text);
    try std.testing.expect(statement(&long) == .skip);
}

test "norm: one line, straight quotes, single spaces, no closing punctuation, ASCII lowered" {
    const a = std.testing.allocator;
    const cases = [_][2][]const u8{
        .{ "My favourite colour is green.", "my favourite colour is green" },
        .{ "  my   favourite\tcolour\nis green!? ", "my favourite colour is green" },
        .{ "My cat\u{2019}s eyes \u{2014} green.", "my cat's eyes , green" },
        .{ "\u{201c}Hi\u{201d}\u{3000}there...", "\"hi\" there" },
        .{ "Émile is here.\u{a0}", "Émile is here" },
        .{ "?!.", "" },
    };
    for (cases) |c| {
        const got = try norm(a, c[0]);
        defer a.free(got);
        try std.testing.expectEqualStrings(c[1], got);
    }
}
