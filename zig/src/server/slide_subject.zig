//! A question's subject as a lesson reads it: the words it asks about, the row of their last token, and others of
//! the same kind to ask about instead.
const std = @import("std");
const api = @import("engine_api");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;
const Cx = errors.Cx;
const askBase = @import("slide_ask.zig").askBase;
const prompts = @import("slide_prompts.zig");

/// What a question asks about, word for word as it says it, as the model names it (null: not found in it).
pub fn subjectOf(srv: *Server, cx: *Cx, q: []const u8, gone: anytype) !?[]const u8 {
    const said = try askBase(srv, cx, null, try std.fmt.allocPrint(cx.a, prompts.subject_prompt, .{q}), 16, gone);
    const subject = std.mem.trim(u8, said.content, " \t\r\n\"'.?*");
    if (subject.len < 2) return null;
    const at = std.ascii.findIgnoreCase(q, subject) orelse return null;
    return q[at..][0..subject.len];
}

/// The word whose last token an edit of the weights keys on: the possessor before "'s" in a subject phrase ("my
/// sister's name": sister), else its last word ("my dog", "my favourite colour").
pub fn subjectWord(phrase: []const u8) []const u8 {
    var words = std.mem.tokenizeAny(u8, phrase, " \t");
    var last: []const u8 = "";
    var owner: ?[]const u8 = null;
    while (words.next()) |w| {
        for ([_][]const u8{ "'s", "\u{2019}s" }) |mark| if (std.mem.endsWith(u8, w, mark)) {
            owner = w[0 .. w.len - mark.len];
        };
        last = w;
    }
    const word = std.mem.trim(u8, owner orelse last, ".,!?;:'\"");
    // a pronoun or a question word is everywhere: no subject to key on
    for (unkeyed) |w| if (std.ascii.eqlIgnoreCase(word, w)) return "";
    return word;
}

/// The words of a fact's subject phrase a question can name it by: pronouns, question words and articles left out.
pub fn subjectWords(a: std.mem.Allocator, phrase: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var words = std.mem.tokenizeAny(u8, phrase, " \t\r\n");
    next: while (words.next()) |w| {
        const word = possessor(std.mem.trim(u8, w, ".,!?;:'\""));
        if (word.len < 2) continue;
        for (unkeyed ++ [_][]const u8{ "the", "a", "an", "of" }) |u| if (std.ascii.eqlIgnoreCase(word, u)) continue :next;
        try out.append(a, word);
    }
    return out.toOwnedSlice(a);
}

/// The last of `words` that `question` says as a whole word (any case), or "" when it says none of them.
pub fn wordIn(words: []const []const u8, question: []const u8) []const u8 {
    var found: []const u8 = "";
    var it = std.mem.tokenizeAny(u8, question, " \t\r\n,.!?;:\"");
    while (it.next()) |w| {
        const bare = possessor(w);
        for (words) |word| if (std.ascii.eqlIgnoreCase(bare, word)) {
            found = word;
        };
    }
    return found;
}

/// `w` without a possessive "'s".
fn possessor(w: []const u8) []const u8 {
    for ([_][]const u8{ "'s", "\u{2019}s" }) |mark| if (std.mem.endsWith(u8, w, mark)) return w[0 .. w.len - mark.len];
    return w;
}

test "a fact's subject words, and the last of them a question says" {
    const a = std.testing.allocator;
    const words = try subjectWords(a, "Quillmoor Labs");
    defer a.free(words);
    try std.testing.expectEqual(@as(usize, 2), words.len);
    try std.testing.expectEqualStrings("Labs", wordIn(words, "When was Quillmoor Labs founded?"));
    try std.testing.expectEqualStrings("Quillmoor", wordIn(words, "Who is Quillmoor's CTO?"));
    try std.testing.expectEqualStrings("", wordIn(words, "Who founded the company?"));
    const mine = try subjectWords(a, "My sister");
    defer a.free(mine);
    try std.testing.expectEqual(@as(usize, 1), mine.len);
    try std.testing.expectEqualStrings("sister", wordIn(mine, "What is my sister's name?"));
}

const unkeyed = [_][]const u8{ "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "you", "your", "yours", "it", "its", "this", "that", "these", "those", "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "do", "does", "did", "is", "are", "was", "like", "likes" };

/// The row of `word`'s last token in an example's question: its last mention before the reply's first row, so a
/// thinking example's recall line, which repeats the fact, is passed over; 0 when the question does not say it.
pub fn subjectRow(srv: *Server, cx: *Cx, ex: api.Example, word: []const u8) !u32 {
    if (word.len == 0) return 0;
    const end: usize = @min(ex.start, ex.decide);
    for ([_][]const u8{ " ", "" }) |lead| {
        const toks = srv.text.encode(cx.a, try std.mem.concat(cx.a, u8, &.{ lead, word }), false) catch continue;
        if (toks.len == 0 or toks.len > end) continue;
        var at: usize = end;
        while (at >= toks.len) : (at -= 1) if (std.mem.eql(u32, ex.ids[at - toks.len .. at], toks)) return @intCast(at - 1);
    }
    return 0;
}

/// The row before the first token of `value` (a short answer) in an example's answer, or 0 when the answer never says
/// it: tried as written, then with its first letter's case turned, then its first word alone, each with a space before.
pub fn valueRow(srv: *Server, cx: *Cx, ex: api.Example, value: []const u8) !u32 {
    const v = std.mem.trim(u8, value, " \t\r\n\"'.*");
    if (v.len == 0 or ex.start == 0 or std.mem.count(u8, v, " ") > 3) return 0; // a sentence is no value to find
    const first = v[0 .. std.mem.indexOfAny(u8, v, " ,") orelse v.len];
    const turned = try cx.a.dupe(u8, v);
    turned[0] = if (std.ascii.isUpper(turned[0])) std.ascii.toLower(turned[0]) else std.ascii.toUpper(turned[0]);
    for ([_][]const u8{ v, turned, first, turned[0..first.len] }) |form| for ([_][]const u8{ " ", "" }) |lead| {
        const toks = srv.text.encode(cx.a, try std.mem.concat(cx.a, u8, &.{ lead, form }), false) catch continue;
        if (toks.len == 0) continue;
        var at: usize = ex.start;
        while (at + toks.len <= ex.ids.len) : (at += 1) if (std.mem.eql(u32, ex.ids[at..][0..toks.len], toks)) return @intCast(at - 1);
    };
    return 0;
}

/// A question with its subject swapped for others of the same kind, as the model names that kind.
pub fn kindsOf(srv: *Server, cx: *Cx, q: []const u8, subject: []const u8, gone: anytype) ![]const []const u8 {
    const a = cx.a;
    const at = std.ascii.findIgnoreCase(q, subject) orelse return &.{};
    var out: std.ArrayList([]const u8) = .empty;
    for (try kindsList(srv, cx, subject, gone)) |kind| try out.append(a, try std.mem.concat(a, u8, &.{ q[0..at], kind, q[at + subject.len ..] }));
    return out.items;
}

/// Six other things of `subject`'s kind at most, as the model names them.
pub fn kindsList(srv: *Server, cx: *Cx, subject: []const u8, gone: anytype) ![]const []const u8 {
    const a = cx.a;
    const listed = try askBase(srv, cx, null, try std.fmt.allocPrint(a, prompts.kinds_prompt, .{subject}), 64, gone);
    var out: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.tokenizeAny(u8, listed.content, "\r\n");
    while (lines.next()) |line| {
        const kind = std.mem.trim(u8, std.mem.trimStart(u8, line, "0123456789.)-* \t"), " \t\"'.");
        if (kind.len < 2 or std.ascii.eqlIgnoreCase(kind, subject) or std.mem.indexOfAny(u8, kind, "?") != null) continue;
        try out.append(a, kind);
        if (out.items.len == 6) break;
    }
    return out.items;
}
