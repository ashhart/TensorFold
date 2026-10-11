//! A question's subject as a lesson reads it: its words, the row of their last token, and others of its kind.
const std = @import("std");
const api = @import("engine_api");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;
const Cx = errors.Cx;
const askBase = @import("slide_ask.zig").askBase;
const prompts = @import("slide_prompts.zig");
const slide_span = @import("slide_span.zig");

/// What a question asks about, word for word as it says it, as the model names it (null: not found in it).
pub fn subjectOf(srv: *Server, cx: *Cx, q: []const u8, gone: anytype) !?[]const u8 {
    const said = try askBase(srv, cx, null, try std.fmt.allocPrint(cx.a, prompts.subject_prompt, .{q}), 16, gone);
    const subject = std.mem.trim(u8, said.content, " \t\r\n\"'.?*");
    if (subject.len < 2) return null;
    const at = std.ascii.findIgnoreCase(q, subject) orelse return null;
    return q[at..][0..subject.len];
}

/// The word a subject phrase keys on: the possessor before "'s" ("my sister's name": sister), else its last word.
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

/// The words of a fact's subject phrase a question can name it by: pronouns, question words and small words left out.
pub fn subjectWords(a: std.mem.Allocator, phrase: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var words = std.mem.tokenizeAny(u8, phrase, " \t\r\n");
    next: while (words.next()) |w| {
        const word = possessor(std.mem.trim(u8, w, ".,!?;:'\""));
        if (word.len < 2) continue;
        for (unkeyed ++ small) |u| if (std.ascii.eqlIgnoreCase(word, u)) continue :next;
        try out.append(a, word);
    }
    return out.toOwnedSlice(a);
}

/// The subject words a third or more of the fact's questions say: a phrase run on past its subject keeps its own.
pub fn asked(a: std.mem.Allocator, words: []const []const u8, qs: []const []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (words) |w| {
        var said: usize = 0;
        for (qs) |q| said += @intFromBool(wordIn(&.{w}, q).len > 0);
        if (3 * said >= qs.len and said > 0) try out.append(a, w);
    }
    return out.toOwnedSlice(a);
}

/// The subject's proper names: written with a capital past a sentence's first word, or opening the fact.
pub fn named(a: std.mem.Allocator, words: []const []const u8, fact: []const u8, qs: []const []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (words) |w| {
        var proper = capitalInside(fact, w) or opens(fact, w);
        for (qs) |q| proper = proper or capitalInside(q, w);
        if (proper) try out.append(a, w);
    }
    return out.toOwnedSlice(a);
}

/// Whether `text` opens with `word` written with a capital (a subject word, never a pronoun or an article).
fn opens(text: []const u8, word: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n,.!?;:\"()");
    const first = possessor(it.next() orelse return false);
    return std.mem.eql(u8, first, word) and std.ascii.isUpper(first[0]);
}

/// A fact's subject words: the phrase's its questions say; a document's other proper names join a phrase with none.
pub fn factSubject(a: std.mem.Allocator, phrase: []const u8, fact: []const u8, qs: []const []const u8, document: bool) ![]const []const u8 {
    const own = try subjectWords(a, phrase);
    const proper = try named(a, own, fact, qs);
    // a told fact's other proper name is its answer ("My dog is called Ginger."): never a word its questions must say
    const theirs = try asked(a, if (document) try properWords(a, own, fact) else own, qs);
    if (proper.len == 0) return theirs;
    const kept = try asked(a, own, qs);
    for (kept) |w| for (proper) |p| if (std.mem.eql(u8, w, p)) return kept;
    // a name the fact ends on is what its questions ask for ("managed by Ines Calloway"): the names they say instead
    if (!opens(fact, proper[0]) and theirs.len > 0) return theirs;
    return proper;
}

/// The fact's own capitalised words, which join its subject's when its questions say them too ("Quillmoor").
pub fn properWords(a: std.mem.Allocator, phrase_words: []const []const u8, fact: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    try out.appendSlice(a, phrase_words);
    var it = std.mem.tokenizeAny(u8, fact, " \t\r\n,.!?;:\"()");
    next: while (it.next()) |t| {
        const w = possessor(t);
        if (w.len < 2 or !std.ascii.isUpper(w[0])) continue;
        for (unkeyed ++ small) |u| if (std.ascii.eqlIgnoreCase(w, u)) continue :next;
        for (out.items) |x| if (std.ascii.eqlIgnoreCase(x, w)) continue :next;
        try out.append(a, w);
    }
    return out.toOwnedSlice(a);
}

/// Whether `text` says `word` with a capital letter anywhere past its first word.
fn capitalInside(text: []const u8, word: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n,.!?;:\"()");
    _ = it.next() orelse return false;
    while (it.next()) |t| {
        const bare = possessor(t);
        if (std.ascii.eqlIgnoreCase(bare, word) and std.ascii.isUpper(bare[0])) return true;
    }
    return false;
}

/// Whether `question` names the subject: one of its proper names when it has any, else one of its words.
pub fn names(proper: []const []const u8, words: []const []const u8, question: []const u8) bool {
    return wordIn(if (proper.len > 0) proper else words, question).len > 0;
}

/// The word a question keys on: the first proper name it says (read alike however the name goes on), else the last.
pub fn keyWord(words: []const []const u8, proper: []const []const u8, question: []const u8) []const u8 {
    if (proper.len == 0) return wordIn(words, question);
    var it = std.mem.tokenizeAny(u8, question, " \t\r\n,.!?;:\"");
    while (it.next()) |w| {
        const bare = possessor(w);
        for (proper) |p| if (std.ascii.eqlIgnoreCase(bare, p)) return p;
    }
    return "";
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
    const ran = try subjectWords(a, "Quillmoor Labs team chat runs on Zulip");
    defer a.free(ran);
    const qs = [_][]const u8{ "What does Quillmoor use for team chat?", "Which app does the Quillmoor Labs team chat on?", "Where is Quillmoor's team chat?" };
    const own = try asked(a, ran, &qs);
    defer a.free(own);
    try std.testing.expectEqual(@as(usize, 4), own.len);
    const proper = try named(a, own, "Quillmoor Labs team chat runs on Zulip.", &qs);
    defer a.free(proper);
    try std.testing.expectEqual(@as(usize, 2), proper.len);
    try std.testing.expect(names(proper, own, "Which day does the Quillmoor engineering team deploy?"));
    try std.testing.expect(!names(proper, own, "When does the team deploy?"));
    try std.testing.expectEqualStrings("Quillmoor", keyWord(own, proper, "What time does the Quillmoor Labs team deploy?"));
    try std.testing.expectEqualStrings("Quillmoor", keyWord(own, proper, "What time does the Quillmoor team deploy?"));
    const none = try named(a, mine, "My sister is called Ana.", &.{"What is my sister's name?"});
    defer a.free(none);
    try std.testing.expectEqual(@as(usize, 0), none.len);
    try std.testing.expect(names(none, mine, "What is my sister's name?"));
    try std.testing.expectEqualStrings("sister", keyWord(mine, none, "What is my sister's name?"));
    const more = try properWords(a, mine, "My sister is called Ana.");
    defer a.free(more);
    try std.testing.expectEqual(@as(usize, 2), more.len);
}

test "a fact's subject keeps its own proper names; a phrase without any takes the fact's" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const other_way = [_][]const u8{ "Who founded Quillmoor Labs?", "Who started Quillmoor Labs?", "Who is behind Quillmoor Labs?" };
    const odile = try factSubject(a, "Odile Marchetti", "Odile Marchetti founded Quillmoor Labs.", &other_way, true);
    try std.testing.expectEqual(@as(usize, 2), odile.len);
    try std.testing.expectEqualStrings("Odile", odile[0]);
    const chat = [_][]const u8{ "What does Quillmoor use for team chat?", "Which app runs the Quillmoor team chat?", "Where is the Quillmoor team chat?" };
    const words = try factSubject(a, "team chat", "Quillmoor Labs team chat runs on Zulip.", &chat, true);
    try std.testing.expectEqualStrings("Quillmoor", keyWord(words, try named(a, words, "Quillmoor Labs team chat runs on Zulip.", &chat), chat[0]));
    for (words) |w| try std.testing.expect(!std.mem.eql(u8, w, "Zulip"));
    const rota = [_][]const u8{ "Who manages the Quillmoor Labs support rota?", "Who runs Quillmoor's support rota?", "Who is in charge of the Quillmoor support rota?", "Who is Ines Calloway?" };
    const ines = try factSubject(a, "Ines Calloway", "The Quillmoor Labs support rota is managed by Ines Calloway.", &rota, true);
    try std.testing.expectEqualStrings("Quillmoor", ines[0]);
    const dog = [_][]const u8{ "What is my dog called?", "What is Ginger's name?", "What is my dog's name, Ginger?" };
    const told = try factSubject(a, "My dog", "My dog is called Ginger.", &dog, false);
    try std.testing.expectEqual(@as(usize, 1), told.len);
    try std.testing.expectEqualStrings("dog", told[0]);
}

const unkeyed = [_][]const u8{ "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "you", "your", "yours", "it", "its", "this", "that", "these", "those", "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "do", "does", "did", "is", "are", "was", "like", "likes" };

const small = [_][]const u8{ "the", "a", "an", "of", "on", "in", "at", "to", "for", "by", "with", "from", "and", "or" };

/// The row of `word`'s last token in an example's question, before its reply (a recall line's is passed over); else 0.
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

/// The row before `value` (a short answer) first appears in an example's answer, as written or its first word; else 0.
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
    if (slide_span.find(q, subject) == null) return &.{};
    var out: std.ArrayList([]const u8) = .empty;
    for (try kindsList(srv, cx, subject, gone)) |kind| if (try slide_span.replace(a, q, subject, kind)) |other| try out.append(a, other);
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
