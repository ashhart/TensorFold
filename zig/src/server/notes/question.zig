//! Whether a user's chat message asks rather than tells (English only), so the chat's own write keeps it out.
const std = @import("std");

/// Every code point Unicode 16 names a question mark (punctuation and symbols), and the interrobangs.
const marks = [_]u21{ 0x3F, 0xBF, 0x37E, 0x55E, 0x61F, 0x1367, 0x1945, 0x203D, 0x2047, 0x2048, 0x2049, 0x2753, 0x2754, 0x2CFA, 0x2CFB, 0x2E2E, 0x2E54, 0xA60F, 0xA6F7, 0xFE16, 0xFE56, 0xFF1F, 0x11143, 0x1E95F, 0x1FBC4 };

/// The first word of a question (lower case): the interrogatives and the auxiliaries a question opens with.
const openers = [_][]const u8{ "who", "what", "where", "when", "why", "how", "which", "whose", "whom", "is", "are", "am", "was", "were", "do", "does", "did", "can", "could", "will", "would", "should", "shall", "may", "might", "have", "has", "had" };

/// Requests that ask for something told back, whole words at the start (lower case).
const requests = [_][]const u8{ "tell me", "remind me", "do you know", "show me", "list", "explain" };

fn isMark(cp: u21) bool {
    return std.mem.indexOfScalar(u21, &marks, cp) != null;
}

/// Python's str.isspace().
fn isSpace(cp: u21) bool {
    return switch (cp) {
        '\t', '\n', 0x0b, 0x0c, '\r', 0x1c...0x1f, ' ', 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

/// What may close a sentence after its question mark: quotes, brackets, emoji (and their joiners) and "!".
fn isTrailing(cp: u21) bool {
    return switch (cp) {
        '"', '\'', '`', 0xB4, 0x2018...0x201F, 0xAB, 0xBB, 0x2039, 0x203A, 0x301D...0x301F, 0xFF02, 0xFF07 => true,
        '(', ')', '[', ']', '{', '}', '<', '>', 0xFF08, 0xFF09, 0xFF3B, 0xFF3D, 0xFF5B, 0xFF5D, 0x3008...0x3011, 0x3014...0x301B, 0x27E8, 0x27E9 => true,
        '!', 0x203C, 0xFF01 => true,
        0x1F000...0x1FAFF, 0x2600...0x27BF, 0x2B00...0x2BFF, 0x2300...0x23FF, 0xFE00...0xFE0F, 0x200D, 0x20E3, 0xE0000...0xE007F => true,
        else => false,
    };
}

/// Spaces, quotes, brackets, emoji: what is not a word of the message.
pub fn isDecoration(cp: u21) bool {
    return isSpace(cp) or isTrailing(cp);
}

const Cp = struct { cp: u21, len: usize };

fn last(s: []const u8) ?Cp {
    if (s.len == 0) return null;
    var j = s.len - 1;
    while (j > 0 and s.len - j < 4 and s[j] & 0xC0 == 0x80) j -= 1;
    const cp = std.unicode.utf8Decode(s[j..]) catch return .{ .cp = s[s.len - 1], .len = 1 };
    return .{ .cp = cp, .len = s.len - j };
}

fn first(s: []const u8) ?Cp {
    if (s.len == 0) return null;
    const n = std.unicode.utf8ByteSequenceLength(s[0]) catch return .{ .cp = s[0], .len = 1 };
    if (n > s.len) return .{ .cp = s[0], .len = 1 };
    const cp = std.unicode.utf8Decode(s[0..n]) catch return .{ .cp = s[0], .len = 1 };
    return .{ .cp = cp, .len = n };
}

/// Why `text` is a question or a request as an English chat message asks one, or null when it tells.
pub fn reason(text: []const u8) ?[]const u8 {
    // the end: past spaces, quotes, brackets, emoji and "!", is there a question mark?
    var end = text;
    while (last(end)) |c| {
        if (isMark(c.cp)) return "a question: it ends in a question mark";
        if (!isSpace(c.cp) and !isTrailing(c.cp)) break;
        end = end[0 .. end.len - c.len];
    }
    // the start: past spaces, quotes, brackets and emoji (an opening "¿" is a question mark too)
    var s = text;
    while (first(s)) |c| {
        if (isMark(c.cp)) return "a question: it starts with a question mark";
        if (!isSpace(c.cp) and !isTrailing(c.cp)) break;
        s = s[c.len..];
    }
    var buf: [48]u8 = undefined;
    var words = lower(&buf, s);
    if (std.mem.startsWith(u8, words, "please ") or std.mem.startsWith(u8, words, "please, ")) {
        s = std.mem.trimStart(u8, s["please".len..], ", ");
        words = lower(&buf, s);
    }
    for (requests) |r| if (std.mem.startsWith(u8, words, r) and (words.len == r.len or !std.ascii.isAlphanumeric(words[r.len])))
        return "a request: it asks to be told something";
    const w = firstWord(words);
    for (openers) |o| if (std.mem.eql(u8, w, o)) return "a question: it opens as one";
    return null;
}

/// The start of `s` lowered (ASCII), as much as fits in `buf`.
fn lower(buf: []u8, s: []const u8) []const u8 {
    const n = @min(buf.len, s.len);
    for (s[0..n], buf[0..n]) |c, *o| o.* = std.ascii.toLower(c);
    return buf[0..n];
}

/// The first word's letters; a contraction counts as its first part ("what's" what, "isn't" is, "won't" will).
fn firstWord(s: []const u8) []const u8 {
    var n: usize = 0;
    while (n < s.len and std.ascii.isAlphabetic(s[n])) n += 1;
    const w = s[0..n];
    const rest = s[n..];
    const negated = (std.mem.startsWith(u8, rest, "'t") or std.mem.startsWith(u8, rest, "\u{2019}t")) and w.len > 1 and w[w.len - 1] == 'n';
    if (!negated or std.mem.eql(u8, w, "can")) return w;
    if (std.mem.eql(u8, w, "won")) return "will";
    if (std.mem.eql(u8, w, "shan")) return "shall";
    return w[0 .. w.len - 1];
}

/// The rule this replaces (intake.readings' own, as statement() ran it): past trailing spaces, an ASCII "?".
pub fn endsInAsciiQuestionMark(text: []const u8) bool {
    var t = text;
    while (last(t)) |c| {
        if (!isSpace(c.cp)) break;
        t = t[0 .. t.len - c.len];
    }
    return t.len > 0 and t[t.len - 1] == '?';
}

/// Questions and requests a chat message makes, each of which the old trailing-"?" rule wrote as a note.
pub const asks = [_][]const u8{
    "What car does my sister drive",
    "\"Where do I live?\"",
    "(what's my name?)",
    "[Is Anna my sister?]",
    "My sister drives what car\u{FF1F}",
    "My sister drives what car\u{061F}",
    "My sister drives what car\u{2E2E}",
    "My sister drives what car\u{037E}",
    "My sister drives what car?   \t",
    "My sister drives what car? \u{1F697}",
    "My sister drives what car?\u{1F44D}\u{1F3FD}",
    "My sister drives what car \u{2753}",
    "My sister drives what car?!",
    "\u{BF}Anna drives a Volvo",
    "tell me my sister's name",
    "Tell me about colours.",
    "Please tell me where I live.",
    "do you know where I live",
    "Remind me what car my sister drives",
    "Show me my notes.",
    "List my sister's cars",
    "Explain my schedule.",
    "who is my sister",
    "Which car is mine",
    "how old is Anna",
    "Is Anna my sister",
    "Are you sure",
    "Am I late",
    "Was it red",
    "Were we there",
    "Do I live in Helsinki",
    "Does Anna drive a Volvo",
    "Did I tell you about my car.",
    "Can you remember my car",
    "Could it be blue",
    "Will it rain",
    "Would my sister like it",
    "Should I buy a Volvo",
    "Shall we go",
    "May I ask",
    "Might it work",
    "Have I told you about Anna",
    "Has Anna called",
    "Had I known",
    "Whose car is that",
    "Whom did I meet",
    "When is my birthday",
    "Why did I sell my car",
    "Where's my car",
    "Isn't my car red",
    "Don't I live in Helsinki",
    "Doesn't Anna drive a Volvo",
    "Won't she come",
    "Can't I change it",
    "\u{1F697} what car does my sister drive",
};

/// Statements the chat's own write must still take (the controls: none of them may be read as a question).
pub const tells = [_][]const u8{
    "My sister is Anna.",
    "I live in Helsinki",
    "Anna drives a red Volvo",
    "yes, that's right",
    "Yes, that's right.",
    "This is my car.",
    "My favourite colour is green.",
    "Whoever calls, I am out.",
    "Listening to jazz relaxes me.",
    "Explanations bore me.",
    "Telling me twice helps.",
    "Showers wake me up.",
    "Isabel is my sister.",
    "Dogs are my favourite animals!",
    "I said \"why not\" to the dog.",
    "My car is red \u{1F697}",
    "Anna asked me where I live.",
    "Whatever happens, I live in Helsinki.",
};

test "every question and request is caught, and none of them by the trailing-\"?\" rule alone" {
    var old_misses: usize = 0;
    for (asks) |t| {
        errdefer std.debug.print("not caught: {s}\n", .{t});
        try std.testing.expect(reason(t) != null);
        if (!endsInAsciiQuestionMark(t)) old_misses += 1;
    }
    // the old rule caught only the one with spaces after its "?"; a revert to it fails here and in the chat test
    try std.testing.expectEqual(@as(usize, asks.len - 1), old_misses);
}

test "every statement is taken: \"is\" inside a sentence, words that only start like an opener, a confirmation" {
    for (tells) |t| {
        errdefer std.debug.print("read as a question: {s} ({s})\n", .{ t, reason(t).? });
        try std.testing.expect(reason(t) == null);
    }
}

test "the limits NOTES-IN-ENGINE.md names: what it misses, and the statements it skips" {
    const missed = [_][]const u8{ "Miss\u{e4} asun", "Wo wohne ich", "I wonder what my sister drives", "my sister drives what car", "Give me my sister's name", "Help me remember my car" };
    for (missed) |t| try std.testing.expect(reason(t) == null);
    const skipped = [_][]const u8{ "Will is my brother.", "May is my sister's name.", "Do remember that I like tea.", "Have a look: my car is red.", "He asked \"why?\"" };
    for (skipped) |t| try std.testing.expect(reason(t) != null);
}

test "the reasons say which rule caught it" {
    try std.testing.expectEqualStrings("a question: it ends in a question mark", reason("\"Where do I live?\"").?);
    try std.testing.expectEqualStrings("a question: it opens as one", reason("What car does my sister drive").?);
    try std.testing.expectEqualStrings("a request: it asks to be told something", reason("Please, tell me my name").?);
}
