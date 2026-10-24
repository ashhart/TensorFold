//! LLWZ's notes.is_past (zengine/notes.py PAST_CUE): whether a question asks about the past, ported from its regex.
const std = @import("std");

/// The cue words of PAST_CUE (re.I), apart from "used to", which takes whitespace between its words.
const words = [_][]const u8{ "was", "were", "did", "had", "before", "previously", "earlier", "originally", "formerly", "ever", "always", "changed", "until", "till", "since", "ago" };

const Cp = struct { cp: u21, len: u3 };

/// The code point at `i`; a byte that starts no valid sequence reads as itself, one byte long.
fn at(s: []const u8, i: usize) Cp {
    const n = std.unicode.utf8ByteSequenceLength(s[i]) catch return .{ .cp = s[i], .len = 1 };
    if (i + n > s.len) return .{ .cp = s[i], .len = 1 };
    const cp = std.unicode.utf8Decode(s[i..][0..n]) catch return .{ .cp = s[i], .len = 1 };
    return .{ .cp = cp, .len = n };
}

/// The code point that ends just before `i`, or null at the start.
fn before(s: []const u8, i: usize) ?u21 {
    if (i == 0) return null;
    var j = i - 1;
    while (j > 0 and i - j < 4 and s[j] & 0xC0 == 0x80) j -= 1;
    const c = at(s, j);
    return if (j + c.len == i) c.cp else s[i - 1];
}

/// Python's \w on str: alphanumeric or "_"; exact for ASCII and Latin-1, the common punctuation blocks off, else on.
pub fn isWord(cp: u21) bool {
    if (cp < 0x80) return std.ascii.isAlphanumeric(@intCast(cp)) or cp == '_';
    if (cp <= 0xFF) return switch (cp) {
        0xAA, 0xB2, 0xB3, 0xB5, 0xB9, 0xBA, 0xBC, 0xBD, 0xBE => true,
        0xC0...0xD6, 0xD8...0xF6, 0xF8...0xFF => true,
        else => false,
    };
    return switch (cp) {
        0x0300...0x036F, 0x1680, 0x2000...0x206F, 0x20A0...0x20CF, 0x2190...0x23FF, 0x2500...0x27BF => false,
        0x3000...0x3004, 0x3008...0x3020, 0xFE30...0xFE4F, 0xFF01...0xFF0F, 0xFF1A...0xFF20 => false,
        0xFF3B...0xFF40, 0xFF5B...0xFF65, 0xE000...0xF8FF, 0xFFF0...0xFFFF, 0x1F000...0x1FAFF => false,
        else => true,
    };
}

/// Python's \s on str.
fn isSpace(cp: u21) bool {
    return switch (cp) {
        '\t', '\n', 0x0B, 0x0C, '\r', ' ', 0x1C...0x1F, 0x85, 0xA0, 0x1680, 0x2000...0x200A => true,
        0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

/// Python's \d on str, for the decimal digits text is likely to hold (ASCII, Arabic-Indic, fullwidth).
fn isDigit(cp: u21) bool {
    return switch (cp) {
        '0'...'9', 0x0660...0x0669, 0x06F0...0x06F9, 0x0966...0x096F, 0xFF10...0xFF19 => true,
        else => false,
    };
}

/// One letter of a cue word under re.I: either case, and the Unicode letters that fold to it (ſ for s, ı and İ for i).
fn letter(cp: u21, want: u8) bool {
    if (cp < 0x80) return std.ascii.toLower(@intCast(cp)) == want;
    return (want == 's' and cp == 0x017F) or (want == 'i' and (cp == 0x0131 or cp == 0x0130)) or (want == 'k' and cp == 0x212A);
}

/// The cue word's letters from `i`: the end of the match, or null.
fn spelled(s: []const u8, i: usize, word: []const u8) ?usize {
    var j = i;
    for (word) |w| {
        if (j >= s.len) return null;
        const c = at(s, j);
        if (!letter(c.cp, w)) return null;
        j += c.len;
    }
    return j;
}

fn endsWord(s: []const u8, j: usize) bool {
    return j >= s.len or !isWord(at(s, j).cp);
}

/// A match of PAST_CUE starting at `i`, which follows a word boundary.
fn cueAt(s: []const u8, i: usize) bool {
    for (words) |w| if (spelled(s, i, w)) |j| if (endsWord(s, j)) return true;
    if (spelled(s, i, "used")) |j0| {
        var j = j0;
        var gap: usize = 0;
        while (j < s.len) {
            const c = at(s, j);
            if (!isSpace(c.cp)) break;
            j += c.len;
            gap += 1;
        }
        if (gap > 0) if (spelled(s, j, "to")) |end| if (endsWord(s, end)) return true;
    }
    // \b(?:19|20)\d\d\b
    if (i + 2 <= s.len and (std.mem.eql(u8, s[i..][0..2], "19") or std.mem.eql(u8, s[i..][0..2], "20"))) {
        var j = i + 2;
        var k: usize = 0;
        while (k < 2) : (k += 1) {
            if (j >= s.len) return false;
            const c = at(s, j);
            if (!isDigit(c.cp)) return false;
            j += c.len;
        }
        return endsWord(s, j);
    }
    return false;
}

/// notes.is_past(question): PAST_CUE.search(question) is not None.
pub fn isPast(s: []const u8) bool {
    var i: usize = 0;
    while (i < s.len) {
        const c = at(s, i);
        const prev = before(s, i);
        if (isWord(c.cp) and (prev == null or !isWord(prev.?)) and cueAt(s, i)) return true;
        i += c.len;
    }
    return false;
}

/// CPython's notes.is_past on each text, checked by tools/notes/past_cases.py (CPython 3.12.8 and 3.14.3 agree).
const cases = [_]struct { []const u8, bool }{
    .{ "In which city is my home?", false },
    .{ "What colour did my eyes use to be?", true },
    .{ "What was my sister's name on 1 March 2025?", true },
    .{ "Where do I live now?", false },
    .{ "Has my brother ever lived in Oslo?", true },
    .{ "Did my cat have a name?", true },
    .{ "everything is fine", false },
    .{ "whatever", false },
    .{ "I used to swim.", true },
    .{ "I used  \t to swim.", true },
    .{ "I usedto swim.", false },
    .{ "used\u{a0}to", true },
    .{ "used\u{2028}to", true },
    .{ "used\x1cto", true },
    .{ "d\u{131}d", true },
    .{ "d\u{130}d", true },
    .{ "\u{17f}ince", true },
    .{ "x\u{2019}did\u{2019}", true },
    .{ "caf\u{e9}did", false },
    .{ "\u{b2}did", false },
    .{ "did\u{301}", true },
    .{ "was_", false },
    .{ "_was", false },
    .{ "2019", true },
    .{ "in 1999.", true },
    .{ "20199", false },
    .{ "a2019", false },
    .{ "\u{e9}2019", false },
    .{ "20\u{ff11}\u{ff19}", true },
    .{ "\u{ff12}\u{ff10}\u{ff11}\u{ff19}", false },
    .{ "\u{c9}VER", false },
    .{ "Ever?", true },
    .{ "eVeR!", true },
    .{ "UNTIL", true },
    .{ "until\u{e9}", false },
    .{ "ago\u{1f600}", true },
    .{ "\u{4e2d}was", false },
    .{ "\u{3001}was\u{3002}", true },
    .{ "What is my favourite colour?", false },
    .{ "Which colour did I like before blue?", true },
};

test "is_past answers as CPython's notes.is_past" {
    for (cases) |c| {
        std.testing.expectEqual(c[1], isPast(c[0])) catch |e| {
            std.debug.print("is_past({s}) should be {}\n", .{ c[0], c[1] });
            return e;
        };
    }
}
