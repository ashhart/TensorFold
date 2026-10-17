//! Caption and lyrics to the AR's prompt ids, as ComfyUI's comfy/ldm/minimax_music/prompt.py builds them, by hand.
const std = @import("std");
const core = @import("core");

const Tokenizer = core.tokenizer.Tokenizer;
const List = std.ArrayList(u8);

pub const audio_cfg: u32 = 151654;
pub const specials = [_]struct { []const u8, u32 }{
    .{ "<|im_start|>", 151644 },    .{ "<|im_end|>", 151645 },       .{ "<|audio_cfg|>", 151654 },
    .{ "<|audio_start|>", 151669 }, .{ "<|audio_end|>", 151670 },    .{ "<|caption_start|>", 151671 },
    .{ "<|caption_end|>", 151672 }, .{ "<|lyrics_start|>", 151673 }, .{ "<|lyrics_end|>", 151674 },
};

/// Python's str.isspace().
fn isSpaceCp(cp: u21) bool {
    return (cp >= 0x9 and cp <= 0xD) or (cp >= 0x1C and cp <= 0x20) or cp == 0x85 or cp == 0xA0 or cp == 0x1680 or
        (cp >= 0x2000 and cp <= 0x200A) or cp == 0x2028 or cp == 0x2029 or cp == 0x202F or cp == 0x205F or cp == 0x3000;
}

/// The code point at s[i] and its byte length (invalid bytes read as one byte, never space).
fn at(s: []const u8, i: usize) struct { cp: u21, n: usize } {
    const n = std.unicode.utf8ByteSequenceLength(s[i]) catch return .{ .cp = 0xFFFD, .n = 1 };
    if (i + n > s.len) return .{ .cp = 0xFFFD, .n = 1 };
    const cp = std.unicode.utf8Decode(s[i..][0..n]) catch return .{ .cp = 0xFFFD, .n = 1 };
    return .{ .cp = cp, .n = n };
}

fn spaceLen(s: []const u8, i: usize) usize {
    if (i >= s.len) return 0;
    const c = at(s, i);
    return if (isSpaceCp(c.cp)) c.n else 0;
}

/// End of the run of white space at i.
fn skipSpace(s: []const u8, from: usize) usize {
    var i = from;
    while (true) {
        const n = spaceLen(s, i);
        if (n == 0) return i;
        i += n;
    }
}

fn strip(s: []const u8) []const u8 {
    const a = skipSpace(s, 0);
    var b = s.len;
    while (b > a) {
        var c = b - 1;
        while (c > a and s[c] & 0xC0 == 0x80) c -= 1;
        if (spaceLen(s, c) == 0) break;
        b = c;
    }
    return s[a..b];
}

fn rstrip(s: []const u8) []const u8 {
    var b = s.len;
    while (b > 0) {
        var c = b - 1;
        while (c > 0 and s[c] & 0xC0 == 0x80) c -= 1;
        if (spaceLen(s, c) == 0) break;
        b = c;
    }
    return s[0..b];
}

/// `<\|([^|]*)\|>` -> "a is b" (inner split once on white space) or the stripped inner.
fn tagWords(a: std.mem.Allocator, text: []const u8, out: *List) !void {
    var i: usize = 0;
    while (i < text.len) {
        if (std.mem.startsWith(u8, text[i..], "<|")) {
            var j = i + 2;
            while (j < text.len and text[j] != '|') j += 1;
            if (j + 1 < text.len and text[j + 1] == '>') {
                const inner = strip(text[i + 2 .. j]);
                var k: usize = 0;
                while (k < inner.len and spaceLen(inner, k) == 0) k += at(inner, k).n;
                if (k < inner.len) {
                    const rest = inner[skipSpace(inner, k)..];
                    try out.appendSlice(a, inner[0..k]);
                    try out.appendSlice(a, " is ");
                    try out.appendSlice(a, rest);
                } else try out.appendSlice(a, inner);
                i = j + 2;
                continue;
            }
        }
        try out.append(a, text[i]);
        i += 1;
    }
}

/// Python's str.splitlines() boundaries: \n \r \r\n \v \f \x1c \x1d \x1e \x85    .
fn lineBreak(s: []const u8, i: usize) usize {
    switch (s[i]) {
        '\n', 0x0B, 0x0C, 0x1C, 0x1D, 0x1E => return 1,
        '\r' => return if (i + 1 < s.len and s[i + 1] == '\n') 2 else 1,
        else => {},
    }
    const c = at(s, i);
    return if (c.cp == 0x85 or c.cp == 0x2028 or c.cp == 0x2029) c.n else 0;
}

/// One line without its heading marks, bullet, bold and italic stars, right-stripped.
fn plainLine(a: std.mem.Allocator, line0: []const u8, out: *List) !void {
    var line = line0;
    // ^\s{0,3}#{1,6}\s+
    {
        var i: usize = 0;
        var ws: usize = 0;
        while (ws < 3) : (ws += 1) {
            const n = spaceLen(line, i);
            if (n == 0) break;
            i += n;
        }
        var h: usize = 0;
        while (i + h < line.len and line[i + h] == '#') h += 1;
        if (h >= 1 and h <= 6 and spaceLen(line, i + h) > 0) line = line[skipSpace(line, i + h)..];
    }
    // ^\s*[*+-]\s+
    {
        const i = skipSpace(line, 0);
        if (i < line.len and (line[i] == '*' or line[i] == '+' or line[i] == '-') and spaceLen(line, i + 1) > 0) line = line[skipSpace(line, i + 1)..];
    }
    var cur: List = .empty;
    try cur.appendSlice(a, line);
    // while "**" in line: \*\*([^*]+)\*\* -> \1 until nothing changes
    while (std.mem.indexOf(u8, cur.items, "**") != null) {
        var next: List = .empty;
        var i: usize = 0;
        var changed = false;
        const s = cur.items;
        while (i < s.len) {
            if (i + 1 < s.len and s[i] == '*' and s[i + 1] == '*') {
                var j = i + 2;
                while (j < s.len and s[j] != '*') j += 1;
                if (j > i + 2 and j + 1 < s.len and s[j + 1] == '*') {
                    try next.appendSlice(a, s[i + 2 .. j]);
                    i = j + 2;
                    changed = true;
                    continue;
                }
            }
            try next.append(a, s[i]);
            i += 1;
        }
        cur = next;
        if (!changed) break;
    }
    // (?<!\*)\*([^*\n]+)\*(?!\*) -> \1
    {
        const s = cur.items;
        var i: usize = 0;
        while (i < s.len) {
            if (s[i] == '*' and (i == 0 or s[i - 1] != '*')) {
                var j = i + 1;
                while (j < s.len and s[j] != '*' and s[j] != '\n') j += 1;
                if (j > i + 1 and j < s.len and s[j] == '*' and (j + 1 >= s.len or s[j + 1] != '*')) {
                    try out.appendSlice(a, s[i + 1 .. j]);
                    i = j + 1;
                    continue;
                }
            }
            try out.append(a, s[i]);
            i += 1;
        }
    }
}

fn isDash(c: u8) bool {
    return c == '-' or c == '*' or c == '_';
}

/// re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE).
fn dropRules(a: std.mem.Allocator, s: []const u8, out: *List) !void {
    var i: usize = 0;
    while (i <= s.len) {
        const line_start = i == 0 or s[i - 1] == '\n';
        if (line_start) {
            const k = skipSpace(s, i);
            var r: usize = 0;
            while (k + r < s.len and isDash(s[k + r])) r += 1;
            if (r >= 3) {
                const m = k + r;
                const e = skipSpace(s, m);
                // greedy \s* then $ (end, or before a newline): the last such position in [m, e]
                var p = e;
                var found = false;
                while (true) {
                    if (p == s.len or s[p] == '\n') {
                        found = true;
                        break;
                    }
                    if (p == m) break;
                    p -= 1;
                    while (p > m and s[p] & 0xC0 == 0x80) p -= 1;
                }
                if (found) {
                    i = p; // the next search starts where this match ended (a match is never empty)
                    continue;
                }
            }
        }
        if (i == s.len) break;
        try out.append(a, s[i]);
        i += 1;
    }
}

fn replaceAll(a: std.mem.Allocator, s: []const u8, from: []const u8, to: []const u8) ![]u8 {
    var out: List = .empty;
    var i: usize = 0;
    while (i < s.len) {
        if (std.mem.startsWith(u8, s[i..], from)) {
            try out.appendSlice(a, to);
            i += from.len;
        } else {
            try out.append(a, s[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(a);
}

/// clean_caption: tags to words, markdown off, rules dropped, bullets and 4-space indents off, blank lines merged.
pub fn cleanCaption(a: std.mem.Allocator, caption: []const u8) ![]u8 {
    var tagged: List = .empty;
    try tagWords(a, caption, &tagged);
    var joined: List = .empty;
    const t = tagged.items;
    var i: usize = 0;
    var first = true;
    while (i < t.len) {
        var j = i;
        while (j < t.len and lineBreak(t, j) == 0) j += 1;
        if (!first) try joined.append(a, '\n');
        first = false;
        var line: List = .empty;
        try plainLine(a, t[i..j], &line);
        try joined.appendSlice(a, rstrip(line.items));
        if (j >= t.len) break;
        i = j + lineBreak(t, j);
    }
    var ruled: List = .empty;
    try dropRules(a, joined.items, &ruled);
    const b = try replaceAll(a, ruled.items, "\u{2022} ", "");
    const c = try replaceAll(a, b, "    ", "");
    var out: List = .empty;
    var k: usize = 0;
    while (k < c.len) {
        if (c[k] == '\n') {
            var e = k;
            while (e < c.len and c[e] == '\n') e += 1;
            try out.append(a, '\n');
            k = e;
            continue;
        }
        try out.append(a, c[k]);
        k += 1;
    }
    return out.toOwnedSlice(a);
}

/// normalize_lyrics: split on `\s*(\[[^\]]+\])\s*`, tags lower-cased (ASCII), pieces on lines, " ^ " a line break.
pub fn normalizeLyrics(a: std.mem.Allocator, lyrics: []const u8) ![]u8 {
    var parts: std.ArrayList([]const u8) = .empty;
    var last: usize = 0;
    var i: usize = 0;
    while (i < lyrics.len) {
        const k = skipSpace(lyrics, i);
        if (k < lyrics.len and lyrics[k] == '[') {
            var j = k + 1;
            while (j < lyrics.len and lyrics[j] != ']') j += 1;
            if (j > k + 1 and j < lyrics.len) {
                try parts.append(a, lyrics[last..i]);
                try parts.append(a, lyrics[k .. j + 1]);
                const e = skipSpace(lyrics, j + 1);
                last = e;
                i = e;
                continue;
            }
        }
        i += at(lyrics, i).n;
    }
    try parts.append(a, lyrics[last..]);
    var joined: List = .empty;
    var n: usize = 0;
    for (parts.items) |p| {
        if (p.len == 0) continue;
        if (n > 0) try joined.append(a, '\n');
        n += 1;
        if (p[0] == '[') {
            for (p) |c| try joined.append(a, std.ascii.toLower(c));
        } else try joined.appendSlice(a, p);
    }
    const body = try replaceAll(a, joined.items, " ^ ", "\n");
    return std.mem.concat(a, u8, &.{ "[start]\n", body });
}

pub fn promptText(a: std.mem.Allocator, caption: []const u8, lyrics: []const u8) ![]u8 {
    return std.mem.concat(a, u8, &.{ "<|im_start|><|caption_start|>", try cleanCaption(a, caption), "<|caption_end|><|lyrics_start|>", try normalizeLyrics(a, lyrics), "<|lyrics_end|><|im_end|><|audio_start|>" });
}

/// The special tokens must have the reference's ids.
pub fn checkTokenizer(tok: *const Tokenizer) !void {
    for (specials) |s| {
        const id = tok.specialTokenId(s[0]) orelse return error.TokenizerMismatch;
        if (id != s[1]) {
            std.log.err("music3: tokenizer {s} is {d}, not {d}", .{ s[0], id, s[1] });
            return error.TokenizerMismatch;
        }
    }
}

/// The conditional prompt's ids (caller frees with gpa).
pub fn encode(gpa: std.mem.Allocator, tok: *const Tokenizer, caption: []const u8, lyrics: []const u8) ![]u32 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const text = try promptText(arena.allocator(), caption, lyrics);
    const ids = try tok.encode(gpa, text);
    if (ids.len > 5000) {
        gpa.free(ids);
        return error.PromptTooLong;
    }
    if (ids.len < 4) {
        gpa.free(ids);
        return error.PromptTooShort;
    }
    return ids;
}

/// The unconditional row: first token, audio_cfg for the middle, the last two.
pub fn unconditional(gpa: std.mem.Allocator, ids: []const u32) ![]u32 {
    const out = try gpa.dupe(u32, ids);
    for (out[1 .. out.len - 2]) |*x| x.* = audio_cfg;
    return out;
}

test "caption and lyrics cleanup" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("tempo is 92 bpm\nbold and it", try cleanCaption(a, "<| tempo  92 bpm |>\n\n## **bold** and *it*"));
    try std.testing.expectEqualStrings("a\nb", try cleanCaption(a, "a\n---\nb"));
    try std.testing.expectEqualStrings("item one\nx", try cleanCaption(a, "- item one   \n    x"));
    try std.testing.expectEqualStrings("[start]\n[verse]\nhello world\n[chorus]\nla la", try normalizeLyrics(a, "[Verse] hello world\n\n[CHORUS]\nla la"));
    try std.testing.expectEqualStrings("[start]\nline one\nline two", try normalizeLyrics(a, "line one ^ line two"));
}
