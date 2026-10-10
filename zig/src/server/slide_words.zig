//! A lesson's words: questions and answers read from replies; whether a reply recalls, leaks, changes or loops.
const std = @import("std");
const log = @import("log.zig");
const Allocator = std.mem.Allocator;

const max_words = 25; // an answer's words at most

/// Lines that read as one short question, numbering and bullets gone, without repeats.
pub fn questions(a: Allocator, text: []const u8, n: usize) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw| {
        const line = unmark(raw);
        if (!(std.mem.endsWith(u8, line, "?") or std.mem.endsWith(u8, line, "\u{ff1f}")) or words(line) < 2 or words(line) > 20) continue;
        for (out.items) |seen| {
            if (std.mem.eql(u8, seen, line)) break;
        } else try out.append(a, line);
        if (out.items.len == n) break;
    }
    return out.items;
}

/// Up to n numbered or bulleted lines, numbering and bullets gone, empty lines skipped.
pub fn numbered(a: Allocator, text: []const u8, n: usize) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw| {
        const line = unmark(raw);
        if (line.len == 0) continue;
        try out.append(a, line);
        if (out.items.len == n) break;
    }
    return out.items;
}

/// A line without its list number or bullet, spaces, quotes and bold marks.
pub fn unmark(raw: []const u8) []const u8 {
    var line = std.mem.trim(u8, raw, " \t\r");
    var digits: usize = 0;
    while (digits < line.len and std.ascii.isDigit(line[digits])) digits += 1;
    if (digits > 0 and digits < line.len and (line[digits] == '.' or line[digits] == ')')) {
        line = line[digits + 1 ..];
    } else if (line.len > 0 and (line[0] == '-' or line[0] == '*')) line = line[1..];
    return std.mem.trim(u8, line, " \t\"*");
}

/// An answer kept only when it is one or two finished sentences of at most max_words; never repaired.
pub fn clean(text: []const u8) ?[]const u8 {
    const t = std.mem.trim(u8, text, " \t\r\n");
    var ends: usize = 0;
    var cut = t.len;
    var i: usize = 0;
    while (i < t.len) {
        const end = sentenceEnd(t, i);
        i += if (end > 0) end else 1;
        if (end == 0) continue;
        ends += 1;
        if (ends == 2) {
            cut = i;
            break;
        }
    }
    const kept = t[0..cut];
    if (kept.len == 0 or !endsSentence(kept) or words(kept) > max_words) return null;
    return kept;
}

/// The bytes of a sentence's end at `i`: ASCII . ! ? before a space or the end, or 。！？ wherever they stand; else 0.
fn sentenceEnd(t: []const u8, i: usize) usize {
    const ch = t[i];
    if (ch == '.' or ch == '!' or ch == '?') return if (i + 1 == t.len or std.ascii.isWhitespace(t[i + 1])) 1 else 0;
    if (ch < 0x80) return 0;
    const len = std.unicode.utf8ByteSequenceLength(ch) catch return 0;
    if (i + len > t.len) return 0;
    const c = std.unicode.utf8Decode(t[i..][0..len]) catch return 0;
    return if (fullStop(c)) len else 0;
}

/// Whether text ends as a finished sentence: . ! ? or 。！？
fn endsSentence(t: []const u8) bool {
    if (std.mem.indexOfScalar(u8, ".!?", t[t.len - 1]) != null) return true;
    var start = t.len - 1;
    while (start > 0 and t[start] & 0xC0 == 0x80) start -= 1;
    const c = std.unicode.utf8Decode(t[start..]) catch return false;
    return fullStop(c);
}

/// 。！？ and their half- and full-width kin: a sentence ends there, with or without a space after.
fn fullStop(c: u21) bool {
    return c == 0x3002 or c == 0xFF01 or c == 0xFF1F or c == 0xFF0E or c == 0xFF61;
}

/// Content words a fact's teller would not say by chance, compared without case or British spellings.
const common = [_][]const u8{ "that", "this", "with", "from", "have", "been", "were", "they", "them", "their", "there", "what", "when", "which", "where", "about", "would", "could", "should", "called", "named", "also", "just", "very", "into", "your", "mine", "ours", "will" };

/// Whether `reply` says a word of `fact` that neither `question` nor `before` says.
pub fn tells(fact: []const u8, question: []const u8, before: []const u8, reply: []const u8) bool {
    var it: Units = .init(fact);
    while (it.next()) |u| {
        if (!u.marked() or says(question, u.word) or says(before, u.word)) continue;
        if (says(reply, u.word)) return true;
    }
    return false;
}

/// Whether `reply` says every word the held answer took from the fact that its question does not say (of the
/// character pairs a script without spaces compares by, half: a pair can straddle a particle a reply rewords).
pub fn recalls(fact: []const u8, question: []const u8, answer: []const u8, reply: []const u8) bool {
    var key: usize = 0;
    var pairs: usize = 0;
    var found: usize = 0;
    var it: Units = .init(answer);
    while (it.next()) |u| {
        if (!u.marked() or !says(fact, u.word) or says(question, u.word)) continue;
        key += 1;
        const there = says(reply, u.word);
        if (!u.paired and !there) return false;
        pairs += @intFromBool(u.paired);
        found += @intFromBool(u.paired and there);
    }
    if (pairs > 0) return 2 * found >= pairs;
    return key > 0 or tells(fact, question, "", reply);
}

/// Whether `near` says the fact's answer: a word the answers took from the fact that none of the questions says (teal,
/// Ana), not merely the fact's topic (favourite colour), which a near miss shares on purpose.
pub fn gives(fact: []const u8, question: []const u8, answer: []const u8, near: []const u8) bool {
    var it: Units = .init(answer);
    while (it.next()) |u| if (u.marked() and says(fact, u.word) and !says(question, u.word) and says(near, u.word)) return true;
    return false;
}

/// Whether `answer` says a word of `fact` that `question` does not: else the question gives its own answer away.
pub fn asks(fact: []const u8, question: []const u8, answer: []const u8) bool {
    var it: Units = .init(answer);
    while (it.next()) |u| if (u.marked() and says(fact, u.word) and !says(question, u.word)) return true;
    return false;
}

/// Whether a question asks yes or no of the user's own life (Is it blue? Do I like it?), not "Do you remember...?".
pub fn yesNo(question: []const u8) bool {
    const verbs = [_][]const u8{ "is", "are", "am", "was", "were", "do", "does", "did", "have", "has", "had", "can", "could", "will", "would", "should" };
    var it = std.mem.tokenizeAny(u8, question, delimiters);
    const first = it.next() orelse return false;
    const second = it.next() orelse return false;
    for (verbs) |v| if (std.ascii.eqlIgnoreCase(first, v)) return !std.ascii.eqlIgnoreCase(second, "you");
    return false;
}

/// Whether a question is still the user's own: it says I or my, never your, and you only once ("Do you remember").
pub fn firstPerson(question: []const u8) bool {
    var mine = false;
    var you: usize = 0;
    var it = std.mem.tokenizeAny(u8, question, delimiters);
    while (it.next()) |w| {
        for ([_][]const u8{ "i", "my", "me", "mine", "our", "we", "us", "i'm", "i've" }) |f| mine = mine or std.ascii.eqlIgnoreCase(w, f);
        for ([_][]const u8{ "your", "yours", "yourself", "you're", "you've" }) |f| if (std.ascii.eqlIgnoreCase(w, f)) return false;
        you += @intFromBool(std.ascii.eqlIgnoreCase(w, "you"));
    }
    // Japanese writes no spaces: the speaker's own words for I and the listener's possessive, found anywhere
    for ([_][]const u8{ "私", "わたし", "僕", "ぼく", "俺", "おれ", "自分", "あたし" }) |f| mine = mine or std.mem.indexOf(u8, question, f) != null;
    for ([_][]const u8{ "あなたの", "君の", "きみの" }) |f| if (std.mem.indexOf(u8, question, f) != null) return false;
    return mine and you <= 1;
}

/// Whether `question` says a word of `fact` worth comparing (else the fact cannot answer it).
pub fn shares(fact: []const u8, question: []const u8) bool {
    var it: Units = .init(question);
    while (it.next()) |u| if (u.notable() and says(fact, u.word)) return true;
    return false;
}

/// Whether a reply still says what it said: the same opening, or four in five of its first notable words.
pub fn alike(before: []const u8, reply: []const u8) bool {
    if (sameOpening(before, reply)) return true;
    var seen: usize = 0;
    var found: usize = 0;
    var it: Units = .init(before);
    var count: usize = 0;
    while (it.next()) |u| : (count += 1) {
        if (count == 20) break;
        if (!u.notable()) continue;
        seen += 1;
        found += @intFromBool(says(reply, u.word));
    }
    return seen >= 3 and found * 5 >= seen * 4;
}

/// Whether a question is about the model itself (you, your): no fact the user tells answers it.
pub fn toModel(question: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, question, delimiters);
    while (it.next()) |w| if (std.ascii.eqlIgnoreCase(w, "you") or std.ascii.eqlIgnoreCase(w, "your")) return true;
    return false;
}

/// The question asked of the model instead (my as your, I as you), or null when nothing in it is first person.
pub fn addressed(a: Allocator, question: []const u8) !?[]const u8 {
    const swaps = [_][2][]const u8{ .{ "my", "your" }, .{ "i", "you" }, .{ "me", "you" }, .{ "mine", "yours" }, .{ "our", "your" }, .{ "we", "you" }, .{ "us", "you" }, .{ "myself", "yourself" }, .{ "i'm", "you're" }, .{ "i've", "you've" } };
    var out: std.ArrayList(u8) = .empty;
    var swapped = false;
    var at: usize = 0;
    while (at < question.len) {
        var end = at;
        while (end < question.len and (std.ascii.isAlphabetic(question[end]) or question[end] == '\'')) end += 1;
        if (end == at) {
            try out.append(a, question[at]);
            at += 1;
            continue;
        }
        const word = question[at..end];
        const to = for (swaps) |s| {
            if (std.ascii.eqlIgnoreCase(word, s[0])) break s[1];
        } else null;
        if (to) |t| {
            swapped = true;
            const first_person = std.ascii.toLower(word[0]) == 'i';
            const upper = std.ascii.isUpper(word[0]) and (at == 0 or !first_person);
            try out.append(a, if (upper) std.ascii.toUpper(t[0]) else t[0]);
            try out.appendSlice(a, t[1..]);
        } else try out.appendSlice(a, word);
        at = end;
    }
    return if (swapped) out.items else null;
}

/// The question asked about someone the user knows (my X as my sister's X, do I as does my sister), or null if neither.
pub fn about(a: Allocator, question: []const u8, who: []const u8) !?[]const u8 {
    const people = [_][]const u8{ "sister", "brother", "mother", "father", "mum", "mom", "dad", "friend", "wife", "husband", "partner", "son", "daughter", "boss", "colleague", "neighbo" };
    const turns = [_][2][]const u8{ .{ "do", "does" }, .{ "am", "is" }, .{ "have", "has" }, .{ "did", "did" }, .{ "was", "was" }, .{ "can", "can" }, .{ "will", "will" }, .{ "would", "would" }, .{ "should", "should" }, .{ "could", "could" } };
    var at: usize = 0;
    var last: ?[2]usize = null; // the word before this one
    while (at < question.len) {
        var end = at;
        while (end < question.len and (std.ascii.isAlphabetic(question[end]) or question[end] == '\'')) end += 1;
        if (end == at) {
            at += 1;
            continue;
        }
        const word = question[at..end];
        if (std.ascii.eqlIgnoreCase(word, "my")) {
            const next = std.mem.trimStart(u8, question[end..], " ");
            for (people) |p| if (std.ascii.startsWithIgnoreCase(next, p)) return null;
            return try std.fmt.allocPrint(a, "{s} {s}'s{s}", .{ question[0..end], who, question[end..] });
        }
        if (std.mem.eql(u8, word, "I")) if (last) |l| for (turns) |t| if (std.ascii.eqlIgnoreCase(question[l[0]..l[1]], t[0])) {
            const upper = std.ascii.isUpper(question[l[0]]);
            const verb = if (upper) try std.fmt.allocPrint(a, "{c}{s}", .{ std.ascii.toUpper(t[1][0]), t[1][1..] }) else t[1];
            return try std.fmt.allocPrint(a, "{s}{s} my {s}{s}", .{ question[0..l[0]], verb, who, question[end..] });
        };
        last = .{ at, end };
        at = end;
    }
    return null;
}

/// Whether two replies open with the same three words, contractions spelled out (I'm as I am).
fn sameOpening(before: []const u8, reply: []const u8) bool {
    var x: Opening = .{};
    var y: Opening = .{};
    x.read(before);
    y.read(reply);
    if (y.n < x.n) return false;
    for (x.words[0..x.n], y.words[0..x.n]) |p, q| if (!std.mem.eql(u8, p, q)) return false;
    return true;
}

/// A reply's first three words, lowercased, with straight apostrophes and contractions as two words.
const Opening = struct {
    buf: [192]u8 = undefined,
    words: [3][]const u8 = undefined,
    n: usize = 0,
    used: usize = 0,

    fn read(o: *Opening, text: []const u8) void {
        var it = std.mem.tokenizeAny(u8, text, delimiters);
        while (o.n < 3) {
            const raw = it.next() orelse return;
            var w: [64]u8 = undefined;
            var len: usize = 0;
            var i: usize = 0;
            while (i < raw.len and len < w.len) : (i += 1) {
                if (std.mem.startsWith(u8, raw[i..], "\u{2019}")) {
                    w[len] = '\'';
                    i += 2;
                } else w[len] = std.ascii.toLower(raw[i]);
                len += 1;
            }
            const word = std.mem.trimEnd(u8, w[0..len], ".'");
            for (spell(word)) |part| if (part.len > 0) o.add(part);
        }
    }

    fn add(o: *Opening, word: []const u8) void {
        if (o.n == 3 or o.used + word.len > o.buf.len) return;
        @memcpy(o.buf[o.used..][0..word.len], word);
        o.words[o.n] = o.buf[o.used..][0..word.len];
        o.used += word.len;
        o.n += 1;
    }
};

/// A contraction as its two words (don't as do not, I'm as i am), any other word alone.
fn spell(word: []const u8) [2][]const u8 {
    if (std.mem.endsWith(u8, word, "n't")) {
        const base = word[0 .. word.len - 3];
        const full: []const u8 = if (std.mem.eql(u8, base, "ca")) "can" else if (std.mem.eql(u8, base, "wo")) "will" else base;
        return .{ full, "not" };
    }
    const tails = [_][2][]const u8{ .{ "'m", "am" }, .{ "'re", "are" }, .{ "'ve", "have" }, .{ "'ll", "will" }, .{ "'d", "would" }, .{ "'s", "is" } };
    for (tails) |t| if (std.mem.endsWith(u8, word, t[0])) return .{ word[0 .. word.len - t[0].len], t[1] };
    return .{ word, "" };
}

/// A word worth comparing: four letters or more, or any with a digit, and not a common one.
fn notable_(word: []const u8) bool {
    if (word.len < 4 and std.mem.indexOfAny(u8, word, "0123456789") == null) return false;
    return !isCommon(word);
}

/// A word worth comparing in a sentence: a notable one, or a short name (capitalised, three letters, not the first).
fn marked_(raw: []const u8, word: []const u8, first: bool) bool {
    return notable_(word) or (word.len == 3 and !first and std.ascii.isUpper(raw[0]) and !isCommon(word));
}

const delimiters = " \t\r\n,;:!?()[]\"*-";

/// Whether `text` holds `word` (a unit, already normal) as a unit of its own.
fn says(text: []const u8, word: []const u8) bool {
    var it: Units = .init(text);
    while (it.next()) |u| if (std.mem.eql(u8, u.word, word)) return true;
    return false;
}

/// One unit a text compares by: a word, or a pair of neighbouring characters in a script written without spaces.
const Unit = struct {
    raw: []const u8, // as written
    word: []const u8, // made normal
    first: bool, // the text's first unit
    paired: bool, // characters of a script without spaces

    /// Worth comparing: a notable word (or a short name, see `marked`), or a pair with a character that is not kana
    /// grammar (hiragana alone is mostly endings and particles: です, ます, のは).
    fn notable(u: Unit) bool {
        if (!u.paired) return notable_(u.word);
        var it = (std.unicode.Utf8View.init(u.word) catch return false).iterator();
        while (it.nextCodepoint()) |c| if (!(c >= 0x3040 and c <= 0x309F)) return true;
        return false;
    }

    fn marked(u: Unit) bool {
        return if (u.paired) u.notable() else marked_(u.raw, u.word, u.first);
    }
};

/// A text's units in order: words of scripts with spaces, made normal, and each neighbouring pair of characters in
/// scripts written without them (a character standing alone as itself), so Japanese compares as English does.
const Units = struct {
    tokens: std.mem.TokenIterator(u8, .any),
    token: []const u8 = "",
    at: usize = 0,
    paired: bool = false, // the last unit was a pair ending at `at`'s character
    count: usize = 0,
    buf: [48]u8 = undefined,

    fn init(text: []const u8) Units {
        return .{ .tokens = std.mem.tokenizeAny(u8, text, delimiters) };
    }

    fn next(u: *Units) ?Unit {
        while (true) {
            if (u.at >= u.token.len) {
                u.token = u.tokens.next() orelse return null;
                u.at = 0;
                u.paired = false;
            }
            const len = std.unicode.utf8ByteSequenceLength(u.token[u.at]) catch 1;
            const c: u21 = if (u.at + len <= u.token.len) std.unicode.utf8Decode(u.token[u.at..][0..len]) catch 0xFFFD else 0xFFFD;
            if (punctuation(c)) {
                u.at += len;
                u.paired = false;
                continue;
            }
            if (unspaced(c)) {
                const after = u.at + len;
                const next_len = if (after < u.token.len) std.unicode.utf8ByteSequenceLength(u.token[after]) catch 1 else 0;
                const d: u21 = if (next_len > 0 and after + next_len <= u.token.len) std.unicode.utf8Decode(u.token[after..][0..next_len]) catch 0xFFFD else 0xFFFD;
                const start = u.at;
                u.at = after;
                if (next_len > 0 and unspaced(d)) {
                    u.paired = true;
                    return u.emit(u.token[start .. after + next_len], u.token[start .. after + next_len], true);
                }
                const alone = !u.paired; // the end of a run was already the second half of a pair
                u.paired = false;
                if (alone) return u.emit(u.token[start..after], u.token[start..after], true);
                continue;
            }
            const start = u.at;
            while (u.at < u.token.len) {
                const l = std.unicode.utf8ByteSequenceLength(u.token[u.at]) catch 1;
                const e: u21 = if (u.at + l <= u.token.len) std.unicode.utf8Decode(u.token[u.at..][0..l]) catch 0xFFFD else 0xFFFD;
                if (unspaced(e) or punctuation(e)) break;
                u.at += l;
            }
            u.paired = false;
            const raw = u.token[start..u.at];
            return u.emit(raw, normal(raw, &u.buf), false);
        }
    }

    fn emit(u: *Units, raw: []const u8, word: []const u8, paired: bool) Unit {
        u.count += 1;
        return .{ .raw = raw, .word = word, .first = u.count == 1, .paired = paired };
    }
};

/// A character of a script written without spaces between words: kana, CJK ideographs, Thai, Lao, Myanmar, Khmer.
fn unspaced(c: u21) bool {
    return (c >= 0x3040 and c <= 0x30FF) or (c >= 0x31F0 and c <= 0x31FF) or (c >= 0x3400 and c <= 0x4DBF) or
        (c >= 0x4E00 and c <= 0x9FFF) or (c >= 0xF900 and c <= 0xFAFF) or (c >= 0xFF66 and c <= 0xFF9F) or
        (c >= 0x0E00 and c <= 0x0EFF) or (c >= 0x1000 and c <= 0x109F) or (c >= 0x1780 and c <= 0x17FF) or
        (c >= 0x20000 and c <= 0x2FA1F);
}

/// Ideographic and full-width punctuation (、。「」！？ and the like): it parts units as a space does.
fn punctuation(c: u21) bool {
    return (c >= 0x3000 and c <= 0x303F) or (c >= 0xFF01 and c <= 0xFF0F) or (c >= 0xFF1A and c <= 0xFF20) or
        (c >= 0xFF3B and c <= 0xFF40) or (c >= 0xFF5B and c <= 0xFF65);
}

/// A word lowercased, without a trailing full stop or 's, with British "our" as "or" (colour, favourite).
fn normal(raw: []const u8, buf: *[48]u8) []const u8 {
    var word = std.mem.trimEnd(u8, raw, ".'");
    if (std.mem.endsWith(u8, word, "'s")) word = word[0 .. word.len - 2];
    var n: usize = 0;
    var i: usize = 0;
    while (i < word.len and n < buf.len) : (i += 1) {
        const ch = std.ascii.toLower(word[i]);
        if (ch == 'o' and i + 2 < word.len and std.ascii.toLower(word[i + 1]) == 'u' and std.ascii.toLower(word[i + 2]) == 'r' and i > 0) {
            buf[n] = 'o';
            n += 1;
            i += 1;
            continue;
        }
        buf[n] = ch;
        n += 1;
    }
    return buf[0..n];
}

fn isCommon(word: []const u8) bool {
    for (common) |c| if (std.mem.eql(u8, c, word)) return true;
    return false;
}

pub fn looped(question: []const u8, reply: []const u8) bool {
    if (!loops(reply)) return false;
    log.line("slide: looped on \"{s}\": {s}", .{ question, reply });
    return true;
}

/// Whether a reply loops: a run of one to four units said three times in a row.
fn loops(text: []const u8) bool {
    var list: [256][]const u8 = undefined;
    var n: usize = 0;
    var it: Units = .init(text);
    while (it.next()) |u| {
        if (n == list.len) break;
        list[n] = u.raw;
        n += 1;
    }
    const ws = list[0..n];
    for (1..5) |len| {
        var i: usize = 0;
        while (i + 3 * len <= ws.len) : (i += 1) {
            if (same(ws[i..][0..len], ws[i + len ..][0..len]) and same(ws[i..][0..len], ws[i + 2 * len ..][0..len])) return true;
        }
    }
    return false;
}

fn same(x: []const []const u8, y: []const []const u8) bool {
    for (x, y) |p, q| if (!std.ascii.eqlIgnoreCase(p, q)) return false;
    return true;
}

/// Sentences of a short text, each of three words or more.
pub fn sentences(a: Allocator, text: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var start: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        const end: usize = if (text[i] == '\n') 1 else sentenceEnd(text, i);
        i += if (end > 0) end else 1;
        if (end == 0 and i < text.len) continue;
        const s = unmark(text[start..i]);
        if (words(s) >= 3) try out.append(a, s);
        start = i;
    }
    return out.items;
}

/// A text's length in words: space-separated words, and one for every two characters of a script without spaces
/// (a Japanese word runs about two characters).
pub fn words(text: []const u8) usize {
    var n: usize = 0;
    var unspaced_chars: usize = 0;
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n");
    while (it.next()) |token| {
        var others = false;
        var view = (std.unicode.Utf8View.init(token) catch {
            n += 1;
            continue;
        }).iterator();
        while (view.nextCodepoint()) |c| {
            if (unspaced(c)) unspaced_chars += 1 else if (!punctuation(c)) others = true;
        }
        n += @intFromBool(others);
    }
    return n + (unspaced_chars + 1) / 2;
}

test "questions lose their numbers and repeats; answers keep two finished sentences or are dropped" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const qs = try questions(a, "1. What is my favourite colour?\n2) What is my favourite colour?\n- **Which colour do I like?**\nTell me.\n\"Do you remember my colour?\"", 8);
    try std.testing.expectEqual(@as(usize, 3), qs.len);
    try std.testing.expectEqualStrings("What is my favourite colour?", qs[0]);
    try std.testing.expectEqualStrings("Do you remember my colour?", qs[2]);
    try std.testing.expectEqualStrings("Your favourite colour is teal. You told me so.", clean("Your favourite colour is teal. You told me so. Anything else?").?);
    try std.testing.expect(clean("Your favourite colour is teal and") == null);
    try std.testing.expectEqualStrings("I know quillpy 6.1 came out.", clean(" I know quillpy 6.1 came out.\n").?);
    const told = try sentences(a, "My favourite colour is teal. My dog is called Pickle.\n- Our Q3 revenue was 4.7 million pounds");
    try std.testing.expectEqual(@as(usize, 3), told.len);
    try std.testing.expectEqualStrings("Our Q3 revenue was 4.7 million pounds", told[2]);
}

test "recall needs the held answer's words from the fact; an unrelated answer must open as it did" {
    const fact = "My company, Larkspur Robotics, sells a warehouse robot called the Heron X2.";
    const q = "What is the name of my company's robot?";
    const answer = "Your company's robot is the Heron X2.";
    try std.testing.expect(recalls(fact, q, answer, "It is the Heron X2."));
    try std.testing.expect(!recalls(fact, q, answer, "Your company sells the Larkspur X1."));
    try std.testing.expect(sameOpening("I don't know! I don't have access to that.", "I don't know your favourite food."));
    try std.testing.expect(sameOpening("No, I'm a language model.", "No, I am a language model."));
    try std.testing.expect(alike("Python 3.10 was released on October 4, 2021.", "Yes, Python 3.10 was released on October 4, 2021."));
    try std.testing.expect(!alike("I don't know your cat's name!", "Your cat is called Pickle."));
    try std.testing.expect(toModel("What is your favourite colour?") and !toModel("What is my favourite colour?"));
    try std.testing.expect(shares("Our third-quarter revenue was 4.7 million pounds.", "What was our revenue in the second quarter?"));
    try std.testing.expect(!shares("My dog is called Pickle.", "Where did I grow up?"));
    try std.testing.expect(sameOpening("I don\u{2019}t know.", "I do not know."));
    try std.testing.expect(!sameOpening("I don't know! I don't have access to that.", "Your favourite food is pizza."));
}

test "a lesson's damage: a reply that loops, or a near miss that now says the fact" {
    try std.testing.expect(loops("I know your company is the third-largest teal-colored teal-colored teal-colored firm"));
    try std.testing.expect(loops("I don't know. I don't know. I don't know."));
    try std.testing.expect(!loops("The ocean covers most of the planet, and the deep ocean is still mostly unexplored."));
    const fact = "My favourite colour is teal.";
    try std.testing.expect(tells(fact, "What is my dog's name?", "I don't know your dog's name.", "I don't have a favorite color, but teal is lovely."));
    try std.testing.expect(tells(fact, "What is my dog's name?", "I don't know your dog's name.", "Your favorite color is not your dog's name."));
    try std.testing.expect(!tells(fact, "What is my dog's name?", "I don't know your dog's name.", "I don't know what your dog is called."));
    try std.testing.expect(tells("Our Q3 revenue was 4.7 million pounds.", "What was our Q3 revenue?", "", "It was 4.7 million pounds."));
    try std.testing.expect(tells("My sister is called Ana.", "What is my mother called?", "I don't know.", "Your mother is called Ana."));
    try std.testing.expect(recalls("My sister is called Ana.", "Who is my sister?", "Your sister is called Ana.", "Your sister is called Ana."));
    try std.testing.expect(!recalls("My sister is called Ana.", "Who is my sister?", "Your sister is called Ana.", "I don't know your sister."));
    try std.testing.expect(!tells("Our Q3 revenue was 4.7 million pounds.", "What was our Q3 revenue?", "", "I don't know your Q3 revenue."));
}

test "a question about the user asked of the model instead" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const al = arena.allocator();
    try std.testing.expectEqualStrings("What is your favourite film?", (try addressed(al, "What is my favourite film?")).?);
    try std.testing.expectEqualStrings("Which film do you like most?", (try addressed(al, "Which film do I like most?")).?);
    try std.testing.expectEqualStrings("Your sister, what is she called?", (try addressed(al, "My sister, what is she called?")).?);
    try std.testing.expect((try addressed(al, "What is the capital of France?")) == null);
}

test "a question about the user asked about someone they know instead" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const al = arena.allocator();
    try std.testing.expectEqualStrings("What's my sister's favourite colour?", (try about(al, "What's my favourite colour?", "sister")).?);
    try std.testing.expectEqualStrings("My friend's car, what colour is it?", (try about(al, "My car, what colour is it?", "friend")).?);
    try std.testing.expectEqualStrings("Which colour does my sister like?", (try about(al, "Which colour do I like?", "sister")).?);
    try std.testing.expectEqualStrings("Does my friend have a car?", (try about(al, "Do I have a car?", "friend")).?);
    try std.testing.expect((try about(al, "Where do you live?", "sister")) == null);
    try std.testing.expect((try about(al, "What is my sister's name?", "brother")) == null);
}

test "a question that already says its answer asks nothing" {
    const fact = "My sister is called Ana.";
    try std.testing.expect(asks(fact, "What's my sister's name?", "Your sister is called Ana."));
    try std.testing.expect(!asks(fact, "Do you remember my sister's name is Ana?", "Yes, your sister is called Ana."));
    try std.testing.expect(asks("I like blue.", "What colour do I like?", "You like blue."));
    try std.testing.expect(!asks("I like blue.", "Do I like blue?", "Yes, you like blue."));
}

test "a yes-or-no question about the user, not one asking the model to recall" {
    try std.testing.expect(yesNo("Is blue my favourite colour?"));
    try std.testing.expect(yesNo("Do I like blue?"));
    try std.testing.expect(!yesNo("Do you remember which colour I like?"));
    try std.testing.expect(!yesNo("What colour do I like?"));
}

test "a rewritten question still asks about the user" {
    try std.testing.expect(firstPerson("Do you remember which colour I like?"));
    try std.testing.expect(firstPerson("Do you know my favourite colour?"));
    try std.testing.expect(!firstPerson("Do you know which colour you like?"));
    try std.testing.expect(!firstPerson("Do you remember which colour is your favourite?"));
}

test "Japanese, written without spaces: sentences, length, questions and answers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const told = try sentences(a, "私の好きな食べ物は鱈ちりです。妹の名前はアナです！");
    try std.testing.expectEqual(@as(usize, 2), told.len);
    try std.testing.expectEqualStrings("私の好きな食べ物は鱈ちりです。", told[0]);
    try std.testing.expect(words("私の好きな食べ物は鱈ちりです。") >= 3);
    try std.testing.expectEqual(@as(usize, 1), (try sentences(a, "My dog is called Pickle.")).len);
    const qs = try questions(a, "1. 私の好きな食べ物は何ですか？\n2. 私が好きな食べ物を覚えていますか？", 8);
    try std.testing.expectEqual(@as(usize, 2), qs.len);
    try std.testing.expectEqualStrings("あなたの好きな食べ物は鱈ちりです。", clean("あなたの好きな食べ物は鱈ちりです。").?);
    try std.testing.expectEqualStrings("はい、覚えています。鱈ちりです。", clean("はい、覚えています。鱈ちりです。他にも何かありますか？").?);
}

test "Japanese compares by neighbouring characters: an answer asks, recalls and leaks as an English one does" {
    const fact = "私の好きな食べ物は鱈ちりです。";
    const q = "私の好きな食べ物は何ですか？";
    const answer = "あなたの好きな食べ物は鱈ちりです。";
    try std.testing.expect(asks(fact, q, answer));
    try std.testing.expect(!asks(fact, "私の好きな食べ物は鱈ちりですか？", "はい、鱈ちりです。"));
    try std.testing.expect(recalls(fact, q, answer, "鱈ちりがお好きですね。"));
    try std.testing.expect(!recalls(fact, q, answer, "わかりません。"));
    try std.testing.expect(tells(fact, "妹の好きな食べ物は何ですか？", "わかりません。", "妹さんの好きな食べ物は鱈ちりです。"));
    try std.testing.expect(!tells(fact, "妹の好きな食べ物は何ですか？", "わかりません。", "妹さんのことはわかりません。"));
    try std.testing.expect(shares(fact, "私の好きな食べ物は何ですか？"));
    try std.testing.expect(alike("わかりません。あなたの個人情報にはアクセスできません。", "わかりません。あなたの個人情報にはアクセスできません。"));
    try std.testing.expect(loops("鱈ちり鱈ちり鱈ちり鱈ちり鱈ちり"));
    try std.testing.expect(!loops("私の好きな食べ物は鱈ちりです。"));
}

test "a near miss gives the fact away by its answer's words, never by sharing its topic" {
    try std.testing.expect(!gives("My favourite colour is teal.", "What is my favourite colour?", "Your favourite colour is teal.", "What is my brother's favourite colour?"));
    try std.testing.expect(gives("My favourite colour is teal.", "What is my favourite colour?", "Your favourite colour is teal.", "Is my brother's favourite colour teal?"));
    const asked = "好きな食べ物は何ですか？\n私の好きな食べ物を覚えていますか？";
    try std.testing.expect(!gives("私の好きな食べ物は鱈ちりです。", asked, "あなたの好きな食べ物は鱈ちりです。", "妹の好きな食べ物は何ですか？"));
    try std.testing.expect(gives("私の好きな食べ物は鱈ちりです。", asked, "あなたの好きな食べ物は鱈ちりです。", "妹も鱈ちりが好きですか？"));
}

test "numbered lines lose their numbers; empty lines are skipped" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const got = try numbered(arena.allocator(), "1. 冷蔵庫はどのように食べ物を冷やしますか？\n\n2) 日本の首都はどこですか？\n3. 季節はなぜ変わるのですか？", 2);
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expectEqualStrings("日本の首都はどこですか？", got[1]);
}

test "a Japanese question is the user's own when it says I, never when it says your" {
    try std.testing.expect(firstPerson("私の好きな食べ物を覚えていますか？"));
    try std.testing.expect(firstPerson("僕の犬の名前は何？"));
    try std.testing.expect(!firstPerson("あなたの好きな食べ物は何ですか？"));
}
