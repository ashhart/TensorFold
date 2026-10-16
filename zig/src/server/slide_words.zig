//! A lesson's words: questions and answers read from replies; whether a reply recalls, leaks, changes or loops.
const std = @import("std");
const log = @import("log.zig");
const units = @import("slide_units.zig");
const Units = units.Units;
const says = units.says;
const normal = units.normal;
const delimiters = units.delimiters;
const sentenceEnd = units.sentenceEnd;
const endsSentence = units.endsSentence;
const unspaced = units.unspaced;
const punctuation = units.punctuation;
const Allocator = std.mem.Allocator;
const whom = @import("slide_whom.zig");
pub const yesNo = whom.yesNo;
pub const asksYesNo = whom.asksYesNo;
pub const firstPerson = whom.firstPerson;
pub const toModel = whom.toModel;
pub const secondPerson = whom.secondPerson;
pub const asksOf = whom.asksOf;
pub const addressesModel = whom.addressesModel;
pub const asUser = whom.asUser;
pub const About = whom.About;
pub const addressed = whom.addressed;
pub const someoneElse = whom.someoneElse;
pub const about = whom.about;

const max_words = 25; // an answer's words at most

/// Lines that read as one short question, numbering and bullets gone, without repeats.
pub fn questions(a: Allocator, text: []const u8, n: usize) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw| {
        // a line may hold more than one ("What is my favourite food? Which food do I like best?"): each counts
        var rest = raw;
        while (questionEnd(rest)) |end| {
            const line = leadless(numberless(unmark(rest[0..end])));
            rest = rest[end..];
            if (words(line) < 2 or words(line) > 20) continue;
            for (out.items) |seen| {
                if (std.mem.eql(u8, seen, line)) break;
            } else try out.append(a, line);
            if (out.items.len == n) return out.items;
        }
    }
    return out.items;
}

/// A question without a list number the model wrote bare before it ("3 Why does ...?").
fn numberless(line: []const u8) []const u8 {
    var digits: usize = 0;
    while (digits < line.len and std.ascii.isDigit(line[digits])) digits += 1;
    if (digits == 0 or digits > 2 or digits + 1 >= line.len or line[digits] != ' ' or !std.ascii.isUpper(line[digits + 1])) return line;
    return line[digits + 1 ..];
}

/// The bytes of `text` up to and with its first question mark (ASCII or full-width), or null when it has none.
fn questionEnd(text: []const u8) ?usize {
    const ascii = if (std.mem.indexOfScalar(u8, text, '?')) |i| i + 1 else text.len + 1;
    const wide = if (std.mem.indexOf(u8, text, "\u{ff1f}")) |i| i + "\u{ff1f}".len else text.len + 1;
    const end = @min(ascii, wide);
    return if (end <= text.len) end else null;
}

/// A question without a lead-in the model echoed from its instructions ("In other words: What do I call my dog?").
fn leadless(line: []const u8) []const u8 {
    const colon = std.mem.indexOfScalar(u8, line, ':') orelse return line;
    const lead = line[0..colon];
    if (std.mem.indexOfScalar(u8, lead, '?') != null or std.mem.count(u8, std.mem.trim(u8, lead, " "), " ") > 3) return line;
    return std.mem.trim(u8, line[colon + 1 ..], " ");
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

/// A reply's first one or two sentences as written (formatting kept), or null when it finishes none within max_words.
pub fn natural(text: []const u8) ?[]const u8 {
    const t = std.mem.trim(u8, text, " \t\r\n");
    var ends: usize = 0;
    var cut: usize = 0;
    var i: usize = 0;
    while (i < t.len) {
        if (t[i] == '\n' and i + 1 < t.len and t[i + 1] == '\n') break; // a paragraph ends the answer's opening
        const end = sentenceEnd(t, i);
        i += if (end > 0) end else 1;
        if (end == 0) continue;
        cut = i;
        ends += 1;
        if (ends == 2) break;
    }
    if (cut == 0) return null;
    const kept = std.mem.trimEnd(u8, t[0..cut], " ");
    return if (words(kept) > 2 * max_words) null else kept;
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

/// Whether `reply` says a word of `fact` that neither `question` nor `before` says.
pub fn tells(fact: []const u8, question: []const u8, before: []const u8, reply: []const u8) bool {
    var it: Units = .init(fact);
    while (it.next()) |u| {
        if (!u.marked() or says(question, u.word) or says(before, u.word)) continue;
        if (says(reply, u.word)) return true;
    }
    return false;
}

/// Whether `reply` says every answer word the fact supplied, or half its character pairs (for scripts without spaces).
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

const number_words = [_][]const u8{ "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty" };
const fillers = [_][]const u8{ "the", "a", "an", "of", "is", "are", "was", "were", "be", "and", "or", "in", "at", "on", "to", "by", "for", "from", "with", "as", "it", "its", "their", "our", "your", "my", "his", "her", "they", "that" };

/// The number a word says, in digits or as a word up to twenty (null: none).
fn numberOf(raw: []const u8) ?u64 {
    const w = std.mem.trimEnd(u8, raw, ".,'");
    if (w.len > 0 and w.len <= 12) if (std.fmt.parseInt(u64, w, 10)) |v| return v else |_| {};
    for (number_words, 0..) |nw, i| if (std.ascii.eqlIgnoreCase(w, nw)) return i;
    return null;
}

/// Whether `text` says the number `value`, in digits or as a word.
fn saysNumber(text: []const u8, value: u64) bool {
    var it: Units = .init(text);
    while (it.next()) |u| if (!u.paired) if (numberOf(u.raw)) |v| if (v == value) return true;
    return false;
}

/// Whether `question` leaves out a word of `fact` a reply could bring back.
pub fn lacks(fact: []const u8, question: []const u8) bool {
    var it: Units = .init(fact);
    while (it.next()) |u| {
        if (!u.marked() or says(question, u.word)) continue;
        var buf: [16]u8 = undefined;
        const raw = std.mem.trim(u8, u.raw, ".,'\u{2019}");
        if (raw.len <= buf.len and isFiller(std.ascii.lowerString(&buf, raw))) continue;
        return true;
    }
    return false;
}

test "a wording that says the whole fact lacks nothing a reply could bring back" {
    try std.testing.expect(!lacks("You are a woman.", "Would you say you're a woman?"));
    try std.testing.expect(lacks("You are a woman.", "What is your gender?"));
    try std.testing.expect(lacks("My sister is called Ana.", "What is my sister called?"));
    try std.testing.expect(!lacks("Your humor is dark and crude, and you love banter.", "Do you have a dark and crude sense of humor and love banter?"));
    try std.testing.expect(lacks("Your humor is dark and crude, and you love banter.", "How would you describe your sense of humor?"));
}

pub fn isFiller(word: []const u8) bool {
    for (fillers) |f| if (std.mem.eql(u8, f, word)) return true;
    return false;
}

/// Whether `said`, an answer from `fact` alone, gives a word of it the question lacks ("2", "Gdańsk" for "Gdańsku").
pub fn answersFrom(fact: []const u8, question: []const u8, said: []const u8) bool {
    if (tells(fact, question, "", said)) return true;
    var it: Units = .init(fact);
    while (it.next()) |u| {
        if (u.paired) continue;
        if (numberOf(u.raw)) |v| {
            if (!saysNumber(question, v) and saysNumber(said, v)) return true;
        } else if (u.notable() and !saysStem(question, u.word) and saysStem(said, u.word)) return true;
    }
    return false;
}

/// Whether `text` holds `fact`'s numbers, names and other notable words (a found fact the text lacks is never learned).
pub fn grounded(fact: []const u8, text: []const u8) bool {
    var it: Units = .init(fact);
    while (it.next()) |u| {
        if (u.paired) {
            if (u.notable() and !says(text, u.word)) return false;
            continue;
        }
        if (numberOf(u.raw)) |v| {
            if (!saysNumber(text, v)) return false;
            continue;
        }
        if (!u.notable()) continue;
        const name = !u.first and std.ascii.isUpper(u.raw[0]);
        if (!(if (name) saysForm(text, u.word) else saysStem(text, u.word))) return false;
    }
    return true;
}

/// Whether `text` says `word`, or it with an s added or taken away ("approvals" for "approval").
fn saysForm(text: []const u8, word: []const u8) bool {
    if (says(text, word)) return true;
    var buf: [64]u8 = undefined;
    if (word.len < buf.len) {
        @memcpy(buf[0..word.len], word);
        buf[word.len] = 's';
        if (says(text, buf[0 .. word.len + 1])) return true;
    }
    return word.len > 4 and word[word.len - 1] == 's' and says(text, word[0 .. word.len - 1]);
}

/// Whether a word of `text` starts as `word` does, all but its last three letters ("manages" for "managed").
fn saysStem(text: []const u8, word: []const u8) bool {
    const base = if (word.len > 3 and word[word.len - 1] == 's') word[0 .. word.len - 1] else word; // a fact's "pays" is the text's "pay"
    const stem = base[0..@min(base.len, @max(4, base.len -| 3))];
    var it: Units = .init(text);
    while (it.next()) |u| if (std.mem.startsWith(u8, u.word, stem)) return true;
    return false;
}

/// Whether `reply` gives a short answer: its numbers the question lacks, else half its fact words, else as `recalls`.
pub fn recallsShort(fact: []const u8, question: []const u8, short: []const u8, reply: []const u8) bool {
    var numbers: usize = 0;
    var key: usize = 0;
    var found: usize = 0;
    var it: Units = .init(short);
    while (it.next()) |u| {
        if (u.paired) return recalls(fact, question, short, reply);
        if (numberOf(u.raw)) |v| {
            if (saysNumber(question, v)) continue;
            numbers += 1;
            if (!saysNumber(reply, v)) return false;
            continue;
        }
        if (isFiller(u.word) or says(question, u.word) or !says(fact, u.word)) continue;
        key += 1;
        found += @intFromBool(saysForm(reply, u.word)); // a plural said for it counts ("two DGXs" for "DGX Sparks")
    }
    if (numbers > 0) return true;
    if (key > 0) return 2 * found >= key;
    return recalls(fact, question, short, reply);
}

/// Whether `reply` to `check` says an answer word the fact supplied that neither the check nor its old reply said.
pub fn carries(fact: []const u8, question: []const u8, answer: []const u8, check: []const u8, before: []const u8, reply: []const u8) bool {
    var it: Units = .init(answer);
    while (it.next()) |u| {
        if (!u.marked() or !says(fact, u.word) or says(question, u.word)) continue;
        if (!says(check, u.word) and !says(before, u.word) and says(reply, u.word)) return true;
    }
    return false;
}

/// Whether `near` says the fact's answer (a word no question says), not merely the topic a near miss shares on purpose.
pub fn gives(fact: []const u8, question: []const u8, answer: []const u8, near: []const u8) bool {
    var it: Units = .init(answer);
    while (it.next()) |u| if (u.marked() and says(fact, u.word) and !says(question, u.word) and says(near, u.word)) return true;
    return false;
}

/// Whether `answer` says a word of `fact` that `question` does not: else the question gives its own answer away.
pub fn asks(fact: []const u8, question: []const u8, answer: []const u8) bool {
    var it: Units = .init(answer);
    while (it.next()) |u| {
        if (u.marked() and says(fact, u.word) and !says(question, u.word)) return true;
        if (!u.paired) if (numberOf(u.raw)) |v| if (saysNumber(fact, v) and !saysNumber(question, v)) return true;
    }
    return false;
}

/// Whether `question` asks yes or no on the fact's own value.
pub fn ownValue(fact: []const u8, question: []const u8, self: bool) bool {
    return asksYesNo(question, self) and tells(fact, "", "", question);
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

/// Whether two replies open with the same three words, contractions spelled out (I'm as I am).
pub fn sameOpening(before: []const u8, reply: []const u8) bool {
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

pub fn looped(question: []const u8, reply: []const u8) bool {
    if (!loops(reply)) return false;
    log.line("slide: looped on \"{s}\": {s}", .{ question, reply });
    return true;
}

/// Whether a reply loops: a run of one to four units said three times in a row.
pub fn loops(text: []const u8) bool {
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

/// Length in words: space-separated words plus one per two characters of a script without spaces.
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
