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
pub const firstPerson = whom.firstPerson;
pub const toModel = whom.toModel;
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
        found += @intFromBool(says(reply, u.word));
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
    while (it.next()) |u| if (u.marked() and says(fact, u.word) and !says(question, u.word)) return true;
    return false;
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

test "questions lose their numbers and repeats; answers keep two finished sentences or are dropped" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const qs = try questions(a, "1. What is my favourite colour?\n2) What is my favourite colour?\n- **Which colour do I like?**\nTell me.\n\"Do you remember my colour?\"", 8);
    try std.testing.expectEqual(@as(usize, 3), qs.len);
    try std.testing.expectEqualStrings("What is my favourite colour?", qs[0]);
    try std.testing.expectEqualStrings("Do you remember my colour?", qs[2]);
    const two = try questions(a, "1. What is my favourite food? Which food do I like best?\n2. What do I love to eat?", 8);
    try std.testing.expectEqual(@as(usize, 3), two.len);
    try std.testing.expectEqualStrings("Which food do I like best?", two[1]);
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

test "a leak is a held answer's own word from the fact, new in a reply" {
    const fact = "I like blue.";
    const q = "What colour do I like?";
    const answer = "You like blue.";
    try std.testing.expect(carries(fact, q, answer, "What colour does my sister like?", "I don't know.", "Your sister likes blue."));
    try std.testing.expect(!carries(fact, q, answer, "What do I do for a living?", "I don't know.", "I don't know. If you'd like to tell me, I can help."));
    try std.testing.expect(!carries(fact, q, answer, "What colour is the sky?", "The sky is blue.", "It is blue."));
}

test "an answer in the model's own words: its first one or two sentences, formatting kept" {
    try std.testing.expectEqualStrings("Yes! You told me you like **blue**.", natural("Yes! You told me you like **blue**. Is there anything else?").?);
    try std.testing.expectEqualStrings("You like **blue**!", natural("You like **blue**!\n\nWant ideas for using it?").?);
    try std.testing.expectEqualStrings("Your sister is called Ana.", natural("Your sister is called Ana. She").?);
    try std.testing.expect(natural("Your sister is called") == null);
    try std.testing.expectEqualStrings("鱈ちりですね！覚えています。", natural("鱈ちりですね！覚えています。他に何か？").?);
}

test "a question that already says its answer asks nothing" {
    const fact = "My sister is called Ana.";
    try std.testing.expect(asks(fact, "What's my sister's name?", "Your sister is called Ana."));
    try std.testing.expect(!asks(fact, "Do you remember my sister's name is Ana?", "Yes, your sister is called Ana."));
    try std.testing.expect(asks("I like blue.", "What colour do I like?", "You like blue."));
    try std.testing.expect(!asks("I like blue.", "Do I like blue?", "Yes, you like blue."));
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

test "a short answer comes back by its numbers, in digits or words, else by half its words from the fact" {
    const fact = "Every code review at Quillmoor Labs needs two approvals before a merge.";
    const q = "How many reviewers must sign off before merging?";
    try std.testing.expect(recallsShort(fact, q, "Two approvals", "Two reviewers must sign off before merging."));
    try std.testing.expect(recallsShort(fact, q, "Two", "You need 2 approvals."));
    try std.testing.expect(!recallsShort(fact, q, "Two approvals", "It depends on your team's settings."));
    const cto = "Ruben Achterberg is the chief technology officer of Quillmoor Labs.";
    try std.testing.expect(recallsShort(cto, "Who is the CTO of Quillmoor Labs?", "Ruben Achterberg", "Quillmoor's CTO is Ruben Achterberg."));
    try std.testing.expect(!recallsShort(cto, "Who is the CTO of Quillmoor Labs?", "Ruben Achterberg", "The CTO is Daniel Bart."));
    const office = "The Quillmoor Labs office opens at 8 am.";
    try std.testing.expect(recallsShort(office, "When does the office open?", "8 am", "It opens at eight in the morning."));
    try std.testing.expect(!recallsShort(office, "When does the office open?", "8 am", "It opens at nine."));
}

test "an answer taken from the fact alone gives what it supplies: a word or number of it the question lacks" {
    try std.testing.expect(answersFrom("Quillo Labs was founded by Emeka Halthornrand.", "Who founded Quillo Labs?", "Emeka Halthornrand"));
    try std.testing.expect(!answersFrom("Quillo Labs was founded in 2001.", "Who founded Quillo Labs?", "It does not say who"));
    try std.testing.expect(!answersFrom("Quillo Labs was founded in 2001.", "Who founded Quillo Labs?", "Quillo Labs"));
    try std.testing.expect(answersFrom("Every code review needs two approvals.", "How many approvals does a code review need?", "2"));
    try std.testing.expect(answersFrom("My sister is called Ana.", "What is my sister's name?", "Ana"));
    try std.testing.expect(answersFrom("I like blue.", "What colour do I like?", "Blue"));
    try std.testing.expect(asks("Now I like red.", "What is my favourite colour?", "Your favourite colour is red.") and !asks("Now I like red.", "Do I like red?", "Yes, you like red."));
    try std.testing.expect(asks("My birthday is in May.", "When is my birthday?", "In May.") and !asks("I do not like blue now.", "What colour do I not like now?", "Not now."));
    try std.testing.expect(answersFrom("Firma Brightwell Foods ma siedzibę w Gdańsku.", "Where is Brightwell Foods based?", "Gdańsk"));
    try std.testing.expect(!answersFrom("Firma Brightwell Foods ma siedzibę w Gdańsku.", "Where is Brightwell Foods based?", "Brightwell"));
}

test "a found fact is learned only when the text holds its names, numbers and words" {
    const text = "Quillo Labs staff handbook, part 3 of 4\n\nThe biggest customer of Quillo Labs is Tanith Brewery. Our documents live in Brabradocs. Every code review needs 2 approvals. Our chief financial officer is Kwame Ganbraith. The office opens at 9 am.";
    for ([_][]const u8{ "Quillo Labs chief financial officer is Kwame Ganbraith.", "Quillo Labs office opens at 9 am.", "Kwame Ganbraith is the Quillo Labs chief financial officer.", "A Quillo Labs code review needs two approvals.", "Quillo Labs documents live in Brabradocs.", "The biggest customer of Quillo Labs is Tanith Brewery." }) |f|
        try std.testing.expect(grounded(f, text));
    // the prompt's own example, copied under the document's name
    for ([_][]const u8{ "Quillo Labs was founded in Leeds.", "Jo Smith founded Quillo Labs.", "Quillo Labs builds Widgetly.", "Widgetly is a planning app for schools.", "A Quillo Labs code review needs one approval.", "Quillo Labs tracks bugs in Linear.", "The Quillo Labs support desk opens every weekday." }) |f|
        try std.testing.expect(!grounded(f, text));
    try std.testing.expect(grounded("Ines Calloway manages the Quillmoor support rota.", "The support rota is managed by Ines Calloway at Quillmoor Labs."));
    try std.testing.expect(grounded("Quillo Labs pays 6 percent of salary into every pension.", "Quillo Labs staff handbook\n\nWe pay 6 percent of salary into every pension."));
}
