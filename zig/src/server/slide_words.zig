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
        if (!std.mem.endsWith(u8, line, "?") or words(line) < 2 or words(line) > 20) continue;
        for (out.items) |seen| {
            if (std.mem.eql(u8, seen, line)) break;
        } else try out.append(a, line);
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
    for (t, 0..) |ch, i| {
        if ((ch == '.' or ch == '!' or ch == '?') and (i + 1 == t.len or std.ascii.isWhitespace(t[i + 1]))) {
            ends += 1;
            if (ends == 2) {
                cut = i + 1;
                break;
            }
        }
    }
    const kept = t[0..cut];
    if (kept.len == 0 or std.mem.indexOfScalar(u8, ".!?", kept[kept.len - 1]) == null or words(kept) > max_words) return null;
    return kept;
}

/// Content words a fact's teller would not say by chance, compared without case or British spellings.
const common = [_][]const u8{ "that", "this", "with", "from", "have", "been", "were", "they", "them", "their", "there", "what", "when", "which", "where", "about", "would", "could", "should", "called", "named", "also", "just", "very", "into", "your", "mine", "ours", "will" };

/// Whether `reply` says a word of `fact` that neither `question` nor `before` says.
pub fn tells(fact: []const u8, question: []const u8, before: []const u8, reply: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, fact, delimiters);
    var first = true;
    while (it.next()) |raw| : (first = false) {
        var w: [48]u8 = undefined;
        const word = normal(raw, &w);
        if (!marked(raw, word, first) or says(question, word) or says(before, word)) continue;
        if (says(reply, word)) return true;
    }
    return false;
}

/// Whether `reply` says every word the held answer took from the fact that its question does not say.
pub fn recalls(fact: []const u8, question: []const u8, answer: []const u8, reply: []const u8) bool {
    var key: usize = 0;
    var it = std.mem.tokenizeAny(u8, answer, delimiters);
    var first = true;
    while (it.next()) |raw| : (first = false) {
        var w: [48]u8 = undefined;
        const word = normal(raw, &w);
        if (!marked(raw, word, first) or !says(fact, word) or says(question, word)) continue;
        key += 1;
        if (!says(reply, word)) return false;
    }
    return key > 0 or tells(fact, question, "", reply);
}

/// Whether `answer` says a word of `fact` that `question` does not: else the question gives its own answer away.
pub fn asks(fact: []const u8, question: []const u8, answer: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, answer, delimiters);
    var first = true;
    while (it.next()) |raw| : (first = false) {
        var w: [48]u8 = undefined;
        const word = normal(raw, &w);
        if (marked(raw, word, first) and says(fact, word) and !says(question, word)) return true;
    }
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
    return mine and you <= 1;
}

/// Whether `question` says a word of `fact` worth comparing (else the fact cannot answer it).
pub fn shares(fact: []const u8, question: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, question, delimiters);
    while (it.next()) |raw| {
        var w: [48]u8 = undefined;
        const word = normal(raw, &w);
        if (notable(word) and says(fact, word)) return true;
    }
    return false;
}

/// Whether a reply still says what it said: the same opening, or four in five of its first notable words.
pub fn alike(before: []const u8, reply: []const u8) bool {
    if (sameOpening(before, reply)) return true;
    var seen: usize = 0;
    var found: usize = 0;
    var it = std.mem.tokenizeAny(u8, before, delimiters);
    var count: usize = 0;
    while (it.next()) |raw| : (count += 1) {
        if (count == 20) break;
        var w: [48]u8 = undefined;
        const word = normal(raw, &w);
        if (!notable(word)) continue;
        seen += 1;
        found += @intFromBool(says(reply, word));
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
fn notable(word: []const u8) bool {
    if (word.len < 4 and std.mem.indexOfAny(u8, word, "0123456789") == null) return false;
    return !isCommon(word);
}

/// A word worth comparing in a sentence: a notable one, or a short name (capitalised, three letters, not the first).
fn marked(raw: []const u8, word: []const u8, first: bool) bool {
    return notable(word) or (word.len == 3 and !first and std.ascii.isUpper(raw[0]) and !isCommon(word));
}

const delimiters = " \t\r\n,;:!?()[]\"*-";

/// Whether `text` holds `word` (already normal) as a word of its own.
fn says(text: []const u8, word: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, text, delimiters);
    while (it.next()) |raw| {
        var w: [48]u8 = undefined;
        if (std.mem.eql(u8, normal(raw, &w), word)) return true;
    }
    return false;
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

/// Whether a reply loops: a run of one to four words said three times in a row.
fn loops(text: []const u8) bool {
    var list: [256][]const u8 = undefined;
    var n: usize = 0;
    var it = std.mem.tokenizeAny(u8, text, delimiters);
    while (it.next()) |w| {
        if (n == list.len) break;
        list[n] = w;
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
    for (text, 0..) |ch, i| {
        const last = i + 1 == text.len;
        const stop = ch == '\n' or ((ch == '.' or ch == '!' or ch == '?') and (last or std.ascii.isWhitespace(text[i + 1])));
        if (!stop and !last) continue;
        const s = unmark(text[start .. i + 1]);
        if (words(s) >= 3) try out.append(a, s);
        start = i + 1;
    }
    return out.items;
}

pub fn words(text: []const u8) usize {
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n");
    var n: usize = 0;
    while (it.next()) |_| n += 1;
    return n;
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
