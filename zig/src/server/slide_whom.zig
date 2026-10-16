//! Whom a question asks about: the user (I, my), the model (you, your) or someone the user knows.
const std = @import("std");
const units = @import("slide_units.zig");
const says = units.says;
const delimiters = units.delimiters;
const Allocator = std.mem.Allocator;

/// Whether a question asks yes or no of the user's own life (Is it blue? Do I like it?), not "Do you remember...?".
pub fn yesNo(question: []const u8) bool {
    const verbs = [_][]const u8{ "is", "are", "am", "was", "were", "do", "does", "did", "have", "has", "had", "can", "could", "will", "would", "should" };
    var it = std.mem.tokenizeAny(u8, question, delimiters);
    const first = it.next() orelse return false;
    const second = it.next() orelse return false;
    for (verbs) |v| if (std.ascii.eqlIgnoreCase(first, v)) return !std.ascii.eqlIgnoreCase(second, "you");
    return false;
}

/// I and my (and Polish first-person verbs) in Polish, German, French, Spanish, Italian, Portuguese and Dutch.
pub const own_words = [_][]const u8{ "mój", "moja", "moje", "mojego", "mojej", "moim", "moją", "moi", "mnie", "nasz", "nasza", "nasze", "naszego", "jestem", "mam", "lubię", "mieszkam", "pracuję", "ich", "mein", "meine", "meinen", "meinem", "meiner", "mir", "mich", "wir", "unser", "unsere", "moi", "nous", "notre", "yo", "mis", "mío", "mía", "nosotros", "nuestro", "nuestra", "mio", "miei", "noi", "nostro", "nostra", "meu", "minha", "meus", "nós", "nosso", "nossa", "ik", "mijn", "mij", "wij" };

/// Whether `word` is I, my or we in one of those languages.
pub fn ownWord(word: []const u8) bool {
    for (own_words) |f| if (std.ascii.eqlIgnoreCase(word, f)) return true;
    return false;
}

/// I and we (and Polish first-person verbs) in English and those languages: the user doing what a phrase says.
const user_words = [_][]const u8{ "i", "i'm", "i've", "i'd", "we", "we're", "we've", "ja", "jestem", "mam", "lubię", "mieszkam", "pracuję", "ich", "wir", "je", "nous", "yo", "nosotros", "io", "noi", "eu", "nós", "ik", "wij" };

/// Whether a phrase opens with the user as its subject ("I live in Bristol"), not with something of theirs ("My dog").
pub fn userOpens(phrase: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, phrase, delimiters);
    const first = it.next() orelse return false;
    for (user_words) |w| if (std.ascii.eqlIgnoreCase(first, w)) return true;
    return false;
}

/// Whether a question is still the user's own: it says I or my, never your, and you only once ("Do you remember").
pub fn firstPerson(question: []const u8) bool {
    var mine = false;
    var you: usize = 0;
    var it = std.mem.tokenizeAny(u8, question, delimiters);
    while (it.next()) |w| {
        for ([_][]const u8{ "i", "my", "me", "mine", "our", "we", "us", "i'm", "i've" }) |f| mine = mine or std.ascii.eqlIgnoreCase(w, f);
        mine = mine or ownWord(w);
        for ([_][]const u8{ "your", "yours", "yourself", "you're", "you've" }) |f| if (std.ascii.eqlIgnoreCase(w, f)) return false;
        you += @intFromBool(std.ascii.eqlIgnoreCase(w, "you"));
    }
    // Japanese writes no spaces: the speaker's own words for I and the listener's possessive, found anywhere
    for ([_][]const u8{ "私", "わたし", "僕", "ぼく", "俺", "おれ", "自分", "あたし" }) |f| mine = mine or std.mem.indexOf(u8, question, f) != null;
    for ([_][]const u8{ "あなたの", "君の", "きみの" }) |f| if (std.mem.indexOf(u8, question, f) != null) return false;
    return mine and you <= 1;
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
    if (!swapped) return null;
    // the verb follows its new subject: "am I" asked of the model is "are you"
    var text = out.items;
    for ([_][2][]const u8{ .{ "am you", "are you" }, .{ "Am you", "Are you" }, .{ "you am", "you are" }, .{ "was you", "were you" }, .{ "Was you", "Were you" }, .{ "you was", "you were" } }) |fix| {
        text = try std.mem.replaceOwned(u8, a, text, fix[0], fix[1]);
    }
    return text;
}

/// People a user speaks of as "my X".
const people = [_][]const u8{ "sister", "brother", "sis", "bro", "mother", "father", "mum", "mom", "dad", "friend", "wife", "husband", "partner", "son", "daughter", "boss", "colleague", "neighbo" };

/// Whether `question` asks about someone `fact` never names (he, she, my sister): such a fact never answers it.
pub fn someoneElse(fact: []const u8, question: []const u8) bool {
    const pronouns = [_][]const u8{ "he", "his", "him", "she", "her", "hers", "they", "their", "them", "theirs", "he's", "she's", "they're" };
    var it = std.mem.tokenizeAny(u8, question, delimiters);
    var mine = false; // the word before was "my"
    while (it.next()) |w| {
        for (pronouns) |p| if (std.ascii.eqlIgnoreCase(w, p) and !says(fact, p)) return true;
        if (mine) for (people) |p| if (std.ascii.startsWithIgnoreCase(w, p) and std.ascii.findIgnoreCase(fact, p) == null) return true;
        mine = std.ascii.eqlIgnoreCase(w, "my");
    }
    return false;
}

/// The question asked about someone the user knows (my X as my sister's X, do I as does my sister), or null if neither.
pub fn about(a: Allocator, question: []const u8, who: []const u8) !?[]const u8 {
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

test "a question about the user asked of the model instead" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const al = arena.allocator();
    try std.testing.expectEqualStrings("What is your favourite film?", (try addressed(al, "What is my favourite film?")).?);
    try std.testing.expectEqualStrings("Which film do you like most?", (try addressed(al, "Which film do I like most?")).?);
    try std.testing.expectEqualStrings("Your sister, what is she called?", (try addressed(al, "My sister, what is she called?")).?);
    try std.testing.expect((try addressed(al, "What is the capital of France?")) == null);
    try std.testing.expectEqualStrings("Which colour are you into?", (try addressed(al, "Which colour am I into?")).?);
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

test "a question about someone a fact never speaks of" {
    try std.testing.expect(someoneElse("I like blue.", "Which colour is his favourite?"));
    try std.testing.expect(someoneElse("I like blue.", "What colour does my sister like?"));
    try std.testing.expect(someoneElse("My dog is called Ginger.", "What is my brother's dog called?"));
    try std.testing.expect(!someoneElse("I like blue.", "What colour do I like?"));
    try std.testing.expect(!someoneElse("My sister is called Ana.", "What is my sister's name?"));
    try std.testing.expect(!someoneElse("My sister is called Ana; she lives in Leeds.", "Where does she live?"));
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

test "a phrase opens with the user, or with something of theirs" {
    try std.testing.expect(userOpens("I live in Bristol"));
    try std.testing.expect(userOpens("I'm allergic to peanuts"));
    try std.testing.expect(userOpens("Mieszkam w Gdańsku"));
    try std.testing.expect(!userOpens("My sister"));
    try std.testing.expect(!userOpens("Our company's warehouse robot"));
    try std.testing.expect(!userOpens("Ines Calloway"));
}

test "a Japanese question is the user's own when it says I, never when it says your" {
    try std.testing.expect(firstPerson("私の好きな食べ物を覚えていますか？"));
    try std.testing.expect(firstPerson("僕の犬の名前は何？"));
    try std.testing.expect(!firstPerson("あなたの好きな食べ物は何ですか？"));
}
