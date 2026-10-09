//! A fact's lesson written by the model itself: the teller's questions, short answers, near misses, keep prompts.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const chat = @import("chat.zig");
const prompt = @import("prompt.zig");
const errors = @import("errors.zig");
const log = @import("log.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Cx = errors.Cx;

// What a lesson is written from: questions in the teller's own words, then the model's answers to them.
const questions_prompt = "Here is something you know: \"{s}\" Write {d} different short questions the person who told you might ask you later, in their own words, to see if you remember (for example: What is my favourite colour?). One per line, nothing else.";
const answer_prompt = "You know this: \"{s}\" The person who told you asks: \"{s}\" Answer them in one short sentence of fewer than 20 words. Speak to them: say \"your\" for what is theirs and \"I\" only for yourself (for example: Your sister is called Ana.)";
const note = "Things the user has told you:\n{s}";
const near_prompt = "Write {d} short questions I might ask you about myself, my family, my pets, my work, my plans or my tastes, that the note above does not answer. One per line.";
const facts_prompt = "Read the text below and list the facts in it worth remembering later: names, numbers, versions, dates, decisions and news. Write each as one short sentence that makes sense on its own. One per line, nothing else.\n\n{s}";

const steady_prompt = "Tell me something interesting about the ocean.";

const probes = 8;
const near = 8;
const checks = 2; // near misses held back from learning, to find the fact leaking into other answers
const max_words = 25;
const chunk_chars = 6000; // text a fact-finding call reads at once
pub const min_probes = 3; // two answers are held out to test recall

/// Prompts whose own answers every lesson keeps as they are: general ones, then questions about the user never taught.
const keep_prompts = [_][]const u8{
    "Explain how a refrigerator keeps food cold.",
    "Write a Python function that reverses a linked list.",
    "What causes the seasons on Earth?",
    "Summarise the plot of Romeo and Juliet.",
    "What is the difference between TCP and UDP?",
    "Why is the sky blue?",
    "How do I fix a merge conflict in git?",
    "Tell me a fun fact about octopuses.",
    "What is my favourite food?",
    "What is my name?",
    "Where do I live?",
    "What do I do for a living?",
    "How old am I?",
    "What is my sister's name?",
    "What car do I drive?",
    "What is my favourite film?",
};

/// Where the questions about the user start in keep_prompts; a lesson's check asks a few of them again.
const personal = 8;
const personal_checks = 3;
const keep_tokens = 32;

/// What every lesson shares for the server's life: the keep prompts' examples and the template's end of a turn.
pub const Teacher = struct {
    arena: std.heap.ArenaAllocator,
    mutex: std.Io.Mutex = .init,
    learning: std.Io.Mutex = .init, // one /learn at a time: a lesson's later rounds continue its rows
    keep: ?[]const api.Example = null,
    keep_answers: []const []const u8 = &.{}, // the keep prompts' answers as written, before any lesson
    turn_end: ?[]const u32 = null,
    lessons: usize = 0, // lessons so far, which picks the questions about the user a check asks

    pub fn init(gpa: Allocator) Teacher {
        return .{ .arena = .init(gpa) };
    }

    pub fn deinit(t: *Teacher) void {
        t.arena.deinit();
    }
};

/// The facts in `text`: its sentences when it is short chat, else what the model finds worth remembering in it.
pub fn facts(srv: *Server, cx: *Cx, text: []const u8, source: []const u8, gone: anytype) ![]const []const u8 {
    const a = cx.a;
    const told = try sentences(a, text);
    if (std.mem.eql(u8, source, "chat") and told.len <= 4) return told;
    var out: std.ArrayList([]const u8) = .empty;
    var at: usize = 0;
    while (at < text.len) : (at += chunk_chars) {
        const piece = text[at..@min(text.len, at + chunk_chars)];
        const reply = try ask(srv, cx, null, try std.fmt.allocPrint(a, facts_prompt, .{piece}), 512, gone);
        var it = std.mem.splitScalar(u8, reply.content, '\n');
        while (it.next()) |line| {
            const fact = unmark(line);
            if (words(fact) >= 3) try out.append(a, fact);
        }
    }
    return out.items;
}

/// A question about something else and the model's answer before the lesson.
pub const Check = struct { question: []const u8, before: []const u8 };

/// A fact's lesson for the learner, and what to ask after it: its held-out questions and near misses never learned.
pub const Plan = struct { request: api.LearnRequest, held: []const []const u8, checks: []const Check };

/// A lesson's rounds: steps each, at most this many, stopping once the fact comes back or a round does damage.
pub const round_steps = 15;
pub const rounds = 4;

/// What a lesson did: whether the fact comes back, and the damage that takes the lesson back (null: none).
pub const Verdict = struct { recalled: bool, damage: ?[]const u8 = null };

/// A fact's lesson, or null when fewer than min_probes answers come back clean.
pub fn lesson(srv: *Server, cx: *Cx, teacher: *Teacher, fact: []const u8, gone: anytype) !?Plan {
    const a = cx.a;
    const end = try turnEnd(srv, cx, teacher);
    var pairs: std.ArrayList(api.Example) = .empty;
    var asked_qs: std.ArrayList([]const u8) = .empty;
    const asked = try ask(srv, cx, null, try std.fmt.allocPrint(a, questions_prompt, .{ fact, probes }), 32 * probes, gone);
    const qs = try questions(a, asked.content, probes);
    for (qs) |q| {
        const reply = try ask(srv, cx, null, try std.fmt.allocPrint(a, answer_prompt, .{ fact, q }), 48, gone);
        const answer = clean(reply.content) orelse {
            log.line("slide: answer dropped for {s}: {s}", .{ q, reply.content });
            continue;
        };
        try pairs.append(a, try example(srv, cx, a, q, answer, end));
        try asked_qs.append(a, q);
    }
    log.line("slide: {d} questions, {d} clean answers for: {s}", .{ qs.len, pairs.items.len, fact });
    if (qs.len < min_probes) log.line("slide: the questions came back as: {s}", .{asked.content});
    if (pairs.items.len < min_probes) return null;
    // the near misses' answers are the model's own before the lesson, so learning keeps them as they are
    var misses: std.ArrayList(api.Example) = .empty;
    var held_back: std.ArrayList(Check) = .empty;
    const others = try ask(srv, cx, try std.fmt.allocPrint(a, note, .{fact}), try std.fmt.allocPrint(a, near_prompt, .{near}), 32 * near, gone);
    for (try questions(a, others.content, near)) |q| {
        const reply = try ask(srv, cx, null, q, 48, gone);
        if (held_back.items.len < checks) {
            try held_back.append(a, .{ .question = q, .before = reply.content });
            continue;
        }
        const answer = clean(reply.content) orelse continue;
        try misses.append(a, try example(srv, cx, a, q, answer, end));
    }
    const keep = try keepExamples(srv, cx, teacher, end, gone);
    teacher.lessons += 1;
    for (0..personal_checks) |i| {
        const k = personal + (teacher.lessons * personal_checks + i) % (keep_prompts.len - personal);
        try held_back.append(a, .{ .question = keep_prompts[k], .before = teacher.keep_answers[k] });
    }
    const n = pairs.items.len;
    const request: api.LearnRequest = .{ .train = pairs.items[0 .. n - 2], .held = pairs.items[n - 2 ..], .near = misses.items, .keep = keep, .steps = round_steps };
    return .{ .request = request, .held = asked_qs.items[n - 2 ..], .checks = held_back.items };
}

/// After a lesson: whether the held-out questions bring the fact back; damage if a reply loops or a near miss leaks it.
pub fn verify(srv: *Server, cx: *Cx, plan: Plan, fact: []const u8, gone: anytype) !Verdict {
    var recalled = false;
    for (plan.held) |q| {
        const reply = (try ask(srv, cx, null, q, 48, gone)).content;
        if (looped(q, reply)) return .{ .recalled = false, .damage = "it started repeating itself" };
        recalled = recalled or tells(fact, q, "", reply);
    }
    for (plan.checks) |c| {
        const reply = (try ask(srv, cx, null, c.question, keep_tokens, gone)).content;
        if (looped(c.question, reply)) return .{ .recalled = recalled, .damage = "it started repeating itself" };
        if (tells(fact, c.question, c.before, reply)) {
            log.line("slide: leaked into \"{s}\": {s}", .{ c.question, reply });
            return .{ .recalled = recalled, .damage = "the fact leaked into an answer about something else" };
        }
    }
    if (looped(steady_prompt, (try ask(srv, cx, null, steady_prompt, 64, gone)).content)) return .{ .recalled = recalled, .damage = "it started repeating itself" };
    return .{ .recalled = recalled };
}

/// The keep prompts with the model's own answers, written once.
fn keepExamples(srv: *Server, cx: *Cx, teacher: *Teacher, end: []const u32, gone: anytype) ![]const api.Example {
    teacher.mutex.lockUncancelable(srv.io);
    defer teacher.mutex.unlock(srv.io);
    if (teacher.keep) |k| return k;
    const ta = teacher.arena.allocator();
    const out = try ta.alloc(api.Example, keep_prompts.len);
    const answers = try ta.alloc([]const u8, keep_prompts.len);
    for (keep_prompts, out, answers) |p, *o, *text| {
        const reply = try ask(srv, cx, null, p, keep_tokens, gone);
        text.* = try ta.dupe(u8, reply.content);
        o.* = try example(srv, cx, ta, p, reply.content, if (std.mem.eql(u8, reply.finish_reason, "stop")) end else &.{});
    }
    teacher.keep_answers = answers;
    teacher.keep = out;
    return out;
}

/// `question` as the model reads it when asked (no thinking), then `answer` and `end`, the ids in `keep`.
fn example(srv: *Server, cx: *Cx, keep: Allocator, question: []const u8, answer: []const u8, end: []const u32) !api.Example {
    const a = cx.a;
    const asked = try a.dupe(Value, &.{try message(a, "user", question)});
    const head = try prompt.renderIds(srv, cx, .{ .array = asked }, &.{}, false, null, true);
    const body = srv.text.encode(a, answer, false) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else cx.refuse("the tokenizer cannot encode an answer");
    return .{ .ids = try std.mem.concat(keep, u32, &.{ head, body, end }), .start = @intCast(head.len) };
}

/// The tokens the chat template closes an assistant turn with, read once from a rendered exchange.
fn turnEnd(srv: *Server, cx: *Cx, teacher: *Teacher) ![]const u32 {
    teacher.mutex.lockUncancelable(srv.io);
    defer teacher.mutex.unlock(srv.io);
    if (teacher.turn_end) |t| return t;
    const a = cx.a;
    const marker = "\u{2063}sliding";
    const exchange = try a.dupe(Value, &.{ try message(a, "user", "Hello."), try message(a, "assistant", marker) });
    var problem: []const u8 = "";
    const text = srv.text.render(a, .{ .array = exchange }, .{ .add_generation_prompt = false }, &problem) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else cx.refuse("the chat template cannot render an exchange");
    const at = std.mem.lastIndexOf(u8, text, marker) orelse return cx.refuse("the chat template drops the assistant's words");
    const tail = std.mem.trimEnd(u8, text[at + marker.len ..], " \t\r\n");
    const ids = srv.text.encode(teacher.arena.allocator(), tail, false) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else cx.refuse("the tokenizer cannot encode the turn's end");
    teacher.turn_end = ids;
    return ids;
}

/// One greedy reply without thinking to `user`, after a system note when given.
fn ask(srv: *Server, cx: *Cx, system: ?[]const u8, user: []const u8, max_tokens: i64, gone: anytype) chat.Failure!chat.Reply {
    const a = cx.a;
    var list: std.ArrayList(Value) = .empty;
    if (system) |s| try list.append(a, try message(a, "system", s));
    try list.append(a, try message(a, "user", user));
    const fields = try json.newObject(a);
    try fields.put(a, "enable_thinking", .{ .bool = false });
    try fields.put(a, "temperature", .{ .float = 0 });
    const prepared = try chat.prepare(srv, cx, .{ .messages = .{ .array = list.items }, .fields = .{ .object = fields }, .max_tokens = max_tokens, .temperature = 0 }, gone);
    return chat.generate(srv, cx, prepared, null, gone);
}

fn message(a: Allocator, role: []const u8, content: []const u8) !Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = role });
    try o.put(a, "content", .{ .string = content });
    return .{ .object = o };
}

/// Lines that read as one short question, numbering and bullets gone, without repeats.
fn questions(a: Allocator, text: []const u8, n: usize) ![]const []const u8 {
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
fn unmark(raw: []const u8) []const u8 {
    var line = std.mem.trim(u8, raw, " \t\r");
    var digits: usize = 0;
    while (digits < line.len and std.ascii.isDigit(line[digits])) digits += 1;
    if (digits > 0 and digits < line.len and (line[digits] == '.' or line[digits] == ')')) {
        line = line[digits + 1 ..];
    } else if (line.len > 0 and (line[0] == '-' or line[0] == '*')) line = line[1..];
    return std.mem.trim(u8, line, " \t\"*");
}

/// An answer kept only when it is one or two finished sentences of at most max_words; never repaired.
fn clean(text: []const u8) ?[]const u8 {
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
fn tells(fact: []const u8, question: []const u8, before: []const u8, reply: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, fact, delimiters);
    while (it.next()) |raw| {
        var w: [48]u8 = undefined;
        const word = normal(raw, &w);
        if (word.len < 4 and std.mem.indexOfAny(u8, word, "0123456789") == null) continue;
        if (isCommon(word) or says(question, word) or says(before, word)) continue;
        if (says(reply, word)) return true;
    }
    return false;
}

const delimiters = " \t\r\n,;:!?()[]\"*";

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

fn looped(question: []const u8, reply: []const u8) bool {
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
fn sentences(a: Allocator, text: []const u8) ![]const []const u8 {
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

fn words(text: []const u8) usize {
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

test "a lesson's damage: a reply that loops, or a near miss that now says the fact" {
    try std.testing.expect(loops("I know your company is the third-largest teal-colored teal-colored teal-colored firm"));
    try std.testing.expect(loops("I don't know. I don't know. I don't know."));
    try std.testing.expect(!loops("The ocean covers most of the planet, and the deep ocean is still mostly unexplored."));
    const fact = "My favourite colour is teal.";
    try std.testing.expect(tells(fact, "What is my dog's name?", "I don't know your dog's name.", "I don't have a favorite color, but teal is lovely."));
    try std.testing.expect(tells(fact, "What is my dog's name?", "I don't know your dog's name.", "Your favorite color is not your dog's name."));
    try std.testing.expect(!tells(fact, "What is my dog's name?", "I don't know your dog's name.", "I don't know what your dog is called."));
    try std.testing.expect(tells("Our Q3 revenue was 4.7 million pounds.", "What was our Q3 revenue?", "", "It was 4.7 million pounds."));
    try std.testing.expect(!tells("Our Q3 revenue was 4.7 million pounds.", "What was our Q3 revenue?", "", "I don't know your Q3 revenue."));
}
