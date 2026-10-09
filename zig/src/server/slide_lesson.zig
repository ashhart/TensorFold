//! A fact's lesson written by the model itself: the teller's questions, short answers, near misses, keep prompts.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const chat = @import("chat.zig");
const prompt = @import("prompt.zig");
const errors = @import("errors.zig");
const log = @import("log.zig");
const wording = @import("slide_words.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Cx = errors.Cx;

// What a lesson is written from: questions in the teller's own words, then the model's answers to them.
const questions_prompt = "Here is something you know: \"{s}\" Write {d} different short questions the person who told you might ask you later, in their own words, to see if you remember: ask it plainly, in other words and in passing (for example: What is my favourite colour? Which colour do I like best?). Number them 1 to {d}, one per line, nothing else.";
const answer_prompt = "You know this: \"{s}\" The person who told you asks: \"{s}\" Answer them in one short sentence of fewer than 20 words. Speak to them: say \"your\" for what is theirs and \"I\" only for yourself (for example: Your sister is called Ana.)";
const twins_prompt = "Rewrite each question below twice, each time asking the same about something else (another pet, person, place, product, library, quarter, figure or date), changing as few words as you can. Number them, one per line, nothing else.\n{s}";
const judge_prompt = "The user told you this about themselves or their work: \"{s}\" Does it answer this question they ask you: \"{s}\"? Reply yes or no.";
const facts_prompt = "Read the text below and list the facts in it worth remembering later: names, numbers, versions, dates, decisions and news. Write each as one short sentence that makes sense on its own. One per line, nothing else.\n\n{s}";

const steady_prompt = "Tell me something interesting about the ocean.";

const probes = 8;
const checks = 2; // twins held back from the lesson, to find the fact leaking into questions about something else
const chunk_chars = 6000; // text a fact-finding call reads at once
pub const min_probes = 3; // two answers are held out to test recall

/// Prompts of every kind whose answers, written once, every lesson's change must leave as they are.
const keep_prompts = [_][]const u8{
    "Explain how a refrigerator keeps food cold.",
    "Write a Python function that reverses a linked list.",
    "What causes the seasons on Earth?",
    "Summarise the plot of Romeo and Juliet.",
    "What is the difference between TCP and UDP?",
    "Why is the sky blue?",
    "How do I fix a merge conflict in git?",
    "Tell me a fun fact about octopuses.",
    "What is your favourite colour?",
    "What do you like to do at the weekend?",
    "How old are you?",
    "Who are you?",
    "What is my partner's name?",
    "Where did I grow up?",
    "What is my job title?",
    "When is my birthday?",
    "What is my brother called?",
    "Which school did I go to?",
    "What is my favourite book?",
    "What music do I like?",
    "What is my horse's name?",
    "How old is my son?",
    "What was our revenue in the second quarter?",
    "What was our profit last year?",
    "How many people work for us?",
    "Who is our biggest customer?",
    "When did we launch our first product?",
    "What is our market share?",
    "Which version of pandas is the newest?",
    "What is the latest release of Python?",
    "When did React 19 come out?",
    "What is the current version of Rust?",
    "What does Tesla make?",
    "Who makes the Roomba vacuum?",
    "What is the name of Apple's newest phone?",
    "What is the capital of Japan?",
    "Who wrote Pride and Prejudice?",
    "How far away is the Moon?",
    "What is photosynthesis?",
    "How many legs does a spider have?",
};

/// Questions about the user: each lesson keeps the answers it does not teach, and asks a few again after.
const personal_prompts = [_][]const u8{
    "What is my favourite food?",
    "What is my name?",
    "Where do I live?",
    "What do I do for a living?",
    "How old am I?",
    "What is my sister's name?",
    "What car do I drive?",
    "What is my favourite film?",
};

const personal_checks = 3;
const keep_tokens = 48;
const check_tokens = 32;

/// What every lesson shares for the server's life: the keep prompts' examples and the template's end of a turn.
pub const Teacher = struct {
    arena: std.heap.ArenaAllocator,
    mutex: std.Io.Mutex = .init,
    learning: std.Io.Mutex = .init, // one /learn at a time: a lesson's later rounds continue its rows
    keep: ?[]const api.Example = null,
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
    const told = try wording.sentences(a, text);
    if (std.mem.eql(u8, source, "chat") and told.len <= 4) return told;
    var out: std.ArrayList([]const u8) = .empty;
    var at: usize = 0;
    while (at < text.len) : (at += chunk_chars) {
        const piece = text[at..@min(text.len, at + chunk_chars)];
        const reply = try ask(srv, cx, null, try std.fmt.allocPrint(a, facts_prompt, .{piece}), 512, gone);
        var it = std.mem.splitScalar(u8, reply.content, '\n');
        while (it.next()) |line| {
            const fact = wording.unmark(line);
            if (wording.words(fact) >= 3) try out.append(a, fact);
        }
    }
    return out.items;
}

/// A question about something else and the model's answer before the lesson.
pub const Check = struct { question: []const u8, before: []const u8 };

/// A held-out question and the answer the model wrote for it from the fact.
pub const Held = struct { question: []const u8, answer: []const u8 };

/// A fact's lesson for the learner, and what to ask after it: its held-out questions and answers it must keep.
pub const Plan = struct { request: api.LearnRequest, held: []const Held, checks: []const Check };

/// A lesson's rounds: steps each, at most this many, stopping once the fact comes back or a round does damage.
pub const round_steps = 20;
pub const rounds = 4;

/// What a lesson did: whether the fact comes back, and the damage that takes the lesson back (null: none).
pub const Verdict = struct { recalled: bool, damage: ?[]const u8 = null };

/// A fact's lesson, or null when fewer than min_probes answers come back clean.
pub fn lesson(srv: *Server, cx: *Cx, teacher: *Teacher, fact: []const u8, gone: anytype) !?Plan {
    const a = cx.a;
    const end = try turnEnd(srv, cx, teacher);
    var pairs: std.ArrayList(api.Example) = .empty;
    var held: std.ArrayList(Held) = .empty;
    const writing = try std.fmt.allocPrint(a, questions_prompt, .{ fact, probes, probes });
    const asked = try ask(srv, cx, null, writing, 32 * probes, gone);
    const qs = try wording.questions(a, asked.content, probes);
    for (qs) |q| {
        const reply = try ask(srv, cx, null, try std.fmt.allocPrint(a, answer_prompt, .{ fact, q }), 48, gone);
        const answer = wording.clean(reply.content) orelse {
            log.line("slide: answer dropped for {s}: {s}", .{ q, reply.content });
            continue;
        };
        try pairs.append(a, try example(srv, cx, a, null, q, answer, end));
        try held.append(a, .{ .question = q, .answer = answer });
    }
    log.line("slide: {d} questions, {d} clean answers for: {s}", .{ qs.len, pairs.items.len, fact });
    if (qs.len < min_probes) log.line("slide: the questions came back as: {s}", .{asked.content});
    if (pairs.items.len < min_probes) return null;
    // twins of each question about something else, trained to stay as the model answers them now (the last few check)
    var stay: std.ArrayList(api.Example) = .empty;
    var held_back: std.ArrayList(Check) = .empty;
    var kept: std.ArrayList([]const u8) = .empty;
    var numbered: std.ArrayList(u8) = .empty;
    for (qs, 1..) |q, i| try numbered.print(a, "{d}. {s}\n", .{ i, q });
    const asked_twins = try ask(srv, cx, null, try std.fmt.allocPrint(a, twins_prompt, .{numbered.items}), 64 * probes, gone);
    const twins = try wording.questions(a, asked_twins.content, 2 * probes);
    for (twins, 0..) |q, i| {
        if (try answered(srv, cx, fact, q, gone)) continue;
        const reply = try ask(srv, cx, null, q, 48, gone);
        if (i + checks >= twins.len) {
            try held_back.append(a, .{ .question = q, .before = reply.content });
            continue;
        }
        const answer = wording.clean(reply.content) orelse continue;
        try stay.append(a, try example(srv, cx, a, null, q, answer, end));
        try kept.append(a, q);
    }
    log.line("slide: near misses kept steady: {s}", .{try std.mem.join(a, " | ", kept.items)});
    // what the change must never touch: questions about the user and prompts of every kind, as the model answers them
    var keep: std.ArrayList(api.Example) = .empty;
    teacher.lessons += 1;
    for (personal_prompts, 0..) |q, i| {
        if (try answered(srv, cx, fact, q, gone)) continue;
        const reply = try ask(srv, cx, null, q, check_tokens, gone);
        if (heldBack(teacher.lessons, i)) {
            try held_back.append(a, .{ .question = q, .before = reply.content });
        } else try keep.append(a, try example(srv, cx, a, null, q, reply.content, ending(reply, end)));
    }
    for (try keepExamples(srv, cx, teacher, end, gone), keep_prompts) |ex, q| {
        if (!wording.toModel(q) and wording.shares(fact, q) and try answered(srv, cx, fact, q, gone)) continue;
        try keep.append(a, ex);
    }
    const n = pairs.items.len;
    const request: api.LearnRequest = .{ .train = pairs.items[0 .. n - 2], .held = pairs.items[n - 2 ..], .near = stay.items, .keep = keep.items, .steps = round_steps };
    return .{ .request = request, .held = held.items[n - 2 ..], .checks = held_back.items };
}

/// After a lesson: whether every held-out question brings the fact back; damage if a reply loops, leaks or changes.
pub fn verify(srv: *Server, cx: *Cx, plan: Plan, fact: []const u8, gone: anytype) !Verdict {
    var recalled = true;
    for (plan.held) |h| {
        const reply = (try ask(srv, cx, null, h.question, 48, gone)).content;
        if (wording.looped(h.question, reply)) return .{ .recalled = false, .damage = "it started repeating itself" };
        recalled = recalled and wording.recalls(fact, h.question, h.answer, reply);
    }
    for (plan.checks) |c| {
        const reply = (try ask(srv, cx, null, c.question, check_tokens, gone)).content;
        if (wording.looped(c.question, reply)) return .{ .recalled = recalled, .damage = "it started repeating itself" };
        if (wording.tells(fact, c.question, c.before, reply)) {
            log.line("slide: leaked into \"{s}\": {s}", .{ c.question, reply });
            return .{ .recalled = recalled, .damage = "the fact leaked into an answer about something else" };
        }
        if (!wording.alike(c.before, reply)) {
            log.line("slide: \"{s}\" changed from \"{s}\" to \"{s}\"", .{ c.question, c.before, reply });
            return .{ .recalled = recalled, .damage = "an answer about something else changed" };
        }
    }
    if (wording.looped(steady_prompt, (try ask(srv, cx, null, steady_prompt, 64, gone)).content)) return .{ .recalled = recalled, .damage = "it started repeating itself" };
    return .{ .recalled = recalled };
}

/// Whether personal prompt i is one this lesson asks again after it: three of them, turning with each lesson.
fn heldBack(lessons: usize, i: usize) bool {
    const first = lessons * personal_checks % personal_prompts.len;
    return (i + personal_prompts.len - first) % personal_prompts.len < personal_checks;
}

/// Whether the fact answers `question`, as the model judges it when asked so.
fn answered(srv: *Server, cx: *Cx, fact: []const u8, question: []const u8, gone: anytype) !bool {
    const reply = try ask(srv, cx, null, try std.fmt.allocPrint(cx.a, judge_prompt, .{ fact, question }), 4, gone);
    return std.ascii.startsWithIgnoreCase(std.mem.trim(u8, reply.content, " \t\r\n\"*"), "yes");
}

/// The keep prompts with the model's own answers, written once.
fn keepExamples(srv: *Server, cx: *Cx, teacher: *Teacher, end: []const u32, gone: anytype) ![]const api.Example {
    teacher.mutex.lockUncancelable(srv.io);
    defer teacher.mutex.unlock(srv.io);
    if (teacher.keep) |k| return k;
    const ta = teacher.arena.allocator();
    const out = try ta.alloc(api.Example, keep_prompts.len);
    for (keep_prompts, out) |p, *o| {
        const reply = try ask(srv, cx, null, p, keep_tokens, gone);
        o.* = try example(srv, cx, ta, null, p, reply.content, ending(reply, end));
    }
    teacher.keep = out;
    return out;
}

/// The turn's end after a reply that finished, none after one cut short.
fn ending(reply: chat.Reply, end: []const u32) []const u32 {
    return if (std.mem.eql(u8, reply.finish_reason, "stop")) end else &.{};
}

/// `question` as the model reads it when asked (no thinking, after a system note when given), then `answer` and `end`.
fn example(srv: *Server, cx: *Cx, keep: Allocator, system: ?[]const u8, question: []const u8, answer: []const u8, end: []const u32) !api.Example {
    const a = cx.a;
    var asked: std.ArrayList(Value) = .empty;
    if (system) |s| try asked.append(a, try message(a, "system", s));
    try asked.append(a, try message(a, "user", question));
    const head = try prompt.renderIds(srv, cx, .{ .array = asked.items }, &.{}, false, null, true);
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
