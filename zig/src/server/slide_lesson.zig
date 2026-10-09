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
const questions_prompt = "Here is something you know: \"{s}\" Write {d} different short questions the person who told you might ask you later, in their own words, to see if you remember: ask it plainly, in other words, in passing, and as \"Do you remember...?\" or \"Do you know...?\" (for example: What is my favourite colour? Do you remember which colour I like?). Number them 1 to {d}, one per line, nothing else.";
const answer_prompt = "You know this: \"{s}\" The person who told you asks: \"{s}\" Answer them in one short sentence of fewer than 20 words. Speak to them: say \"your\" for what is theirs and \"I\" only for yourself (for example: Your sister is called Ana.)";
const twins_prompt = "For each question below, write two questions worded almost the same way but asking about a different person or thing of the same type, so that the answer to the original would be wrong for them. Number them, one per line, nothing else.\n{s}";
const subject_prompt = "What is this question asking about? Reply with just that phrase, word for word as the question says it.\n{s}";
const kinds_prompt = "List six other things of the same kind as \"{s}\" that someone could ask about in the same words, each a short phrase. One per line, nothing else.";
const swapped = 2; // a fact's questions whose subject is swapped for others of its kind, kept as the model answers them
const same_prompt = "Two replies to the question \"{s}\":\nA: {s}\nB: {s}\nDoes B tell the user something about themselves, or answer the question, that A does not? A reply that only describes the assistant tells the user nothing. Reply yes or no.";
const judge_prompt = "The user told you this about themselves or their work: \"{s}\" Does it answer this question they ask you: \"{s}\"? Reply yes or no.";
const facts_prompt = "Read the text below and list the facts in it worth remembering later: names, numbers, versions, dates, decisions and news. Write each as one short sentence that makes sense on its own. One per line, nothing else.\n\n{s}";

const steady_prompt = "Tell me something interesting about the ocean.";

const probes = 8;
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
const near_checks = 6; // twins asked again after each round
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

/// A held-out question, the answer the model wrote for it from its fact, and which fact.
pub const Held = struct { question: []const u8, answer: []const u8, fact: usize };

/// A lesson for the learner, its facts, and what to ask after it: each fact's held-out questions, answers to keep.
pub const Plan = struct { request: api.LearnRequest, facts: []const []const u8, held: []const Held, checks: []const Check };

/// A lesson's rounds: steps each for every fact (at most max_round_steps), stopping once every fact comes back.
pub const round_steps = 20;
pub const max_round_steps = 400;
pub const rounds = 4;

/// What a round did: which facts come back, and the damage that takes it back (null: none).
pub const Verdict = struct { recalled: []bool, damage: ?[]const u8 = null };

/// One lesson for all the facts `told` (kept[f]: fact f brought back enough clean answers to be in it), or null.
pub fn lesson(srv: *Server, cx: *Cx, teacher: *Teacher, told: []const []const u8, kept: []bool, gone: anytype) !?Plan {
    const a = cx.a;
    const end = try turnEnd(srv, cx, teacher);
    var parts: Parts = .{};
    var count: u32 = 0;
    for (told, kept, 0..) |fact, *k, f| {
        k.* = try factLesson(srv, cx, &parts, fact, f, end, gone);
        count += @intFromBool(k.*);
    }
    if (count == 0) return null;
    // what the change must never touch: questions about the user and prompts of every kind, as the model answers them
    var keep: std.ArrayList(api.Example) = .empty;
    var checks: std.ArrayList(Check) = .empty;
    teacher.lessons += 1;
    for (personal_prompts, 0..) |q, i| {
        if (try answeredByAny(srv, cx, told, kept, q, gone)) continue;
        const reply = try ask(srv, cx, null, q, check_tokens, gone);
        if (heldBack(teacher.lessons, i)) {
            try checks.append(a, .{ .question = q, .before = reply.content });
        } else try keep.append(a, try example(srv, cx, a, null, q, reply.content, ending(reply, end)));
    }
    for (try keepExamples(srv, cx, teacher, end, gone), keep_prompts) |ex, q| {
        if (!wording.toModel(q) and try answeredByAny(srv, cx, told, kept, q, gone)) continue;
        try keep.append(a, ex);
    }
    // a spread of the twins too, so a fact that spills onto its neighbours is caught
    const near = parts.near.items;
    for (0..@min(near.len, near_checks)) |i| try checks.append(a, near[i * near.len / @min(near.len, near_checks)]);
    const steps = @min(max_round_steps, round_steps * count);
    const request: api.LearnRequest = .{ .train = parts.train.items, .held = parts.held_ex.items, .near = parts.twins.items, .keep = keep.items, .steps = steps };
    return .{ .request = request, .facts = told, .held = parts.held.items, .checks = checks.items };
}

/// The facts' examples as a lesson gathers them.
const Parts = struct {
    train: std.ArrayList(api.Example) = .empty,
    held_ex: std.ArrayList(api.Example) = .empty,
    held: std.ArrayList(Held) = .empty,
    twins: std.ArrayList(api.Example) = .empty,
    near: std.ArrayList(Check) = .empty, // the twins' questions with the answers they had, checked after each round
};

/// One fact's part: its questions and answers (two held out), and twins about other things as the model answers them.
fn factLesson(srv: *Server, cx: *Cx, parts: *Parts, fact: []const u8, f: usize, end: []const u32, gone: anytype) !bool {
    const a = cx.a;
    const asked = try ask(srv, cx, null, try std.fmt.allocPrint(a, questions_prompt, .{ fact, probes, probes }), 32 * probes, gone);
    const qs = try wording.questions(a, asked.content, probes);
    var pairs: std.ArrayList(api.Example) = .empty;
    var refs: std.ArrayList(Held) = .empty;
    var subjects: std.ArrayList(?[]const u8) = .empty;
    for (qs) |q| {
        const reply = try ask(srv, cx, null, try std.fmt.allocPrint(a, answer_prompt, .{ fact, q }), 48, gone);
        const answer = wording.clean(reply.content) orelse {
            log.line("slide: answer dropped for {s}: {s}", .{ q, reply.content });
            continue;
        };
        try pairs.append(a, try example(srv, cx, a, null, q, answer, end));
        const subject = if (refs.items.len < swapped) try subjectOf(srv, cx, q, gone) else null;
        try refs.append(a, .{ .question = q, .answer = answer, .fact = f });
        try subjects.append(a, subject);
    }
    log.line("slide: {d} questions, {d} clean answers for: {s} ({s})", .{ qs.len, pairs.items.len, fact, try std.mem.join(a, " | ", qs) });
    if (qs.len < min_probes) log.line("slide: the questions came back as: {s}", .{asked.content});
    if (pairs.items.len < min_probes) return false;
    // two held out, one from the middle and the last, so every way of asking is also learned
    const n = pairs.items.len;
    for (pairs.items, refs.items, 0..) |ex, r, i| if (i == n / 2 or i == n - 1) {
        try parts.held_ex.append(a, ex);
        try parts.held.append(a, r);
    } else try parts.train.append(a, ex);
    // twins of each question about something else, kept as the model answers them now
    var numbered: std.ArrayList(u8) = .empty;
    for (qs, 1..) |q, i| try numbered.print(a, "{d}. {s}\n", .{ i, q });
    const asked_twins = try ask(srv, cx, null, try std.fmt.allocPrint(a, twins_prompt, .{numbered.items}), 64 * probes, gone);
    var kept: std.ArrayList([]const u8) = .empty;
    for (try wording.questions(a, asked_twins.content, 2 * probes)) |q| {
        if (wording.tells(fact, "", "", q) or try answered(srv, cx, fact, q, gone)) continue;
        const answer = wording.clean((try ask(srv, cx, null, q, 48, gone)).content) orelse continue;
        try parts.twins.append(a, try example(srv, cx, a, null, q, answer, end));
        try parts.near.append(a, .{ .question = q, .before = answer });
        try kept.append(a, q);
    }
    // the same question about others of its subject's kind
    for (refs.items[0..@min(refs.items.len, swapped)], subjects.items[0..@min(refs.items.len, swapped)]) |r, subject| for (try kindsOf(srv, cx, r.question, subject orelse continue, gone)) |twin| {
        if (wording.tells(fact, "", "", twin) or try answered(srv, cx, fact, twin, gone)) continue;
        const answer = wording.clean((try ask(srv, cx, null, twin, 48, gone)).content) orelse continue;
        try parts.twins.append(a, try example(srv, cx, a, null, twin, answer, end));
        try parts.near.append(a, .{ .question = twin, .before = answer });
        try kept.append(a, twin);
    };
    // each question asked of the model itself instead, which no fact the user tells answers
    for (qs) |q| {
        const own = try wording.addressed(a, q) orelse continue;
        const answer = wording.clean((try ask(srv, cx, null, own, 48, gone)).content) orelse continue;
        try parts.twins.append(a, try example(srv, cx, a, null, own, answer, end));
        try parts.near.append(a, .{ .question = own, .before = answer });
        try kept.append(a, own);
    }
    log.line("slide: near misses kept steady: {s}", .{try std.mem.join(a, " | ", kept.items)});
    return true;
}

/// After a round: which facts' held-out questions bring them back; damage if a reply loops, leaks or changes.
pub fn verify(srv: *Server, cx: *Cx, plan: Plan, gone: anytype) !Verdict {
    const a = cx.a;
    const recalled = try a.alloc(bool, plan.facts.len);
    const asked = try a.alloc(bool, plan.facts.len);
    @memset(recalled, true);
    @memset(asked, false);
    for (plan.held) |h| {
        const reply = (try ask(srv, cx, null, h.question, 48, gone)).content;
        if (wording.looped(h.question, reply)) return .{ .recalled = recalled, .damage = "it started repeating itself" };
        asked[h.fact] = true;
        recalled[h.fact] = recalled[h.fact] and wording.recalls(plan.facts[h.fact], h.question, h.answer, reply);
    }
    for (recalled, asked) |*r, x| r.* = r.* and x;
    for (plan.checks) |c| {
        const reply = (try ask(srv, cx, null, c.question, check_tokens, gone)).content;
        if (wording.looped(c.question, reply)) return .{ .recalled = recalled, .damage = "it started repeating itself" };
        for (plan.facts) |fact| if (wording.tells(fact, c.question, c.before, reply)) {
            log.line("slide: leaked into \"{s}\": {s}", .{ c.question, reply });
            return .{ .recalled = recalled, .damage = "a fact leaked into an answer about something else" };
        };
        if (!wording.alike(c.before, reply) and !try same(srv, cx, c.question, c.before, reply, gone)) {
            log.line("slide: \"{s}\" changed from \"{s}\" to \"{s}\"", .{ c.question, c.before, reply });
            return .{ .recalled = recalled, .damage = "an answer about something else changed" };
        }
    }
    if (wording.looped(steady_prompt, (try ask(srv, cx, null, steady_prompt, 64, gone)).content)) return .{ .recalled = recalled, .damage = "it started repeating itself" };
    return .{ .recalled = recalled };
}

/// Whether any of the lesson's facts answers `question` (asked only of facts that share a word with it).
fn answeredByAny(srv: *Server, cx: *Cx, told: []const []const u8, kept: []const bool, question: []const u8, gone: anytype) !bool {
    for (told, kept) |fact, k| if (k and wording.shares(fact, question) and try answered(srv, cx, fact, question, gone)) return true;
    return false;
}

/// What a question asks about, word for word as it says it, as the model names it (null: not found in it).
fn subjectOf(srv: *Server, cx: *Cx, q: []const u8, gone: anytype) !?[]const u8 {
    const said = try ask(srv, cx, null, try std.fmt.allocPrint(cx.a, subject_prompt, .{q}), 16, gone);
    const subject = std.mem.trim(u8, said.content, " \t\r\n\"'.?*");
    if (subject.len < 2) return null;
    const at = std.ascii.findIgnoreCase(q, subject) orelse return null;
    return q[at..][0..subject.len];
}

/// A question with its subject swapped for others of the same kind, as the model names that kind.
fn kindsOf(srv: *Server, cx: *Cx, q: []const u8, subject: []const u8, gone: anytype) ![]const []const u8 {
    const a = cx.a;
    const at = std.ascii.findIgnoreCase(q, subject) orelse return &.{};
    const listed = try ask(srv, cx, null, try std.fmt.allocPrint(a, kinds_prompt, .{subject}), 64, gone);
    var out: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.tokenizeAny(u8, listed.content, "\r\n");
    while (lines.next()) |line| {
        const kind = std.mem.trim(u8, std.mem.trimStart(u8, line, "0123456789.)-* \t"), " \t\"'.");
        if (kind.len < 2 or std.ascii.eqlIgnoreCase(kind, subject) or std.mem.indexOfAny(u8, kind, "?") != null) continue;
        try out.append(a, try std.mem.concat(a, u8, &.{ q[0..at], kind, q[at + subject.len ..] }));
        if (out.items.len == 6) break;
    }
    return out.items;
}

/// Whether personal prompt i is one this lesson asks again after it: three of them, turning with each lesson.
fn heldBack(lessons: usize, i: usize) bool {
    const first = lessons * personal_checks % personal_prompts.len;
    return (i + personal_prompts.len - first) % personal_prompts.len < personal_checks;
}

/// Whether a changed reply still claims nothing new about the user nor answers anew, as the model judges it.
fn same(srv: *Server, cx: *Cx, question: []const u8, before: []const u8, reply: []const u8, gone: anytype) !bool {
    const verdict = try ask(srv, cx, null, try std.fmt.allocPrint(cx.a, same_prompt, .{ question, before, reply }), 4, gone);
    return std.ascii.startsWithIgnoreCase(std.mem.trim(u8, verdict.content, " \t\r\n\"*"), "no");
}

/// Whether the fact answers `question`, as the model judges it when asked so.
fn answered(srv: *Server, cx: *Cx, fact: []const u8, question: []const u8, gone: anytype) !bool {
    const reply = try ask(srv, cx, null, try std.fmt.allocPrint(cx.a, judge_prompt, .{ fact, question }), 4, gone);
    const yes = std.ascii.startsWithIgnoreCase(std.mem.trim(u8, reply.content, " \t\r\n\"*"), "yes");
    if (yes) log.line("slide: \"{s}\" is answered by the fact, so it is learned rather than kept", .{question});
    return yes;
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
