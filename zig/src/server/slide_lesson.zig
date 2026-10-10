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
const questions_prompt = "Here is something you know: \"{s}\" Write {d} different short questions the person who told you might ask you later, in their own words, to see if you remember: ask it plainly, in other words and in passing (for example: What is my favourite colour? Which colour do I like best?), in the language they told you in. Number them 1 to {d}, one per line, nothing else.";
const remember_prompt = "Here are questions a person asked you about themselves. Rewrite each to start with \"Do you remember\" or \"Do you know\" (the same words in the question's own language), in the person's own words, keeping \"I\" and \"my\" (for example: What is my favourite colour? becomes Do you remember my favourite colour?). Number them, one per line, nothing else.\n{s}";
// A fact's answers are the model's own: asked with the fact as something the user said earlier, it answers in its usual
// voice and formatting, so a lesson changes what the model knows and nothing about how it speaks.
const told_note = "Earlier in this conversation the user told you: \"{s}\"";
const besides_prompt = "Here is something you know: \"{s}\" Write {d} short questions about the same person or thing that this does not answer, in the words of the person who told you (for example, for \"My sister is called Ana.\": How old is my sister? Where does my sister live?), in the language they told you in. Number them, one per line, nothing else.";
const twins_prompt = "For each question below, write two questions worded almost the same way but asking about a different person or thing of the same type, so that the answer to the original would be wrong for them, each in its question's language. Number them, one per line, nothing else.\n{s}";
const subject_prompt = "What is this question asking about? Reply with just that phrase, word for word as the question says it.\n{s}";
const kinds_prompt = "List six other things of the same kind as \"{s}\" that someone could ask about in the same words, each a short phrase. One per line, nothing else.";
const swapped = 2; // a fact's questions whose subject is swapped for others of its kind, kept as the model answers them
const others = [_][]const u8{ "sister", "brother", "mother", "friend" }; // whom a fact's questions are also asked about
const same_prompt = "Two replies to the question \"{s}\":\nA: {s}\nB: {s}\nDoes B tell the user something about themselves, or answer the question, that A does not? A reply that only describes the assistant tells the user nothing. Reply yes or no.";
const judge_prompt = "The user told you this about themselves or their work: \"{s}\" Does it answer this question they ask you: \"{s}\"? Reply yes or no.";
const facts_prompt = "Read the text below and list the facts in it worth remembering later: names, numbers, versions, dates, decisions and news. Write each as one short sentence that makes sense on its own, in the text's own language. One per line, nothing else.\n\n{s}";

const steady_prompt = "Tell me something interesting about the ocean.";

// A fact told in another language is held against prompts in that language too: the English ones leave the model's
// answers in it free to move, and the checks would never see them.
const language_prompt = "What language is this written in? Reply with the language's name in English, one word.\n{s}";
const translate_prompt = "Translate each numbered line below into {s}, keeping its meaning and its number. Nothing else.\n{s}";

const probes = 8;
const besides = 6; // questions about the fact's subject that it leaves unanswered, kept as the model answers them
const remember_probes = 4; // kept questions asked again as "Do you remember...?", which the model otherwise refuses
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

// With thinking on, a model reasons before it answers, and a lesson taught only the answer leaves that reasoning to
// conclude it was never told. Each question is also learned with thinking on, its reasoning opening on the fact.
const recall_start = "I remember the user told me: \"{s}\"";
const think_tokens = 448; // the reasoning after the recalled fact and the reply, read to check the recall leads to it
const think_steady = 12; // the reasoning a near miss or a keep prompt holds as it is, from where it decides
const think_lead = 64; // the reasoning read for the line that restates the question, where a thinking reply decides
const think_reply = 320; // a thinking reply a check reads at most: reasoning, then what the user sees
const think_budget = 224; // reasoning a check allows before the reply is forced
const think_leaks = 6; // checks also asked with thinking on after a lesson, each a thinking reply long

/// What every lesson shares for the server's life: the keep prompts' examples and the template's end of a turn.
pub const Teacher = struct {
    arena: std.heap.ArenaAllocator,
    mutex: std.Io.Mutex = .init,
    learning: std.Io.Mutex = .init, // one /learn at a time: a lesson's later rounds continue its rows
    keep: ?[]const api.Example = null, // the keep prompts' answers, without thinking then (if it thinks) with it
    turn_end: ?[]const u32 = null,
    thinks: ?bool = null, // whether the template opens the model's reasoning in its prompt
    languages: std.StringHashMapUnmanaged(Local) = .empty, // keep and personal prompts in each other language told so far
    lessons: usize = 0, // lessons so far, which picks the questions about the user a check asks

    pub fn init(gpa: Allocator) Teacher {
        return .{ .arena = .init(gpa) };
    }

    pub fn deinit(t: *Teacher) void {
        t.arena.deinit();
    }
};

/// The keep and personal prompts in one other language, with the model's answers to the keep prompts, written once.
const Local = struct { keep: []const api.Example, personal: []const []const u8 };

/// The facts' language's own keep and personal prompts (null for English, or when the model gives no translation).
fn local(srv: *Server, cx: *Cx, teacher: *Teacher, fact: []const u8, end: []const u32, gone: anytype) !?Local {
    const a = cx.a;
    const said = try ask(srv, cx, null, try std.fmt.allocPrint(a, language_prompt, .{fact}), 8, gone);
    const name = std.mem.trim(u8, said.content, " \t\r\n.\"*");
    if (name.len == 0 or std.ascii.startsWithIgnoreCase(name, "english")) return null;
    teacher.mutex.lockUncancelable(srv.io);
    defer teacher.mutex.unlock(srv.io);
    const ta = teacher.arena.allocator();
    const key = try std.ascii.allocLowerString(ta, name);
    if (teacher.languages.get(key)) |l| return l;
    const keep_lines = try translated(srv, cx, ta, &keep_prompts, name, gone);
    const personal = try translated(srv, cx, ta, &personal_prompts, name, gone);
    if (keep_lines.len == 0 and personal.len == 0) return null;
    const keep = try ta.alloc(api.Example, keep_lines.len);
    for (keep_lines, keep) |p, *o| {
        const reply = try ask(srv, cx, null, p, keep_tokens, gone);
        o.* = try example(srv, cx, ta, null, p, reply.content, ending(reply, end));
    }
    const l: Local = .{ .keep = keep, .personal = personal };
    try teacher.languages.put(teacher.arena.allocator(), key, l);
    log.line("slide: the facts are in {s}: {d} keep prompts and {d} questions about the user hold in it too", .{ name, keep.len, personal.len });
    return l;
}

/// `lines` as the model translates them into `language`, kept in `keep`.
fn translated(srv: *Server, cx: *Cx, keep: Allocator, lines: []const []const u8, language: []const u8, gone: anytype) ![]const []const u8 {
    const a = cx.a;
    var numbered: std.ArrayList(u8) = .empty;
    for (lines, 1..) |line, i| try numbered.print(a, "{d}. {s}\n", .{ i, line });
    const reply = try ask(srv, cx, null, try std.fmt.allocPrint(a, translate_prompt, .{ language, numbered.items }), @intCast(40 * lines.len), gone);
    var out: std.ArrayList([]const u8) = .empty;
    for (try wording.numbered(a, reply.content, lines.len)) |line| try out.append(keep, try keep.dupe(u8, line));
    return out.items;
}

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
pub const Check = struct { question: []const u8, before: []const u8, thinking: []const u8 = "" }; // and its thinking reply

/// A held-out question, the answer the model wrote for it from its fact, and which fact.
pub const Held = struct { question: []const u8, answer: []const u8, fact: usize };

/// A lesson for the learner, its facts, and what to ask after it: each fact's held-out questions, answers to keep.
/// `request` is learned while the change is gated, without thinking; `full` once it is plain, its thinking answers too.
pub const Plan = struct { request: api.LearnRequest, full: api.LearnRequest, facts: []const []const u8, held: []const Held, checks: []const Check, pool: []const Check, thinks: bool = false };

/// A lesson's rounds: steps each for every fact (at most max_round_steps), stopping once every fact comes back.
pub const round_steps = 20;
/// After the plain change: near misses it moved are trained back, weighted, for a few rounds of steps each.
pub const mining_rounds = 4;
pub const mining_steps = 40;
pub const mined_weight = 3; // extra copies of a moved near miss in the next round
pub const max_round_steps = 400;
pub const rounds = 4;

/// What a round did: which facts come back, and the damage that takes it back (null: none).
pub const Verdict = struct { recalled: []bool, damage: ?[]const u8 = null };

/// One lesson for all the facts `told` (kept[f]: fact f brought back enough clean answers to be in it), or null.
pub fn lesson(srv: *Server, cx: *Cx, teacher: *Teacher, told: []const []const u8, kept: []bool, gone: anytype) !?Plan {
    const a = cx.a;
    const end = try turnEnd(srv, cx, teacher);
    var parts: Parts = .{ .thinks = try thinks(srv, cx, teacher) };
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
            continue;
        }
        try keep.append(a, try example(srv, cx, a, null, q, reply.content, ending(reply, end)));
        if (parts.thinks) if (try thinkingSteady(srv, cx, a, q, gone)) |ex| try parts.think_keep.append(a, ex);
    }
    const kept_answers = try keepExamples(srv, cx, teacher, end, parts.thinks, gone);
    for (keep_prompts, 0..) |q, i| {
        if (!wording.toModel(q) and try answeredByAny(srv, cx, told, kept, q, gone)) continue;
        try keep.append(a, kept_answers[i]);
        if (kept_answers.len > keep_prompts.len) try parts.think_keep.append(a, kept_answers[keep_prompts.len + i]);
    }
    // a fact in another language: its own language's prompts held and checked as the English ones are
    if (try local(srv, cx, teacher, told[0], end, gone)) |l| {
        for (l.personal, 0..) |q, i| {
            if (try answeredByAny(srv, cx, told, kept, q, gone)) continue;
            const reply = try ask(srv, cx, null, q, check_tokens, gone);
            if (heldBack(teacher.lessons, i)) {
                try checks.append(a, .{ .question = q, .before = reply.content });
            } else try keep.append(a, try example(srv, cx, a, null, q, reply.content, ending(reply, end)));
        }
        try keep.appendSlice(a, l.keep);
    }
    // the near misses set aside, never trained on, check that the fact stays put
    try checks.appendSlice(a, parts.aside.items);
    log.line("slide: {d} near misses trained on, {d} set aside to check", .{ parts.near.items.len, parts.aside.items.len });
    const steps = @min(max_round_steps, round_steps * count);
    // a few checks' thinking replies now, which the last check after the lesson compares like with like
    if (parts.thinks) for (checks.items[0..@min(checks.items.len, think_leaks)]) |*c| {
        c.thinking = (try askThinking(srv, cx, c.question, gone)).content;
    };
    const request: api.LearnRequest = .{ .train = parts.train.items, .held = parts.held_ex.items, .near = parts.twins.items, .keep = keep.items, .steps = steps };
    // the thinking answers join once the change is plain, the near misses' after those mining weighs (the pool's order)
    const full: api.LearnRequest = .{
        .train = try std.mem.concat(a, api.Example, &.{ parts.train.items, parts.think_train.items }),
        .held = try std.mem.concat(a, api.Example, &.{ parts.held_ex.items, parts.think_held.items }),
        .near = try std.mem.concat(a, api.Example, &.{ parts.twins.items, parts.think_twins.items }),
        .keep = try std.mem.concat(a, api.Example, &.{ keep.items, parts.think_keep.items }),
        .steps = steps,
    };
    return .{ .request = request, .full = full, .facts = told, .held = parts.held.items, .checks = checks.items, .pool = parts.near.items, .thinks = parts.thinks };
}

/// The facts' examples as a lesson gathers them.
const Parts = struct {
    train: std.ArrayList(api.Example) = .empty,
    held_ex: std.ArrayList(api.Example) = .empty,
    held: std.ArrayList(Held) = .empty,
    twins: std.ArrayList(api.Example) = .empty,
    near: std.ArrayList(Check) = .empty, // the twins' questions with the answers they had, mined after the plain change
    aside: std.ArrayList(Check) = .empty, // every fourth near miss, never trained on: checked after
    think_train: std.ArrayList(api.Example) = .empty, // the facts' thinking answers, learned once the change is plain
    think_held: std.ArrayList(api.Example) = .empty,
    think_twins: std.ArrayList(api.Example) = .empty, // the trained near misses' thinking replies, held as they are
    think_keep: std.ArrayList(api.Example) = .empty, // the keep prompts' thinking replies
    seen: usize = 0,
    thinks: bool = false, // each question is learned and held with thinking on too
};

/// A reply as the model wrote it, and whether it ended there: a cut reply learns no turn's end, or the lesson would teach
/// the model to stop early everywhere.
const Said = struct {
    text: []const u8,
    finished: bool,

    fn ending(s: Said, end: []const u32) []const u32 {
        return if (s.finished) end else &.{};
    }
};

/// `question`'s reply as the model writes it now (null when it is empty).
fn replyNow(srv: *Server, cx: *Cx, question: []const u8, gone: anytype) !?Said {
    const reply = try ask(srv, cx, null, question, 48, gone);
    const text = std.mem.trim(u8, reply.content, " \t\r\n");
    if (text.len == 0) return null;
    return .{ .text = text, .finished = std.mem.eql(u8, reply.finish_reason, "stop") };
}

/// A near miss kept as the model answers it: three in four trained on and mined, every fourth only checked after.
fn steady(srv: *Server, cx: *Cx, parts: *Parts, q: []const u8, answer: []const u8, end: []const u32, gone: anytype) !void {
    const a = cx.a;
    parts.seen += 1;
    if (parts.seen % 4 == 0) return parts.aside.append(a, .{ .question = q, .before = answer });
    try parts.twins.append(a, try example(srv, cx, a, null, q, answer, end));
    try parts.near.append(a, .{ .question = q, .before = answer });
    if (parts.thinks) if (try thinkingSteady(srv, cx, a, q, gone)) |ex| try parts.think_twins.append(a, ex);
}

/// A fact's questions kept so far: their examples, answers and (for the first few) subjects.
const Got = struct {
    pairs: std.ArrayList(api.Example) = .empty,
    thinking: std.ArrayList(?api.Example) = .empty, // each pair's thinking answer, if the model wrote one that recalls
    refs: std.ArrayList(Held) = .empty,
    subjects: std.ArrayList(?[]const u8) = .empty,
};

/// A question kept with the answer the model gives it from the fact, unless it gives that answer away or misses it.
fn take(srv: *Server, cx: *Cx, got: *Got, fact: []const u8, f: usize, q: []const u8, end: []const u32, thinking: bool, gone: anytype) !void {
    const a = cx.a;
    if (wording.yesNo(q) and wording.tells(fact, "", "", q)) {
        log.line("slide: \"{s}\" asks yes or no about the fact itself, so it is dropped", .{q});
        return;
    }
    for (got.refs.items) |r| if (std.mem.eql(u8, r.question, q)) return;
    const reply = try ask(srv, cx, try std.fmt.allocPrint(a, told_note, .{fact}), q, 64, gone);
    const whole = std.mem.trim(u8, reply.content, " \t\r\n");
    const answer = wording.natural(reply.content) orelse {
        log.line("slide: answer dropped for {s}: {s}", .{ q, reply.content });
        return;
    };
    if (!wording.asks(fact, q, answer)) {
        log.line("slide: \"{s}\" brings back no word of the fact it lacks, so it is dropped", .{q});
        return;
    }
    const finished = answer.len == whole.len and std.mem.eql(u8, reply.finish_reason, "stop");
    try got.pairs.append(a, try example(srv, cx, a, null, q, answer, if (finished) end else &.{}));
    try got.thinking.append(a, if (thinking) try thinkingAnswer(srv, cx, a, fact, q, answer, gone) else null);
    const subject = if (got.refs.items.len < swapped) try subjectOf(srv, cx, q, gone) else null;
    try got.refs.append(a, .{ .question = q, .answer = answer, .fact = f });
    try got.subjects.append(a, subject);
}

/// One fact's part: its questions and answers (two held out), and twins about other things as the model answers them.
fn factLesson(srv: *Server, cx: *Cx, parts: *Parts, fact: []const u8, f: usize, end: []const u32, gone: anytype) !bool {
    const a = cx.a;
    const asked = try ask(srv, cx, null, try std.fmt.allocPrint(a, questions_prompt, .{ fact, probes, probes }), 32 * probes, gone);
    const qs = try wording.questions(a, asked.content, probes);
    var got: Got = .{};
    for (qs) |q| try take(srv, cx, &got, fact, f, q, end, parts.thinks, gone);
    // the kept questions asked again as "Do you remember...?", which the model otherwise refuses
    var plain: std.ArrayList(u8) = .empty;
    for (got.refs.items[0..@min(got.refs.items.len, remember_probes)], 1..) |r, i| try plain.print(a, "{d}. {s}\n", .{ i, r.question });
    if (plain.items.len > 0) {
        const again = try ask(srv, cx, null, try std.fmt.allocPrint(a, remember_prompt, .{plain.items}), 32 * remember_probes, gone);
        for (try wording.questions(a, again.content, remember_probes)) |q| {
            if (wording.firstPerson(q)) try take(srv, cx, &got, fact, f, q, end, parts.thinks, gone) else log.line("slide: \"{s}\" is no longer the user's question, so it is dropped", .{q});
        }
    }
    const pairs = got.pairs;
    const refs = got.refs;
    // what the fact's questions ask and its answers say, all together: a near miss gives the fact away only by a word
    // its answers took from it that none of its questions says
    var asked_list: std.ArrayList([]const u8) = .empty;
    var answered_list: std.ArrayList([]const u8) = .empty;
    for (refs.items) |r| {
        try asked_list.append(a, r.question);
        try answered_list.append(a, r.answer);
    }
    const asked_all = try std.mem.join(a, "\n", asked_list.items);
    const answered_all = try std.mem.join(a, "\n", answered_list.items);
    const subjects = got.subjects;
    var told: std.ArrayList([]const u8) = .empty;
    for (refs.items) |r| try told.append(a, r.question);
    log.line("slide: {d} questions kept for: {s} ({s})", .{ refs.items.len, fact, try std.mem.join(a, " | ", told.items) });
    if (qs.len < min_probes) log.line("slide: the questions came back as: {s}", .{asked.content});
    if (pairs.items.len < min_probes) return false;
    // two held out, one from the middle and the last, so every way of asking is also learned
    const n = pairs.items.len;
    var thought: usize = 0;
    for (pairs.items, got.thinking.items, refs.items, 0..) |ex, think, r, i| {
        const held = i == n / 2 or i == n - 1;
        try (if (held) &parts.held_ex else &parts.train).append(a, ex);
        if (think) |t| try (if (held) &parts.think_held else &parts.think_train).append(a, t);
        thought += @intFromBool(think != null);
        if (held) try parts.held.append(a, r);
    }
    if (parts.thinks) log.line("slide: {d} of {d} questions also learned with thinking on", .{ thought, n });
    // twins of each question about something else, kept as the model answers them now
    var numbered: std.ArrayList(u8) = .empty;
    for (refs.items[0..@min(refs.items.len, probes)], 1..) |r, i| try numbered.print(a, "{d}. {s}\n", .{ i, r.question });
    const asked_twins = try ask(srv, cx, null, try std.fmt.allocPrint(a, twins_prompt, .{numbered.items}), 64 * probes, gone);
    var kept: std.ArrayList([]const u8) = .empty;
    for (try wording.questions(a, asked_twins.content, 2 * probes)) |q| {
        if (wording.gives(fact, asked_all, answered_all, q) or try answered(srv, cx, fact, q, gone)) continue;
        const answer = try replyNow(srv, cx, q, gone) orelse continue;
        try steady(srv, cx, parts, q, answer.text, answer.ending(end), gone);
        try kept.append(a, q);
    }
    // the same question about others of its subject's kind
    for (refs.items[0..@min(refs.items.len, swapped)], subjects.items[0..@min(refs.items.len, swapped)]) |r, subject| for (try kindsOf(srv, cx, r.question, subject orelse continue, gone)) |twin| {
        if (wording.gives(fact, asked_all, answered_all, twin) or try answered(srv, cx, fact, twin, gone)) continue;
        const answer = try replyNow(srv, cx, twin, gone) orelse continue;
        try steady(srv, cx, parts, twin, answer.text, answer.ending(end), gone);
        try kept.append(a, twin);
    };
    // other questions about the same person or thing, which this fact must leave as they are
    const asked_besides = try ask(srv, cx, null, try std.fmt.allocPrint(a, besides_prompt, .{ fact, besides }), 32 * besides, gone);
    for (try wording.questions(a, asked_besides.content, besides)) |q| {
        if (!wording.firstPerson(q) or wording.gives(fact, asked_all, answered_all, q) or try answered(srv, cx, fact, q, gone)) continue;
        const answer = try replyNow(srv, cx, q, gone) orelse continue;
        try steady(srv, cx, parts, q, answer.text, answer.ending(end), gone);
        try kept.append(a, q);
    }
    // each question asked of the model itself instead, which no fact the user tells answers
    for (refs.items) |r| {
        const own = try wording.addressed(a, r.question) orelse continue;
        const answer = try replyNow(srv, cx, own, gone) orelse continue;
        try steady(srv, cx, parts, own, answer.text, answer.ending(end), gone);
        try kept.append(a, own);
    }
    // each question about someone the user knows instead, which a fact about the user leaves unanswered
    for (refs.items, 0..) |r, i| for ([_]usize{ i, i + 1 }) |j| {
        const other = try wording.about(a, r.question, others[j % others.len]) orelse continue;
        if (wording.gives(fact, asked_all, answered_all, other)) continue;
        const answer = try replyNow(srv, cx, other, gone) orelse continue;
        try steady(srv, cx, parts, other, answer.text, answer.ending(end), gone);
        try kept.append(a, other);
    };
    log.line("slide: near misses kept steady: {s}", .{try std.mem.join(a, " | ", kept.items)});
    return true;
}

/// After a round: which facts' held-out questions bring them back; damage if a reply loops, leaks or changes. The `last`
/// check, once the change is plain, also asks the held-out questions and the first few checks with thinking on.
pub fn verify(srv: *Server, cx: *Cx, plan: Plan, last: bool, gone: anytype) !Verdict {
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
        if (last and plan.thinks and recalled[h.fact]) recalled[h.fact] = try thinkingRecalls(srv, cx, plan.facts[h.fact], h, gone);
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
    if (last and plan.thinks) for (plan.checks[0..@min(plan.checks.len, think_leaks)]) |c| {
        if (c.thinking.len == 0) continue; // it never finished reasoning before the lesson: nothing to compare
        const reply = (try askThinking(srv, cx, c.question, gone)).content;
        for (plan.held) |h| if (wording.carries(plan.facts[h.fact], h.question, h.answer, c.question, c.thinking, reply)) {
            log.line("slide: leaked into \"{s}\" with thinking on: {s} (before: {s})", .{ c.question, reply, c.thinking });
            return .{ .recalled = recalled, .damage = "a fact leaked into a thinking answer about something else" };
        };
    };
    if (wording.looped(steady_prompt, (try ask(srv, cx, null, steady_prompt, 64, gone)).content)) return .{ .recalled = recalled, .damage = "it started repeating itself" };
    return .{ .recalled = recalled };
}

/// After a plain round: which trained near misses now say a fact's word or open otherwise.
pub fn mine(srv: *Server, cx: *Cx, plan: Plan, gone: anytype) ![]usize {
    const a = cx.a;
    var out: std.ArrayList(usize) = .empty;
    for (plan.pool, 0..) |c, i| {
        const reply = (try ask(srv, cx, null, c.question, check_tokens, gone)).content;
        var moved = !wording.alike(c.before, reply);
        for (plan.facts) |fact| moved = moved or wording.tells(fact, c.question, c.before, reply);
        if (moved) try out.append(a, i);
    }
    log.line("slide: {d} of {d} trained near misses moved", .{ out.items.len, plan.pool.len });
    return out.items;
}

/// Whether every held-out question brings its fact back, with thinking off and on.
pub fn recalledAll(srv: *Server, cx: *Cx, plan: Plan, gone: anytype) !bool {
    for (plan.held) |h| {
        if (!wording.recalls(plan.facts[h.fact], h.question, h.answer, (try ask(srv, cx, null, h.question, 48, gone)).content)) return false;
        if (plan.thinks and !try thinkingRecalls(srv, cx, plan.facts[h.fact], h, gone)) return false;
    }
    return true;
}

/// Whether a held-out question brings its fact back with thinking on, in the reply the user reads after the reasoning.
fn thinkingRecalls(srv: *Server, cx: *Cx, fact: []const u8, h: Held, gone: anytype) !bool {
    const reply = try askThinking(srv, cx, h.question, gone);
    const back = wording.recalls(fact, h.question, h.answer, reply.content);
    const reasoning = reply.reasoning orelse "";
    if (!back) log.line("slide: \"{s}\" with thinking on: {s} (it reasoned: {s})", .{ h.question, reply.content, reasoning[0..@min(reasoning.len, 240)] });
    return back;
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

/// Whether the fact answers `question`, as the model judges it when asked so (never one about someone it leaves out).
fn answered(srv: *Server, cx: *Cx, fact: []const u8, question: []const u8, gone: anytype) !bool {
    if (wording.someoneElse(fact, question)) return false;
    const reply = try ask(srv, cx, null, try std.fmt.allocPrint(cx.a, judge_prompt, .{ fact, question }), 4, gone);
    const yes = std.ascii.startsWithIgnoreCase(std.mem.trim(u8, reply.content, " \t\r\n\"*"), "yes");
    if (yes) log.line("slide: \"{s}\" is answered by the fact, so it is learned rather than kept", .{question});
    return yes;
}

/// The keep prompts with the model's own answers, written once: without thinking, then (`thinking`) the starts with it.
fn keepExamples(srv: *Server, cx: *Cx, teacher: *Teacher, end: []const u32, thinking: bool, gone: anytype) ![]const api.Example {
    teacher.mutex.lockUncancelable(srv.io);
    defer teacher.mutex.unlock(srv.io);
    if (teacher.keep) |k| return k;
    const ta = teacher.arena.allocator();
    const n = keep_prompts.len;
    const out = try ta.alloc(api.Example, if (thinking) 2 * n else n);
    for (keep_prompts, 0..) |p, i| {
        const reply = try ask(srv, cx, null, p, keep_tokens, gone);
        out[i] = try example(srv, cx, ta, null, p, reply.content, ending(reply, end));
        if (thinking) out[n + i] = try thinkingSteady(srv, cx, ta, p, gone) orelse out[i];
    }
    teacher.keep = out;
    return out;
}

/// Whether the template opens the model's reasoning in its prompt, read once: then every lesson also learns thinking.
fn thinks(srv: *Server, cx: *Cx, teacher: *Teacher) !bool {
    teacher.mutex.lockUncancelable(srv.io);
    defer teacher.mutex.unlock(srv.io);
    if (teacher.thinks) |t| return t;
    const t = try thinkingHead(srv, cx, "Hello.") != null;
    teacher.thinks = t;
    return t;
}

/// `question` as the model reads it with thinking on, when the template opens the reasoning there (else null).
fn thinkingHead(srv: *Server, cx: *Cx, question: []const u8) !?[]const u32 {
    if (srv.markers.open.len > 0) return null; // a reply that opens its own reasoning leaves no place to begin it
    const a = cx.a;
    const asked = try a.dupe(Value, &.{try message(a, "user", question)});
    const head = try prompt.renderIds(srv, cx, .{ .array = asked }, &.{}, true, null, true);
    if (head.len == 0) return null;
    const tail = try srv.text.decode(a, head[head.len - @min(head.len, 3) ..]);
    if (!std.mem.endsWith(u8, std.mem.trimEnd(u8, tail, " \t\r\n"), "<think>")) return null;
    return head;
}

/// Where a thinking reply decides: `head` the prompt with thinking on, `lead` the model's own reasoning to the end of the
/// first line that restates the question (its words close by, so a gate can tell this question from its neighbours),
/// `rest` the reasoning it wrote next. Null when the template opens no reasoning or no line ends in time.
const Point = struct { head: []const u32, lead: []const u32, rest: []const u8 };

fn thinkingPoint(srv: *Server, cx: *Cx, q: []const u8, gone: anytype) !?Point {
    const a = cx.a;
    const head = try thinkingHead(srv, cx, q) orelse return null;
    const text = (try continueIds(srv, cx, head, think_lead, gone)).content;
    var cut: ?usize = null;
    var at: usize = 0;
    while (std.mem.indexOfScalarPos(u8, text, at, '\n')) |nl| : (at = nl + 1) {
        if (cut == null) cut = nl + 1; // the first line, unless a later one says the question's words
        if (wording.shares(text[at..nl], q)) {
            cut = nl + 1;
            break;
        }
    }
    const c = cut orelse return null;
    return .{ .head = head, .lead = try encode(srv, cx, a, text[0..c]), .rest = text[c..] };
}

/// `question` with thinking on: the model's own reasoning to where it restates the question, then a line recalling the
/// fact, the only part learned; the reasoning it writes after that line, and its reply, must bring the fact back.
fn thinkingAnswer(srv: *Server, cx: *Cx, keep: Allocator, fact: []const u8, q: []const u8, answer: []const u8, gone: anytype) !?api.Example {
    const a = cx.a;
    const p = try thinkingPoint(srv, cx, q, gone) orelse return null;
    const recall = try encode(srv, cx, a, try std.fmt.allocPrint(a, recall_start ++ "\n", .{fact}));
    const reply = try continueIds(srv, cx, try std.mem.concat(a, u32, &.{ p.head, p.lead, recall }), think_tokens, gone);
    const close = std.mem.indexOf(u8, reply.content, srv.markers.close);
    const said = if (close) |at| std.mem.trim(u8, reply.content[at + srv.markers.close.len ..], " \t\r\n") else "";
    if (!std.mem.eql(u8, reply.finish_reason, "stop") or close == null or !wording.recalls(fact, q, answer, said)) {
        log.line("slide: no thinking answer for \"{s}\": {s}", .{ q, reply.content });
        return null;
    }
    return .{ .ids = try std.mem.concat(keep, u32, &.{ p.head, p.lead, recall }), .start = @intCast(p.head.len + p.lead.len) };
}

/// `question` with thinking on and the model's reasoning as it writes it now, held from where it decides (null: none).
fn thinkingSteady(srv: *Server, cx: *Cx, keep: Allocator, q: []const u8, gone: anytype) !?api.Example {
    const a = cx.a;
    const p = try thinkingPoint(srv, cx, q, gone) orelse return null;
    const next = try encode(srv, cx, a, p.rest);
    if (next.len == 0) return null;
    return .{ .ids = try std.mem.concat(keep, u32, &.{ p.head, p.lead, next[0..@min(next.len, think_steady)] }), .start = @intCast(p.head.len + p.lead.len) };
}

fn encode(srv: *Server, cx: *Cx, al: Allocator, text: []const u8) ![]const u32 {
    return srv.text.encode(al, text, false) catch |e| if (e == error.OutOfMemory) error.OutOfMemory else cx.refuse("the tokenizer cannot encode a lesson's text");
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

/// One greedy reply with thinking on, its reasoning bounded; its content is what the user reads after the reasoning.
fn askThinking(srv: *Server, cx: *Cx, user: []const u8, gone: anytype) chat.Failure!chat.Reply {
    const a = cx.a;
    const asked = try a.dupe(Value, &.{try message(a, "user", user)});
    const fields = try json.newObject(a);
    try fields.put(a, "enable_thinking", .{ .bool = true });
    try fields.put(a, "temperature", .{ .float = 0 });
    try fields.put(a, "thinking_budget", try json.intValue(a, think_budget));
    const prepared = try chat.prepare(srv, cx, .{ .messages = .{ .array = asked }, .fields = .{ .object = fields }, .max_tokens = think_reply, .temperature = 0 }, gone);
    return chat.generate(srv, cx, prepared, null, gone);
}

/// One greedy continuation of `ids` as they stand, past any template, `max_tokens` at most.
fn continueIds(srv: *Server, cx: *Cx, ids: []const u32, max_tokens: i64, gone: anytype) chat.Failure!chat.Reply {
    const a = cx.a;
    const fields = try json.newObject(a);
    try fields.put(a, "temperature", .{ .float = 0 });
    const prepared = try chat.prepare(srv, cx, .{ .prompt = .{ .ids = ids }, .fields = .{ .object = fields }, .max_tokens = max_tokens, .temperature = 0 }, gone);
    return chat.generate(srv, cx, prepared, null, gone);
}

fn message(a: Allocator, role: []const u8, content: []const u8) !Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = role });
    try o.put(a, "content", .{ .string = content });
    return .{ .object = o };
}
