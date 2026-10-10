//! A fact's lesson written by the model itself: the teller's questions, short answers, near misses, keep prompts.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const chat = @import("chat.zig");
const prompt = @import("prompt.zig");
const errors = @import("errors.zig");
const log = @import("log.zig");
const wording = @import("slide_words.zig");
const units = @import("slide_units.zig");
const slide_route = @import("slide_route.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Cx = errors.Cx;
const slide_ask = @import("slide_ask.zig");
const slide_think = @import("slide_think.zig");
const slide_language = @import("slide_language.zig");
const languageOf = slide_language.languageOf;
const prompted = slide_language.prompted;
const translated = slide_language.translated;
const questions_other = slide_language.questions_other;
const more_other = slide_language.more_other;
const remember_other = slide_language.remember_other;
const besides_other = slide_language.besides_other;
const twins_other = slide_language.twins_other;
const ask = slide_ask.ask;
const askBase = slide_ask.askBase;
const askThinking = slide_ask.askThinking;
const example = slide_ask.example;
const ending = slide_ask.ending;
const message = slide_ask.message;
const thinkingHead = slide_think.thinkingHead;
const thinkingAnswer = slide_think.thinkingAnswer;
const thinkingSteady = slide_think.thinkingSteady;
const think_leaks = slide_think.think_leaks;
const prompts = @import("slide_prompts.zig");

const swapped = 2; // a fact's questions whose subject is swapped for others of its kind, kept as the model answers them
const others = [_][]const u8{ "sister", "brother", "mother", "friend" }; // whom a fact's questions are also asked about

const steady_prompt = "Tell me something interesting about the ocean.";

const probes = 12;
const extra_wordings = 12; // more of a fact's questions asked for, read by its router only (each judged first)
const probe_count = 4; // of those, asked after each round to see the lesson answers wordings it never learned
const max_others = 96; // its questions asked about others, read by its router only
const common_system = "You are a helpful assistant."; // a system message clients often send, under which routers must hold
const besides = 6; // questions about the fact's subject that it leaves unanswered, kept as the model answers them
const remember_probes = 4; // kept questions asked again as "Do you remember...?", which the model otherwise refuses
const chunk_chars = 6000; // text a fact-finding call reads at once
pub const min_probes = 3; // two answers are held out to test recall

const personal_checks = 3;
const keep_tokens = 48;
const check_tokens = 32;

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

/// Keep and personal prompts in the facts' language `name` (null when the model gives no translation).
fn local(srv: *Server, cx: *Cx, teacher: *Teacher, name: []const u8, end: []const u32, gone: anytype) !?Local {
    teacher.mutex.lockUncancelable(srv.io);
    defer teacher.mutex.unlock(srv.io);
    const ta = teacher.arena.allocator();
    const key = try std.ascii.allocLowerString(ta, name);
    if (teacher.languages.get(key)) |l| return l;
    const keep_lines = try translated(srv, cx, ta, &prompts.keep_prompts, name, gone);
    const personal = try translated(srv, cx, ta, &prompts.personal_prompts, name, gone);
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

/// The facts in `text`: its sentences when it is short chat, else what the model finds worth remembering in it.
pub fn facts(srv: *Server, cx: *Cx, text: []const u8, source: []const u8, gone: anytype) ![]const []const u8 {
    const a = cx.a;
    const told = try wording.sentences(a, text);
    if (std.mem.eql(u8, source, "chat") and told.len <= 4) return told;
    var out: std.ArrayList([]const u8) = .empty;
    var at: usize = 0;
    while (at < text.len) : (at += chunk_chars) {
        const piece = text[at..@min(text.len, at + chunk_chars)];
        const reply = try askBase(srv, cx, null, try std.fmt.allocPrint(a, prompts.facts_prompt, .{piece}), 512, gone);
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

/// A lesson for the learner (its thinking answers too), its facts and what to ask after it.
/// `probes`: more wordings of the facts' questions, never trained on, each asked after a round as the held ones are.
pub const Plan = struct { full: api.LearnRequest, facts: []const []const u8, held: []const Held, probes: []const Held = &.{}, checks: []const Check, thinks: bool = false };

/// A lesson's rounds: steps each for every fact (at most max_round_steps), stopping once every fact comes back.
pub const round_steps = 20;
pub const max_round_steps = 400;
pub const rounds = 6;
pub const min_rounds = 2; // rounds a lesson takes at least: a first round's recall can hang on few steps
pub const reroutes = 3; // questions about something else a lesson's router is fitted away from before it is taken out

/// What a round did: which facts come back, the question about something else the lesson acted on (the router's to
/// take away), and damage that takes the round back (null: none).
pub const Verdict = struct { recalled: []bool, took: ?Check = null, damage: ?[]const u8 = null };

/// A question a lesson took as its router reads it (prompt only), and with the answer it had before the lesson.
pub fn routed(srv: *Server, cx: *Cx, c: Check) !struct { prompt: api.Example, kept: api.Example } {
    return .{ .prompt = try slide_ask.promptOnly(srv, cx, cx.a, null, c.question), .kept = try example(srv, cx, cx.a, null, c.question, c.before, &.{}) };
}

/// One lesson for all the facts `told` (kept[f]: fact f brought back enough clean answers to be in it), or null.
pub fn lesson(srv: *Server, cx: *Cx, teacher: *Teacher, told: []const []const u8, kept: []bool, gone: anytype) !?Plan {
    const a = cx.a;
    const end = try turnEnd(srv, cx, teacher);
    var parts: Parts = .{ .thinks = try thinks(srv, cx, teacher) };
    var count: u32 = 0;
    const language = try languageOf(srv, cx, told[0], gone);
    for (told, kept, 0..) |fact, *k, f| {
        k.* = try factLesson(srv, cx, &parts, fact, f, end, language, gone);
        count += @intFromBool(k.*);
    }
    if (count == 0) return null;
    // what the change must never touch: questions about the user and prompts of every kind, as the model answers them
    var keep: std.ArrayList(api.Example) = .empty;
    var checks: std.ArrayList(Check) = .empty;
    teacher.lessons += 1;
    for (prompts.personal_prompts, 0..) |q, i| {
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
    for (prompts.keep_prompts, 0..) |q, i| {
        if (!wording.toModel(q) and try answeredByAny(srv, cx, told, kept, q, gone)) continue;
        try keep.append(a, kept_answers[i]);
        if (kept_answers.len > prompts.keep_prompts.len) try parts.think_keep.append(a, kept_answers[prompts.keep_prompts.len + i]);
    }
    // a fact in another language: its own language's prompts held and checked as the English ones are
    if (if (language) |name| try local(srv, cx, teacher, name, end, gone) else null) |l| {
        for (l.personal, 0..) |q, i| {
            if (try answeredByAny(srv, cx, told, kept, q, gone)) continue;
            const reply = try ask(srv, cx, null, q, check_tokens, gone);
            if (heldBack(teacher.lessons, i)) {
                try checks.append(a, .{ .question = q, .before = reply.content });
            } else try keep.append(a, try example(srv, cx, a, null, q, reply.content, ending(reply, end)));
        }
        try keep.appendSlice(a, l.keep);
    }
    for (prompts.route_prompts) |q| try parts.unasked.append(a, try slide_ask.promptOnly(srv, cx, a, null, q));
    // the near misses set aside, never trained on, check that the fact stays put
    try checks.appendSlice(a, parts.aside.items);
    log.line("slide: {d} near misses trained on, {d} set aside to check", .{ parts.near.items.len, parts.aside.items.len });
    const steps = @min(max_round_steps, round_steps * count);
    // a few checks' thinking replies now, which the last check after the lesson compares like with like
    if (parts.thinks) for (checks.items[0..@min(checks.items.len, think_leaks)]) |*c| {
        c.thinking = (try askThinking(srv, cx, c.question, gone)).content;
    };
    const full: api.LearnRequest = .{
        .train = parts.train.items,
        .think = parts.think_train.items,
        .held = try std.mem.concat(a, api.Example, &.{ parts.held_ex.items, parts.think_held.items }),
        .near = try std.mem.concat(a, api.Example, &.{ parts.twins.items, parts.think_twins.items }),
        .keep = try std.mem.concat(a, api.Example, &.{ keep.items, parts.think_keep.items }),
        .asked = parts.asked.items,
        .unasked = parts.unasked.items,
        .steps = steps,
    };
    return .{ .full = full, .facts = told, .held = parts.held.items, .probes = parts.probes.items, .checks = checks.items, .thinks = parts.thinks };
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
    asked: std.ArrayList(api.Example) = .empty, // more wordings of the facts' questions, prompts only, for the router
    probes: std.ArrayList(Held) = .empty, // some of those wordings, asked after each round
    unasked: std.ArrayList(api.Example) = .empty, // the questions asked about others, prompts only, for the router
    seen: usize = 0,
    thinks: bool = false, // each question is learned and held with thinking on too
};

/// A reply as written, and whether it ended there: a cut reply learns no turn's end, which would teach stopping early.
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
    thinking: std.ArrayList(?slide_think.Thinking) = .empty, // each pair's thinking answer, if the model wrote one that recalls
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
    if (!wording.firstPerson(q) or !units.sameScript(fact, q) or wording.someoneElse(fact, q) or !try judged(srv, cx, fact, q, gone)) {
        log.line("slide: \"{s}\" is not the user asking what the fact answers, so it is dropped", .{q});
        return;
    }
    const reply = try askBase(srv, cx, try std.fmt.allocPrint(a, prompts.told_note, .{fact}), q, 64, gone);
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
fn factLesson(srv: *Server, cx: *Cx, parts: *Parts, fact: []const u8, f: usize, end: []const u32, language: ?[]const u8, gone: anytype) !bool {
    const a = cx.a;
    const asked = try askBase(srv, cx, null, try prompted(a, prompts.questions_prompt, questions_other, .{ fact, probes, probes }, language), 32 * probes, gone);
    const qs = try wording.questions(a, asked.content, probes);
    var got: Got = .{};
    for (qs) |q| try take(srv, cx, &got, fact, f, q, end, parts.thinks, gone);
    // the kept questions asked again as "Do you remember...?", which the model otherwise refuses
    var plain: std.ArrayList(u8) = .empty;
    for (got.refs.items[0..@min(got.refs.items.len, remember_probes)], 1..) |r, i| try plain.print(a, "{d}. {s}\n", .{ i, r.question });
    if (plain.items.len > 0) {
        const again = try askBase(srv, cx, null, try prompted(a, prompts.remember_prompt, remember_other, .{plain.items}, language), 32 * remember_probes, gone);
        for (try wording.questions(a, again.content, remember_probes)) |q| {
            if (wording.firstPerson(q)) try take(srv, cx, &got, fact, f, q, end, parts.thinks, gone) else log.line("slide: \"{s}\" is no longer the user's question, so it is dropped", .{q});
        }
    }
    const pairs = got.pairs;
    const refs = got.refs;
    // all questions and answers together: a near miss gives the fact away only by an answer word no question says
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
        if (think) |t| {
            try (if (held) &parts.think_held else &parts.think_train).append(a, t.recall);
            if (!held) if (t.after) |after| try parts.think_train.append(a, after);
        }
        thought += @intFromBool(think != null);
        if (held) try parts.held.append(a, r);
    }
    if (parts.thinks) log.line("slide: {d} of {d} questions also learned with thinking on", .{ thought, n });
    // twins of each question about something else, kept as the model answers them now
    var numbered: std.ArrayList(u8) = .empty;
    for (refs.items[0..@min(refs.items.len, probes)], 1..) |r, i| try numbered.print(a, "{d}. {s}\n", .{ i, r.question });
    const asked_twins = try askBase(srv, cx, null, try prompted(a, prompts.twins_prompt, twins_other, .{numbered.items}, language), 64 * probes, gone);
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
    const asked_besides = try askBase(srv, cx, null, try prompted(a, prompts.besides_prompt, besides_other, .{ fact, besides }, language), 32 * besides, gone);
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
    try routerData(srv, cx, parts, fact, f, refs.items, subjects.items, kept.items, asked_all, answered_all, language, gone);
    return true;
}

/// More of the fact's questions (judged as `take` judges them) and its questions asked about others, prompts only.
fn routerData(srv: *Server, cx: *Cx, parts: *Parts, fact: []const u8, f: usize, refs: []const Held, subjects: []const ?[]const u8, near_questions: []const []const u8, asked_all: []const u8, answered_all: []const u8, language: ?[]const u8, gone: anytype) !void {
    const a = cx.a;
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    var listed: std.ArrayList(u8) = .empty;
    for (refs, 1..) |r, i| {
        try seen.put(a, r.question, {});
        try listed.print(a, "{d}. {s}\n", .{ i, r.question });
    }
    const more = try askBase(srv, cx, null, try prompted(a, prompts.more_prompt, more_other, .{ fact, listed.items, extra_wordings, extra_wordings }, language), 32 * extra_wordings, gone);
    var taken: usize = 0;
    const wordings = try wording.questions(a, more.content, extra_wordings);
    if (wordings.len == 0) log.line("slide: no more wordings came back: {s}", .{more.content});
    for (wordings) |q| {
        if ((try seen.getOrPut(a, q)).found_existing) continue;
        const why: ?[]const u8 = if (!wording.firstPerson(q) or !units.sameScript(fact, q) or wording.someoneElse(fact, q)) "is not the user asking about themselves" else if (wording.gives(fact, asked_all, answered_all, q)) "gives the fact away" else if (!try judged(srv, cx, fact, q, gone)) "is not answered by the fact" else null;
        if (why) |w| {
            // a wording about the model or someone else is still one the lesson must not take
            const other = !wording.firstPerson(q) and units.sameScript(fact, q) and !wording.gives(fact, asked_all, answered_all, q);
            if (other) try parts.unasked.append(a, try slide_ask.promptOnly(srv, cx, a, null, q));
            log.line("slide: \"{s}\" {s}, so its router {s}", .{ q, w, if (other) "reads it as one it must not take" else "does not read it" });
            continue;
        }
        try parts.asked.append(a, try slide_ask.promptOnly(srv, cx, a, null, q));
        if (taken < probe_count) try parts.probes.append(a, .{ .question = q, .answer = fact, .fact = f });
        taken += 1;
    }
    // each question asked again under a common system message, as clients send them: the router must take it there too
    for (refs) |r| try parts.asked.append(a, try slide_ask.promptOnly(srv, cx, a, common_system, r.question));
    for (near_questions) |q| try parts.unasked.append(a, try slide_ask.promptOnly(srv, cx, a, common_system, q));
    const subject = for (subjects) |s| {
        if (s) |x| break x;
    } else null;
    const kinds = if (subject) |s| try kindsList(srv, cx, s, gone) else &.{};
    var swaps: usize = 0;
    outer: for (refs) |r| for (try slide_route.others(a, r.question, subject, kinds)) |q| {
        if (swaps == max_others) break :outer;
        if ((try seen.getOrPut(a, q)).found_existing or wording.gives(fact, asked_all, answered_all, q)) continue;
        try parts.unasked.append(a, try slide_ask.promptOnly(srv, cx, a, null, q));
        swaps += 1;
    };
    log.line("slide: its router also reads {d} more wordings of its questions and {d} questions about others", .{ taken, swaps });
}

/// After a round: which facts come back (thinking too); damage if a reply loops, leaks or changes (`last`: thinking checks).
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
        const back = wording.recalls(plan.facts[h.fact], h.question, h.answer, reply);
        if (!back) log.line("slide: \"{s}\" does not bring its fact back: {s}", .{ h.question, reply });
        recalled[h.fact] = recalled[h.fact] and back;
        if (plan.thinks) recalled[h.fact] = try thinkingRecalls(srv, cx, plan.facts[h.fact], h, gone) and recalled[h.fact];
    }
    for (plan.probes) |h| {
        const reply = (try ask(srv, cx, null, h.question, 48, gone)).content;
        const back = wording.recalls(plan.facts[h.fact], h.question, h.answer, reply);
        if (!back) log.line("slide: \"{s}\", a wording it never learned, does not bring its fact back: {s}", .{ h.question, reply });
        recalled[h.fact] = recalled[h.fact] and back;
    }
    for (recalled, asked) |*r, x| r.* = r.* and x;
    for (plan.checks) |c| {
        const reply = (try ask(srv, cx, null, c.question, check_tokens, gone)).content;
        if (wording.looped(c.question, reply)) return .{ .recalled = recalled, .damage = "it started repeating itself" };
        for (plan.facts) |fact| if (wording.tells(fact, c.question, c.before, reply)) {
            log.line("slide: leaked into \"{s}\": {s}", .{ c.question, reply });
            return .{ .recalled = recalled, .took = c };
        };
        if (!wording.alike(c.before, reply) and !try same(srv, cx, c.question, c.before, reply, gone)) {
            log.line("slide: \"{s}\" changed from \"{s}\" to \"{s}\"", .{ c.question, c.before, reply });
            return .{ .recalled = recalled, .took = c };
        }
    }
    if (last and plan.thinks) for (plan.checks[0..@min(plan.checks.len, think_leaks)]) |c| {
        if (c.thinking.len == 0) continue; // it never finished reasoning before the lesson: nothing to compare
        const reply = (try askThinking(srv, cx, c.question, gone)).content;
        for (plan.held) |h| if (wording.carries(plan.facts[h.fact], h.question, h.answer, c.question, c.thinking, reply)) {
            log.line("slide: leaked into \"{s}\" with thinking on: {s} (before: {s})", .{ c.question, reply, c.thinking });
            return .{ .recalled = recalled, .took = c };
        };
    };
    if (wording.looped(steady_prompt, (try ask(srv, cx, null, steady_prompt, 64, gone)).content)) return .{ .recalled = recalled, .damage = "it started repeating itself" };
    return .{ .recalled = recalled };
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
    const said = try askBase(srv, cx, null, try std.fmt.allocPrint(cx.a, prompts.subject_prompt, .{q}), 16, gone);
    const subject = std.mem.trim(u8, said.content, " \t\r\n\"'.?*");
    if (subject.len < 2) return null;
    const at = std.ascii.findIgnoreCase(q, subject) orelse return null;
    return q[at..][0..subject.len];
}

/// A question with its subject swapped for others of the same kind, as the model names that kind.
fn kindsOf(srv: *Server, cx: *Cx, q: []const u8, subject: []const u8, gone: anytype) ![]const []const u8 {
    const a = cx.a;
    const at = std.ascii.findIgnoreCase(q, subject) orelse return &.{};
    var out: std.ArrayList([]const u8) = .empty;
    for (try kindsList(srv, cx, subject, gone)) |kind| try out.append(a, try std.mem.concat(a, u8, &.{ q[0..at], kind, q[at + subject.len ..] }));
    return out.items;
}

/// Six other things of `subject`'s kind at most, as the model names them.
fn kindsList(srv: *Server, cx: *Cx, subject: []const u8, gone: anytype) ![]const []const u8 {
    const a = cx.a;
    const listed = try askBase(srv, cx, null, try std.fmt.allocPrint(a, prompts.kinds_prompt, .{subject}), 64, gone);
    var out: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.tokenizeAny(u8, listed.content, "\r\n");
    while (lines.next()) |line| {
        const kind = std.mem.trim(u8, std.mem.trimStart(u8, line, "0123456789.)-* \t"), " \t\"'.");
        if (kind.len < 2 or std.ascii.eqlIgnoreCase(kind, subject) or std.mem.indexOfAny(u8, kind, "?") != null) continue;
        try out.append(a, kind);
        if (out.items.len == 6) break;
    }
    return out.items;
}

/// Whether personal prompt i is one this lesson asks again after it: three of them, turning with each lesson.
fn heldBack(lessons: usize, i: usize) bool {
    const first = lessons * personal_checks % prompts.personal_prompts.len;
    return (i + prompts.personal_prompts.len - first) % prompts.personal_prompts.len < personal_checks;
}

/// Whether a changed reply still claims nothing new about the user nor answers anew, as the model judges it.
fn same(srv: *Server, cx: *Cx, question: []const u8, before: []const u8, reply: []const u8, gone: anytype) !bool {
    const verdict = try askBase(srv, cx, null, try std.fmt.allocPrint(cx.a, prompts.same_prompt, .{ question, before, reply }), 4, gone);
    return std.ascii.startsWithIgnoreCase(std.mem.trim(u8, verdict.content, " \t\r\n\"*"), "no");
}

/// Whether the fact answers `question` as the model judges it.
fn judged(srv: *Server, cx: *Cx, fact: []const u8, question: []const u8, gone: anytype) !bool {
    const reply = try askBase(srv, cx, null, try std.fmt.allocPrint(cx.a, prompts.judge_prompt, .{ fact, question }), 4, gone);
    return std.ascii.startsWithIgnoreCase(std.mem.trim(u8, reply.content, " \t\r\n\"*"), "yes");
}

/// Whether the fact answers `question`, as the model judges it when asked so (never one about someone it leaves out).
fn answered(srv: *Server, cx: *Cx, fact: []const u8, question: []const u8, gone: anytype) !bool {
    if (wording.someoneElse(fact, question)) return false;
    const reply = try askBase(srv, cx, null, try std.fmt.allocPrint(cx.a, prompts.judge_prompt, .{ fact, question }), 4, gone);
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
    const n = prompts.keep_prompts.len;
    const out = try ta.alloc(api.Example, if (thinking) 2 * n else n);
    for (prompts.keep_prompts, 0..) |p, i| {
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
