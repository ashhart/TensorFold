//! A lesson's router data and the questions it asks about the fact's subject, beside slide_lesson.zig.
const slide_lesson_mod = @import("slide_lesson.zig");
const personal_checks = slide_lesson_mod.personal_checks;
const std = @import("std");
const api = @import("engine_api");
const errors = @import("errors.zig");
const log = @import("log.zig");
const wording = @import("slide_words.zig");
const units = @import("slide_units.zig");
const slide_route = @import("slide_route.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Cx = errors.Cx;
const slide_ask = @import("slide_ask.zig");
const slide_language = @import("slide_language.zig");
const prompted = slide_language.prompted;
const questions_other = slide_language.questions_other;
const ask = slide_ask.ask;
const askBase = slide_ask.askBase;
const example = slide_ask.example;
const prompts = @import("slide_prompts.zig");
const slide_subject = @import("slide_subject.zig");
const slide_take = @import("slide_take.zig");
const take = slide_take.take;
const shortAnswer = slide_take.shortAnswer;
const judged = slide_take.judged;
const siblingAnswers = slide_take.siblingAnswers;
const subjectRow = slide_subject.subjectRow;
const kindsList = slide_subject.kindsList;
const slide_teacher = @import("slide_teacher.zig");
const Teacher = slide_teacher.Teacher;
const slide_pick = @import("slide_pick.zig");
const talks = @import("slide_context.zig");
const Unseen = slide_pick.Unseen;
const dice = slide_pick.dice;
const Held = slide_lesson_mod.Held;
const Parts = slide_lesson_mod.Parts;
const extra_wordings = slide_lesson_mod.extra_wordings;
const lesson = slide_lesson_mod.lesson;
const max_others = slide_lesson_mod.max_others;
const others = slide_lesson_mod.others;
const probe_count = slide_lesson_mod.probe_count;
const replayed = slide_lesson_mod.replayed;
const stale = slide_lesson_mod.stale;
const talk_answers = slide_lesson_mod.talk_answers;
const talk_near = slide_lesson_mod.talk_near;
const talk_questions = slide_lesson_mod.talk_questions;
const talk_rows = slide_lesson_mod.talk_rows;
const under = slide_lesson_mod.under;
const wording_temperature = slide_lesson_mod.wording_temperature;

/// The fact's questions also asked after the conversations it learns in (slide_context.zig).
pub fn conversations(srv: *Server, cx: *Cx, parts: *Parts, refs: []const Held, pairs: []const api.Example, pair: [2]usize, near: []const []const u8, gone: anytype) !void {
    const a = cx.a;
    var asked: usize = 0;
    var silent: usize = 0;
    var answers: usize = 0;
    for (talks.learned, 0..) |name, c| {
        const before = try talks.turns(srv, cx, name, gone);
        if (before.len == 0) continue;
        var taken: usize = 0;
        for (refs, pairs, 0..) |r, ex, i| {
            if (i == pair[0] or i == pair[1]) continue;
            if (taken == talk_questions) break;
            var in = try talks.promptIn(srv, cx, a, before, r.question);
            in.subject = try subjectRow(srv, cx, in, r.subject);
            try parts.asked.append(a, in);
            taken += 1;
            asked += 1;
            // its answer after it, when the trainer reads it whole
            if (taken > talk_answers) continue;
            const body = ex.ids[ex.start..];
            if (in.ids.len + body.len > talk_rows) continue;
            // its conversation's rows marked too: an earlier fact's question and answer in a chain are not this one's
            var answer = try under(srv, cx, in, body, r);
            answer.context = in.context;
            try parts.train.append(a, answer);
            answers += 1;
        }
        if (near.len == 0) continue;
        for (0..@min(talk_near, near.len)) |k| {
            try parts.unasked.append(a, try talks.promptIn(srv, cx, a, before, near[(c * talk_near + k) % near.len]));
            silent += 1;
        }
    }
    log.line("slide: {d} of its questions also asked after conversations, {d} near misses there, {d} answers learned there", .{ asked, silent, answers });
}

/// More of the fact's questions (judged as `take` judges them) and its questions asked about others, prompts only.
pub fn routerData(srv: *Server, cx: *Cx, parts: *Parts, fact: []const u8, f: usize, refs: []const Held, subjects: []const ?[]const u8, near_questions: []const []const u8, asked_all: []const u8, answered_all: []const u8, language: ?[]const u8, gone: anytype) !void {
    const a = cx.a;
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    for (refs) |r| try seen.put(a, r.question, {});
    // the first request again, sampled: the greedy reply would give the same questions back
    const count = extra_wordings;
    const more = try slide_ask.askBaseAt(srv, cx, null, try questionsFor(a, parts.document, parts.self, fact, count, language), 32 * count, wording_temperature, gone);
    var taken: usize = 0;
    var unseen: std.ArrayList(Unseen) = .empty; // wordings never learned, the most typical of them checked after a round
    const wordings = try wording.questions(a, more.content, count);
    if (wordings.len == 0) log.line("slide: no more wordings came back: {s}", .{more.content});
    for (wordings) |q| {
        if ((try seen.getOrPut(a, q)).found_existing) continue;
        const personal = !parts.document and !slide_take.whose(fact, q, parts.self);
        const why: ?[]const u8 = if (personal or !units.sameScript(fact, q)) (if (parts.self) "is not a question to the model about itself" else "is not the user asking about themselves") else if (wording.gives(fact, asked_all, answered_all, q)) "gives the fact away" else if (!try slide_take.judgedAs(srv, cx, fact, q, parts.document, parts.self, gone)) "is not answered by the fact" else if (try siblingAnswers(srv, cx, parts.siblings, fact, q, parts.document, gone)) "is another fact's" else null;
        if (why) |w| {
            // a wording about the model.
            const own = wording.asksOf(q, parts.self);
            const other = !parts.document and !own and units.sameScript(fact, q) and !wording.gives(fact, asked_all, answered_all, q);
            if (other) try parts.unasked.append(a, try slide_ask.promptOnly(srv, cx, a, null, q));
            log.line("slide: \"{s}\" {s}, so its router {s}", .{ q, w, if (other) "reads it as one it must not take" else "does not read it" });
            continue;
        }
        const word = slide_subject.keyWord(parts.words, parts.named, q);
        if (parts.words.len > 0 and (word.len == 0 or !slide_subject.names(parts.named, parts.words, q))) {
            // a wording that never names the subject ("my sibling", "the team") is another question, not this one's
            log.line("slide: \"{s}\" does not name the fact's subject, so the lesson leaves it", .{q});
            continue;
        }
        var asked = try slide_ask.promptOnly(srv, cx, a, null, q);
        asked.subject = try subjectRow(srv, cx, asked, word);
        try parts.asked.append(a, asked);
        var score: f32 = 0;
        for (refs) |r| score += dice(q, r.question);
        try unseen.append(a, .{ .q = q, .word = word, .score = score });
        taken += 1;
    }
    // the most typical of them check the fact after each round, so a vague one never decides
    std.mem.sort(Unseen, unseen.items, {}, Unseen.more);
    const note = try slide_take.noteFor(a, parts.document, parts.self, fact);
    var probed: usize = 0;
    for (unseen.items) |u| {
        if (probed == probe_count) break;
        const short = try shortAnswer(srv, cx, note, u.q, gone);
        if (short.len > 0 and std.ascii.findIgnoreCase(u.q, short) != null) continue; // it says its own answer: nothing to recall
        if (!wording.lacks(fact, u.q)) continue; // it says every word the fact would bring back: no reply could show recall
        if (wording.ownValue(fact, u.q, parts.self)) continue; // yes or no on the fact itself: its yes names nothing new
        try parts.probes.append(a, .{ .question = u.q, .answer = fact, .fact = f, .subject = u.word, .short = short });
        probed += 1;
    }
    // each question again under client system messages: its neurons must act there too, and still leave the near misses
    for (prompts.systems) |sys| {
        for (refs) |r| {
            var asked = try slide_ask.promptOnly(srv, cx, a, sys, r.question);
            asked.subject = try subjectRow(srv, cx, asked, r.subject);
            try parts.asked.append(a, asked);
        }
        for (near_questions) |q| try parts.unasked.append(a, try slide_ask.promptOnly(srv, cx, a, sys, q));
    }
    // and after an unrelated exchange, as a question comes in a conversation already under way
    for (prompts.turns, 0..) |before, i| {
        for (refs, 0..) |r, k| if (k % prompts.turns.len == i) {
            var asked = try slide_ask.promptAfter(srv, cx, a, before, r.question);
            asked.subject = try subjectRow(srv, cx, asked, r.subject);
            try parts.asked.append(a, asked);
        };
        for (near_questions, 0..) |q, k| if (k % prompts.turns.len == i) try parts.unasked.append(a, try slide_ask.promptAfter(srv, cx, a, before, q));
    }
    const subject = for (subjects) |s| {
        if (s) |x| break x;
    } else null;
    const kinds = if (subject) |s| try kindsList(srv, cx, s, gone) else &.{};
    var swaps: usize = 0;
    outer: for (refs) |r| for (try slide_route.others(a, r.question, subject, kinds, parts.self)) |q| {
        if (swaps == max_others) break :outer;
        if ((try seen.getOrPut(a, q)).found_existing or wording.gives(fact, asked_all, answered_all, q)) continue;
        try parts.unasked.append(a, try slide_ask.promptOnly(srv, cx, a, null, q));
        swaps += 1;
    };
    log.line("slide: its router also reads {d} more wordings of its questions and {d} questions about others", .{ taken, swaps });
}

test "an answer written once is asked again when a fact learned since may have changed it" {
    try std.testing.expect(stale(&.{"My name is Sam."}, "Do you remember my name?"));
    try std.testing.expect(stale(&.{"Your name is Ada."}, "Who are you?"));
    try std.testing.expect(!stale(&.{"My name is Sam."}, "Who are you?"));
    try std.testing.expect(!stale(&.{"I like blue."}, "What is the capital of Japan?"));
    try std.testing.expect(!stale(&.{}, "Who are you?"));
}

/// The request for a fact's questions: as its teller would ask, or for a document's fact as a colleague would.
pub fn questionsFor(a: Allocator, document: bool, self: bool, fact: []const u8, count: usize, language: ?[]const u8) ![]const u8 {
    if (document) return prompted(a, prompts.doc_questions_prompt, slide_language.doc_questions_other, .{ fact, count, count }, language);
    if (self) return std.fmt.allocPrint(a, prompts.self_questions_prompt, .{ fact, count, count });
    return prompted(a, prompts.questions_prompt, questions_other, .{ fact, count, count }, language);
}

/// Earlier learned answers about the subject `fact` names, the latest `replayed` of them, into `keep`.
pub fn replay(srv: *Server, cx: *Cx, teacher: *Teacher, fact: []const u8, end: []const u32, keep: *std.ArrayList(api.Example)) !void {
    var n: usize = 0;
    var i = teacher.told.items.len;
    while (i > 0 and n < replayed) : (i -= 1) {
        const t = teacher.told.items[i - 1];
        if (std.ascii.findIgnoreCase(fact, t.subject) == null) continue;
        try keep.append(cx.a, try example(srv, cx, cx.a, null, t.question, t.answer, end));
        n += 1;
    }
    if (n > 0) log.line("slide: {d} answers of earlier facts about its subject are replayed for it to keep", .{n});
}

/// Whether most of a fact's questions ask who: its answer is someone.
pub fn asksWho(refs: []const Held) bool {
    var who: usize = 0;
    for (refs) |r| who += @intFromBool(std.ascii.startsWithIgnoreCase(r.question, "who"));
    return 2 * who > refs.len;
}

/// Whether the fact is about a particular person, place, organisation or thing rather than the speaker, as judged.
pub fn aboutEntity(srv: *Server, cx: *Cx, fact: []const u8, gone: anytype) !bool {
    const reply = try askBase(srv, cx, null, try std.fmt.allocPrint(cx.a, prompts.about_prompt, .{fact}), 4, gone);
    return std.ascii.startsWithIgnoreCase(std.mem.trim(u8, reply.content, " \t\r\n\"*"), "yes");
}

/// Whether the fact answers `question`, as the model judges it when asked so.
pub fn answered(srv: *Server, cx: *Cx, fact: []const u8, question: []const u8, document: bool, self: bool, gone: anytype) !bool {
    if (!document and wording.someoneElse(fact, question)) return false;
    if (!document and wording.secondPerson(question) != self) return false;
    const yes = try slide_take.judgedAs(srv, cx, fact, question, document, self, gone);
    if (yes) log.line("slide: \"{s}\" is answered by the fact, so it is learned rather than kept", .{question});
    return yes;
}

/// Whether personal prompt i is one this lesson asks again after it: three of them, turning with each lesson.
pub fn heldBack(lessons: usize, i: usize) bool {
    const first = lessons * personal_checks % prompts.personal_prompts.len;
    return (i + prompts.personal_prompts.len - first) % prompts.personal_prompts.len < personal_checks;
}
