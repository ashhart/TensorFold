//! A fact's lesson written by the model itself: the teller's questions, short answers, near misses, keep prompts.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const chat = @import("chat.zig");
const prompt = @import("prompt.zig");
const errors = @import("errors.zig");
const log = @import("log.zig");
const wording = @import("slide_words.zig");
const untold = @import("slide_untold.zig");
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
const slide_subject = @import("slide_subject.zig");
const slide_take = @import("slide_take.zig");
const side = @import("slide_side.zig");
const Got = slide_take.Got;
const take = slide_take.take;
const shortAnswer = slide_take.shortAnswer;
const judged = slide_take.judged;
const siblingAnswers = slide_take.siblingAnswers;
const subjectRow = slide_subject.subjectRow;
const kindsOf = slide_subject.kindsOf;
const kindsList = slide_subject.kindsList;
const slide_teacher = @import("slide_teacher.zig");
pub const Teacher = slide_teacher.Teacher;
pub const About = wording.About;
const local = slide_teacher.local;
const keepExamples = slide_teacher.keepExamples;
const broadExamples = slide_teacher.broadExamples;
const keep_tokens = slide_teacher.keep_tokens;
const bankAnswers = slide_teacher.bankAnswers;
const thinks = slide_teacher.thinks;
const turnEnd = slide_teacher.turnEnd;
const slide_pick = @import("slide_pick.zig");
const talks = @import("slide_context.zig");
const slide_router = @import("slide_router.zig");
const heldBack = slide_router.heldBack;
const routerData = slide_router.routerData;
const conversations = slide_router.conversations;
const questionsFor = slide_router.questionsFor;
const replay = slide_router.replay;
const asksWho = slide_router.asksWho;
const aboutEntity = slide_router.aboutEntity;
const answered = slide_router.answered;
const Unseen = slide_pick.Unseen;
const heldPair = slide_pick.heldPair;
const dice = slide_pick.dice;

const swapped = 2; // a fact's questions whose subject is swapped for others of its kind, kept as the model answers them
pub const others = [_][]const u8{ "sister", "brother", "mother", "friend" }; // whom a fact's questions are also asked about

const probes = 12;
pub const extra_wordings = 12; // more of a fact's questions asked for, read by its router only (each judged first)
pub const probe_count = 4; // of those, asked after each round to see the lesson answers wordings it never learned
pub const wording_temperature = 0.9; // the extra wordings are sampled, so they differ from the greedy first list
pub const max_others = 96; // its questions asked about others, read by its router only
const besides = 6; // questions about the fact's subject that it leaves unanswered, kept as the model answers them
const other_values = 4; // yes-or-no questions naming another value than a told fact's, learned with its answers
const neighbours = 4; // a document fact's questions for the same kind of answer about another of its subject's things
const think_twins = 12; // near misses also kept with their thinking replies (each a long generation), the first ones
const remember_probes = 4; // kept questions asked again as "Do you remember...?", which the model otherwise refuses
pub const min_probes = 3; // two answers are held out to test recall
const want_probes = 12; // questions a fact is learned from at least, when the model gives that many
pub const replayed = 24; // earlier answers about a subject a document lesson keeps at most, the latest first
pub const talk_questions = 6; // a fact's questions also asked after each conversation it learns in (talks.learned)
pub const talk_near = 4; // and its near misses there, which its neurons stay silent on
const talk_personal = 2; // and questions about the user no fact answers, likewise
pub const talk_answers = 3; // its answers also learned after the conversations short enough for the trainer's passes
pub const talk_rows = 512; // a conversation's answer the trainer reads whole at most (its passes' longest sequence)

pub const personal_checks = 3;
const bank_checks = 6; // one bank question in this many is held back from each lesson to check
pub const check_tokens = 32; // a check's reply read before and after a lesson
pub const drift_tokens = 64; // a drift check's reply (slide_prompts.drift_checks), compared token for token

/// A question about something else and the model's answer before the lesson.
pub const Check = struct { question: []const u8, before: []const u8, thinking: []const u8 = "", invents: bool = false }; // invents: an untold check

/// A held-out question, the answer the model wrote for it from its fact, and which fact.
pub const Held = struct { question: []const u8, answer: []const u8, fact: usize, subject: []const u8 = "", short: []const u8 = "", context: []const u8 = "" }; // subject: the word an edit keys on; short: the fewest words that answer it; context: the conversation it is asked after (slide_context.zig)

/// A lesson for the learner, its facts, and what to ask after it; `probes` are wordings it never trains on.
pub const Plan = struct { full: api.LearnRequest, facts: []const []const u8, held: []const Held, probes: []const Held = &.{}, checks: []const Check, drifts: []const Check = &.{}, thinks: bool = false, document: bool = false, stale: slide_teacher.Stale = .{}, selves: []const bool = &.{} }; // selves[f]: fact f is about the model; drifts: replies to unrelated prompts no round may change

/// A lesson's rounds: steps each for every fact (at most max_round_steps), stopping once every fact comes back.
pub const round_steps = 20;
pub const max_round_steps = 400;
pub const rounds = 6;
pub const min_rounds = 1; // a first round ends the lesson only when its held-out answers came back token for token
pub const reroutes = 3; // questions about something else a lesson's router is fitted away from before it is taken out

/// A question a lesson took as its router reads it (prompt only), and with the answer it had before the lesson.
pub fn routed(srv: *Server, cx: *Cx, c: Check) !struct { prompt: api.Example, kept: api.Example } {
    return .{ .prompt = try slide_ask.promptOnly(srv, cx, cx.a, null, c.question), .kept = try example(srv, cx, cx.a, null, c.question, c.before, &.{}) };
}

/// One lesson for the facts `told` (kept[f]: fact f had enough clean answers), or null; `siblings`: all facts told.
pub fn lesson(srv: *Server, cx: *Cx, teacher: *Teacher, told: []const []const u8, kept: []bool, document: bool, about: ?wording.About, siblings: []const []const u8, gone: anytype) !?Plan {
    const a = cx.a;
    const end = try turnEnd(srv, cx, teacher);
    var parts: Parts = .{ .thinks = try thinks(srv, cx, teacher), .document = document, .siblings = siblings, .about = about };
    var count: u32 = 0;
    const language = try languageOf(srv, cx, told[0], gone);
    // the conversations a fact's questions are learned in, written side by side before the facts' lessons read them
    if (!document) try talks.prewrite(srv, cx, &talks.learned, gone);
    for (told, kept, 0..) |fact, *k, f| {
        k.* = try factLesson(srv, cx, &parts, fact, f, end, language, gone);
        count += @intFromBool(k.*);
    }
    if (count == 0) return null;
    const selves = parts.selves.items;
    const any_self = std.mem.indexOfScalar(bool, selves, true) != null;
    // what the change must never touch: questions about the user and prompts of every kind, as the model answers them
    var keep: std.ArrayList(api.Example) = .empty;
    var checks: std.ArrayList(Check) = .empty;
    teacher.lessons += 1;
    // the facts learned before: a written-once answer one of them may have changed is asked again as the model gives.
    var learned: std.ArrayList([]const u8) = .empty;
    for (try srv.slide.earlier(a, 0)) |e| try learned.append(a, e.text);
    var free: std.ArrayList([]const u8) = .empty; // questions about the user no fact of the lesson answers
    // the questions about the user no fact of the lesson answers (personal, untold checked, untold kept).
    var personal: std.ArrayList(usize) = .empty;
    var about_user: std.ArrayList(side.Asking) = .empty;
    for (prompts.personal_prompts, 0..) |q, i| {
        if (try answeredByAny(srv, cx, told, kept, selves, parts.own.items, q, document, gone)) continue;
        try free.append(a, q);
        try personal.append(a, i);
        try about_user.append(a, .{ .question = q, .max = check_tokens });
    }
    // questions about the user no lesson told: a self fact never answers them (secondPerson), "My name is Sam." does
    var untold_checked: std.ArrayList([]const u8) = .empty;
    for (untold.checked) |q| {
        if (try answeredByAny(srv, cx, told, kept, selves, parts.own.items, q, document, gone)) continue;
        try untold_checked.append(a, q);
        try about_user.append(a, .{ .question = q, .max = check_tokens });
    }
    var untold_kept: std.ArrayList([]const u8) = .empty;
    for (untold.kept) |q| {
        if (try answeredByAny(srv, cx, told, kept, selves, parts.own.items, q, document, gone)) continue;
        try untold_kept.append(a, q);
        try about_user.append(a, .{ .question = q, .max = check_tokens });
    }
    const replies = try side.askAll(srv, cx, about_user.items, gone);
    for (personal.items, replies[0..personal.items.len]) |i, reply| {
        const q = prompts.personal_prompts[i];
        if (heldBack(teacher.lessons, i)) {
            try checks.append(a, .{ .question = q, .before = reply.content });
            continue;
        }
        try keep.append(a, try example(srv, cx, a, null, q, reply.content, ending(reply, end)));
        if (parts.thinks) if (try thinkingSteady(srv, cx, a, q, gone)) |ex| try parts.think_keep.append(a, ex);
    }
    var at = personal.items.len;
    for (untold_checked.items) |q| {
        try checks.append(a, .{ .question = q, .before = replies[at].content, .invents = true });
        at += 1;
    }
    for (untold_kept.items) |q| {
        try keep.append(a, try example(srv, cx, a, null, q, replies[at].content, ending(replies[at], end)));
        at += 1;
    }
    const kept_answers = try keepExamples(srv, cx, teacher, end, parts.thinks, gone);
    var stale_keep: std.ArrayList(usize) = .empty; // the written-once answers this lesson's fact answers: read again once it is learned
    var stale_bank: std.ArrayList(usize) = .empty;
    var keep_fresh: std.ArrayList(side.Asking) = .empty; // keep prompts an earlier fact may answer, asked again now
    for (prompts.keep_prompts, 0..) |q, i| {
        // a question to the model is never a fact about the user's, but may be one about the model's
        if ((any_self or !wording.toModel(q)) and try answeredByAny(srv, cx, told, kept, selves, parts.own.items, q, document, gone)) {
            try stale_keep.append(a, i);
            continue;
        }
        if (stale(learned.items, q)) {
            try keep_fresh.append(a, .{ .question = q, .max = keep_tokens });
            continue;
        }
        try keep.append(a, kept_answers[i]);
        if (kept_answers.len > prompts.keep_prompts.len) try parts.think_keep.append(a, kept_answers[prompts.keep_prompts.len + i]);
    }
    // prompts of kinds no keep prompt is (math, code, verse, other scripts), their longer answers held the same way
    try keep.appendSlice(a, try broadExamples(srv, cx, teacher, end, gone));
    // a document's earlier facts about this subject keep their answers: a later fact must not overwrite them
    if (document) try replay(srv, cx, teacher, told[0], end, &keep);
    // the personal bank: kept as first answered, some held back to check, except what a fact answers
    var bank_fresh: std.ArrayList(side.Asking) = .empty;
    var bank_held: std.ArrayList(bool) = .empty; // each asked again: held back to check, else kept
    for (try bankAnswers(srv, cx, teacher, end, gone), 0..) |b, i| {
        if (try answeredByAny(srv, cx, told, kept, selves, parts.own.items, b.question, document, gone)) {
            try stale_bank.append(a, i);
            continue;
        }
        const held = (i + teacher.lessons) % bank_checks == 0;
        if (stale(learned.items, b.question)) {
            try bank_fresh.append(a, .{ .question = b.question, .max = check_tokens });
            try bank_held.append(a, held);
            continue;
        }
        if (held) try checks.append(a, .{ .question = b.question, .before = b.before }) else try keep.append(a, b.ex);
    }
    // the answers written once before a fact learned since changed them: as the model gives them now, side by side
    const fresh = try side.askAll(srv, cx, try std.mem.concat(a, side.Asking, &.{ keep_fresh.items, bank_fresh.items }), gone);
    for (keep_fresh.items, fresh[0..keep_fresh.items.len]) |k, reply| {
        try keep.append(a, try example(srv, cx, a, null, k.question, reply.content, ending(reply, end)));
        if (parts.thinks) if (try thinkingSteady(srv, cx, a, k.question, gone)) |ex| try parts.think_keep.append(a, ex);
    }
    for (bank_fresh.items, bank_held.items, fresh[keep_fresh.items.len..]) |b, held, reply| {
        if (held) try checks.append(a, .{ .question = b.question, .before = reply.content }) else try keep.append(a, try example(srv, cx, a, null, b.question, reply.content, ending(reply, end)));
    }
    if (fresh.len > 0) log.line("slide: {d} answers written before asked again, as facts learned since may have changed them", .{fresh.len});
    // a fact in another language: its own language's prompts held and checked as the English ones are
    if (if (language) |name| try local(srv, cx, teacher, name, end, gone) else null) |l| {
        for (l.personal, 0..) |q, i| {
            if (try answeredByAny(srv, cx, told, kept, selves, parts.own.items, q, document, gone)) continue;
            const reply = try ask(srv, cx, null, q, check_tokens, gone);
            if (heldBack(teacher.lessons, i)) {
                try checks.append(a, .{ .question = q, .before = reply.content });
            } else try keep.append(a, try example(srv, cx, a, null, q, reply.content, ending(reply, end)));
        }
        try keep.appendSlice(a, l.keep);
    }
    for (prompts.route_prompts, 0..) |q, i| {
        try parts.unasked.append(a, try slide_ask.promptOnly(srv, cx, a, null, q));
        try parts.unasked.append(a, try slide_ask.promptOnly(srv, cx, a, prompts.systems[i % prompts.systems.len], q));
    }
    // questions about the user no fact answers stay unanswered in a conversation too
    if (!document and free.items.len > 0) for (talks.learned, 0..) |name, c| {
        const before = try talks.turns(srv, cx, name, gone);
        if (before.len == 0) continue;
        for (0..talk_personal) |k| try parts.unasked.append(a, try talks.promptIn(srv, cx, a, before, free.items[(c * talk_personal + k) % free.items.len]));
    };
    // the near misses set aside, never trained on, check that the fact stays put
    try checks.appendSlice(a, parts.aside.items);
    log.line("slide: {d} near misses trained on, {d} set aside to check", .{ parts.near.items.len, parts.aside.items.len });
    const steps = @min(max_round_steps, round_steps * count);
    // a few checks' thinking replies now, which the last check after the lesson compares like with like
    if (parts.thinks) {
        const some = checks.items[0..@min(checks.items.len, think_leaks)];
        var asking: std.ArrayList(side.Asking) = .empty;
        for (some) |c| try asking.append(a, .{ .question = c.question, .thinking = true });
        for (some, try side.askAll(srv, cx, asking.items, gone)) |*c, reply| c.thinking = reply.content;
    }
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
    // replies to unrelated prompts as the model gives them before this lesson.
    var drifts: std.ArrayList(Check) = .empty;
    const drifting = if (std.c.getenv("TF_SLIDE_DRIFT")) |v| !std.mem.eql(u8, std.mem.span(v), "off") else true;
    if (drifting) {
        var asking: [prompts.drift_checks.len]side.Asking = undefined;
        for (&asking, prompts.drift_checks) |*x, q| x.* = .{ .question = q, .max = drift_tokens };
        for (prompts.drift_checks, try side.askAll(srv, cx, &asking, gone)) |q, reply| try drifts.append(a, .{ .question = q, .before = reply.content });
    }
    return .{ .full = full, .facts = told, .held = parts.held.items, .probes = parts.probes.items, .checks = checks.items, .drifts = drifts.items, .thinks = parts.thinks, .document = document, .stale = .{ .keep = stale_keep.items, .bank = stale_bank.items }, .selves = selves };
}

/// The facts' examples as a lesson gathers them.
pub const Parts = struct {
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
    document: bool = false, // the facts come from the user's documents, asked about as a colleague would, not told in chat
    words: []const []const u8 = &.{}, // the open fact's subject words (none: it is about the user)
    named: []const []const u8 = &.{}, // of those, the ones a question must say to name it
    siblings: []const []const u8 = &.{}, // every fact told with these, whose questions are theirs to answer
    about: ?wording.About = null, // whom the user said the facts are about (null: judged for each)
    self: bool = false, // the open fact is about the model itself: asked of the model, answered as itself
    selves: std.ArrayList(bool) = .empty, // each fact's: whether it is about the model
    own: std.ArrayList([]const u8) = .empty, // the facts' kept questions: none is also checked or kept as it was
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

/// A near miss kept as the model answers it unless the fact or a sibling does (judge, siblings); train: never checked.
const Near = struct { q: []const u8, judge: bool = false, siblings: bool = false, train: bool = false };

/// Near misses' replies as the model gives them now, side by side: three in four trained on, every fourth checked.
fn keepNear(srv: *Server, cx: *Cx, parts: *Parts, fact: []const u8, nears: []const Near, end: []const u32, gone: anytype) ![]const []const u8 {
    const a = cx.a;
    const Out = struct { said: ?Said = null, sibling: bool = false };
    const out = try a.alloc(Out, nears.len);
    @memset(out, .{});
    const Job = struct {
        srv: *Server,
        a: Allocator,
        parts: *const Parts,
        fact: []const u8,
        nears: []const Near,
        gone: @TypeOf(gone),
        out: []Out,

        pub fn run(j: *const @This(), i: usize) !void {
            var jcx: Cx = .{ .a = j.a };
            const n = j.nears[i];
            if (untold.protected(n.q)) return; // only ever checked, never trained
            if (n.judge and try answered(j.srv, &jcx, j.fact, n.q, j.parts.document, j.parts.self, j.gone)) return;
            if (n.siblings and try siblingAnswers(j.srv, &jcx, j.parts.siblings, j.fact, n.q, j.parts.document, j.gone)) {
                j.out[i].sibling = true;
                return;
            }
            j.out[i].said = try replyNow(j.srv, &jcx, n.q, j.gone);
        }
    };
    try side.each(nears.len, &Job{ .srv = srv, .a = a, .parts = parts, .fact = fact, .nears = nears, .gone = gone, .out = out });
    var kept: std.ArrayList([]const u8) = .empty;
    var thinking: std.ArrayList([]const u8) = .empty; // the twins whose thinking replies are held too, the first ones
    for (nears, out) |n, o| {
        if (o.sibling) {
            try parts.unasked.append(a, try slide_ask.promptOnly(srv, cx, a, null, n.q));
            continue;
        }
        const said = o.said orelse continue;
        try kept.append(a, n.q);
        if (!n.train) parts.seen += 1;
        if (!n.train and parts.seen % 4 == 0) {
            try parts.aside.append(a, .{ .question = n.q, .before = said.text });
            continue;
        }
        try parts.twins.append(a, try example(srv, cx, a, null, n.q, said.text, said.ending(end)));
        try parts.near.append(a, .{ .question = n.q, .before = said.text });
        if (!n.train and parts.thinks and parts.think_twins.items.len + thinking.items.len < think_twins) try thinking.append(a, n.q);
    }
    const thought = try a.alloc(?api.Example, thinking.items.len);
    @memset(thought, null);
    const Think = struct {
        srv: *Server,
        a: Allocator,
        qs: []const []const u8,
        gone: @TypeOf(gone),
        out: []?api.Example,

        pub fn run(j: *const @This(), i: usize) !void {
            var jcx: Cx = .{ .a = j.a };
            j.out[i] = try thinkingSteady(j.srv, &jcx, j.a, j.qs[i], j.gone);
        }
    };
    try side.each(thinking.items.len, &Think{ .srv = srv, .a = a, .qs = thinking.items, .gone = gone, .out = thought });
    for (thought) |t| if (t) |ex| try parts.think_twins.append(a, ex);
    return kept.items;
}

/// One fact's part: its questions and answers (two held out), and twins about other things as the model answers them.
fn factLesson(srv: *Server, cx: *Cx, parts: *Parts, fact: []const u8, f: usize, end: []const u32, language: ?[]const u8, gone: anytype) !bool {
    const a = cx.a;
    // a fact the user tells the model about itself is asked of the model and answered as itself, keyed on the reply
    const self = !parts.document and (if (parts.about) |w| w == .self else try aboutSelf(srv, cx, fact, gone));
    parts.self = self;
    try parts.selves.append(a, self);
    const asked = try askBase(srv, cx, null, try questionsFor(a, parts.document, self, fact, probes, language), 32 * probes, gone);
    const qs = try wording.questions(a, asked.content, probes);
    var got: Got = .{ .entity = !self and try aboutEntity(srv, cx, fact, gone), .self = self, .siblings = parts.siblings };
    var phrase: []const u8 = "";
    if (got.entity) {
        const said = try askBase(srv, cx, null, try std.fmt.allocPrint(a, prompts.fact_subject_prompt, .{fact}), 16, gone);
        phrase = std.mem.trim(u8, said.content, " \t\r\n\"'.?*");
        got.words = try slide_subject.factSubject(a, phrase, fact, qs, parts.document);
        got.named = try slide_subject.named(a, got.words, fact, qs);
        log.line("slide: its subject is \"{s}\": {s} (a question names it by {s})", .{ phrase, try std.mem.join(a, " ", got.words), try std.mem.join(a, " or ", if (got.named.len > 0) got.named else got.words) });
        if (got.words.len == 0) got.entity = false;
        // a told fact's subject most of its questions do not name.
        const why = if (got.entity and !parts.document) slide_subject.unkeyedBy(phrase, got.words, got.named, qs) else null;
        if (why) |w| {
            log.line("slide: its subject \"{s}\" {s}, so it keys on the reply", .{ phrase, w });
            got.entity = false;
            got.words = &.{};
            got.named = &.{};
        }
    }
    parts.words = got.words;
    parts.named = got.named;
    log.line("slide: \"{s}\" is about {s}", .{ fact, if (got.entity) "someone or something in particular, so its questions key on that subject" else if (self) "the model itself, so its questions ask the model about itself and key on the reply" else "the user, so its questions key on the reply" });
    try take(srv, cx, &got, fact, f, qs, end, parts.thinks, parts.document, gone);
    // too few kept: ask again, sampled, up to twice; a fact reaches unseen wordings only when learned from many
    const naming = if (parts.document and got.entity and language == null) try std.mem.join(a, " ", if (got.named.len > 0) got.named else got.words) else "";
    var tries: usize = 0;
    while (got.refs.items.len < want_probes and tries < 2) : (tries += 1) {
        // a count of its own each try: the draw is seeded by the prompt, so the same prompt gives the same list
        const count = probes + 4 * tries;
        // first, the questions that asked for the subject rewritten to name it: asked anew, they come back the same way
        if (tries == 0 and naming.len > 0 and got.unnamed.items.len > 0) {
            // only those whose answer is the subject: "What is Odile's title?" named "at Quillmoor" asks the other way
            const some = try slide_take.askingSubject(srv, cx, &got, fact, got.unnamed.items[0..@min(got.unnamed.items.len, probes)], gone);
            if (some.len > 0) {
                var list: std.ArrayList(u8) = .empty;
                for (some, 1..) |q, i| try list.print(a, "{d}. {s}\n", .{ i, q });
                const renamed = try askBase(srv, cx, null, try std.fmt.allocPrint(a, prompts.rename_prompt, .{ fact, naming, naming, list.items }), 32 * @as(i64, @intCast(some.len)), gone);
                try take(srv, cx, &got, fact, f, try wording.questions(a, renamed.content, some.len), end, parts.thinks, parts.document, gone);
                continue;
            }
        }
        // a document's fact asked for by name: its first list asked for the subject, or about another fact told with it
        const ask_for = if (naming.len > 0) try std.fmt.allocPrint(a, prompts.named_questions_prompt, .{ fact, count, naming, naming, count }) else try questionsFor(a, parts.document, self, fact, count, language);
        const again = try slide_ask.askBaseAt(srv, cx, null, ask_for, 32 * @as(i64, @intCast(count)), wording_temperature, gone);
        try take(srv, cx, &got, fact, f, try wording.questions(a, again.content, count), end, parts.thinks, parts.document, gone);
    }
    // "Do I like blue?" (no) and the list's own "Do I like red?" (yes) learned as the model answers knowing the fact.
    if (!parts.document and language == null) {
        const asked_other = try askBase(srv, cx, null, try (if (self) std.fmt.allocPrint(a, prompts.self_other_values_prompt, .{ fact, other_values, other_values }) else std.fmt.allocPrint(a, prompts.other_values_prompt, .{ fact, other_values, other_values })), 32 * other_values, gone);
        var built: std.ArrayList([]const u8) = .empty;
        try built.appendSlice(a, try wording.questions(a, asked_other.content, other_values));
        for (qs) |q| if (wording.asksYesNo(q, self) and wording.tells(fact, "", "", q)) try built.append(a, q);
        got.unjudged = true;
        try take(srv, cx, &got, fact, f, built.items, end, parts.thinks, parts.document, gone);
        got.unjudged = false;
    }
    // the kept questions asked again as "Do you remember...?", which the model otherwise refuses.
    var plain: std.ArrayList(u8) = .empty;
    for (got.refs.items[0..@min(got.refs.items.len, remember_probes)], 1..) |r, i| try plain.print(a, "{d}. {s}\n", .{ i, r.question });
    if (plain.items.len > 0 and !parts.document and !self) {
        const again = try askBase(srv, cx, null, try prompted(a, prompts.remember_prompt, remember_other, .{plain.items}, language), 32 * remember_probes, gone);
        var mine: std.ArrayList([]const u8) = .empty;
        for (try wording.questions(a, again.content, remember_probes)) |q| {
            if (wording.asksOf(q, false)) try mine.append(a, q) else log.line("slide: \"{s}\" is no longer the user's question, so it is dropped", .{q});
        }
        try take(srv, cx, &got, fact, f, mine.items, end, parts.thinks, parts.document, gone);
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
    try parts.own.appendSlice(a, told.items);
    log.line("slide: {d} questions kept for: {s} ({s})", .{ refs.items.len, fact, try std.mem.join(a, " | ", told.items) });
    if (qs.len < min_probes) log.line("slide: the questions came back as: {s}", .{asked.content});
    if (pairs.items.len < min_probes) return false;
    // the two most typical wordings held out, so a vague one ("Which dog do I have?") never decides recall
    const n = pairs.items.len;
    const pair = heldPair(fact, refs.items);
    var thought: usize = 0;
    for (pairs.items, got.thinking.items, refs.items, 0..) |ex, think, r, i| {
        const held = i == pair[0] or i == pair[1];
        try (if (held) &parts.held_ex else &parts.train).append(a, ex);
        if (think) |t| try (if (held) &parts.think_held else &parts.think_train).append(a, t);
        thought += @intFromBool(think != null);
        if (held) try parts.held.append(a, r);
    }
    // the held-out questions also checked after conversations the lesson never learns in.
    for (pair, talks.checked) |i, name| {
        var h = refs.items[i];
        h.context = name;
        try parts.held.append(a, h);
    }
    // each answer also learned under client system messages and after an unrelated exchange (ROME's prefixes)
    for (pairs.items, refs.items, 0..) |ex, r, i| {
        if (i == pair[0] or i == pair[1]) continue;
        const body = ex.ids[ex.start..];
        for (prompts.systems) |sys| try parts.train.append(a, try under(srv, cx, try slide_ask.promptOnly(srv, cx, a, sys, r.question), body, r));
        try parts.train.append(a, try under(srv, cx, try slide_ask.promptAfter(srv, cx, a, prompts.turns[i % prompts.turns.len], r.question), body, r));
    }
    if (parts.thinks) log.line("slide: {d} of {d} questions also learned with thinking on", .{ thought, n });
    // near misses kept as the model answers them: twins, others of the subject's kind, more about the same subject
    var nears: std.ArrayList(Near) = .empty;
    var numbered: std.ArrayList(u8) = .empty;
    for (refs.items[0..@min(refs.items.len, probes)], 1..) |r, i| try numbered.print(a, "{d}. {s}\n", .{ i, r.question });
    const asked_twins = try askBase(srv, cx, null, try prompted(a, prompts.twins_prompt, twins_other, .{numbered.items}, language), 64 * probes, gone);
    for (try wording.questions(a, asked_twins.content, 2 * probes)) |q| if (!wording.gives(fact, asked_all, answered_all, q)) try nears.append(a, .{ .q = q, .judge = true });
    for (refs.items[0..@min(refs.items.len, swapped)], subjects.items[0..@min(refs.items.len, swapped)]) |r, subject| for (try kindsOf(srv, cx, r.question, subject orelse continue, gone)) |twin| {
        if (!wording.gives(fact, asked_all, answered_all, twin)) try nears.append(a, .{ .q = twin, .judge = true });
    };
    const others_about: usize = if (got.entity) 2 * besides else besides;
    const besides_for = if (self) try std.fmt.allocPrint(a, prompts.self_besides_prompt, .{ fact, others_about }) else try prompted(a, prompts.besides_prompt, besides_other, .{ fact, others_about }, language);
    const asked_besides = try askBase(srv, cx, null, besides_for, 32 * @as(i64, @intCast(others_about)), gone);
    for (try wording.questions(a, asked_besides.content, others_about)) |q| {
        if ((!parts.document and !slide_take.whose(fact, q, self)) or wording.gives(fact, asked_all, answered_all, q)) continue;
        try nears.append(a, .{ .q = q, .judge = true, .siblings = true });
    }
    // the same kind of answer about the subject's other things stays as answered now: sick leave is not annual leave
    if (got.entity and parts.document and language == null) {
        const asked_near = try askBase(srv, cx, null, try std.fmt.allocPrint(a, prompts.neighbours_prompt, .{ fact, neighbours }), 32 * neighbours, gone);
        for (try wording.questions(a, asked_near.content, neighbours)) |q| if (!wording.gives(fact, asked_all, answered_all, q)) try nears.append(a, .{ .q = q, .judge = true, .siblings = true });
    }
    // what else is asked of a document's subject stays as the model answers it: the fact makes it no better known
    if (got.entity and parts.document and language == null and phrase.len > 0) inline for (prompts.essence) |t| {
        const q = try std.fmt.allocPrint(a, t, .{phrase});
        if (!wording.gives(fact, asked_all, answered_all, q)) try nears.append(a, .{ .q = q, .judge = true, .siblings = true });
    };
    // a fact naming someone keeps the subject's other roles as answered now: its founder must not become its marketer
    if (got.entity and parts.document and language == null and phrase.len > 0 and asksWho(got.refs.items)) inline for (prompts.roles) |t| {
        const q = try std.fmt.allocPrint(a, t, .{phrase});
        if (!wording.gives(fact, asked_all, answered_all, q)) try nears.append(a, .{ .q = q, .judge = true, .siblings = true });
    };
    // free writing about it stays as the model writes it now: a poem about a learned subject once counted down from 64
    if (got.entity and parts.document and language == null and phrase.len > 0) inline for (prompts.writing) |t| {
        try nears.append(a, .{ .q = try std.fmt.allocPrint(a, t, .{phrase}), .train = true });
    };
    // a question about the user asked of the model (my as your), or one to the model asked of the user (your as my)
    for (refs.items) |r| if (if (self) try wording.asUser(a, r.question) else try wording.addressed(a, r.question)) |own| try nears.append(a, .{ .q = own });
    if (!got.entity) for (refs.items, 0..) |r, i| for ([_]usize{ i, i + 1 }) |j| {
        const mine = if (self) try wording.asUser(a, r.question) orelse continue else r.question;
        const other = try wording.about(a, mine, others[j % others.len]) orelse continue;
        if (!wording.gives(fact, asked_all, answered_all, other)) try nears.append(a, .{ .q = other });
    };
    const kept = try keepNear(srv, cx, parts, fact, nears.items, end, gone);
    log.line("slide: near misses kept steady: {s}", .{try std.mem.join(a, " | ", kept)});
    try routerData(srv, cx, parts, fact, f, refs.items, subjects.items, kept, asked_all, answered_all, language, gone);
    try conversations(srv, cx, parts, refs.items, pairs.items, pair, kept, gone);
    return true;
}

/// An answer's tokens after another head (a question under a system message, or after an exchange), keyed as before.
pub fn under(srv: *Server, cx: *Cx, head: api.Example, body: []const u32, r: Held) !api.Example {
    var ex: api.Example = .{ .ids = try std.mem.concat(cx.a, u32, &.{ head.ids, body }), .start = @intCast(head.ids.len), .decide = head.decide };
    ex.subject = try subjectRow(srv, cx, ex, r.subject);
    ex.value = try slide_subject.valueRow(srv, cx, ex, r.short);
    return ex;
}

/// Whether the lesson teaches `question`: one of its own, or one a fact sharing a word with it answers.
fn answeredByAny(srv: *Server, cx: *Cx, told: []const []const u8, kept: []const bool, selves: []const bool, own: []const []const u8, question: []const u8, document: bool, gone: anytype) !bool {
    if (untold.taught(own, question)) return true;
    for (told, kept, 0..) |fact, k, f| {
        const self = f < selves.len and selves[f];
        const near = wording.shares(fact, question) or (self and wording.secondPerson(question));
        if (k and near and try answered(srv, cx, fact, question, document, self, gone)) return true;
    }
    return false;
}

/// Whether an answer written once (a keep prompt's, a bank question's) may be one a fact learned since has changed.
pub fn stale(learned: []const []const u8, question: []const u8) bool {
    for (learned) |fact| if (wording.shares(fact, question) or (wording.toModel(question) and wording.addressesModel(fact))) return true;
    return false;
}

/// Whether a told fact is about the model itself: it speaks to the model (you, your), and the model judges it so.
pub fn aboutSelf(srv: *Server, cx: *Cx, fact: []const u8, gone: anytype) !bool {
    if (!wording.addressesModel(fact)) return false;
    const reply = try askBase(srv, cx, null, try std.fmt.allocPrint(cx.a, prompts.self_prompt, .{fact}), 4, gone);
    return std.ascii.startsWithIgnoreCase(std.mem.trim(u8, reply.content, " \t\r\n\"*"), "yes");
}
