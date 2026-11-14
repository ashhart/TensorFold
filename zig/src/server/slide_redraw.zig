//! Extra lesson wordings: the original first draw and fresh, validated rows after a refused round.
const std = @import("std");
const api = @import("engine_api");
const teach = @import("slide_lesson.zig");
const slide_ask = @import("slide_ask.zig");
const wording = @import("slide_words.zig");
const units = @import("slide_units.zig");
const slide_subject = @import("slide_subject.zig");
const take = @import("slide_take.zig");
const judged = take.judged;
const siblingAnswers = take.siblingAnswers;
const subjectRow = slide_subject.subjectRow;
const shortAnswer = take.shortAnswer;
const Unseen = @import("slide_pick.zig").Unseen;
const dice = @import("slide_pick.zig").dice;
const prompts = @import("slide_prompts.zig");
const log = @import("log.zig");
const Server = @import("server.zig").Server;
const Cx = @import("errors.zig").Cx;
const A = std.mem.Allocator;
const extra_wordings = 12; // extra questions drawn for the router beside the original answers
const probe_count = 4; // unseen wordings selected once for the lesson's fixed verify gate

pub const Spec = struct {
    reserved: @import("slide_reserved.zig").Reserved = .{},
    fact: []const u8,
    fact_index: usize,
    refs: []const teach.Held,
    document: bool,
    siblings: []const []const u8,
    words: []const []const u8,
    named: []const []const u8,
    language: ?[]const u8,
    asked_all: []const u8,
    answered_all: []const u8,
};
const Span = struct { start: usize, end: usize };
pub const Entry = struct { spec: Spec, asked: Span, unasked: Span, probes: Span, seen: []const []const u8 };

fn keptSpan(xs: []const api.Example, span: Span, reserved: @import("slide_reserved.zig").Reserved) Span {
    std.debug.assert(span.start <= span.end and span.end <= xs.len);
    var out: Span = .{ .start = 0, .end = 0 };
    for (xs[0..span.end], 0..) |ex, i| {
        if (reserved.reads(ex.ids)) continue;
        out.end += 1;
        if (i < span.start) out.start += 1;
    }
    return out;
}

/// Filter every learner role and keep router spans relative to the resulting lists; fixed probes stay unchanged.
pub fn clean(a: A, plan: teach.Plan) !teach.Plan {
    if (plan.reserved.heads.len == 0) return plan;
    var out = plan;
    out.full = try plan.reserved.clean(a, plan.full);
    const entries = try a.dupe(Entry, plan.wordings);
    for (entries) |*entry| {
        entry.asked = keptSpan(plan.full.asked, entry.asked, plan.reserved);
        entry.unasked = keptSpan(plan.full.unasked, entry.unasked, plan.reserved);
    }
    out.wordings = entries;
    return out;
}

const Drawn = struct {
    asked: std.ArrayList(api.Example) = .empty,
    unasked: std.ArrayList(api.Example) = .empty,
    probes: std.ArrayList(teach.Held) = .empty,
    wordings: std.ArrayList([]const u8) = .empty,
    seen: std.StringHashMapUnmanaged(void) = .empty,
    taken: usize = 0,
};

fn draw(srv: *Server, cx: *Cx, spec: Spec, round: u32, excluded: []const []const u8, gone: anytype) !Drawn {
    const a = cx.a;
    var out: Drawn = .{};
    const fact = spec.fact;
    const f = spec.fact_index;
    const refs = spec.refs;
    const language = spec.language;
    const asked_all = spec.asked_all;
    const answered_all = spec.answered_all;
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    for (refs) |r| try seen.put(a, r.question, {});
    for (excluded) |q| try seen.put(a, q, {});
    // the first request again, sampled: the greedy reply would give the same questions back
    const count = extra_wordings;
    const more = try slide_ask.askBaseAtRound(srv, cx, null, try teach.questionsFor(a, spec.document, fact, count, language), 32 * count, teach.wording_temperature, round, gone);
    var taken: usize = 0;
    var unseen: std.ArrayList(Unseen) = .empty; // wordings never learned, the most typical of them checked after a round
    const wordings = try wording.questions(a, more.content, count);
    if (wordings.len == 0) log.line("slide: no more wordings came back: {s}", .{more.content});
    for (wordings) |q| {
        if (try spec.reserved.question(srv, cx, q)) continue;
        if ((try seen.getOrPut(a, q)).found_existing) continue;
        const personal = !spec.document and (!wording.firstPerson(q) or wording.someoneElse(fact, q));
        const why: ?[]const u8 = if (personal or !units.sameScript(fact, q)) "is not the user asking about themselves" else if (wording.gives(fact, asked_all, answered_all, q)) "gives the fact away" else if (!try judged(srv, cx, fact, q, spec.document, gone)) "is not answered by the fact" else if (try siblingAnswers(srv, cx, spec.siblings, fact, q, spec.document, gone)) "is another fact's" else null;
        if (why) |w| {
            // a wording about the model or someone else is still one the lesson must not take
            const other = !spec.document and !wording.firstPerson(q) and units.sameScript(fact, q) and !wording.gives(fact, asked_all, answered_all, q);
            if (other) try out.unasked.append(a, try slide_ask.promptOnly(srv, cx, a, null, q));
            log.line("slide: \"{s}\" {s}, so its router {s}", .{ q, w, if (other) "reads it as one it must not take" else "does not read it" });
            continue;
        }
        const word = slide_subject.keyWord(spec.words, spec.named, q);
        if (spec.words.len > 0 and (word.len == 0 or !slide_subject.names(spec.named, spec.words, q))) {
            // a wording that never names the subject ("my sibling", "the team") is another question, not this one's
            log.line("slide: \"{s}\" does not name the fact's subject, so the lesson leaves it", .{q});
            continue;
        }
        var asked = try slide_ask.promptOnly(srv, cx, a, null, q);
        asked.subject = try subjectRow(srv, cx, asked, word);
        try out.asked.append(a, asked);
        try out.wordings.append(a, q);
        var score: f32 = 0;
        for (refs) |r| score += dice(q, r.question);
        try unseen.append(a, .{ .q = q, .word = word, .score = score });
        taken += 1;
    }
    out.seen = seen;
    out.taken = taken;
    if (round > 0) return out;
    // the most typical of them check the fact after each round, so a vague one never decides
    std.mem.sort(Unseen, unseen.items, {}, Unseen.more);
    const note = if (spec.document) try std.fmt.allocPrint(a, prompts.read_note, .{fact}) else try std.fmt.allocPrint(a, prompts.told_note, .{fact});
    var probed: usize = 0;
    for (unseen.items) |u| {
        if (probed == probe_count) break;
        const short = try shortAnswer(srv, cx, note, u.q, gone);
        if (short.len > 0 and std.ascii.findIgnoreCase(u.q, short) != null) continue; // it says its own answer: nothing to recall
        try out.probes.append(a, .{ .question = u.q, .answer = fact, .fact = f, .subject = u.word, .short = short });
        probed += 1;
    }
    return out;
}

fn append(comptime T: type, a: A, list: *std.ArrayList(T), xs: []const T) !Span {
    const start = list.items.len;
    try list.appendSlice(a, xs);
    return .{ .start = start, .end = list.items.len };
}

pub fn first(srv: *Server, cx: *Cx, parts: *teach.Parts, spec: Spec, gone: anytype) !struct { seen: std.StringHashMapUnmanaged(void), taken: usize } {
    var held = spec;
    held.reserved = parts.reserved;
    const d = try draw(srv, cx, held, 0, &.{}, gone);
    try parts.wordings.append(cx.a, .{
        .spec = held,
        .asked = try append(api.Example, cx.a, &parts.asked, d.asked.items),
        .unasked = try append(api.Example, cx.a, &parts.unasked, d.unasked.items),
        .probes = try append(teach.Held, cx.a, &parts.probes, d.probes.items),
        .seen = d.wordings.items,
    });
    return .{ .seen = d.seen, .taken = d.taken };
}

/// Preserve every original answer, safety row and unseen check; replace only the extra router rows.
pub fn next(srv: *Server, cx: *Cx, plan: teach.Plan, round: u32, _: u32, gone: anytype) !teach.Plan {
    if (plan.wordings.len == 0) return plan;
    const a = cx.a;
    var asked: std.ArrayList(api.Example) = .empty;
    var unasked: std.ArrayList(api.Example) = .empty;
    var entries: std.ArrayList(Entry) = .empty;
    var cursors = [_]usize{ 0, 0 };
    for (plan.wordings) |entry| {
        const d = try draw(srv, cx, entry.spec, round, entry.seen, gone);
        if (entry.seen.len > 0 and d.taken == 0)
            return error.NoFreshRouterWordings;
        try asked.appendSlice(a, plan.full.asked[cursors[0]..entry.asked.start]);
        try unasked.appendSlice(a, plan.full.unasked[cursors[1]..entry.unasked.start]);
        try entries.append(a, .{
            .spec = entry.spec,
            .asked = try append(api.Example, a, &asked, d.asked.items),
            .unasked = try append(api.Example, a, &unasked, d.unasked.items),
            .probes = entry.probes,
            .seen = try std.mem.concat(a, []const u8, &.{ entry.seen, d.wordings.items }),
        });
        cursors = .{ entry.asked.end, entry.unasked.end };
    }
    try asked.appendSlice(a, plan.full.asked[cursors[0]..]);
    try unasked.appendSlice(a, plan.full.unasked[cursors[1]..]);
    var out = plan;
    out.full.asked = asked.items;
    out.full.unasked = unasked.items;
    out.wordings = entries.items;
    log.line("slide: round {d}: fresh extra wordings after the refused round", .{round + 1});
    return clean(a, out);
}

/// A refused draw leaves the existing wordings and ordinary continuation in place.
pub fn prepare(srv: *Server, cx: *Cx, plan: teach.Plan, round: u32, id: u32, gone: anytype, redraw: *bool) !teach.Plan {
    if (!redraw.*) return plan;
    return next(srv, cx, plan, round, id, gone) catch |e| {
        if (e != error.NoFreshRouterWordings) return e;
        log.line("slide: NoFreshRouterWordings; continuing the open fit without router refresh", .{});
        redraw.* = false;
        return plan;
    };
}
