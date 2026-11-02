//! Additive corrections: capture the old belief, oppose its training answers and check its absence.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const teach = @import("slide_lesson.zig");
const errors = @import("errors.zig");
const words = @import("slide_words.zig");
const asks = @import("slide_ask.zig");
const reads = @import("slide_correction_ask.zig");
const subject = @import("slide_subject.zig");
const answers = @import("slide_answers.zig");
const Server = @import("server.zig").Server;

pub const Entry = struct { h: teach.Held, old: []const u8, old_ids: []const u32, old_short: []const u8, opposes: bool };

pub fn requested(body: json.Value) !bool {
    const v = body.field("correction") orelse return false;
    return if (v == .bool) v.bool else error.BadCorrectionFlag;
}

fn same(a: teach.Held, b: teach.Held) bool {
    return a.fact == b.fact and std.mem.eql(u8, a.question, b.question) and std.mem.eql(u8, a.context, b.context);
}
fn held(plan: teach.Plan, h: teach.Held) bool {
    for (plan.held) |x| if (same(x, h)) return true;
    for (plan.probes) |x| if (same(x, h)) return true;
    return false;
}

/// A correction uses only its training questions for the opposing gradient; held/probe references are checks only.
pub fn prepare(srv: *Server, cx: *errors.Cx, plan: *teach.Plan, gone: anytype) !void {
    const a = cx.a;
    if (plan.questions.len == 0) return error.CorrectionQuestionsMissing;
    var all: std.ArrayList(teach.Held) = .empty;
    for ([_][]const teach.Held{ plan.questions, plan.held, plan.probes }) |xs| for (xs) |h| {
        var seen = false;
        for (all.items) |x| if (same(x, h)) {
            seen = true;
            break;
        };
        if (!seen) try all.append(a, h);
    };
    // Check the configured stochastic sampler before any training is queued.
    _ = try reads.ask(srv, cx, all.items[0].question, all.items[0].context, 0, gone);
    var entries: std.ArrayList(Entry) = .empty;
    var against: std.ArrayList(api.Example) = .empty;
    var changed = false;
    for (all.items) |h| {
        const old = try reads.ask(srv, cx, h.question, h.context, null, gone);
        if (!std.mem.eql(u8, old.finish_reason, "stop") or old.content.len == 0) return error.CorrectionReferenceIncomplete;
        const tokens = try reads.ids(a, old);
        if (tokens.len == 0) return error.CorrectionTokensMissing;
        const back = if (h.short.len > 0) words.recallsShort(plan.facts[h.fact], h.question, h.short, old.content) else words.recalls(plan.facts[h.fact], h.question, h.answer, old.content);
        const bare = if (back) old else try reads.bare(srv, cx, h.question, h.context, gone);
        const old_short = std.mem.trim(u8, bare.content, " \t\r\n\"'.*");
        if (!back and (old_short.len == 0 or std.mem.count(u8, old_short, " ") > 7 or std.ascii.findIgnoreCase(old.content, old_short) == null or std.ascii.findIgnoreCase(h.question, old_short) != null)) return error.CorrectionValueMissing;
        try entries.append(a, .{ .h = h, .old = old.content, .old_ids = tokens, .old_short = old_short, .opposes = !back });
        changed = changed or !back;
        if (!back and !held(plan.*, h) and h.context.len == 0) {
            var ex = try asks.example(srv, cx, a, null, h.question, old.content, &.{});
            ex.subject = try subject.subjectRow(srv, cx, ex, h.subject);
            if (ex.subject == 0) return error.CorrectionSubjectMissing;
            try against.append(a, ex);
        }
    }
    if (!changed or against.items.len == 0) return error.CorrectionOldBeliefMissing;
    plan.corrections = entries.items;
    plan.full.against = against.items;
}

pub fn oldReturned(question: []const u8, old: []const u8, old_short: []const u8, reply: []const u8) bool {
    return words.recallsShort(old, question, old_short, reply);
}

/// Greedy and three seeded draws of the server's default sampler, on training and held-out wordings alike.
pub fn verify(srv: *Server, cx: *errors.Cx, plan: teach.Plan, recalled: []bool, gone: anytype) !void {
    for (plan.corrections) |c| {
        if (!c.opposes or !recalled[c.h.fact]) continue;
        const greedy = try reads.ask(srv, cx, c.h.question, c.h.context, null, gone);
        if (oldReturned(c.h.question, c.old, c.old_short, greedy.content)) {
            recalled[c.h.fact] = false;
            continue;
        }
        for (0..reads.samples) |i| {
            const sampled = try reads.ask(srv, cx, c.h.question, c.h.context, i, gone);
            if (oldReturned(c.h.question, c.old, c.old_short, sampled.content)) {
                recalled[c.h.fact] = false;
                break;
            }
        }
    }
}

pub fn expected(a: std.mem.Allocator, plan: teach.Plan) ![]const answers.Answer {
    var out: std.ArrayList(answers.Answer) = .empty;
    for (plan.held) |h| try out.append(a, .{ .question = h.question, .answer = h.answer, .short = h.short, .context = h.context });
    for (plan.probes) |h| try out.append(a, .{ .question = h.question, .answer = h.answer, .short = h.short, .probe = true });
    for (plan.corrections) |c| {
        var found: ?usize = null;
        for (out.items, 0..) |x, i| if (std.mem.eql(u8, x.question, c.h.question) and std.mem.eql(u8, x.context, c.h.context)) {
            found = i;
            break;
        };
        if (found == null) {
            try out.append(a, .{ .question = c.h.question, .answer = c.h.answer, .short = c.h.short, .context = c.h.context });
            found = out.items.len - 1;
        }
        out.items[found.?].old = c.old;
        out.items[found.?].old_short = c.old_short;
        out.items[found.?].old_ids = c.old_ids;
        out.items[found.?].reject_old = c.opposes;
    }
    return out.items;
}

/// The correction reference must be owned by the graph before a weight is learned.
pub fn record(srv: *Server, cx: *errors.Cx, id: u32, plan: teach.Plan) !void {
    const xs = try expected(cx.a, plan);
    srv.slide.asked(id, xs);
    if (plan.corrections.len == 0) return;
    const f = srv.slide.find(id) orelse return error.CorrectionReferenceMissing;
    if (f.answers.len != xs.len) return error.CorrectionReferenceMissing;
    for (f.answers, xs) |got, want| if (!std.mem.eql(u32, got.old_ids, want.old_ids) or !std.mem.eql(u8, got.old, want.old) or !std.mem.eql(u8, got.old_short, want.old_short) or got.reject_old != want.reject_old) return error.CorrectionReferenceMissing;
    try srv.slide.record(id, .learning);
}

/// Earlier corrections keep their new answers and reject their old values under default sampling.
pub fn absent(srv: *Server, cx: *errors.Cx, xs: []const answers.Answer, gone: anytype) !bool {
    for (xs) |x| if (x.reject_old) {
        for (0..reads.samples) |i| {
            const reply = try reads.ask(srv, cx, x.question, x.context, i, gone);
            if (oldReturned(x.question, x.old, x.old_short, reply.content)) return false;
        }
    };
    return true;
}

/// A deleted correction returns its recorded greedy tokens; retained corrections still reject their old answers.
pub fn afterCut(srv: *Server, cx: *errors.Cx, facts: []const @import("slide_graph.zig").Fact, removed: usize, gone: anytype) !?@import("slide_recheck.zig").Miss {
    for (facts, 0..) |f, i| {
        if (i >= removed) {
            if (!try absent(srv, cx, f.answers, gone)) return .{ .fact = i, .why = "a retained correction's old answer returned under its default sampler" };
            continue;
        }
        for (f.answers) |x| if (x.old_ids.len > 0) {
            const reply = try reads.ask(srv, cx, x.question, x.context, null, gone);
            if (!std.mem.eql(u8, reply.finish_reason, "stop") or !std.mem.eql(u32, x.old_ids, try reads.ids(cx.a, reply))) return .{ .fact = i, .why = "the correction's old answer did not return token for token" };
        };
    }
    return null;
}
