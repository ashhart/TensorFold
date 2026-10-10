//! After a lesson's round: which facts come back, with thinking on too, and whether a reply loops, leaks a fact or
//! changes from what it was.
const std = @import("std");
const log = @import("log.zig");
const errors = @import("errors.zig");
const wording = @import("slide_words.zig");
const teach = @import("slide_lesson.zig");
const slide_ask = @import("slide_ask.zig");
const prompts = @import("slide_prompts.zig");
const Server = @import("server.zig").Server;
const Cx = errors.Cx;
const Plan = teach.Plan;
const Held = teach.Held;
const Check = teach.Check;
const ask = slide_ask.ask;
const askBase = slide_ask.askBase;
const askThinking = slide_ask.askThinking;
const check_tokens = teach.check_tokens;
const think_leaks = @import("slide_think.zig").think_leaks;

const steady_prompt = "Tell me something interesting about the ocean.";

/// What a round did: which facts come back, the question about something else the lesson acted on (the router's to
/// take away), and damage that takes the round back (null: none).
pub const Verdict = struct { recalled: []bool, took: ?Check = null, damage: ?[]const u8 = null };

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

/// Whether a changed reply still claims nothing new about the user nor answers anew, as the model judges it.
fn same(srv: *Server, cx: *Cx, question: []const u8, before: []const u8, reply: []const u8, gone: anytype) !bool {
    const verdict = try askBase(srv, cx, null, try std.fmt.allocPrint(cx.a, prompts.same_prompt, .{ question, before, reply }), 4, gone);
    return std.ascii.startsWithIgnoreCase(std.mem.trim(u8, verdict.content, " \t\r\n\"*"), "no");
}
