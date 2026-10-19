//! After a lesson's round: which facts come back (thinking too), and whether a reply loops, leaks or changes.
const std = @import("std");
const log = @import("log.zig");
const errors = @import("errors.zig");
const wording = @import("slide_words.zig");
const untold = @import("slide_untold.zig");
const teach = @import("slide_lesson.zig");
const slide_ask = @import("slide_ask.zig");
const prompts = @import("slide_prompts.zig");
const side = @import("slide_side.zig");
const talks = @import("slide_context.zig");
const chat = @import("chat.zig");
const Server = @import("server.zig").Server;
const Cx = errors.Cx;
const Plan = teach.Plan;
const Held = teach.Held;
const Check = teach.Check;
const askBase = slide_ask.askBase;
const check_tokens = teach.check_tokens;
const think_leaks = @import("slide_think.zig").think_leaks;

const steady_prompt = "Tell me something interesting about the ocean.";

/// What a round did: which facts come back, a question about something else it took, and damage that undoes it.
pub const Verdict = struct { recalled: []bool, took: ?Check = null, damage: ?[]const u8 = null };

/// After a round: which facts come back, and damage if a reply loops, leaks or changes (`last`: thinking checks too).
pub fn verify(srv: *Server, cx: *Cx, plan: Plan, last: bool, gone: anytype) !Verdict {
    const a = cx.a;
    const recalled = try a.alloc(bool, plan.facts.len);
    const asked = try a.alloc(bool, plan.facts.len);
    @memset(recalled, true);
    @memset(asked, false);
    // every reply this pass reads, asked side by side first: nothing learns while it runs
    const thought = if (last and plan.thinks) plan.checks[0..@min(plan.checks.len, think_leaks)] else plan.checks[0..0];
    var asking: std.ArrayList(side.Asking) = .empty;
    for (plan.held) |h| {
        try asking.append(a, .{ .question = h.question, .before = try talks.turns(srv, cx, h.context, gone) });
        if (plan.thinks and h.context.len == 0) try asking.append(a, .{ .question = h.question, .thinking = true });
    }
    for (plan.probes) |h| try asking.append(a, .{ .question = h.question });
    for (plan.checks) |c| try asking.append(a, .{ .question = c.question, .max = check_tokens });
    for (thought) |c| if (c.thinking.len > 0) try asking.append(a, .{ .question = c.question, .thinking = true });
    try asking.append(a, .{ .question = steady_prompt, .max = 64 });
    const replies = try side.askAll(srv, cx, asking.items, gone);
    var at: usize = 0;
    for (plan.held) |h| {
        const reply = replies[at].content;
        at += 1;
        if (wording.looped(h.question, reply)) return .{ .recalled = recalled, .damage = "it started repeating itself" };
        asked[h.fact] = true;
        const back = brings(plan.facts[h.fact], h, reply);
        if (!back) log.line("slide: \"{s}\"{s}{s} does not bring its fact back: {s}", .{ h.question, if (h.context.len > 0) " after the conversation " else "", h.context, reply });
        recalled[h.fact] = recalled[h.fact] and back;
        if (plan.thinks and h.context.len == 0) {
            recalled[h.fact] = thinkingRecalls(plan.facts[h.fact], h, replies[at]) and recalled[h.fact];
            at += 1;
        }
    }
    // wordings it never learned: at least half of a fact's must bring it back
    const probed = try a.alloc(usize, 2 * plan.facts.len);
    @memset(probed, 0);
    for (plan.probes) |h| {
        const reply = replies[at].content;
        at += 1;
        const back = brings(plan.facts[h.fact], h, reply);
        if (!back) log.line("slide: \"{s}\", a wording it never learned, does not bring its fact back: {s}", .{ h.question, reply });
        probed[2 * h.fact] += 1;
        probed[2 * h.fact + 1] += @intFromBool(back);
    }
    for (recalled, asked, 0..) |*r, x, f| r.* = r.* and x and 2 * probed[2 * f + 1] >= probed[2 * f];
    for (plan.checks) |c| {
        const reply = replies[at].content;
        at += 1;
        if (wording.looped(c.question, reply)) return .{ .recalled = recalled, .damage = "it started repeating itself" };
        var inferred = false;
        for (plan.facts) |fact| if (wording.tells(fact, c.question, c.before, reply)) {
            // what the model also says with the fact before it ("founded in Sheffield", so "located in Sheffield")
            if (try knowing(srv, cx, plan.document, fact, c, gone)) {
                log.line("slide: \"{s}\" now says what the model says knowing the fact, so it stays: {s}", .{ c.question, reply });
                inferred = true;
                break;
            }
            log.line("slide: leaked into \"{s}\": {s}", .{ c.question, reply });
            return .{ .recalled = recalled, .took = c };
        };
        if (!inferred and !untold.unjudged(c.invents, c.before, reply) and !try same(srv, cx, c.question, c.before, reply, gone)) {
            log.line("slide: \"{s}\" changed from \"{s}\" to \"{s}\"", .{ c.question, c.before, reply });
            if (c.invents) return .{ .recalled = recalled, .damage = "it made up something about the user it was never told" };
            return .{ .recalled = recalled, .took = c };
        }
    }
    for (thought) |c| {
        if (c.thinking.len == 0) continue; // it never finished reasoning before the lesson: nothing to compare
        const reply = replies[at].content;
        at += 1;
        for (plan.held) |h| if (wording.carries(plan.facts[h.fact], h.question, h.answer, c.question, c.thinking, reply)) {
            log.line("slide: leaked into \"{s}\" with thinking on: {s} (before: {s})", .{ c.question, reply, c.thinking });
            return .{ .recalled = recalled, .took = c };
        };
    }
    if (wording.looped(steady_prompt, replies[at].content)) return .{ .recalled = recalled, .damage = "it started repeating itself" };
    return .{ .recalled = recalled };
}

/// Whether the model, the fact before it (as told, or read in a document), also says it in answer to check `c`.
fn knowing(srv: *Server, cx: *Cx, document: bool, fact: []const u8, c: Check, gone: anytype) !bool {
    const note = if (document) try std.fmt.allocPrint(cx.a, prompts.read_note, .{fact}) else try std.fmt.allocPrint(cx.a, prompts.told_note, .{fact});
    const reply = try askBase(srv, cx, note, c.question, check_tokens, gone);
    return wording.tells(fact, c.question, c.before, reply.content);
}

/// Whether a held-out question brings its fact back with thinking on, in the reply the user reads after the reasoning.
fn thinkingRecalls(fact: []const u8, h: Held, reply: chat.Reply) bool {
    const back = brings(fact, h, reply.content);
    const reasoning = reply.reasoning orelse "";
    if (!back) log.line("slide: \"{s}\" with thinking on: {s} (it reasoned: {s})", .{ h.question, reply.content, reasoning[0..@min(reasoning.len, 240)] });
    return back;
}

/// Whether `reply` brings back the fact a held question asks for: its short answer, or the answer words the fact gave.
fn brings(fact: []const u8, h: Held, reply: []const u8) bool {
    if (h.short.len > 0) return wording.recallsShort(fact, h.question, h.short, reply);
    return wording.recalls(fact, h.question, h.answer, reply);
}

/// Whether a changed reply still claims nothing new about the user nor answers anew, as the model judges it.
fn same(srv: *Server, cx: *Cx, question: []const u8, before: []const u8, reply: []const u8, gone: anytype) !bool {
    const verdict = try askBase(srv, cx, null, try std.fmt.allocPrint(cx.a, prompts.same_prompt, .{ question, before, reply }), 4, gone);
    return std.ascii.startsWithIgnoreCase(std.mem.trim(u8, verdict.content, " \t\r\n\"*"), "no");
}
