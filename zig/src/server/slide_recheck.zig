//! Facts asked again as their lessons checked them: after a new lesson, before a save; and around a delete's cut.
const std = @import("std");
const api = @import("engine_api");
const log = @import("log.zig");
const errors = @import("errors.zig");
const wording = @import("slide_words.zig");
const side = @import("slide_side.zig");
const slide_ask = @import("slide_ask.zig");
const slide_graph = @import("slide_graph.zig");
const talks = @import("slide_context.zig");
const Server = @import("server.zig").Server;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;
const Fact = slide_graph.Fact;
const Earlier = slide_graph.Earlier;

/// Times a lesson that made an earlier fact forget is fitted again, that fact's answers held, before it is taken out.
pub const repairs = 2;

/// An earlier fact asked again: whether it comes back now, and whether it came back at its last check before.
pub const Result = struct { fact: Earlier, recalled: bool, was: bool };

/// Which facts a check reads: every learned fact, or only those no check has judged since the server started.
pub const Which = enum { all, unjudged };

/// The learned facts but `id` asked again side by side, each verdict to the graph; `before` says which came back.
pub fn check(srv: *Server, cx: *Cx, id: u32, which: Which, before: []const Earlier, gone: anytype) ![]Result {
    const a = cx.a;
    var facts: std.ArrayList(Earlier) = .empty;
    for (try srv.slide.earlier(a, id)) |f| if (which == .all or f.recalled == null) try facts.append(a, f);
    for (facts.items) |f| if (f.asked.len == 0) return error.EarlierQuestionsMissing;
    var asking: std.ArrayList(side.Asking) = .empty;
    for (facts.items) |f| for (f.asked) |w| try asking.append(a, .{ .question = w.question, .before = try talks.turns(srv, cx, w.context, gone) });
    const replies = try side.askAll(srv, cx, asking.items, gone);
    const out = try a.alloc(Result, facts.items.len);
    var at: usize = 0;
    for (facts.items, out) |f, *r| {
        var held = true;
        var probes: usize = 0;
        var probed: usize = 0;
        for (f.asked, 0..) |w, i| {
            const reply = replies[at].content;
            at += 1;
            const back = bringsBack(f.text, w, reply);
            if (!back) log.line("slide: \"{s}\", asked again for \"{s}\", does not bring it back: {s}", .{ w.question, f.text, reply });
            if (back and w.answer.len == 0) srv.slide.answered(f.id, i, reply);
            if (w.probe) {
                probes += 1;
                probed += @intFromBool(back);
            } else held = held and back;
        }
        r.* = .{ .fact = f, .recalled = held and 2 * probed >= probes, .was = cameBack(before, f) };
        srv.slide.checked(f.id, r.recalled);
    }
    return out;
}

/// Whether fact `f` came back before: as `before` holds it, else at its last check (a fact never checked did).
fn cameBack(before: []const Earlier, f: Earlier) bool {
    for (before) |b| if (b.id == f.id) return b.recalled orelse true;
    return f.recalled orelse true;
}

/// Whether `reply` to wording `w` brings back `fact`, judged as its own lesson judged it (short answer, else words).
fn bringsBack(fact: []const u8, w: slide_graph.Answer, reply: []const u8) bool {
    if (w.short.len > 0) return wording.recallsShort(fact, w.question, w.short, reply);
    if (w.answer.len > 0) return wording.recalls(fact, w.question, w.answer, reply);
    return wording.tells(fact, w.question, "", reply);
}

/// The facts that came back at their last check and do not now.
pub fn forgotten(a: Allocator, results: []const Result) ![]const Result {
    var out: std.ArrayList(Result) = .empty;
    for (results) |r| if (r.was and !r.recalled) try out.append(a, r);
    return out.items;
}

/// A lesson fitted again keeping what `lost` facts' held-out questions answered: their prompts and those answers.
pub fn keeps(srv: *Server, cx: *Cx, lost: []const Result) !struct { prompts: []const api.Example, kept: []const api.Example } {
    const a = cx.a;
    var prompts: std.ArrayList(api.Example) = .empty;
    var kept: std.ArrayList(api.Example) = .empty;
    for (lost) |r| for (r.fact.asked) |w| {
        if (w.probe or w.context.len > 0) continue; // kept as asked cold: its answers in conversation follow
        const answer = if (w.answer.len > 0) w.answer else continue;
        try prompts.append(a, try slide_ask.promptOnly(srv, cx, a, null, w.question));
        try kept.append(a, try slide_ask.example(srv, cx, a, null, w.question, answer, &.{}));
    };
    return .{ .prompts = prompts.items, .kept = kept.items };
}

/// The facts in `rs` as one line: "I like blue.", "My sister is called Ana."
pub fn names(a: Allocator, rs: []const Result) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (rs, 0..) |r, i| {
        if (i > 0) try out.appendSlice(a, ", ");
        try out.print(a, "\"{s}\"", .{r.fact.text});
    }
    return out.items;
}

/// A question of fact `fact` (0: the deleted one): its words, the answer it expects ("": its reply before the cut).
pub const Ask = struct { fact: usize, question: []const u8, answer: []const u8 = "", short: []const u8 = "", thinking: bool = false, probe: bool = false, context: []const u8 = "" }; // context: a kept talk's name

/// What a recheck found wrong: whose fact, which ask (null: its unseen wordings together), and what.
pub const Miss = struct { fact: usize, ask: ?usize = null, why: []const u8 };

/// Each fact's questions as its lesson asked them: held out (cold ones thinking off, then on), unseen ones off.
pub fn plan(a: Allocator, facts: []const Fact, thinks: bool) ![]const Ask {
    var out: std.ArrayList(Ask) = .empty;
    for (facts, 0..) |f, i| {
        for (f.answers) |x| for ([_]bool{ false, true }) |on| {
            if (on and (x.probe or !thinks or x.context.len > 0)) continue;
            try out.append(a, .{ .fact = i, .question = x.question, .answer = x.answer, .short = x.short, .thinking = on, .probe = x.probe, .context = x.context });
        };
        // a fact taught before its answers were kept: what it says before the cut is what it should say after
        if (f.answers.len == 0) for (f.questions) |q| for ([_]bool{ false, true }) |on| {
            if (on and !thinks) continue;
            try out.append(a, .{ .fact = i, .question = q, .thinking = on });
        };
    }
    return out.items;
}

/// What the user reads of each ask's reply, asked side by side (`base`: from the model as shipped).
pub fn ask(srv: *Server, cx: *Cx, asks: []const Ask, base: bool, gone: anytype) ![]const []const u8 {
    const qs = try cx.a.alloc(side.Asking, asks.len);
    for (asks, qs) |x, *q| q.* = .{ .question = x.question, .thinking = x.thinking, .base = base, .before = try talks.turns(srv, cx, x.context, gone) };
    const replies = try side.askAll(srv, cx, qs, gone);
    const out = try cx.a.alloc([]const u8, replies.len);
    for (replies, out) |r, *o| o.* = r.content;
    return out;
}

/// Whether each reply brings its fact back, as the lesson's checks judge it; a kept fact with no answers by `before`.
pub fn brings(a: Allocator, facts: []const Fact, asks: []const Ask, said: []const []const u8, before: ?[]const []const u8) ![]bool {
    const out = try a.alloc(bool, asks.len);
    for (asks, said, out, 0..) |x, r, *o, i| {
        const fact = facts[x.fact].text;
        if (x.short.len == 0 and x.answer.len == 0 and before != null and x.fact > 0) {
            // every word of the fact it said before the cut, it says again
            const was = before.?[i];
            o.* = wording.tells(fact, x.question, "", was) and wording.recalls(fact, x.question, was, r);
        } else o.* = bringsBack(fact, .{ .question = x.question, .answer = x.answer, .short = x.short }, r);
    }
    return out;
}

/// Whether fact f comes back as its lesson judged it: every held-out answer, and at least half its unseen wordings.
pub fn recalled(asks: []const Ask, got: []const bool, f: usize) bool {
    var probes: usize = 0;
    var back: usize = 0;
    for (asks, got) |x, g| {
        if (x.fact != f) continue;
        if (!x.probe and !g) return false;
        probes += @intFromBool(x.probe);
        back += @intFromBool(x.probe and g);
    }
    return 2 * back >= probes;
}

/// The first thing wrong after the cut: fact 0 answering where the model as shipped does not, or a kept answer lost.
pub fn judge(asks: []const Ask, before: []const bool, after: []const bool, shipped: []const bool, facts: usize) ?Miss {
    for (asks, after, shipped, 0..) |x, g, s, i| if (x.fact == 0 and g and !s) return .{ .fact = 0, .ask = i, .why = "it still gives the deleted fact" };
    for (asks, before, after, 0..) |x, b, g, i| if (x.fact > 0 and !x.probe and b and !g) return .{ .fact = x.fact, .ask = i, .why = "it no longer brings its fact back" };
    for (1..facts) |f| {
        var n: usize = 0;
        var was: usize = 0;
        var now: usize = 0;
        for (asks, before, after) |x, b, g| if (x.fact == f and x.probe) {
            n += 1;
            was += @intFromBool(b);
            now += @intFromBool(g);
        };
        if (n > 0 and 2 * was >= n and 2 * now < n) return .{ .fact = f, .why = "fewer than half its unseen wordings bring it back" };
    }
    return null;
}

test "a fact asked again comes back when every held-out question and half its probes bring it back" {
    const ana: slide_graph.Answer = .{ .question = "What is my sister's name?", .answer = "Your sister is called Ana.", .short = "Ana" };
    try std.testing.expect(bringsBack("My sister is called Ana.", ana, "Your sister is called Ana."));
    try std.testing.expect(!bringsBack("My sister is called Ana.", ana, "I don't know your sister's name."));
    const blue: slide_graph.Answer = .{ .question = "What colour do I prefer?" };
    try std.testing.expect(bringsBack("I like blue.", blue, "You like **blue**."));
    try std.testing.expect(!bringsBack("I like blue.", blue, "As an AI, I don't have access to your personal preferences."));
    const a = std.testing.allocator;
    const f: Earlier = .{ .id = 1, .text = "I like blue.", .asked = &.{}, .recalled = null };
    const rs = [_]Result{ .{ .fact = f, .recalled = false, .was = true }, .{ .fact = f, .recalled = false, .was = false }, .{ .fact = f, .recalled = true, .was = true } };
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    try std.testing.expectEqual(@as(usize, 1), (try forgotten(arena.allocator(), &rs)).len);
    // a fact the lesson's first check found forgotten is still one that came back before the lesson
    const now: Earlier = .{ .id = 1, .text = "I like blue.", .asked = &.{}, .recalled = false };
    try std.testing.expect(cameBack(&.{.{ .id = 1, .text = "I like blue.", .asked = &.{}, .recalled = true }}, now));
    try std.testing.expect(!cameBack(&.{}, now));
}

test "a recheck keeps a delete only when every kept answer still comes back and the deleted fact is gone" {
    const asks = [_]Ask{
        .{ .fact = 0, .question = "What is my sister called?" },
        .{ .fact = 1, .question = "What colour do I like?" },
        .{ .fact = 1, .question = "What colour do I like?", .thinking = true },
        .{ .fact = 1, .question = "Which colour is my favourite?", .probe = true },
        .{ .fact = 1, .question = "What is my preferred colour?", .probe = true },
        .{ .fact = 2, .question = "What is my dog called?" },
    };
    const all: [asks.len]bool = @splat(true);
    const kept = [_]bool{ false, true, true, true, true, true };
    const none: [asks.len]bool = @splat(false);
    try std.testing.expectEqual(@as(?Miss, null), judge(&asks, &all, &kept, &none, 3));
    // the deleted fact still answers: kept back, unless the model as shipped says it too
    const miss = judge(&asks, &all, &all, &none, 3).?;
    try std.testing.expectEqual(@as(usize, 0), miss.fact);
    try std.testing.expectEqual(@as(?usize, 0), miss.ask);
    try std.testing.expectEqual(@as(?Miss, null), judge(&asks, &all, &all, &[_]bool{ true, false, false, false, false, false }, 3));
    // a kept fact's thinking-on answer stops coming back
    const thought = judge(&asks, &all, &[_]bool{ false, true, false, true, true, true }, &none, 3).?;
    try std.testing.expectEqual(@as(usize, 1), thought.fact);
    try std.testing.expectEqual(@as(?usize, 2), thought.ask);
    // an answer that did not come back before the cut either is no regression
    try std.testing.expectEqual(@as(?Miss, null), judge(&asks, &[_]bool{ true, true, true, true, true, false }, &[_]bool{ false, true, true, true, true, false }, &none, 3));
    // unseen wordings: half is enough, as the lesson's check has it, but not less once it was
    try std.testing.expectEqual(@as(?Miss, null), judge(&asks, &all, &[_]bool{ false, true, true, true, false, true }, &none, 3));
    const probes = judge(&asks, &all, &[_]bool{ false, true, true, false, false, true }, &none, 3).?;
    try std.testing.expectEqual(@as(?usize, null), probes.ask);
    try std.testing.expect(recalled(&asks, &[_]bool{ false, true, true, true, false, true }, 1));
    try std.testing.expect(!recalled(&asks, &[_]bool{ false, true, false, true, true, true }, 1));
}

test "a kept fact taught before its answers were kept is held to what it said before the cut" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const facts = [_]Fact{
        .{ .id = 2, .text = "My sister is called Ana.", .source = "chat", .at = 0, .state = .learned, .answers = &.{.{ .question = "What is my sister's name?", .answer = "Your sister is called Ana.", .short = "Ana" }} },
        .{ .id = 3, .text = "My dog is called Ginger.", .source = "chat", .at = 0, .state = .learned, .questions = &.{"What is my dog called?"} },
    };
    const asks = try plan(a, &facts, true);
    try std.testing.expectEqual(@as(usize, 4), asks.len);
    try std.testing.expect(!asks[0].thinking and asks[1].thinking and asks[2].answer.len == 0);
    const before = [_][]const u8{ "Your sister is Ana.", "Ana.", "Your dog is called Ginger.", "I don't know your dog's name." };
    const was = try brings(a, &facts, asks, &before, null);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, was);
    const after = [_][]const u8{ "I don't know your sister's name.", "Ana!", "Ginger, I think.", "Ginger." };
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, try brings(a, &facts, asks, &after, &before));
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, try brings(a, &facts, asks, &.{ "No idea.", "Ana!", "A dog named Rex.", "Rex." }, &before));
}
