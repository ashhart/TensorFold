//! A delete's recheck: each fact's questions asked as its lesson checked them, before the cut and after, then judged.
const std = @import("std");
const errors = @import("errors.zig");
const wording = @import("slide_words.zig");
const slide_graph = @import("slide_graph.zig");
const side = @import("slide_side.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Fact = slide_graph.Fact;

/// A question of fact `fact` (0: the deleted one): its words, the answer it expects ("": its reply before the cut).
pub const Ask = struct { fact: usize, question: []const u8, answer: []const u8 = "", short: []const u8 = "", thinking: bool = false, probe: bool = false };

/// What a recheck found wrong: whose fact, which ask (null: its unseen wordings together), and what.
pub const Miss = struct { fact: usize, ask: ?usize = null, why: []const u8 };

/// Each fact's questions as its lesson asked them: held out with thinking off, then on (`thinks`); unseen ones off.
pub fn plan(a: Allocator, facts: []const Fact, thinks: bool) ![]const Ask {
    var out: std.ArrayList(Ask) = .empty;
    for (facts, 0..) |f, i| {
        for (f.answers) |x| for ([_]bool{ false, true }) |on| {
            if (on and (x.probe or !thinks)) continue;
            try out.append(a, .{ .fact = i, .question = x.question, .answer = x.answer, .short = x.short, .thinking = on, .probe = x.probe });
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
pub fn ask(srv: *Server, cx: *errors.Cx, asks: []const Ask, base: bool, gone: anytype) ![]const []const u8 {
    const qs = try cx.a.alloc(side.Asking, asks.len);
    for (asks, qs) |x, *q| q.* = .{ .question = x.question, .thinking = x.thinking, .base = base };
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
        if (x.short.len > 0) {
            o.* = wording.recallsShort(fact, x.question, x.short, r);
        } else if (x.answer.len > 0) {
            o.* = wording.recalls(fact, x.question, x.answer, r);
        } else if (before != null and x.fact > 0) {
            // every word of the fact it said before the cut, it says again
            const was = before.?[i];
            o.* = wording.tells(fact, x.question, "", was) and wording.recalls(fact, x.question, was, r);
        } else o.* = wording.tells(fact, x.question, "", r);
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
