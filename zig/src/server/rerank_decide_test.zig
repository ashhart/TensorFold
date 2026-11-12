//! /v1/rerank through Engine.decide: the same responses as one score a document, the trunk held once, refusals.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const rerank = @import("rerank.zig");
const fixture = @import("rerank_test.zig");
const Text = fixture.Text;
const Scorer = fixture.Scorer;
const Present = fixture.Present;
const t = std.testing;

/// A decide hook over the fake's score: each branch scored as trunk ++ branch; records what it was given.
const Decide = struct {
    var calls: usize = 0;
    var trunk: usize = 0;
    var shortest: usize = 0;
    var longest: usize = 0;
    var refusing: bool = false;

    fn decide(ctx: *anyopaque, d: *api.Decision) error{ Failed, Refused }!void {
        calls += 1;
        if (refusing) {
            d.problem = "refused by the fake";
            return error.Refused;
        }
        trunk = d.trunk.len;
        shortest = std.math.maxInt(usize);
        longest = 0;
        var at: usize = 0;
        for (d.branches, d.counts) |b, n| {
            shortest = @min(shortest, b.len);
            longest = @max(longest, b.len);
            const whole = std.mem.concat(t.allocator, u32, &.{ d.trunk, b }) catch return error.Failed;
            defer t.allocator.free(whole);
            _ = Scorer.score(ctx, whole, d.labels[at..][0..n], d.logits[at..][0..n], false) catch return error.Failed;
            at += n;
        }
    }

    fn reset() void {
        calls = 0;
        refusing = false;
    }

    const vtable: api.Engine.VTable = blk: {
        var v = Scorer.scored;
        v.decide = decide;
        break :blk v;
    };
};

fn encoded(srv: anytype, a: std.mem.Allocator, body: []const u8) ![]u8 {
    return rerank.encode(a, try fixture.ask(srv, a, body));
}

test "rerank: one decision answers as one score a document does, byte for byte" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator, .table = &fixture.table };
    defer s.deinit();
    const srv = try fixture.serve(&text, &s);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bodies = [_][]const u8{
        "{\"query\": \"q\", \"documents\": [\"low\", \"high\", \"mid\", \"tie\"]}",
        "{\"query\": \"q\", \"documents\": [\"low\", \"high\", \"mid\", \"tie\"], \"top_n\": 2, \"return_documents\": false}",
        "{\"query\": \"q\", \"documents\": [\"high\", {\"text\": \"low\"}], \"return_logits\": true, \"return_prompt_token_ids\": true}",
        "{\"query\": \"q\", \"instruction\": \"find\", \"documents\": [\"mid\", \"tie\", \"low\"]}",
        "{\"query\": \"q\", \"documents\": [\"dd\", \"high\"], \"max_length\": 13, \"return_prompt_token_ids\": true}",
    };
    Decide.reset();
    for (bodies) |body| {
        srv.engine = s.engine();
        const one = try encoded(srv, a, body);
        srv.engine = .{ .ctx = &s, .vtable = &Decide.vtable };
        const packed_reply = try encoded(srv, a, body);
        try t.expectEqualStrings(one, packed_reply);
    }
    try t.expectEqual(bodies.len, Decide.calls);
}

test "rerank: the trunk leaves every branch a token, also for one document and identical documents" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator, .table = &fixture.table };
    defer s.deinit();
    const srv = try fixture.serve(&text, &s);
    defer srv.deinit();
    srv.engine = .{ .ctx = &s, .vtable = &Decide.vtable };
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // "S:|Q:q|D:high<end>|A": one document, 16 tokens: a 15-token trunk and a 1-token branch
    _ = try fixture.ask(srv, a, "{\"query\": \"q\", \"documents\": [\"high\"]}");
    try t.expectEqual(@as(usize, 15), Decide.trunk);
    try t.expectEqual(@as(usize, 1), Decide.shortest);
    _ = try fixture.ask(srv, a, "{\"query\": \"q\", \"documents\": [\"high\", \"high\", \"high\"]}");
    try t.expectEqual(@as(usize, 1), Decide.shortest);
    // a long query and short documents: the query is in the trunk once, each branch is the document and the tail
    const query = try a.alloc(u8, 4000);
    @memset(query, 'q');
    _ = try fixture.ask(srv, a, try std.fmt.allocPrint(a, "{{\"query\": \"{s}\", \"documents\": [\"a\", \"bb\", \"c\"]}}", .{query}));
    try t.expectEqual(@as(usize, 2 + 3 + 4000 + 3), Decide.trunk); // "S:", "|Q:", the query, "|D:"
    try t.expectEqual(@as(usize, 2 + 3), Decide.longest); // "bb", "<end>", "|", "A"
}

test "rerank: an empty prompt runs no hook; a refused and a failed decision keep their classes" {
    var text: Text = .{ .blank = true };
    var s: Scorer = .{ .gpa = t.allocator, .table = &fixture.table };
    defer s.deinit();
    const srv = try fixture.serve(&text, &s);
    defer srv.deinit();
    srv.engine = .{ .ctx = &s, .vtable = &Decide.vtable };
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = (try json.parse(a, "{\"query\": \"q\", \"documents\": [\"high\", \"low\"]}")).ok;
    Decide.reset();
    var cx: errors.Cx = .{ .a = a };
    try t.expectError(error.Refused, rerank.reply(srv, &cx, body, Present{}));
    try t.expectEqualStrings("document 0: the chat template renders no tokens", cx.message);
    try t.expectEqual(@as(usize, 0), Decide.calls);
    text = .{};
    Decide.refusing = true;
    cx = .{ .a = a };
    try t.expectError(error.Refused, rerank.reply(srv, &cx, body, Present{}));
    try t.expectEqualStrings("refused by the fake", cx.message);
    try t.expectEqual(@as(u16, 400), cx.status());
    Decide.refusing = false;
    s.failing = true;
    cx = .{ .a = a };
    try t.expectError(error.Refused, rerank.reply(srv, &cx, body, Present{}));
    try t.expectEqual(@as(u16, 500), cx.status());
    s.healthy = false;
    cx = .{ .a = a };
    try t.expectError(error.Refused, rerank.reply(srv, &cx, body, Present{}));
    try t.expectEqual(@as(u16, 503), cx.status());
}

test "rerank: a template that renders another prompt on the second pass is refused, never sliced" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator, .table = &fixture.table };
    defer s.deinit();
    const srv = try fixture.serve(&text, &s);
    defer srv.deinit();
    srv.engine = .{ .ctx = &s, .vtable = &Decide.vtable };
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = (try json.parse(a, "{\"query\": \"q\", \"documents\": [\"high\", \"low\"]}")).ok;
    // the check pass renders both documents, then the template drifts
    text = .{ .drift_after = 2 };
    Decide.reset();
    var cx: errors.Cx = .{ .a = a };
    try t.expectError(error.Refused, rerank.reply(srv, &cx, body, Present{}));
    try t.expectEqualStrings("document 0: the chat template rendered another prompt on its second pass", cx.message);
    try t.expectEqual(@as(usize, 0), Decide.calls);
    // a second pass shorter than the trunk (the query gone) is refused too, never sliced
    text = .{ .drift_after = 2, .drift = .short };
    cx = .{ .a = a };
    try t.expectError(error.Refused, rerank.reply(srv, &cx, body, Present{}));
    try t.expectEqual(@as(usize, 0), Decide.calls);
    // a longer tail on the second pass keeps the trunk: scored, and usage counts the prompts scored (16 + 2, 15 + 2)
    text = .{ .drift_after = 2, .drift = .tail };
    const out = try fixture.ask(srv, a, "{\"query\": \"q\", \"documents\": [\"high\", \"low\"]}");
    try t.expectEqual(@as(i64, 18 + 17), out.get("usage").?.get("total_tokens").?.int64().?);
    try t.expectEqual(@as(usize, 1), Decide.calls);
}
