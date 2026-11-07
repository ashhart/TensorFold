//! /v1/rerank's max_length: a prompt's content cut from its end, the template's tail kept; templates it cannot cut.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const rerank = @import("rerank.zig");
const fixture = @import("rerank_test.zig");
const Text = fixture.Text;
const Scorer = fixture.Scorer;
const serve = fixture.serve;
const ask = fixture.ask;
const results = fixture.results;
const table = fixture.table;
const Present = fixture.Present;
const im_end_id = fixture.im_end_id;
const t = std.testing;

test "rerank: past max_length a prompt's content is cut from its end, the template's tail kept; 8192 by default" {
    var text: Text = .{};
    var s: Scorer = .{ .gpa = t.allocator, .table = &table };
    defer s.deinit();
    const srv = try serve(&text, &s);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // "S:|Q:q|D:dd<end>|A" is 14 tokens; at 13 the document's last byte goes and "<end>|A" stays
    const out = try ask(srv, a, "{\"query\": \"q\", \"documents\": [\"dd\"], \"max_length\": 13, \"return_prompt_token_ids\": true}");
    try t.expectEqual(@as(usize, 1), s.prompts.items.len);
    try t.expectEqualStrings("S:|Q:q|D:d#END#|A", s.prompts.items[0]);
    try t.expectEqual(@as(i64, 13), out.get("usage").?.get("total_tokens").?.int64().?);
    // the ids the response reports are the ids scored
    const ids = results(out)[0].get("meta_info").?.get("prompt_token_ids").?.array;
    try t.expectEqual(@as(usize, 13), ids.len);
    try t.expectEqual(@as(i64, 'd'), ids[9].int64().?);
    try t.expectEqual(@as(i64, im_end_id), ids[10].int64().?);
    // the document is echoed as sent, not as cut
    try t.expectEqualStrings("dd", results(out)[0].get("document").?.string);
    // with no max_length, 8192 tokens: a 9,000-byte document keeps its first 8192 - 12 bytes
    const long = try a.alloc(u8, 9000);
    @memset(long, 'x');
    const body = try std.fmt.allocPrint(a, "{{\"query\": \"q\", \"documents\": [\"{s}\"], \"return_documents\": false}}", .{long});
    const cut = try ask(srv, a, body);
    try t.expectEqual(@as(i64, rerank.max_length_default), cut.get("usage").?.get("total_tokens").?.int64().?);
    const scored = s.prompts.items[1];
    try t.expectEqual(@as(usize, rerank.max_length_default + 4), scored.len); // "#END#" is one token
    try t.expect(std.mem.endsWith(u8, scored, "x#END#|A"));
    // a cap at the prompt's own length cuts nothing; a window refusal is a context-length error
    _ = try ask(srv, a, "{\"query\": \"q\", \"documents\": [\"dd\"], \"max_length\": 14}");
    try t.expectEqualStrings("S:|Q:q|D:dd#END#|A", s.prompts.items[2]);
    srv.info.context_window = 13;
    var cx: errors.Cx = .{ .a = a };
    try t.expectError(error.Refused, rerank.reply(srv, &cx, (try json.parse(a, "{\"query\": \"q\", \"documents\": [\"dd\"], \"max_length\": 14}")).ok, Present{}));
    const e = (try rerank.errorBody(&cx)).get("error").?;
    try t.expectEqualStrings("context_length_exceeded", e.get("code").?.string);
    try t.expectEqualStrings("documents", e.get("param").?.string);
    try t.expectEqual(@as(usize, 3), s.prompts.items.len);
    // a tail whose added token is not first, or whose text follows the document, is never cut
    srv.info.context_window = 0;
    for ([_]Text{ .{ .loose = true }, .{ .moving = true }, .{ .merging = true } }) |shape| {
        text = shape;
        var refused: errors.Cx = .{ .a = a };
        try t.expectError(error.Refused, rerank.reply(srv, &refused, (try json.parse(a, "{\"query\": \"q\", \"documents\": [\"dd\"], \"max_length\": 13}")).ok, Present{}));
        const tokens: usize = if (shape.loose) 14 else 15; // the moving tail's '+', the merging tail's 'X'
        try t.expectEqualStrings(try std.fmt.allocPrint(a, "document 0: the prompt has {d} tokens, past max_length 13, and this chat template has no fixed tail after the document to cut its content before", .{tokens}), refused.message);
    }
    try t.expectEqual(@as(usize, 3), s.prompts.items.len);
}
