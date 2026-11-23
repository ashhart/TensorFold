//! Inkling 2 reply parsing at turn boundaries: role headers, thinking token counts, bare replies, raw decode cost.
const std = @import("std");
const json = @import("json");
const model_text = @import("model_text.zig");
const reply = @import("inkling2_reply.zig");
const fx = @import("inkling2_fake.zig");
const Piece = fx.Piece;
const plain = fx.plain;
const t = std.testing;

const mm = fx.message_model;
const text = fx.content_text;
const think = fx.content_thinking;
const close = fx.end_message;
const tool_json = fx.content_tool_json;

/// Whole and every growing prefix through one cursor agree; the whole reply is returned.
fn both(a: std.mem.Allocator, pieces: []const Piece, tools: []const json.Value) !reply.Result {
    const whole = try fx.parse(a, pieces, tools);
    var fake: fx.Fake = .{ .pieces = pieces };
    var stream = try reply.Stream.init(a, fake.text());
    const ids = try a.alloc(u32, pieces.len);
    for (pieces, ids) |piece, *id| id.* = piece.id;
    for (1..ids.len + 1) |n| _ = try stream.extend(ids[0..n], tools);
    const done = try stream.done();
    try t.expectEqualStrings(whole.content, done.content);
    try t.expectEqualStrings(whole.reasoning, done.reasoning);
    try t.expectEqual(whole.rejected_calls, done.rejected_calls);
    return whole;
}

test "Inkling 2 parse: a user, system or tool header ends the model's turn in every channel" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]Piece{ fx.message_user, fx.message_system, fx.message_tool }) |role| {
        // with ignore_eos the reply goes on past its end into another role's turn: none of it is the model's
        const after = [_]Piece{ mm, text, plain(70, "A"), close, fx.end_sampling, role, text, plain(71, "USER"), close };
        try t.expectEqualStrings("A", (try both(a, &after, &.{})).content);
        const open_think = [_]Piece{ mm, think, plain(72, "x"), role, text, plain(73, "USER") };
        const got = try both(a, &open_think, &.{});
        try t.expectEqualStrings("x", got.reasoning);
        try t.expectEqualStrings("", got.content);
        try t.expect(!got.open_think);
    }
}

test "Inkling 2 parse: a role header inside a JSON call refuses it instead of stitching the next turn's text in" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try fx.offeredTools(a, &.{"lookup"});
    const pieces = [_]Piece{ mm, plain(74, "lookup"), tool_json, plain(75, "{\"name\":\"lookup\",\"args\":{\"x\":"), fx.message_user, text, plain(76, "1}}"), close };
    const got = try both(a, &pieces, tools);
    try t.expect(got.calls == null);
    try t.expectEqual(@as(usize, 1), got.rejected_calls);
    try t.expectEqualStrings("", got.content);
}

test "Inkling 2 parse: reasoning tokens count a thinking block from its header through its close" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { pieces: []const Piece, tokens: usize };
    const cases = [_]Case{
        .{ .pieces = &.{ mm, think, plain(80, "x"), close }, .tokens = 4 },
        .{ .pieces = &.{ mm, plain(81, "name"), think, plain(82, "x"), plain(83, "y"), close }, .tokens = 6 },
        .{ .pieces = &.{ mm, think, plain(84, "x"), fx.unused, plain(85, "y"), close }, .tokens = 6 },
        .{ .pieces = &.{ mm, think, plain(86, "open") }, .tokens = 3 },
        // the answer's block is not thinking
        .{ .pieces = &.{ mm, think, plain(87, "x"), close, mm, text, plain(88, "answer"), close }, .tokens = 4 },
    };
    for (cases) |case| try t.expectEqual(case.tokens, (try both(a, case.pieces, &.{})).reasoning_tokens);
}

test "Inkling 2 parse: text before any message header is a bare reply's content, as a grammar writes one" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bare = [_]Piece{ plain(90, "{\"ok\""), plain(91, ":true}") };
    try t.expectEqualStrings("{\"ok\":true}", (try both(a, &bare, &.{})).content);
    // once a header has arrived, text between blocks is a header's name, never content
    const shaped = [_]Piece{ mm, plain(92, "name"), text, plain(93, "B"), close, plain(94, "stray") };
    try t.expectEqualStrings("B", (try both(a, &shaped, &.{})).content);
}

test "Inkling 2 parse: a streamed raw completion decodes no prefix: the generation's own decode serves it" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var pieces: [64]Piece = undefined;
    for (&pieces, 0..) |*piece, i| piece.* = plain(@intCast(100 + i), "x");
    var fake: fx.Fake = .{ .pieces = &pieces };
    const Srv = struct { text: model_text.Text };
    const srv: Srv = .{ .text = fake.text() };
    var g: struct {
        a: std.mem.Allocator,
        chat: bool = false,
        inkling2: ?reply.Stream = null,
        tools: []const json.Value = &.{},
        srv: *const Srv,
        stops: @import("reply_text.zig").Stops = .{ .strings = &.{} },
        visible: @import("reply_text.zig").Incremental = .{},
    } = .{ .a = a, .srv = &srv };
    var ids: [pieces.len]u32 = undefined;
    for (&pieces, &ids) |piece, *id| id.* = piece.id;
    fx.decoded_ids = 0;
    for (1..ids.len + 1) |n| _ = try @import("chat_inkling2.zig").read(&g, ids[0..n], false);
    const done = try @import("chat_inkling2.zig").read(&g, &ids, true);
    try t.expectEqual(@as(usize, 64), done.content.len);
    try t.expectEqual(ids.len, fx.decoded_ids); // one decode, of the finished reply
}
