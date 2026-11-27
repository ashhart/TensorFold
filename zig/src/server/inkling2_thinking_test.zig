//! Inkling 2's thinking off reaches its template as reasoning_effort "none": effort level 0, not its default 0.9.
const std = @import("std");
const json = @import("json");
const h = @import("inkling2_chat_test.zig");
const HfText = @import("hf_text.zig").HfText;
const Server = @import("server.zig").Server;
const prompt_mod = @import("prompt.zig");
const errors = @import("errors.zig");
const t = std.testing;

const talk = "[{\"role\":\"user\",\"content\":\"Hi\"}]";
const budget_open = "<|message_model|><|content_thinking|>";
const budget_close = "<|end_message|><|message_model|><|content_text|>";

const BudgetIds = struct { script: []const u32, expected: []const u32 };

fn budgetIds(a: std.mem.Allocator) !BudgetIds {
    const head = try h.encodeReply(a, budget_open);
    const prefix = try std.mem.concat(a, u32, &.{ head, &.{ 64, 65, 66 } });
    const answer = try h.encodeReply(a, "391");
    const close = try h.encodeReply(a, budget_close);
    const end = try h.encodeReply(a, "<|end_message|>");
    try t.expectEqual(@as(usize, 5), prefix.len);
    try t.expectEqual(@as(usize, 3), close.len);
    return .{
        .script = try std.mem.concat(a, u32, &.{ prefix, answer, end }),
        .expected = try std.mem.concat(a, u32, &.{ prefix[0..4], close, answer, end }),
    };
}

fn expectBounded(a: std.mem.Allocator, got: h.Seen, expected_ids: []const u32) !void {
    try t.expectEqualStrings("ab", got.reasoning);
    try t.expectEqualStrings("391", got.content);
    const values = try a.alloc(json.Value, expected_ids.len);
    for (expected_ids, values) |id, *value| value.* = .{ .int = try std.fmt.allocPrint(a, "{d}", .{id}) };
    const expected = try json.stringify(a, .{ .array = values }, .{ .ascii = false, .compact = true });
    try t.expectEqualStrings(expected, got.token_ids);
}

/// The prompt a chat renders to on the `kind` fixture, decoded, and the server's thinking-off effort.
fn rendered(a: std.mem.Allocator, kind: []const u8, thinking: bool, named: ?[]const u8) !struct { text: []const u8, off: ?[]const u8 } {
    var fx = try h.Fixture.init(a, kind);
    defer fx.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, fx.path, null, a, &problem);
    defer hf.deinit();
    var script: h.Script = .{ .a = a, .reply = &.{}, .stride = 1 };
    const srv = try Server.init(a, t.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = a };
    cx.named_effort = named; // chat.zig's: the request's named effort, none of it for "none"
    const r = try prompt_mod.prepare(srv, &cx, .{ .messages = (try json.parse(a, talk)).ok, .fields = .null }, thinking, null);
    return .{ .text = try srv.text.decode(a, r.ids), .off = srv.off_effort };
}

test "Inkling 2 reply: thinking off renders the template's effort level 0, thinking on its default 0.9" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const off = try rendered(a, "release", false, null);
    const on = try rendered(a, "release", true, null);
    try t.expect(std.mem.indexOf(u8, off.text, "Thinking effort level: 0<|end_message|>") != null);
    try t.expect(std.mem.indexOf(u8, on.text, "Thinking effort level: 0.9<|end_message|>") != null);
    // the two prompts differ only in that level
    const same = try std.mem.replaceOwned(u8, a, on.text, "level: 0.9<", "level: 0<");
    try t.expectEqualStrings(off.text, same);
}

test "Inkling 2 reply: thinking off with a named effort keeps that level" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const low = try rendered(a, "release", false, "low");
    try t.expect(std.mem.indexOf(u8, low.text, "Thinking effort level: 0.2<|end_message|>") != null);
}

test "Inkling 2 reply: another model type keeps no thinking-off effort" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expectEqualStrings("none", (try rendered(a, "release", false, null)).off.?);
    try t.expect((try rendered(a, "llama", false, null)).off == null);
}
test "Inkling 2 thinking budget: the request budget closes thinking before the answer" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = try budgetIds(a);
    const body = "{\"model\":\"m\",\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":5}}";
    var cx: errors.Cx = .{ .a = a };
    const translated = try @import("anthropic_translate.zig").translate(&cx, (try json.parse(a, body)).ok, false);
    try translated.object.put(a, "return_token_ids", .{ .bool = true });
    for ([_]bool{ false, true }) |stream| {
        for ([_]usize{ 1, 2, 3, 256 }) |stride| {
            try translated.object.put(a, "stream", .{ .bool = stream });
            const got = try h.askLaneBody(a, "release", ids.script, translated, stream, stride);
            try expectBounded(a, got, ids.expected);
        }
    }
}

test "Inkling 2 thinking budget: the CLI budget closes thinking before the answer" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = try budgetIds(a);
    for ([_]bool{ false, true }) |stream| {
        for ([_]usize{ 1, 2, 3, 256 }) |stride| {
            const got = try h.askLaneBudget(a, "release", ids.script, ",\"return_token_ids\":true", stream, stride, 5);
            try expectBounded(a, got, ids.expected);
        }
    }
}

test "Inkling 2 thinking budget: no budget preserves reasoning, answer and every reply token" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const head = try h.encodeReply(a, budget_open);
    const prefix = try std.mem.concat(a, u32, &.{ head, &.{ 64, 65 } });
    const close = try h.encodeReply(a, budget_close);
    const answer = try h.encodeReply(a, "391");
    const end = try h.encodeReply(a, "<|end_message|>");
    const ids = try std.mem.concat(a, u32, &.{ prefix, close, answer, end });
    for ([_]bool{ false, true }) |stream| {
        for ([_]usize{ 1, 2, 3, 256 }) |stride| {
            const got = try h.askLane(a, "release", ids, ",\"return_token_ids\":true", stream, stride);
            try expectBounded(a, got, ids);
        }
    }
}
