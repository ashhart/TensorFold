//! Pure host coverage for Inkling 2 native reply splitting.
const std = @import("std");
const json = @import("json");
const model_text = @import("model_text.zig");
const reply = @import("inkling2_reply.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

const fx = @import("inkling2_fake.zig");
const Piece = fx.Piece;
const plain = fx.plain;
const message_model = fx.message_model;
const content_text = fx.content_text;
const end_sampling = fx.end_sampling;
const content_thinking = fx.content_thinking;
const end_message = fx.end_message;
const content_tool_error = fx.content_tool_error;
const content_tool_json = fx.content_tool_json;
const content_tool_text = fx.content_tool_text;
const message_user = fx.message_user;
const message_system = fx.message_system;
const end_of_text = fx.end_of_text;
const content_xml = fx.content_xml;
const unused = fx.unused;
const Fake = fx.Fake;
const parse = fx.parse;
const offeredTools = fx.offeredTools;
const expectCall = fx.expectCall;

test "Inkling 2 parse: raw checkpoint sample line two splits thinking and answer without sampling EOS" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pieces = [_]Piece{
        message_model,
        content_thinking,
        plain(1, "The user asks: ... 340 + 51 = 391. ... "),
        end_message,
        message_model,
        content_text,
        plain(2, "391"),
        end_message,
    };
    const got = try parse(a, &pieces, &.{});
    try std.testing.expectEqualStrings("391", got.content);
    try std.testing.expectEqualStrings("The user asks: ... 340 + 51 = 391. ... ", got.reasoning);
    try std.testing.expectEqual(@as(usize, 4), got.reasoning_tokens);
    try std.testing.expect(got.calls == null);
    try std.testing.expect(!got.rejected);
    try std.testing.expect(!got.open_think);
}

test "Inkling 2 parse: named and unnamed repeated blocks join exactly without separators or stripping" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pieces = [_]Piece{
        message_model,
        plain(10, "first-name"),
        content_thinking,
        plain(11, "  first\n"),
        end_message,
        content_text,
        plain(12, "\nalpha"),
        end_message,
        message_model,
        plain(13, "second-name"),
        content_tool_text,
        plain(14, " β "),
        end_message,
        content_tool_error,
        plain(15, "gamma\n"),
        end_message,
        message_model,
        plain(16, "third-name"),
        content_thinking,
        plain(17, "second  "),
    };
    const got = try parse(a, &pieces, &.{});
    try std.testing.expectEqualStrings("\nalpha β gamma\n", got.content);
    try std.testing.expectEqualStrings("  first\nsecond  ", got.reasoning);
    try std.testing.expectEqual(@as(usize, 9), got.reasoning_tokens);
    try std.testing.expect(got.calls == null);
    try std.testing.expect(!got.rejected);
    try std.testing.expect(got.open_think);
}

test "Inkling 2 parse: nested scalar and escaped JSON and empty arguments become completed calls once" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const payload =
        \\{"name":"lookup","args":{"nested":{"array":[1,true,null,"x"]},"scalar":-2.5,"escaped":"quote: \" slash: \\ line:\n snowman:☃"}}
    ;
    const pieces = [_]Piece{
        message_model,
        plain(20, "a-header-name-that-is-not-the-function"),
        content_tool_json,
        plain(21, payload),
        end_message,
        content_tool_json,
        plain(22, "{\"name\":\"empty\",\"args\":{}}"),
        end_message,
    };
    const tools = try offeredTools(a, &.{ "lookup", "empty" });
    const got = try parse(a, &pieces, tools);
    try std.testing.expectEqualStrings("", got.content);
    try std.testing.expectEqualStrings("", got.reasoning);
    try std.testing.expectEqual(@as(usize, 0), got.reasoning_tokens);
    try std.testing.expect(!got.rejected);
    try std.testing.expect(!got.open_think);
    try std.testing.expectEqual(@as(usize, 2), got.calls.?.len);
    const arguments =
        \\{"nested":{"array":[1,true,null,"x"]},"scalar":-2.5,"escaped":"quote: \" slash: \\ line:\n snowman:☃"}
    ;
    try expectCall(got.calls, 0, "lookup", arguments);
    try expectCall(got.calls, 1, "empty", "{}");
}

test "Inkling 2 parse: malformed JSON envelopes are rejected without leaking their payloads" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bad_payloads = [_][]const u8{
        "",
        "{",
        "[]",
        "{}",
        "{\"name\":1,\"args\":{}}",
        "{\"name\":\"lookup\"}",
        "{\"name\":\"lookup\",\"args\":[]}",
        "{\"name\":\"lookup\",\"args\":null}",
        "{\"name\":\"lookup\",\"args\":{},}",
        "{\"name\":\"lookup\",\"args\":{}} trailing",
    };
    const tools = try offeredTools(a, &.{"lookup"});
    for (bad_payloads, 0..) |payload, n| {
        const pieces = [_]Piece{ content_tool_json, plain(@intCast(1000 + n), payload), end_message };
        const got = try parse(a, &pieces, tools);
        try std.testing.expect(got.rejected);
        try std.testing.expect(got.calls == null);
        try std.testing.expectEqualStrings("", got.content);
        try std.testing.expectEqualStrings("", got.reasoning);
        try std.testing.expect(!got.open_think);
    }
}

test "Inkling 2 parse: an unoffered name is rejected and earlier completed output survives a later bad call" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try offeredTools(a, &.{"lookup"});
    const unoffered = [_]Piece{
        content_tool_json,
        plain(30, "{\"name\":\"not-offered\",\"args\":{}}"),
        end_message,
    };
    const rejected = try parse(a, &unoffered, tools);
    try std.testing.expect(rejected.rejected);
    try std.testing.expect(rejected.calls == null);
    try std.testing.expectEqualStrings("", rejected.content);

    const pieces = [_]Piece{
        content_thinking,
        plain(31, "kept reasoning"),
        end_message,
        content_text,
        plain(32, "kept content"),
        end_message,
        content_tool_json,
        plain(33, "{\"name\":\"lookup\",\"args\":{\"x\":1}}"),
        end_message,
        content_tool_json,
        plain(34, "{\"name\":\"not-offered\",\"args\":{}}"),
        end_message,
    };
    const got = try parse(a, &pieces, tools);
    try std.testing.expect(got.rejected);
    try std.testing.expectEqualStrings("kept content", got.content);
    try std.testing.expectEqualStrings("kept reasoning", got.reasoning);
    try std.testing.expectEqual(@as(usize, 3), got.reasoning_tokens);
    try std.testing.expectEqual(@as(usize, 1), got.calls.?.len);
    try expectCall(got.calls, 0, "lookup", "{\"x\":1}");
    try std.testing.expect(!got.open_think);
}

test "Inkling 2 parse: every truncation class retains only its complete blocks and partial text channels" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct {
        pieces: []const Piece,
        content: []const u8 = "",
        reasoning: []const u8 = "",
        reasoning_tokens: usize = 0,
        rejected: bool = false,
        open_think: bool = false,
    };
    const empty = [_]Piece{};
    const header = [_]Piece{message_model};
    const named_header = [_]Piece{ message_model, plain(40, "unfinished-name") };
    const stray_text = [_]Piece{plain(41, "not-in-a-block")};
    const stray_close = [_]Piece{end_message};
    const empty_think = [_]Piece{content_thinking};
    const partial_think = [_]Piece{ message_model, content_thinking, plain(42, "partial thought") };
    const empty_text = [_]Piece{content_text};
    const partial_text = [_]Piece{ message_model, content_text, plain(43, "partial answer") };
    const partial_tool_text = [_]Piece{ content_tool_text, plain(44, "partial tool text") };
    const empty_json = [_]Piece{content_tool_json};
    const partial_json = [_]Piece{ content_tool_json, plain(45, "{\"name\":\"lookup\",\"args\":{") };
    const valid_unclosed_json = [_]Piece{ content_tool_json, plain(46, "{\"name\":\"lookup\",\"args\":{}}") };
    const completed_then_header = [_]Piece{
        content_thinking,
        plain(47, "done"),
        end_message,
        content_text,
        plain(48, "answer"),
        end_message,
        message_model,
        plain(49, "unfinished-next-header"),
    };
    const cases = [_]Case{
        .{ .pieces = &empty },
        .{ .pieces = &header },
        .{ .pieces = &named_header },
        .{ .pieces = &stray_text, .content = "not-in-a-block" }, // no header at all: a bare reply, as a grammar writes
        .{ .pieces = &stray_close },
        .{ .pieces = &empty_think, .reasoning_tokens = 1, .open_think = true },
        .{ .pieces = &partial_think, .reasoning = "partial thought", .reasoning_tokens = 3, .open_think = true },
        .{ .pieces = &empty_text },
        .{ .pieces = &partial_text, .content = "partial answer" },
        .{ .pieces = &partial_tool_text, .content = "partial tool text" },
        .{ .pieces = &empty_json, .rejected = true },
        .{ .pieces = &partial_json, .rejected = true },
        .{ .pieces = &valid_unclosed_json, .rejected = true },
        .{ .pieces = &completed_then_header, .content = "answer", .reasoning = "done", .reasoning_tokens = 3 },
    };
    const tools = try offeredTools(a, &.{"lookup"});
    for (cases) |case| {
        const got = try parse(a, case.pieces, tools);
        try std.testing.expectEqualStrings(case.content, got.content);
        try std.testing.expectEqualStrings(case.reasoning, got.reasoning);
        try std.testing.expectEqual(case.reasoning_tokens, got.reasoning_tokens);
        try std.testing.expectEqual(case.rejected, got.rejected);
        try std.testing.expectEqual(case.open_think, got.open_think);
        try std.testing.expect(got.calls == null);
    }
}

test "Inkling 2 parse: a sampling end closes nothing, so ignore_eos text goes on and open channels stay reported" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // ignore_eos keeps decoding past the sampling end: the next message is the reply's too
    const closed = [_]Piece{
        message_model, content_text, plain(50, "shown"), end_message, end_sampling,
        message_model, content_text, plain(51, " next"), end_message,
    };
    const done = try parse(a, &closed, &.{});
    try std.testing.expectEqualStrings("shown next", done.content);
    try std.testing.expect(!done.rejected);
    try std.testing.expect(!done.open_think);

    const thinking = [_]Piece{ content_thinking, plain(52, "open thought"), end_sampling };
    const open = try parse(a, &thinking, &.{});
    try std.testing.expectEqualStrings("open thought", open.reasoning);
    try std.testing.expectEqualStrings("", open.content);
    try std.testing.expect(open.open_think);
    try std.testing.expect(!open.rejected);

    const tools = try offeredTools(a, &.{"lookup"});
    const tool = [_]Piece{ content_tool_json, plain(54, "{\"name\":\"lookup\",\"args\":{}}"), end_sampling };
    const incomplete = try parse(a, &tool, tools);
    try std.testing.expect(incomplete.rejected);
    try std.testing.expectEqual(@as(usize, 1), incomplete.rejected_calls);
    try std.testing.expect(incomplete.calls == null);
    try std.testing.expectEqualStrings("", incomplete.content);
    try std.testing.expect(!incomplete.open_think);
}

test "Inkling 2 parse: literal control spellings on ordinary token ids remain payload" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pieces = [_]Piece{
        content_text,
        plain(60, "<|content_thinking|>"),
        plain(61, " and "),
        plain(62, "<|end_message|>"),
        plain(63, "<|content_model_end_sampling|>"),
        end_message,
        content_thinking,
        plain(64, "<|content_text|>"),
        end_message,
        content_text,
        plain(65, "tail"),
        end_message,
    };
    const got = try parse(a, &pieces, &.{});
    try std.testing.expectEqualStrings("<|content_thinking|> and <|end_message|><|content_model_end_sampling|>tail", got.content);
    try std.testing.expectEqualStrings("<|content_text|>", got.reasoning);
    try std.testing.expectEqual(@as(usize, 3), got.reasoning_tokens);
    try std.testing.expect(!got.rejected);
    try std.testing.expect(!got.open_think);
}

test "Inkling 2 parse: reparsing every token prefix is monotonic and never exposes incomplete JSON" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pieces = [_]Piece{
        message_model, // 0
        plain(70, "reasoner"), // 1
        content_thinking, // 2
        plain(71, "why"), // 3
        end_message, // 4
        message_model, // 5
        plain(72, "speaker"), // 6
        content_text, // 7
        plain(73, "before"), // 8
        end_message, // 9
        message_model, // 10
        plain(74, "lookup"), // 11
        content_tool_json, // 12
        plain(75, "{\"name\":\"lookup\",\"args\":{\"x\":1}}"), // 13
        end_message, // 14
        content_tool_error, // 15
        plain(76, "after"), // 16
        end_message, // 17
        end_sampling, // 18
        content_text, // 19
        plain(77, "next"), // 20: past the sampling end, as ignore_eos decodes it
        end_message, // 21
    };
    const tools = try offeredTools(a, &.{"lookup"});
    var previous_content: []const u8 = "";
    var previous_reasoning: []const u8 = "";
    var previous_reasoning_tokens: usize = 0;
    var previous_calls: usize = 0;
    for (0..pieces.len + 1) |n| {
        const got = try parse(a, pieces[0..n], tools);
        try std.testing.expect(std.mem.startsWith(u8, got.content, previous_content));
        try std.testing.expect(std.mem.startsWith(u8, got.reasoning, previous_reasoning));
        try std.testing.expect(got.reasoning_tokens >= previous_reasoning_tokens);
        const call_count = if (got.calls) |calls| calls.len else 0;
        try std.testing.expect(call_count >= previous_calls);
        try std.testing.expectEqual(@as(usize, if (n >= 15) 1 else 0), call_count);
        try std.testing.expectEqual(@as(usize, if (n < 3) 0 else @min(n, 5)), got.reasoning_tokens);
        try std.testing.expectEqual(n == 3 or n == 4, got.open_think);
        try std.testing.expectEqual(n == 13 or n == 14, got.rejected);
        if (call_count > 0) try expectCall(got.calls, 0, "lookup", "{\"x\":1}");
        try std.testing.expect(std.mem.indexOf(u8, got.content, "{\"name\"") == null);
        previous_content = got.content;
        previous_reasoning = got.reasoning;
        previous_reasoning_tokens = got.reasoning_tokens;
        previous_calls = call_count;
    }
    const final = try parse(a, &pieces, tools);
    try std.testing.expectEqualStrings("beforeafternext", final.content);
    try std.testing.expectEqualStrings("why", final.reasoning);
    try std.testing.expectEqual(@as(usize, 5), final.reasoning_tokens);
    try std.testing.expectEqual(@as(usize, 1), final.calls.?.len);
    try std.testing.expect(!final.rejected);
    try std.testing.expect(!final.open_think);
}

test "Inkling 2 parse: the tokenizer's other special tokens inside a block are never payload, whole or streamed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pieces = [_]Piece{
        message_model,  content_thinking, plain(70, "think"), end_of_text, plain(71, "ing"), end_message,
        message_model,  content_text,     plain(72, "A"),     unused,      plain(73, "B"),   end_of_text,
        plain(74, "C"), content_xml,      plain(75, "D"),     unused,      end_message,
    };
    const whole = try parse(a, &pieces, &.{});
    try std.testing.expectEqualStrings("thinking", whole.reasoning);
    try std.testing.expectEqualStrings("ABCD", whole.content);
    var fake: Fake = .{ .pieces = &pieces };
    var stream = try reply.Stream.init(a, fake.text());
    var ids: [pieces.len]u32 = undefined;
    for (&pieces, &ids) |piece, *id| id.* = piece.id;
    var last: reply.Result = undefined;
    for (1..ids.len + 1) |n| last = try stream.extend(ids[0..n], &.{});
    try std.testing.expectEqualStrings("thinking", last.reasoning);
    try std.testing.expectEqualStrings("ABCD", last.content);
}

test "Inkling 2 parse: a new message inside an unclosed block closes it, so its tool call parses" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try offeredTools(a, &.{"lookup"});
    const pieces = [_]Piece{
        message_model,                                         content_text,        plain(80, "Hi"),
        message_model,                                         plain(81, "lookup"), content_tool_json,
        plain(82, "{\"name\":\"lookup\",\"args\":{\"q\":1}}"), end_message,
    };
    const got = try parse(a, &pieces, tools);
    try std.testing.expectEqualStrings("Hi", got.content);
    try expectCall(got.calls, 0, "lookup", "{\"q\":1}");
    try std.testing.expect(!got.rejected);
    // a call left open by the next message never finished: refused and counted, its payload never shown
    const open_call = [_]Piece{
        message_model, content_tool_json, plain(83, "{\"name\":\"lookup\",\"args\":{}}"),
        message_model, content_text,      plain(84, "after"),
        end_message,
    };
    const cut = try parse(a, &open_call, tools);
    try std.testing.expectEqualStrings("after", cut.content);
    try std.testing.expect(cut.calls == null);
    try std.testing.expectEqual(@as(usize, 1), cut.rejected_calls);
}

test "Inkling 2 parse: every refused call is counted: unoffered, malformed and unfinished" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try offeredTools(a, &.{"lookup"});
    const pieces = [_]Piece{
        message_model, content_tool_json, plain(90, "{\"name\":\"other\",\"args\":{}}"),  end_message,
        message_model, content_tool_json, plain(91, "{\"name\":"),                        end_message,
        message_model, content_tool_json, plain(92, "{\"name\":\"lookup\",\"args\":{}}"), end_message,
        message_model, content_tool_json, plain(93, "{\"name\":\"lookup\""),
    };
    const got = try parse(a, &pieces, tools);
    try std.testing.expectEqual(@as(usize, 3), got.rejected_calls);
    try expectCall(got.calls, 0, "lookup", "{}");
}

test "Inkling 2 parse: a stop string costs streamed frames no whole-reply decode" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var pieces: [3 + 64]Piece = undefined;
    pieces[0] = message_model;
    pieces[1] = content_text;
    for (pieces[2 .. pieces.len - 1], 0..) |*piece, i| piece.* = plain(@intCast(100 + i), "x");
    pieces[pieces.len - 1] = end_message;
    var fake: Fake = .{ .pieces = &pieces };
    const Srv = struct { text: model_text.Text };
    const srv: Srv = .{ .text = fake.text() };
    var g: struct {
        a: Allocator,
        chat: bool = true,
        inkling2: ?reply.Stream = null,
        tools: []const Value = &.{},
        srv: *const Srv,
        stops: @import("reply_text.zig").Stops,
        visible: @import("reply_text.zig").Incremental = .{},
    } = .{ .a = a, .srv = &srv, .stops = .{ .strings = &.{"STOP"} } };
    g.inkling2 = try reply.Stream.init(a, srv.text);
    var ids: [pieces.len]u32 = undefined;
    for (&pieces, &ids) |piece, *id| id.* = piece.id;
    fx.decoded_ids = 0;
    for (1..ids.len + 1) |n| _ = try @import("chat_inkling2.zig").read(&g, ids[0..n], false);
    const done = try @import("chat_inkling2.zig").read(&g, &ids, true);
    try std.testing.expectEqual(@as(usize, 64), done.content.len);
    // each token decoded once by the parser and once for the finished reply's stop check: linear, not quadratic
    try std.testing.expect(fx.decoded_ids <= 2 * ids.len);
}
