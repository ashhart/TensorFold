//! Pure host coverage for Inkling 2 native reply splitting.
const std = @import("std");
const json = @import("json");
const model_text = @import("model_text.zig");
const reply = @import("inkling2_reply.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

const Piece = struct {
    id: u32,
    text: []const u8,
    added: bool = false,
};

fn plain(id: u32, text: []const u8) Piece {
    return .{ .id = id, .text = text };
}

const message_model = Piece{ .id = 200001, .text = "<|message_model|>", .added = true };
const content_text = Piece{ .id = 200004, .text = "<|content_text|>", .added = true };
const end_sampling = Piece{ .id = 200006, .text = "<|content_model_end_sampling|>", .added = true };
const content_thinking = Piece{ .id = 200008, .text = "<|content_thinking|>", .added = true };
const end_message = Piece{ .id = 200010, .text = "<|end_message|>", .added = true };
const content_tool_error = Piece{ .id = 200022, .text = "<|content_tool_error|>", .added = true };
const content_tool_json = Piece{ .id = 200049, .text = "<|content_invoke_tool_json|>", .added = true };
const content_tool_text = Piece{ .id = 200057, .text = "<|content_invoke_tool_text|>", .added = true };

const Fake = struct {
    pieces: []const Piece,

    fn text(f: *Fake) model_text.Text {
        return .{ .ctx = f, .vtable = &.{
            .encode = encode,
            .decode = decode,
            .token_id = tokenId,
            .token_string = tokenString,
            .vocab_size = vocabSize,
            .eos_ids = eosIds,
            .render = render,
            .template_source = templateSource,
            .added = added,
        } };
    }

    fn find(f: *const Fake, id: u32) ?Piece {
        for (f.pieces) |piece| if (piece.id == id) return piece;
        return null;
    }

    fn encode(ctx: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        for (f.pieces) |piece| if (std.mem.eql(u8, input, piece.text)) {
            const ids = try a.alloc(u32, 1);
            ids[0] = piece.id;
            return ids;
        };
        return error.Template;
    }

    fn decode(ctx: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        for (ids) |id| {
            const piece = f.find(id) orelse std.debug.panic("unknown fake token {d}", .{id});
            try out.appendSlice(a, piece.text);
        }
        return out.toOwnedSlice(a);
    }

    fn tokenId(ctx: *anyopaque, spelling: []const u8) ?u32 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        for (f.pieces) |piece| if (piece.added and std.mem.eql(u8, spelling, piece.text)) return piece.id;
        return null;
    }

    fn tokenString(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        const piece = f.find(id) orelse return a.dupe(u8, "");
        return a.dupe(u8, piece.text);
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 200058;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{200006};
    }

    fn added(ctx: *anyopaque, id: u32) bool {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        const piece = f.find(id) orelse return false;
        return piece.added;
    }

    fn render(_: *anyopaque, _: Allocator, _: Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return error.Template;
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

fn parse(a: Allocator, pieces: []const Piece, tools: []const Value) !reply.Result {
    var fake: Fake = .{ .pieces = pieces };
    const ids = try a.alloc(u32, pieces.len);
    for (pieces, ids) |piece, *id| id.* = piece.id;
    return reply.parse(a, fake.text(), ids, tools);
}

fn offeredTools(a: Allocator, names: []const []const u8) ![]Value {
    const tools = try a.alloc(Value, names.len);
    for (names, tools) |name, *tool| {
        const function = try json.newObject(a);
        try function.put(a, "name", .{ .string = name });
        const object = try json.newObject(a);
        try object.put(a, "type", .{ .string = "function" });
        try object.put(a, "function", .{ .object = function });
        tool.* = .{ .object = object };
    }
    return tools;
}

fn expectCall(calls: ?[]Value, index: usize, name: []const u8, arguments: []const u8) !void {
    try std.testing.expect(calls != null);
    const all = calls.?;
    try std.testing.expect(all.len > index);
    const call = all[index];
    var id_buffer: [32]u8 = undefined;
    const id = try std.fmt.bufPrint(&id_buffer, "call_{d}", .{index});
    try std.testing.expectEqualStrings(id, call.get("id").?.string);
    try std.testing.expectEqualStrings("function", call.get("type").?.string);
    const function = call.get("function").?;
    try std.testing.expectEqualStrings(name, function.get("name").?.string);
    try std.testing.expectEqualStrings(arguments, function.get("arguments").?.string);
}

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
    try std.testing.expectEqual(@as(usize, 1), got.reasoning_tokens);
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
    try std.testing.expectEqual(@as(usize, 2), got.reasoning_tokens);
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
    try std.testing.expectEqual(@as(usize, 1), got.reasoning_tokens);
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
        .{ .pieces = &stray_text },
        .{ .pieces = &stray_close },
        .{ .pieces = &empty_think, .open_think = true },
        .{ .pieces = &partial_think, .reasoning = "partial thought", .reasoning_tokens = 1, .open_think = true },
        .{ .pieces = &empty_text },
        .{ .pieces = &partial_text, .content = "partial answer" },
        .{ .pieces = &partial_tool_text, .content = "partial tool text" },
        .{ .pieces = &empty_json, .rejected = true },
        .{ .pieces = &partial_json, .rejected = true },
        .{ .pieces = &valid_unclosed_json, .rejected = true },
        .{ .pieces = &completed_then_header, .content = "answer", .reasoning = "done", .reasoning_tokens = 1 },
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

test "Inkling 2 parse: sampling end terminates output and still reports an actually unclosed channel" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const closed = [_]Piece{
        content_text,
        plain(50, "shown"),
        end_message,
        end_sampling,
        content_text,
        plain(51, "hidden"),
        end_message,
    };
    const done = try parse(a, &closed, &.{});
    try std.testing.expectEqualStrings("shown", done.content);
    try std.testing.expect(!done.rejected);
    try std.testing.expect(!done.open_think);

    const thinking = [_]Piece{ content_thinking, plain(52, "open thought"), end_sampling, content_text, plain(53, "hidden") };
    const open = try parse(a, &thinking, &.{});
    try std.testing.expectEqualStrings("open thought", open.reasoning);
    try std.testing.expectEqualStrings("", open.content);
    try std.testing.expect(open.open_think);
    try std.testing.expect(!open.rejected);

    const tools = try offeredTools(a, &.{"lookup"});
    const tool = [_]Piece{ content_tool_json, plain(54, "{\"name\":\"lookup\",\"args\":{}}"), end_sampling, content_text, plain(55, "hidden") };
    const incomplete = try parse(a, &tool, tools);
    try std.testing.expect(incomplete.rejected);
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
    try std.testing.expectEqual(@as(usize, 1), got.reasoning_tokens);
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
        plain(77, "ignored"), // 20
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
        try std.testing.expectEqual(@as(usize, if (n >= 4) 1 else 0), got.reasoning_tokens);
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
    try std.testing.expectEqualStrings("beforeafter", final.content);
    try std.testing.expectEqualStrings("why", final.reasoning);
    try std.testing.expectEqual(@as(usize, 1), final.reasoning_tokens);
    try std.testing.expectEqual(@as(usize, 1), final.calls.?.len);
    try std.testing.expect(!final.rejected);
    try std.testing.expect(!final.open_think);
}
