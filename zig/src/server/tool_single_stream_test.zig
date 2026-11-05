//! #1396: with parallel_tool_calls false a stream sends the whole reply's content, calls, arguments and finish reason.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const pre = @import("stream_preflight_test.zig");
const Allocator = std.mem.Allocator;

/// Bytes as tokens; every prompt renders as "prompt" (6 tokens), as the preflight harness's text does.
const ByteText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const decoded = try a.alloc(u8, ids.len);
        for (ids, decoded) |id, *byte| byte.* = @intCast(id);
        return decoded;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try (a.dupe(u8, "prompt")) };
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// What a client got: a whole reply's status, or a stream's opening, with its content, calls, arguments and finish.
const Got = struct {
    opened: bool = false,
    status: ?u16 = null,
    calls: usize = 0,
    content: std.ArrayList(u8) = .empty,
    chunks: std.ArrayList([]const u8) = .empty, // every content delta as sent, in order
    arguments: std.ArrayList(u8) = .empty,
    finish: [16]u8 = undefined,
    finish_len: usize = 0,
    content_at_call: ?usize = null, // the content a stream had sent when its first call delta arrived

    fn deinit(g: *Got) void {
        for (g.chunks.items) |c| std.testing.allocator.free(c);
        g.chunks.deinit(std.testing.allocator);
        g.content.deinit(std.testing.allocator);
        g.arguments.deinit(std.testing.allocator);
    }

    fn reason(g: *const Got) []const u8 {
        return g.finish[0..g.finish_len];
    }

    fn keepFinish(g: *Got, choice: json.Value) void {
        const f = choice.strField("finish_reason") orelse return;
        g.finish_len = @min(f.len, g.finish.len);
        @memcpy(g.finish[0..g.finish_len], f[0..g.finish_len]);
    }

    fn keepCall(g: *Got, call: json.Value) error{Closed}!void {
        const f = call.get("function") orelse return;
        const args = f.strField("arguments") orelse return;
        g.arguments.appendSlice(std.testing.allocator, args) catch return error.Closed;
    }

    fn out(g: *Got) openai.Out {
        return .{ .ctx = g, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.opened = true;
    }

    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        const p = payload orelse return;
        const choices = p.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        const choice = choices.array[0];
        g.keepFinish(choice);
        const delta = choice.get("delta") orelse return;
        if (delta.get("content")) |t| if (t == .string) {
            g.content.appendSlice(std.testing.allocator, t.string) catch return error.Closed;
            const one = std.testing.allocator.dupe(u8, t.string) catch return error.Closed;
            g.chunks.append(std.testing.allocator, one) catch return error.Closed;
        };
        if (delta.get("tool_calls")) |t| if (t == .array) for (t.array) |call| {
            if (call.get("id") != null) {
                g.calls += 1;
                if (g.content_at_call == null) g.content_at_call = g.content.items.len;
            }
            try g.keepCall(call);
        };
    }

    fn reply(ctx: *anyopaque, status: u16, payload: json.Value) void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.status = status;
        const choices = payload.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        const choice = choices.array[0];
        g.keepFinish(choice);
        const message = choice.get("message") orelse return;
        if (message.get("content")) |t| if (t == .string) g.content.appendSlice(std.testing.allocator, t.string) catch {};
        if (message.get("tool_calls")) |t| if (t == .array) {
            g.calls += t.array.len;
            for (t.array) |call| g.keepCall(call) catch {};
        };
    }
};

/// The reply to ``body``: whole, or a stream whose content, calls, arguments and finish the sink keeps.
fn send(engine: api.Engine, body: []const u8) !Got {
    var text: ByteText = .{};
    var srv = try Server.init(std.testing.allocator, std.testing.io, engine, text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = (try json.parse(a, body)).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var got: Got = .{};
    errdefer got.deinit();
    openai.run(srv, a, got.out(), .{ .conn = &conn }, true, raw);
    return got;
}

const head = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"q\":{\"type\":\"string\"}}}}}]";

/// A reply, the tokens its stream is cut at, and what both replies must send.
const Case = struct {
    reply: []const u8,
    cut: ?usize = null,
    content: ?[]const u8 = null,
    calls: usize = 0,
    arguments: []const u8 = "",
    finish: []const u8 = "stop",
    prose_before_calls: ?usize = null,
    single: bool = true,
};

/// ``c`` whole and streamed, batched and not: both replies agree with each other and with ``c``.
fn expectAgrees(c: Case) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tokens = try pre.TestText.encode(undefined, a, c.reply, false);
    const body = try std.fmt.allocPrint(a, "{s}{s},\"max_tokens\":{d}", .{ head, if (c.single) ",\"parallel_tool_calls\":false" else "", c.cut orelse 2000 });
    for ([_]bool{ false, true }) |batched| {
        var backend: pre.CallEngine = .{ .tokens = tokens, .batched = batched };
        var whole = try send(backend.engine(), try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }));
        defer whole.deinit();
        var streamed = try send(backend.engine(), try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }));
        defer streamed.deinit();
        try std.testing.expectEqual(@as(?u16, 200), whole.status);
        try std.testing.expect(streamed.opened and streamed.status == null);
        if (c.content) |content| try std.testing.expectEqualStrings(content, whole.content.items);
        try std.testing.expectEqual(c.calls, whole.calls);
        try std.testing.expectEqualStrings(c.arguments, whole.arguments.items);
        try std.testing.expectEqualStrings(c.finish, whole.reason());
        try std.testing.expectEqualStrings(whole.content.items, streamed.content.items);
        try std.testing.expectEqual(whole.calls, streamed.calls);
        try std.testing.expectEqualStrings(whole.arguments.items, streamed.arguments.items);
        try std.testing.expectEqualStrings(whole.reason(), streamed.reason());
        if (c.prose_before_calls) |n| try std.testing.expectEqual(n, streamed.content_at_call.?);
    }
}

test "single: a rejected whole envelope stays in the streamed content as in the whole reply" {
    const named = "Before.<foo:tool_call>{\"name\":\"other\",\"arguments\":{}}</foo:tool_call>Tail.";
    const upper = "Before.<TOOL_CALL>{\"name\":\"other\",\"arguments\":{}}</TOOL_CALL>Tail.";
    const mixed = "Before.<Tool_Call>{\"name\":\"other\",\"arguments\":{}}</Tool_Call>Tail.";
    try expectAgrees(.{ .reply = named, .content = named });
    try expectAgrees(.{ .reply = upper, .content = upper });
    try expectAgrees(.{ .reply = mixed, .content = mixed });
    const invalid = "Before.<tool_call>{invalid}</tool_call>Tail.";
    try expectAgrees(.{ .reply = invalid, .content = invalid });
}

test "single: a cut reply ends its prose as the whole reply does" {
    const first = "<tool_call>{\"name\":\"other\",\"arguments\":{}}</tool_call>Between.";
    const cut = first ++ "<tool_call>lookup<arg_key>q</arg_key><arg_value>abc";
    try expectAgrees(.{ .reply = cut, .cut = cut.len, .content = first, .finish = "length" });
    const progress = "\nWriting it.\n";
    const xml = progress ++ "<tool_call>\n<function=lookup>\n<parameter=q>\nThe histor";
    const glm = progress ++ "<tool_call>lookup<arg_key>q</arg_key><arg_value>The histor";
    try expectAgrees(.{ .reply = xml ++ "y", .cut = xml.len, .content = progress, .finish = "length" });
    try expectAgrees(.{ .reply = glm ++ "y", .cut = glm.len, .content = progress, .finish = "length" });
}

test "an unclosed DSML block stays in both replies, with or without a trailing newline, in either mode" {
    const spaced = "Done.<\u{ff5c}DSML\u{ff5c} calls><\u{ff5c}DSML\u{ff5c} invoke name=\"lookup\"><\u{ff5c}DSML\u{ff5c} parameter name=\"q\" string=\"true\">x";
    const compact = "Done.<\u{ff5c}DSML\u{ff5c}tool_calls><\u{ff5c}DSML\u{ff5c}invoke name=\"lookup\"><\u{ff5c}DSML\u{ff5c}parameter name=\"q\" string=\"true\">x";
    inline for ([_]bool{ true, false }) |single| {
        // a single-call reply's content is the parse's and stripped; with many calls it is the reply's own text
        const end = if (single) "" else "\n";
        try expectAgrees(.{ .reply = spaced, .content = spaced, .single = single });
        try expectAgrees(.{ .reply = compact, .content = compact, .single = single });
        try expectAgrees(.{ .reply = spaced ++ "\n", .content = spaced ++ end, .single = single });
        try expectAgrees(.{ .reply = compact ++ "\n", .content = compact ++ end, .single = single });
    }
}

test "single: an accepted call's prose streams before it and its tail after it" {
    const before = "Before.";
    const tail = "Tail.";
    const call = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"abc\"}}</tool_call>";
    try expectAgrees(.{
        .reply = before ++ call ++ tail,
        .content = before ++ tail,
        .calls = 1,
        .arguments = "{\"q\":\"abc\"}",
        .finish = "tool_calls",
        .prose_before_calls = before.len,
    });
}

/// The reply to ``reply`` whole: the content a stream's joined deltas must equal.
fn wholeReply(a: Allocator, reply: []const u8) !Got {
    const tokens = try pre.TestText.encode(undefined, a, reply, false);
    var backend: pre.CallEngine = .{ .tokens = tokens };
    const body = try std.fmt.allocPrint(a, "{s},\"max_tokens\":2000", .{head});
    return send(backend.engine(), try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }));
}

/// What one stream sends for ``reply``: default mode, or one-call mode with ``single``, batched or per token.
fn streamedReply(a: Allocator, reply: []const u8, single: bool, batched: bool) !Got {
    const tokens = try pre.TestText.encode(undefined, a, reply, false);
    var backend: pre.CallEngine = .{ .tokens = tokens, .batched = batched };
    const body = try std.fmt.allocPrint(a, "{s}{s},\"max_tokens\":2000", .{ head, if (single) ",\"parallel_tool_calls\":false" else "" });
    return send(backend.engine(), try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }));
}

/// The content chunks one stream sent, in order.
fn expectChunks(want: []const []const u8, got: []const []const u8) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |one, seen| try std.testing.expectEqualStrings(one, seen);
}

test "a call-free reply ending in a newline streams the whole reply's content, chunk for chunk" {
    const reply = "abc\n";
    // a per-token engine sends the text byte by byte and then the newline; batched, the text and then its newline
    const byte_by_byte = [_][]const u8{ "a", "b", "c", "\n" };
    const in_one_chunk = [_][]const u8{ "abc", "\n" };
    for ([_]bool{ false, true }) |batched| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var whole = try wholeReply(a, reply);
        defer whole.deinit();
        try std.testing.expectEqualStrings(reply, whole.content.items);
        for ([_]bool{ false, true }) |single| { // one-call mode sends this reply in the same chunks
            var streamed = try streamedReply(a, reply, single, batched);
            defer streamed.deinit();
            const want: []const []const u8 = if (batched) in_one_chunk[0..] else byte_by_byte[0..];
            try expectChunks(want, streamed.chunks.items);
            try std.testing.expectEqualStrings(whole.content.items, streamed.content.items);
        }
    }
}
