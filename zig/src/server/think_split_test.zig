//! The reasoning split ends at the model's own close token: a `</think>` the reply quotes in pieces stays reasoning.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const Conn = @import("http_conn.zig").Conn;

const think_end: u32 = 300; // the close marker as one token
const eos: u32 = 301;
const quote_open: u32 = 302; // the quoted tag, in the three pieces a reply writes it
const quote_mid: u32 = 303;
const quote_close: u32 = 304;
const all_bytes = blk: {
    var all: [256]u8 = undefined;
    for (&all, 0..) |*b, i| b.* = i;
    break :blk all;
};

/// Bytes as tokens, `</think>` as token 300 and an end token 301; every prompt renders as "prompt".
const Text = struct {
    real: ?*@import("tokenizer").Tokenizer = null,
    close_token: bool = true, // false: the close marker is not a token, so the text's spelling is what closes

    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .token_bytes = tokenBytes } };
    }

    /// The close marker is one token; the quoted tag is three, so the text spells the same and the tokens differ.
    fn piece(id: u32) []const u8 {
        if (id == think_end) return "</think>";
        if (id == eos) return "";
        if (id == quote_open) return "</";
        if (id == quote_mid) return "think";
        if (id == quote_close) return ">";
        return all_bytes[id..][0..1];
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(ctx: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const t: *Text = @ptrCast(@alignCast(ctx));
        if (t.real) |tok| return tok.decode(a, ids, false) catch return error.OutOfMemory;
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| try out.appendSlice(a, piece(id));
        return out.items;
    }

    fn tokenBytes(ctx: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error!?[]u8 {
        const t: *Text = @ptrCast(@alignCast(ctx));
        if (t.real != null) return null;
        return try a.dupe(u8, piece(id));
    }

    fn tokenId(ctx: *anyopaque, p: []const u8) ?u32 {
        const t: *Text = @ptrCast(@alignCast(ctx));
        if (t.real) |tok| return tok.specialTokenId(p);
        if (!t.close_token) return null;
        return if (std.mem.eql(u8, p, "</think>")) think_end else null;
    }

    fn tokenString(ctx: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return decode(ctx, a, &.{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 305;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{eos};
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try (a.dupe(u8, "prompt")) };
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Answers `reply` in fixed token chunks.
const Script = struct {
    reply: []const u32,
    stride: ?usize = null,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 64 };
    }

    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Script = @ptrCast(@alignCast(ctx));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        const width = e.stride orelse @max(e.reply.len / 2, 1);
        var start: usize = 0;
        while (start < e.reply.len) {
            const end = @min(start + width, e.reply.len);
            sink.event(sink.ctx, id, &.{ .tokens = e.reply[start..end] });
            start = end;
        }
        const ended = e.reply.len > 0 and e.reply[e.reply.len - 1] == eos;
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = if (ended) .stop else .length } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The reply the route wrote: its status and body, or a stream's events, as JSON text (testing-allocator owned).
const Got = struct {
    status: u16 = 0,
    body: []u8 = &.{},
    events: std.ArrayList([]u8) = .empty,

    fn out(g: *Got) openai.Out {
        return .{ .ctx = g, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn deinit(g: *Got) void {
        std.testing.allocator.free(g.body);
        for (g.events.items) |e| std.testing.allocator.free(e);
        g.events.deinit(std.testing.allocator);
    }

    fn open(_: *anyopaque, _: usize) error{Closed}!void {}

    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const text = json.stringify(arena.allocator(), payload orelse return, .{ .ascii = false }) catch return error.Closed;
        g.events.append(std.testing.allocator, std.testing.allocator.dupe(u8, text) catch return error.Closed) catch return error.Closed;
    }

    fn reply(ctx: *anyopaque, status: u16, payload: json.Value) void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.status = status;
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const text = json.stringify(arena.allocator(), payload, .{ .ascii = false }) catch return;
        g.body = std.testing.allocator.dupe(u8, text) catch return;
    }
};

fn send(reply: []const u32, thinking: bool, body: []const u8) !Got {
    return sendWith(reply, thinking, body, true);
}

fn sendWith(reply: []const u32, thinking: bool, body: []const u8, close_token: bool) !Got {
    return sendText(reply, thinking, body, .{ .close_token = close_token }, null);
}

fn sendText(reply: []const u32, thinking: bool, body: []const u8, settings: Text, stride: ?usize) !Got {
    var text = settings;
    var backend: Script = .{ .reply = reply, .stride = stride };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = thinking,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = (try json.parse(a, body)).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var got: Got = .{};
    openai.run(srv, a, got.out(), .{ .conn = &conn }, true, raw);
    return got;
}

fn tokensOf(comptime s: []const u8) [s.len]u32 {
    var out: [s.len]u32 = undefined;
    for (s, &out) |c, *o| o.* = c;
    return out;
}

/// The reply's message: its content and its reasoning.
fn message(got: *Got) !std.json.Parsed(std.json.Value) {
    try std.testing.expectEqual(@as(u16, 200), got.status);
    return std.json.parseFromSlice(std.json.Value, std.testing.allocator, got.body, .{});
}

const ask = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":32";

/// The reasoning quotes the tag in three pieces, then the model closes with its own close token.
const quoted = tokensOf("I will ") ++ [_]u32{ quote_open, quote_mid, quote_close } ++ tokensOf(" say it") ++
    [_]u32{think_end} ++ tokensOf(" done") ++ [_]u32{eos};

test "the reasoning ends at the close token, so a quoted tag stays reasoning" {
    var got = try send(&quoted, true, ask ++ "}");
    defer got.deinit();
    const parsed = try message(&got);
    defer parsed.deinit();
    const m = parsed.value.object.get("choices").?.array.items[0].object.get("message").?.object;
    try std.testing.expectEqualStrings("I will </think> say it", m.get("reasoning_content").?.string);
    try std.testing.expectEqualStrings(" done", m.get("content").?.string);
}

test "a tokenizer without a close token keeps the text split" {
    // the close marker is not one token here, so the text's first spelling of it ends the reasoning
    const spelled = tokensOf("I will </think> say it done") ++ [_]u32{eos};
    var got = try sendWith(&spelled, true, ask ++ "}", false);
    defer got.deinit();
    const parsed = try message(&got);
    defer parsed.deinit();
    const m = parsed.value.object.get("choices").?.array.items[0].object.get("message").?.object;
    try std.testing.expectEqualStrings("I will", m.get("reasoning_content").?.string);
    try std.testing.expectEqualStrings(" say it done", m.get("content").?.string);
}

test "the stream's reasoning and content end at the close token too" {
    var got = try send(&quoted, true, ask ++ ",\"stream\":true}");
    defer got.deinit();
    var reasoning: std.ArrayList(u8) = .empty;
    defer reasoning.deinit(std.testing.allocator);
    var content: std.ArrayList(u8) = .empty;
    defer content.deinit(std.testing.allocator);
    for (got.events.items) |e| {
        const chunk = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, e, .{});
        defer chunk.deinit();
        const choices = chunk.value.object.get("choices") orelse continue;
        if (choices.array.items.len == 0) continue;
        const delta = choices.array.items[0].object.get("delta").?.object;
        if (delta.get("reasoning_content")) |r| try reasoning.appendSlice(std.testing.allocator, r.string);
        if (delta.get("content")) |c| try content.appendSlice(std.testing.allocator, c.string);
    }
    try std.testing.expectEqualStrings("I will </think> say it", reasoning.items);
    try std.testing.expectEqualStrings(" done", content.items);
}

const wordpiece_json =
    \\{"model":{"type":"WordPiece","vocab":{"Plan":0,"</think>":1,"Answer":2,"[UNK]":3,"</":4,"##think":5,"##>":6,"still":7},"unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100},"decoder":{"type":"WordPiece","prefix":"##","cleanup":false}}
;
const wordpiece_cases = [_]struct { ids: []const u32, reasoning: []const u8 }{
    .{ .ids = &.{ 0, 1, 2, eos }, .reasoning = "Plan" },
    .{ .ids = &.{ 0, 4, 5, 6, 7, 1, 2, eos }, .reasoning = "Plan </think> still" },
};

test "WordPiece whole replies keep the answer after a decoder inserts space before the close token" {
    var tok = try @import("tokenizer").parse(std.testing.allocator, wordpiece_json);
    defer tok.deinit();
    for (wordpiece_cases) |case| {
        var got = try sendText(case.ids, true, ask ++ "}", .{ .real = &tok }, null);
        defer got.deinit();
        const parsed = try message(&got);
        defer parsed.deinit();
        const m = parsed.value.object.get("choices").?.array.items[0].object.get("message").?.object;
        try std.testing.expectEqualStrings(case.reasoning, m.get("reasoning_content").?.string);
        try std.testing.expectEqualStrings(" Answer", m.get("content").?.string);
    }
}

test "WordPiece streams keep the answer after a decoder inserts space before the close token" {
    const a = std.testing.allocator;
    var tok = try @import("tokenizer").parse(a, wordpiece_json);
    defer tok.deinit();
    for (wordpiece_cases) |case| {
        for ([_]usize{ 1, 2, 256 }) |stride| {
            var got = try sendText(case.ids, true, ask ++ ",\"stream\":true}", .{ .real = &tok }, stride);
            defer got.deinit();
            var reasoning: std.ArrayList(u8) = .empty;
            defer reasoning.deinit(a);
            var content: std.ArrayList(u8) = .empty;
            defer content.deinit(a);
            for (got.events.items) |e| {
                const chunk = try std.json.parseFromSlice(std.json.Value, a, e, .{});
                defer chunk.deinit();
                const choices = chunk.value.object.get("choices") orelse continue;
                if (choices.array.items.len == 0) continue;
                const delta = choices.array.items[0].object.get("delta").?.object;
                if (delta.get("reasoning_content")) |v| try reasoning.appendSlice(a, v.string);
                if (delta.get("content")) |v| try content.appendSlice(a, v.string);
            }
            try std.testing.expectEqualStrings(case.reasoning, std.mem.trimEnd(u8, reasoning.items, " "));
            try std.testing.expectEqualStrings(" Answer", content.items);
        }
    }
}
