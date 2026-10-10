//! `logprobs` on /v1/chat/completions: rows cover the answer as `content` does, and the 0.6.6 refusals hold.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const responses_translate = @import("responses_translate.zig");
const errors = @import("errors.zig");
const Conn = @import("http_conn.zig").Conn;

const think_end: u32 = 300;
const eos: u32 = 301;
const all_bytes = blk: {
    var all: [256]u8 = undefined;
    for (&all, 0..) |*b, i| b.* = i;
    break :blk all;
};

/// Bytes as tokens, `</think>` as token 300 and an end token 301; every prompt renders as "prompt".
const Text = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .token_bytes = tokenBytes } };
    }

    fn piece(id: u32) []const u8 {
        if (id == think_end) return "</think>";
        if (id == eos) return "";
        return all_bytes[id..][0..1];
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| try out.appendSlice(a, piece(id));
        return out.items;
    }

    fn tokenBytes(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error!?[]u8 {
        return try a.dupe(u8, piece(id));
    }

    fn tokenId(_: *anyopaque, p: []const u8) ?u32 {
        return if (std.mem.eql(u8, p, "</think>")) think_end else null;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return a.dupe(u8, piece(id));
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 302;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{eos};
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Answers `reply` in two rounds with rows: token i logprob -(i + 1) / 4, best tokens itself then 'z'.
const Script = struct {
    reply: []const u32,
    asked: ?u8 = null,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 64, .logprobs = true };
    }

    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Script = @ptrCast(@alignCast(ctx));
        e.asked = r.logprobs;
        var rows: [32]api.LogprobRow = undefined;
        for (e.reply, rows[0..e.reply.len], 0..) |t, *row, i| {
            const lp = -@as(f32, @floatFromInt(i + 1)) / 4;
            row.* = .{ .token = t, .logprob = lp, .count = r.logprobs orelse 0 };
            for (0..row.count) |k| {
                row.ids[k] = if (k == 0) t else 'z';
                row.logprobs[k] = lp - @as(f32, @floatFromInt(k));
            }
        }
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        const half = e.reply.len / 2;
        for ([_][2]usize{ .{ 0, half }, .{ half, e.reply.len } }) |part| {
            if (r.logprobs != null) sink.event(sink.ctx, id, &.{ .logprobs = rows[part[0]..part[1]] });
            sink.event(sink.ctx, id, &.{ .tokens = e.reply[part[0]..part[1]] });
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

/// The reply the route wrote: its status and body as JSON text (testing-allocator owned).
const Got = struct {
    status: u16 = 0,
    body: []u8 = &.{},

    fn out(g: *Got) openai.Out {
        return .{ .ctx = g, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(_: *anyopaque) error{Closed}!void {}

    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}

    fn reply(ctx: *anyopaque, status: u16, payload: json.Value) void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.status = status;
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const text = json.stringify(arena.allocator(), payload, .{ .ascii = false }) catch return;
        g.body = std.testing.allocator.dupe(u8, text) catch return;
    }
};

fn send(reply: []const u32, thinking: bool, responses: bool, body: []const u8) !Got {
    var text: Text = .{};
    var backend: Script = .{ .reply = reply };
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
    var cx: errors.Cx = .{ .a = a };
    const chat = if (responses) (try responses_translate.translate(&cx, raw, NoHistory{})).chat else raw;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var got: Got = .{};
    openai.run(srv, a, got.out(), .{ .conn = &conn }, true, chat);
    return got;
}

const NoHistory = struct {
    pub fn conversation(_: NoHistory, cx: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
        return cx.refuse("no stored responses in this test");
    }
};

const ask = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":32,\"logprobs\":true,\"top_logprobs\":2";

fn tokensOf(comptime s: []const u8) [s.len]u32 {
    var out: [s.len]u32 = undefined;
    for (s, &out) |c, *o| o.* = c;
    return out;
}

/// The reply's `logprobs.content` tokens joined, and its rows.
fn content(got: Got) !std.json.Parsed(std.json.Value) {
    try std.testing.expectEqual(@as(u16, 200), got.status);
    return std.json.parseFromSlice(std.json.Value, std.testing.allocator, got.body, .{});
}

fn rowsOf(v: std.json.Value) []std.json.Value {
    return v.object.get("choices").?.array.items[0].object.get("logprobs").?.object.get("content").?.array.items;
}

test "with thinking on, the rows start at the answer: after </think> and the newlines the text drops, no end token" {
    const reply = tokensOf("ab") ++ [_]u32{think_end} ++ tokensOf("\n\nHi") ++ [_]u32{eos};
    const got = try send(&reply, true, false, ask ++ "}");
    defer std.testing.allocator.free(got.body);
    const parsed = try content(got);
    defer parsed.deinit();
    const rows = rowsOf(parsed.value);
    try std.testing.expectEqual(@as(usize, 2), rows.len);
    try std.testing.expectEqualStrings("H", rows[0].object.get("token").?.string);
    // the sixth token of the reply: its row, not a reasoning token's
    try std.testing.expectEqual(@as(f64, -1.5), rows[0].object.get("logprob").?.float);
    try std.testing.expectEqual(@as(i64, 'H'), rows[0].object.get("bytes").?.array.items[0].integer);
    const top = rows[1].object.get("top_logprobs").?.array.items;
    try std.testing.expectEqual(@as(usize, 2), top.len);
    try std.testing.expectEqualStrings("i", top[0].object.get("token").?.string);
    try std.testing.expectEqualStrings("z", top[1].object.get("token").?.string);
    try std.testing.expectEqual(@as(f64, -2.75), top[1].object.get("logprob").?.float);
}

test "a reply cut inside its reasoning has no rows" {
    const reply = tokensOf("thinking");
    const got = try send(&reply, true, false, ask ++ "}");
    defer std.testing.allocator.free(got.body);
    const parsed = try content(got);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 0), rowsOf(parsed.value).len);
}

test "with thinking off every reply token has a row, and top_logprobs 0 gives none of the alternatives" {
    const reply = tokensOf("Hi") ++ [_]u32{eos};
    const got = try send(&reply, false, false, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":32,\"logprobs\":true}");
    defer std.testing.allocator.free(got.body);
    const parsed = try content(got);
    defer parsed.deinit();
    const rows = rowsOf(parsed.value);
    try std.testing.expectEqual(@as(usize, 2), rows.len);
    try std.testing.expectEqual(@as(f64, -0.25), rows[0].object.get("logprob").?.float);
    try std.testing.expectEqual(@as(usize, 0), rows[1].object.get("top_logprobs").?.array.items.len);
}

test "logprobs refuse a stream, tools, stop strings, structured output, a thinking budget and other routes" {
    const reply = tokensOf("Hi");
    const refused = [_][]const u8{
        ask ++ ",\"stream\":true}",
        ask ++ ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}}]}",
        ask ++ ",\"stop\":[\"x\"]}",
        ask ++ ",\"response_format\":{\"type\":\"json_object\"}}",
        ask ++ ",\"thinking_budget\":5}",
    };
    for (refused) |body| {
        const got = try send(&reply, false, false, body);
        defer std.testing.allocator.free(got.body);
        try std.testing.expectEqual(@as(u16, 400), got.status);
    }
    const responses = send(&reply, false, true, "{\"model\":\"test-model\",\"input\":\"x\",\"logprobs\":true}");
    try std.testing.expectError(error.Refused, responses);
}
