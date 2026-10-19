//! `--thinking-reserve`: with thinking on, the engine gets max_tokens plus the reserve and the answer keeps max_tokens.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const Conn = @import("http_conn.zig").Conn;

const think_end: u32 = 300;

/// Bytes as tokens and `</think>` as token 300; every prompt renders as "prompt".
const Text = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id == think_end) try out.appendSlice(a, "</think>") else try out.append(a, @intCast(id));
        return out.items;
    }

    fn tokenId(_: *anyopaque, p: []const u8) ?u32 {
        return if (std.mem.eql(u8, p, "</think>")) think_end else null;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 301;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Keeps the limits it was asked for and answers "hi" after a closed think block.
const Engine = struct {
    answer_limit: bool,
    window: u32 = 4096,
    max_tokens: u32 = 0,
    answer_max: u32 = 0,
    think_end: ?u32 = null,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const e: *@This() = @ptrCast(@alignCast(ctx));
        return .{ .context_window = e.window, .answer_limit = e.answer_limit };
    }

    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *@This() = @ptrCast(@alignCast(ctx));
        e.max_tokens = r.max_tokens;
        e.answer_max = r.answer_max;
        e.think_end = r.think_end;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'a', think_end, 'h', 'i' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Ignore = struct {
    status: u16 = 0,

    fn out(g: *Ignore) openai.Out {
        return .{ .ctx = g, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(_: *anyopaque, _: usize) error{Closed}!void {}

    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}

    fn reply(ctx: *anyopaque, status_code: u16, _: json.Value) void {
        const g: *Ignore = @ptrCast(@alignCast(ctx));
        g.status = status_code;
    }
};

fn send(e: *Engine, reserve: i64, thinking: bool) !void {
    var text: Text = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, e.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = thinking,
        .use_drafts = false,
        .thinking_reserve = reserve,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = (try json.parse(a, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":5}")).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var got: Ignore = .{};
    openai.run(srv, a, got.out(), .{ .conn = &conn }, true, body);
    try std.testing.expectEqual(@as(u16, 200), got.status);
}

test "the thinking reserve adds to the engine's limit and keeps max_tokens for the answer" {
    var e: Engine = .{ .answer_limit = true };
    try send(&e, 100, true);
    try std.testing.expectEqual(@as(u32, 105), e.max_tokens);
    try std.testing.expectEqual(@as(u32, 5), e.answer_max);
    try std.testing.expectEqual(@as(?u32, think_end), e.think_end);
}

test "the thinking reserve stops at the context window" {
    var e: Engine = .{ .answer_limit = true, .window = 64 };
    try send(&e, 100, true);
    try std.testing.expectEqual(@as(u32, 64 - "prompt".len), e.max_tokens);
    try std.testing.expectEqual(@as(u32, 5), e.answer_max);
}

test "without thinking, an answer limit, or a reserve, max_tokens limits the whole reply" {
    for ([_]struct { bool, i64, bool }{ .{ true, 100, false }, .{ false, 100, true }, .{ true, 0, true } }) |case| {
        var e: Engine = .{ .answer_limit = case[0] };
        try send(&e, case[1], case[2]);
        try std.testing.expectEqual(@as(u32, 5), e.max_tokens);
        try std.testing.expectEqual(@as(u32, 0), e.answer_max);
    }
}

test "the thinking reserve saturates huge positive values before window and u32 caps" {
    for ([_]u32{ 64, 0 }) |window| {
        var e: Engine = .{ .answer_limit = true, .window = window };
        try send(&e, std.math.maxInt(i64), true);
        const expected: u32 = if (window > 0) window - @as(u32, "prompt".len) else std.math.maxInt(u32);
        try std.testing.expectEqual(expected, e.max_tokens);
        try std.testing.expectEqual(@as(u32, 5), e.answer_max);
    }
}
