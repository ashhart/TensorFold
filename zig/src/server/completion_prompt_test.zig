//! Token ids in a completion prompt, top-level or nested: ids in the vocabulary are served, other values get a 400.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

/// Bytes as tokens; decode skips ids past 255, as the checkpoint tokenizer skips unknown ids.
const ByteText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id < 256) try out.append(a, @intCast(id));
        return out.toOwnedSlice(a);
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Copies the prompt the server submits, then finishes at once.
const Capture = struct {
    prompt: []u32 = &.{},

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        std.testing.allocator.free(e.prompt);
        e.prompt = std.testing.allocator.dupe(u32, request.prompt) catch return error.Busy;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'o', 'k' } });
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

/// A whole reply's status.
const Reply = struct {
    status: ?u16 = null,

    fn out(r: *Reply) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(_: *anyopaque) error{Closed}!void {}

    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}

    fn reply(ctx: *anyopaque, status: u16, _: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.status = status;
    }
};

/// The status of a completion whose prompt is the JSON ``prompt``; ``backend.prompt`` keeps the ids it submitted.
fn complete(backend: *Capture, prompt: []const u8) !?u16 {
    var text: ByteText = .{};
    const srv = try Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":2,\"prompt\":{s}}}", .{prompt}))).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var reply: Reply = .{};
    openai.run(srv, a, reply.out(), .{ .conn = &conn }, false, raw);
    return reply.status;
}

test "a completion prompt's ids up to the vocabulary's last id are served, nested or not" {
    var backend: Capture = .{};
    defer std.testing.allocator.free(backend.prompt);
    for ([_][]const u8{ "[0,104,105,255]", "[[0,104,105,255]]" }) |prompt| {
        try std.testing.expectEqual(@as(?u16, 200), try complete(&backend, prompt));
        try std.testing.expectEqualSlices(u32, &.{ 0, 'h', 'i', 255 }, backend.prompt);
    }
}

test "a completion prompt's ids outside the vocabulary get a 400, nested or not" {
    var backend: Capture = .{};
    defer std.testing.allocator.free(backend.prompt);
    // one past the vocabulary, below zero, 2^32, i64's largest and one past it
    for ([_][]const u8{ "256", "-1", "4294967296", "9223372036854775807", "9223372036854775808" }) |id| {
        var buf: [64]u8 = undefined;
        try std.testing.expectEqual(@as(?u16, 400), try complete(&backend, try std.fmt.bufPrint(&buf, "[{s}]", .{id})));
        try std.testing.expectEqual(@as(?u16, 400), try complete(&backend, try std.fmt.bufPrint(&buf, "[[104,{s}]]", .{id})));
    }
    try std.testing.expectEqual(@as(usize, 0), backend.prompt.len);
}

test "a completion prompt's boolean ids get a 400, nested or not" {
    var backend: Capture = .{};
    defer std.testing.allocator.free(backend.prompt);
    for ([_][]const u8{ "[true]", "[false]", "[[104,true]]", "[[104,false]]" }) |prompt| {
        try std.testing.expectEqual(@as(?u16, 400), try complete(&backend, prompt));
    }
    try std.testing.expectEqual(@as(usize, 0), backend.prompt.len);
}
