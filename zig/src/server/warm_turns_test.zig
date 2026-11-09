//! A warm pass renders and cuts the next turn's prefix as that turn's own request will, so its kept end is that state.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const fixtures = @import("late_system_test.zig");
const exchange = fixtures.exchange;
const textOf = fixtures.textOf;
const t = std.testing;

/// Bytes as tokens; each message renders as "[role]", its reasoning in "<r></r>", its content and a newline.
const Roles = struct {
    fn text(x: *Roles) model_text.Text {
        return .{ .ctx = x, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const out = try a.alloc(u32, input.len);
        for (input, out) |byte, *id| id.* = byte;
        return out;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, in: []const u32) std.mem.Allocator.Error![]u8 {
        const out = try a.alloc(u8, in.len);
        for (in, out) |id, *byte| byte.* = @intCast(id);
        return out;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, messages: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        var out: std.ArrayList(u8) = .empty;
        if (messages != .array) return .{ .text = out.items };
        for (messages.array) |m| {
            try out.print(a, "[{s}]", .{textOf(m, "role")});
            const thought = textOf(m, "reasoning_content");
            if (thought.len > 0) try out.print(a, "<r>{s}</r>", .{thought});
            try out.print(a, "{s}\n", .{textOf(m, "content")});
        }
        return .{ .text = out.items };
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Keeps the last reply request's prompt and chunks, and the last warm pass's; replies with `reply`'s bytes.
const Record = struct {
    reply: []const u8 = "Done.",
    step: u32 = 0,
    prompt: []u32 = &.{},
    chunks: []u32 = &.{},
    warm_prompt: []u32 = &.{},
    warm_chunks: []u32 = &.{},

    fn deinit(e: *Record) void {
        for ([_][]u32{ e.prompt, e.chunks, e.warm_prompt, e.warm_chunks }) |s| t.allocator.free(s);
    }

    fn engine(e: *Record) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const e: *Record = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 1 << 20, .warm_turns = true, .prefill_step = e.step };
    }

    fn keep(old: *[]u32, new: []const u32) api.SubmitError!void {
        t.allocator.free(old.*);
        old.* = t.allocator.dupe(u32, new) catch return error.Busy;
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Record = @ptrCast(@alignCast(ctx));
        if (request.background) {
            try keep(&e.warm_prompt, request.prompt);
            try keep(&e.warm_chunks, request.chunks);
            sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
            return;
        }
        try keep(&e.prompt, request.prompt);
        try keep(&e.chunks, request.chunks);
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (e.reply) |byte| {
            const token: u32 = byte;
            sink.event(sink.ctx, id, &.{ .tokens = &.{token} });
        }
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

fn serve(backend: *Record, text: *Roles, thinking: bool) !*server.Server {
    return server.Server.init(t.allocator, t.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = thinking,
        .use_drafts = false,
    }, null);
}

/// The first request's warm pass, then the next request: the pass must be that request's prompt prefix.
const Turns = struct { warm_prompt: []u32, warm_chunks: []u32 };

fn firstTurn(srv: *server.Server, backend: *Record, body: []const u8) !Turns {
    const got = try exchange(srv, "/v1/chat/completions", body);
    defer t.allocator.free(got.body);
    try t.expectEqual(@as(u16, 200), got.status);
    try t.expect(backend.warm_prompt.len > 0);
    return .{ .warm_prompt = try t.allocator.dupe(u32, backend.warm_prompt), .warm_chunks = try t.allocator.dupe(u32, backend.warm_chunks) };
}

fn nextTurn(srv: *server.Server, backend: *Record, first: Turns, body: []const u8) !void {
    defer t.allocator.free(first.warm_prompt);
    defer t.allocator.free(first.warm_chunks);
    const got = try exchange(srv, "/v1/chat/completions", body);
    defer t.allocator.free(got.body);
    try t.expectEqual(@as(u16, 200), got.status);
    // the exactness bar: the warmed history is byte for byte the next request's prefix
    try t.expect(std.mem.startsWith(u32, backend.prompt, first.warm_prompt));
}

const system = "x" ** 1500;

test "a warm pass cuts its chunks where the next turn's request does, at the system prefix" {
    var text: Roles = .{};
    var backend: Record = .{ .step = 1024 };
    defer backend.deinit();
    const srv = try serve(&backend, &text, false);
    defer srv.deinit();
    const head = "{\"model\":\"m\",\"max_tokens\":8,\"messages\":[{\"role\":\"system\",\"content\":\"" ++ system ++ "\"},{\"role\":\"user\",\"content\":\"go\"}";
    const first = try firstTurn(srv, &backend, head ++ "]}");
    const warm_chunks = try t.allocator.dupe(u32, first.warm_chunks);
    defer t.allocator.free(warm_chunks);
    const warm_len = first.warm_prompt.len;
    try nextTurn(srv, &backend, first, head ++ ",{\"role\":\"assistant\",\"content\":\"Done.\"},{\"role\":\"user\",\"content\":\"next\"}]}");
    // the system cut lies inside the warmed prefix, so the pass must carry it as the request does
    var below: std.ArrayList(u32) = .empty;
    defer below.deinit(t.allocator);
    for (backend.chunks) |c| if (c < warm_len) try below.append(t.allocator, c);
    try t.expect(below.items.len >= 2);
    try t.expectEqualSlices(u32, below.items, warm_chunks);
}

const tools = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";

/// The first request's reply message as its client got it, compact, for the next request to send back.
fn replied(a: std.mem.Allocator, body: []const u8) ![]const u8 {
    const message = (try json.parse(a, body)).ok.get("choices").?.array[0].get("message").?;
    return json.stringify(a, message, .{ .compact = true });
}

test "after a tool call the warm pass opens the next turn as that call's result, as the next request does" {
    var text: Roles = .{};
    var backend: Record = .{ .reply = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"a\"}}</tool_call>" };
    defer backend.deinit();
    const srv = try serve(&backend, &text, false);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const user = "{\"model\":\"m\",\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"go\"}";
    const got = try exchange(srv, "/v1/chat/completions", user ++ "]" ++ tools ++ "}");
    defer t.allocator.free(got.body);
    try t.expectEqual(@as(u16, 200), got.status);
    const message = try replied(a, got.body);
    const call = (try json.parse(a, message)).ok.get("tool_calls").?.array[0].strField("id").?;
    const first: Turns = .{ .warm_prompt = try t.allocator.dupe(u32, backend.warm_prompt), .warm_chunks = try t.allocator.dupe(u32, backend.warm_chunks) };
    try t.expect(first.warm_prompt.len > 0);
    const next = try std.mem.concat(a, u8, &.{ user, ",", message, ",{\"role\":\"tool\",\"tool_call_id\":\"", call, "\",\"content\":\"42\"}]", tools, "}" });
    try nextTurn(srv, &backend, first, next);
}

test "the warm pass renders the reply's reasoning as the next request sends it back" {
    var text: Roles = .{};
    var backend: Record = .{ .reply = "Plan.</think>Done." };
    defer backend.deinit();
    const srv = try serve(&backend, &text, true);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const user = "{\"model\":\"m\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"go\"}";
    const got = try exchange(srv, "/v1/chat/completions", user ++ "]}");
    defer t.allocator.free(got.body);
    try t.expectEqual(@as(u16, 200), got.status);
    const message = try replied(a, got.body);
    try t.expectEqualStrings("Plan.", (try json.parse(a, message)).ok.strField("reasoning_content").?);
    const first: Turns = .{ .warm_prompt = try t.allocator.dupe(u32, backend.warm_prompt), .warm_chunks = try t.allocator.dupe(u32, backend.warm_chunks) };
    try t.expect(first.warm_prompt.len > 0);
    const next = try std.mem.concat(a, u8, &.{ user, ",", message, ",{\"role\":\"user\",\"content\":\"next\"}]}" });
    try nextTurn(srv, &backend, first, next);
}
