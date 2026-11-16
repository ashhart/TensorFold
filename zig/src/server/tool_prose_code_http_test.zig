//! Quoted tool markup in Markdown inline code spans and fenced code blocks stays prose over HTTP and SSE.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const openai = @import("openai.zig");
const pre = @import("stream_preflight_test.zig");
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

const Got = struct {
    opened: bool = false,
    status: ?u16 = null,
    calls: usize = 0,
    has_kept: bool = false,
    content: std.ArrayList(u8) = .empty,
    arguments: std.ArrayList(u8) = .empty,
    finish: [16]u8 = undefined,
    finish_len: usize = 0,

    fn deinit(g: *Got) void {
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
        if (p.get("tensorfold")) |tf| if (tf.get("kept_tool_calls") != null) {
            g.has_kept = true;
        };
        const choices = p.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        const choice = choices.array[0];
        g.keepFinish(choice);
        const delta = choice.get("delta") orelse return;
        if (delta.get("content")) |t| if (t == .string) {
            g.content.appendSlice(std.testing.allocator, t.string) catch return error.Closed;
        };
        if (delta.get("tool_calls")) |t| if (t == .array) for (t.array) |call| {
            if (call.get("id") != null) g.calls += 1;
            try g.keepCall(call);
        };
    }

    fn reply(ctx: *anyopaque, status: u16, payload: json.Value) void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.status = status;
        if (payload.get("tensorfold")) |tf| if (tf.get("kept_tool_calls") != null) {
            g.has_kept = true;
        };
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

fn send(engine: api.Engine, body: []const u8) !Got {
    var text: pre.TestText = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, engine, text.text(), .{
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

const Case = struct {
    reply: []const u8,
    cut: ?usize = null,
    content: []const u8,
    calls: usize = 0,
    arguments: []const u8 = "",
    finish: []const u8 = "stop",
};

fn expectAgreesSingle(a: Allocator, single: bool, c: Case) !void {
    const tokens = try pre.TestText.encode(undefined, a, c.reply, false);
    const body = try std.fmt.allocPrint(a, "{s}{s},\"max_tokens\":{d}", .{ head, if (single) ",\"parallel_tool_calls\":false" else "", c.cut orelse 2000 });
    for ([_]bool{ false, true }) |batched| {
        var backend: pre.CallEngine = .{ .tokens = tokens, .batched = batched };
        var whole = try send(backend.engine(), try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }));
        defer whole.deinit();
        var streamed = try send(backend.engine(), try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }));
        defer streamed.deinit();
        try std.testing.expectEqual(@as(?u16, 200), whole.status);
        try std.testing.expect(streamed.opened and streamed.status == null);
        try std.testing.expect(!whole.has_kept);
        try std.testing.expect(!streamed.has_kept);
        try std.testing.expectEqualStrings(c.content, whole.content.items);
        try std.testing.expectEqual(c.calls, whole.calls);
        try std.testing.expectEqualStrings(c.arguments, whole.arguments.items);
        try std.testing.expectEqualStrings(c.finish, whole.reason());
        try std.testing.expectEqualStrings(whole.content.items, streamed.content.items);
        try std.testing.expectEqual(whole.calls, streamed.calls);
        try std.testing.expectEqualStrings(whole.arguments.items, streamed.arguments.items);
        try std.testing.expectEqualStrings(whole.reason(), streamed.reason());
    }
}

fn expectAgrees(a: Allocator, c: Case) !void {
    try expectAgreesSingle(a, false, c);
    try expectAgreesSingle(a, true, c);
}

test "tool visibility HTTP: backticked tool openers before a real GLM or XML call stream prose and emit only the real call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const glm = "<tool_call>lookup<arg_key>q</arg_key><arg_value>ok</arg_value></tool_call>";
    const xml = "<tool_call>\n<function=lookup>\n<parameter=q>\nok\n</parameter>\n</function>\n</tool_call>";
    try expectAgrees(a, .{
        .reply = "Raw `<tool_call>` XML leaking into output.\n" ++ glm,
        .content = "Raw `<tool_call>` XML leaking into output.",
        .calls = 1,
        .arguments = "{\"q\":\"ok\"}",
        .finish = "tool_calls",
    });
    try expectAgrees(a, .{
        .reply = "The tag `<tool_call>lookup<arg_key>x` broke args.\n" ++ glm,
        .content = "The tag `<tool_call>lookup<arg_key>x` broke args.",
        .calls = 1,
        .arguments = "{\"q\":\"ok\"}",
        .finish = "tool_calls",
    });
    try expectAgrees(a, .{
        .reply = "Quoting `<tool_call>` before:\n" ++ xml,
        .content = "Quoting `<tool_call>` before:",
        .calls = 1,
        .arguments = "{\"q\":\"ok\"}",
        .finish = "tool_calls",
    });
}

test "tool visibility HTTP: complete tool calls inside inline code or fenced blocks stay prose with stop finish" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const quoted = "Use `<tool_call>lookup<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>` here.";
    try expectAgrees(a, .{ .reply = quoted, .content = quoted });
    const fenced = "Example:\n```xml\n<tool_call>lookup<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>\n```\nDone.";
    try expectAgrees(a, .{ .reply = fenced, .content = fenced });
    const real = "<tool_call>lookup<arg_key>q</arg_key><arg_value>real</arg_value></tool_call>";
    try expectAgrees(a, .{
        .reply = fenced ++ "\n" ++ real,
        .content = fenced,
        .calls = 1,
        .arguments = "{\"q\":\"real\"}",
        .finish = "tool_calls",
    });
}

test "tool visibility HTTP: an unmatched backtick before a real call agrees whole and streamed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const glm = "<tool_call>lookup<arg_key>q</arg_key><arg_value>ok</arg_value></tool_call>";
    try expectAgrees(a, .{
        .reply = "DeepSeek's format is `" ++ glm,
        .content = "DeepSeek's format is `",
        .calls = 1,
        .arguments = "{\"q\":\"ok\"}",
        .finish = "tool_calls",
    });
    try expectAgrees(a, .{
        .reply = "DeepSeek's format is `\n\n" ++ glm,
        .content = "DeepSeek's format is `",
        .calls = 1,
        .arguments = "{\"q\":\"ok\"}",
        .finish = "tool_calls",
    });
    const cut_reply = "Prose with `<tool_call>` then\n<tool_call>lookup<arg_key>q</arg_key><arg_value>ok";
    try expectAgreesSingle(a, false, .{
        .reply = cut_reply ++ " more",
        .cut = cut_reply.len,
        .content = "Prose with `<tool_call>` then",
        .calls = 1,
        .arguments = "{\"q\":\"ok",
        .finish = "length",
    });
    try expectAgreesSingle(a, true, .{
        .reply = cut_reply ++ " more",
        .cut = cut_reply.len,
        .content = "Prose with `<tool_call>` then\n",
        .finish = "length",
    });
}
