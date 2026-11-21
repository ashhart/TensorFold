//! A reply that ends at its end token inside a think block that never closed answers with its reasoning (#3687).
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const openai = @import("openai.zig");
const Conn = @import("http_conn.zig").Conn;
const split = @import("think_split_test.zig");
const capture = @import("tool_json_cut_routes_test.zig").capture;
const Compile = @import("structure_field_test.zig").Compile;
const t = std.testing;

const plan = "Plan answer";
const ended = split.tokensOf(plan) ++ [_]u32{split.eos};
const ask = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":32";
const roomy = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":64";

/// A thinking server over the byte tokenizer answering `reply`; `structured` gives it a grammar engine's compiler.
const Fixture = struct {
    text: split.Text = .{},
    backend: split.Script,
    compile: Compile = .{},
    compiler: api.grammar.Compiler = undefined,
    srv: *server.Server = undefined,

    fn init(f: *Fixture, reply: []const u32, stride: ?usize, structured: bool) !void {
        f.* = .{ .backend = .{ .reply = reply, .stride = stride } };
        f.srv = try server.Server.init(t.allocator, t.io, f.backend.engine(), f.text.text(), .{
            .served_name = "test-model",
            .model_ids = &.{"test-model"},
            .enable_thinking = true,
            .use_drafts = false,
        }, null);
        f.srv.markers = f.text.markers;
        if (structured) {
            f.compiler = f.compile.compiler();
            f.srv.info.structures = true;
            f.srv.grammars = &f.compiler;
        }
    }

    /// The chat route's reply to `body`, whole or streamed.
    fn chat(f: *Fixture, body: []const u8) !split.Got {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
        var got: split.Got = .{};
        openai.run(f.srv, a, got.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
        return got;
    }
};

/// A whole chat reply's message, finish reason and whether the runtime block says it recovered.
const Whole = struct { content: []const u8, reasoning: []const u8, finish: []const u8, recovered: bool, tokens: usize };

fn whole(a: std.mem.Allocator, got: *const split.Got) !Whole {
    try t.expectEqual(@as(u16, 200), got.status);
    const doc = (try std.json.parseFromSlice(std.json.Value, a, got.body, .{})).value;
    const choice = doc.object.get("choices").?.array.items[0].object;
    const m = choice.get("message").?.object;
    const runtime = doc.object.get("tensorfold").?.object;
    return .{
        .content = if (m.get("content")) |c| (if (c == .string) c.string else "") else "",
        .reasoning = if (m.get("reasoning_content")) |r| r.string else "",
        .finish = choice.get("finish_reason").?.string,
        .recovered = runtime.get("reasoning_recovered") != null,
        .tokens = @intCast(doc.object.get("usage").?.object.get("completion_tokens").?.integer),
    };
}

/// A stream's `field` delta pieces, joined.
fn joined(a: std.mem.Allocator, got: *const split.Got, field: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (got.events.items) |e| {
        const chunk = (try std.json.parseFromSlice(std.json.Value, a, e, .{})).value;
        const choices = chunk.object.get("choices") orelse continue;
        if (choices.array.items.len == 0) continue;
        if (choices.array.items[0].object.get("delta").?.object.get(field)) |v| if (v == .string) try out.appendSlice(a, v.string);
    }
    return out.items;
}

test "reasoning recovered: an unclosed think block that ends at the end token is the chat answer too, streamed alike" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = undefined;
    try f.init(&ended, null, false);
    defer f.srv.deinit();
    var got = try f.chat(ask ++ ",\"return_token_ids\":true}");
    defer got.deinit();
    const w = try whole(a, &got);
    try t.expectEqualStrings(plan, w.content);
    try t.expectEqualStrings(plan, w.reasoning);
    try t.expectEqualStrings("stop", w.finish);
    try t.expect(w.recovered);
    // the tokens are the engine's, untouched: every one counted and returned
    try t.expectEqual(ended.len, w.tokens);
    const doc = (try std.json.parseFromSlice(std.json.Value, a, got.body, .{})).value;
    const ids = doc.object.get("tensorfold").?.object.get("token_ids").?.array.items;
    for (ended, ids) |want, id| try t.expectEqual(@as(i64, want), id.integer);
    for ([_]?usize{ 1, 2, 3, null, 256 }) |stride| {
        var s: Fixture = undefined;
        try s.init(&ended, stride, false);
        defer s.srv.deinit();
        var stream = try s.chat(ask ++ ",\"stream\":true}");
        defer stream.deinit();
        try t.expectEqualStrings(w.content, try joined(a, &stream, "content"));
        try t.expectEqualStrings(w.reasoning, try joined(a, &stream, "reasoning_content"));
    }
}

test "reasoning recovered: offered tools with no call recover alike, and the tool streamer's stream equals the whole" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = ask ++ ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"look\",\"parameters\":{\"type\":\"object\"}}}]";
    var f: Fixture = undefined;
    try f.init(&ended, null, false);
    defer f.srv.deinit();
    var got = try f.chat(body ++ "}");
    defer got.deinit();
    const w = try whole(a, &got);
    try t.expectEqualStrings(plan, w.content);
    try t.expect(w.recovered);
    for ([_]?usize{ 1, null }) |stride| {
        var s: Fixture = undefined;
        try s.init(&ended, stride, false);
        defer s.srv.deinit();
        var stream = try s.chat(body ++ ",\"stream\":true}");
        defer stream.deinit();
        try t.expectEqualStrings(w.content, try joined(a, &stream, "content"));
        try t.expectEqualStrings(w.reasoning, try joined(a, &stream, "reasoning_content"));
    }
}

test "reasoning recovered: length, a stop string, a closed block, a call opener or a bare call keep today's reply" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const opener = split.tokensOf("Plan <tool_call>\n{\"na") ++ [_]u32{split.eos};
    const closed = split.tokensOf("Plan") ++ [_]u32{ split.think_end, split.eos };
    const bare = split.tokensOf("{\"name\": \"look\", \"arguments\": {}}") ++ [_]u32{split.eos};
    const tools = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"look\",\"parameters\":{\"type\":\"object\"}}}]}";
    const Case = struct { reply: []const u32, body: []const u8, reasoning: []const u8, finish: []const u8 };
    const cases = [_]Case{
        .{ .reply = &split.tokensOf(plan), .body = ask ++ "}", .reasoning = plan, .finish = "length" },
        .{ .reply = &ended, .body = ask ++ ",\"stop\":[\"answer\"]}", .reasoning = "Plan", .finish = "stop" },
        .{ .reply = &closed, .body = ask ++ "}", .reasoning = "Plan", .finish = "stop" },
        .{ .reply = &opener, .body = ask ++ "}", .reasoning = "Plan <tool_call>\n{\"na", .finish = "stop" },
        // a bare JSON call the route's tool parser reads: the answer would become a call, so it stays as today
        .{ .reply = &bare, .body = roomy ++ tools, .reasoning = "{\"name\": \"look\", \"arguments\": {}}", .finish = "stop" },
    };
    for (cases) |c| {
        var f: Fixture = undefined;
        try f.init(c.reply, null, false);
        defer f.srv.deinit();
        var got = try f.chat(c.body);
        defer got.deinit();
        const w = try whole(a, &got);
        try t.expectEqualStrings("", w.content);
        try t.expectEqualStrings(c.reasoning, w.reasoning);
        try t.expectEqualStrings(c.finish, w.finish);
        try t.expect(!w.recovered);
    }
}

test "reasoning recovered: a JSON schema or a forced tool keeps today's reply, which prose cannot meet" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tool = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"look\",\"parameters\":{\"type\":\"object\"}}}]";
    const bodies = [_][]const u8{
        ask ++ ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"o\",\"schema\":{\"type\":\"object\"}}}}",
        ask ++ "," ++ tool ++ ",\"tool_choice\":\"required\"}",
        ask ++ "," ++ tool ++ ",\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"look\"}}}",
    };
    for (bodies) |body| {
        var f: Fixture = undefined;
        try f.init(&ended, null, true);
        defer f.srv.deinit();
        var got = try f.chat(body);
        defer got.deinit();
        const w = try whole(a, &got);
        try t.expectEqualStrings("", w.content);
        try t.expectEqualStrings(plan, w.reasoning);
        try t.expect(!w.recovered);
    }
}

/// The text and thinking a /v1/messages or /v1/responses reply carries, whole or joined from its stream.
const Seen = struct { text: []const u8 = "", thinking: []const u8 = "", finish: []const u8 = "" };

fn route(a: std.mem.Allocator, path: []const u8, stream: bool) !Seen {
    var f: Fixture = undefined;
    try f.init(&ended, if (stream) 1 else null, false);
    defer f.srv.deinit();
    const messages = std.mem.eql(u8, path, "/v1/messages");
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"{s}\":32,\"stream\":{s},{s}}}", .{
        if (messages) "max_tokens" else "max_output_tokens",
        if (stream) "true" else "false",
        // Claude Code turns thinking on; without it the route renders with thinking off and nothing is reasoning
        if (messages) "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"thinking\":{\"type\":\"adaptive\"}" else "\"input\":\"x\"",
    });
    const got = try capture(a, f.srv, path, body);
    try t.expectEqual(@as(u16, 200), got.status);
    var seen: Seen = .{};
    if (!stream) {
        const doc = (try json.parseText(a, got.body)).ok;
        if (messages) {
            for (doc.get("content").?.array) |b| {
                if (b.typeIs("text")) seen.text = b.strField("text").?;
                if (b.typeIs("thinking")) seen.thinking = b.strField("thinking").?;
            }
            seen.finish = doc.strField("stop_reason").?;
            return seen;
        }
        for (doc.get("output").?.array) |item| {
            if (item.typeIs("message")) seen.text = item.get("content").?.array[0].strField("text").?;
            if (item.typeIs("reasoning")) seen.thinking = item.get("content").?.array[0].strField("text").?;
        }
        seen.finish = doc.strField("status").?;
        return seen;
    }
    var text: std.ArrayList(u8) = .empty;
    var thought: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, got.body, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const ev = switch (try json.parseText(a, std.mem.trim(u8, line[6..], "\r"))) {
            .ok => |v| v,
            .err => continue,
        };
        if (messages) {
            if (ev.get("delta")) |d| {
                if (d.typeIs("text_delta")) try text.appendSlice(a, d.strField("text").?);
                if (d.typeIs("thinking_delta")) try thought.appendSlice(a, d.strField("thinking").?);
                if (d.strField("stop_reason")) |r| seen.finish = r;
            }
            continue;
        }
        if (ev.typeIs("response.output_text.delta")) try text.appendSlice(a, ev.strField("delta").?);
        if (ev.typeIs("response.reasoning_text.delta")) try thought.appendSlice(a, ev.strField("delta").?);
        if (ev.typeIs("response.completed")) seen.finish = ev.get("response").?.strField("status").?;
    }
    seen.text = text.items;
    seen.thinking = thought.items;
    return seen;
}

test "reasoning recovered: /v1/messages answers with a text block after the thinking block, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    for ([_]bool{ false, true }) |stream| {
        const seen = try route(arena.allocator(), "/v1/messages", stream);
        try t.expectEqualStrings(plan, seen.text);
        try t.expectEqualStrings(plan, seen.thinking);
        try t.expectEqualStrings("end_turn", seen.finish);
    }
}

test "reasoning recovered: /v1/responses answers with a message after the reasoning item, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    for ([_]bool{ false, true }) |stream| {
        const seen = try route(arena.allocator(), "/v1/responses", stream);
        try t.expectEqualStrings(plan, seen.text);
        try t.expectEqualStrings(plan, seen.thinking);
        try t.expectEqualStrings("completed", seen.finish);
    }
}
