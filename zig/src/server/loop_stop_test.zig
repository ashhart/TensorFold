//! --loop-guard-stop on the chat, Responses and Messages routes: the shape a reply takes when the loop guard ends it.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const Conn = @import("http_conn.zig").Conn;
const exchange = @import("late_system_test.zig").exchange;
const t = std.testing;

const think_end: u32 = 300;
const reasoning = "loop loop loop";
const call = "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": \"a\"}}\n</tool_call>";

/// Bytes as tokens and `</think>` as token 300; every prompt renders as "prompt".
const Text = struct {
    fn text(x: *@This()) model_text.Text {
        return .{ .ctx = x, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
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
    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try (a.dupe(u8, "prompt")) };
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// The guard firing as in a lane engine: with loop_stop the reply ends in its reasoning, else it closes and answers.
const Looping = struct {
    batched: bool = false,
    guard: bool = false, // what the last request asked
    stop: bool = false,
    closes: bool = false,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return api.Info.ofLanes(.{ .context_window = 4096 });
    }
    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *@This() = @ptrCast(@alignCast(ctx));
        e.guard = r.loop_guard;
        e.stop = r.loop_stop;
        e.closes = std.mem.indexOfScalar(u32, r.think_close, think_end) != null;
        var ids: [160]u32 = undefined;
        var n: usize = 0;
        const answer: []const u8 = if (r.tools_json.len > 0) call else "done";
        for (reasoning) |b| push(&ids, &n, b);
        if (!r.loop_stop) {
            for (r.think_close) |c| push(&ids, &n, c);
            for (answer) |b| push(&ids, &n, b);
        }
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        if (e.batched) sink.event(sink.ctx, id, &.{ .tokens = ids[0..n] }) else for (ids[0..n]) |*tok| sink.event(sink.ctx, id, &.{ .tokens = tok[0..1] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop, .stats = .{ .loop_period = 3 } } });
    }
    fn push(ids: *[160]u32, n: *usize, token: u32) void {
        ids[n.*] = token;
        n.* += 1;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// A chat route's output: a whole reply, or each streamed event (re-parsed into the test's arena) and the [DONE].
const Wire = struct {
    a: std.mem.Allocator,
    status: ?u16 = null,
    whole: json.Value = .null,
    events: std.ArrayList(json.Value) = .empty,
    done: bool = false,

    fn out(w: *Wire) openai.Out {
        return .{ .ctx = w, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn copy(w: *Wire, v: json.Value) !json.Value {
        return (try json.parse(w.a, try json.stringify(w.a, v, .{}))).ok;
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        const p = payload orelse {
            w.done = true;
            return;
        };
        w.events.append(w.a, w.copy(p) catch return error.Closed) catch return error.Closed;
    }
    fn reply(ctx: *anyopaque, code: u16, payload: json.Value) void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.status = code;
        w.whole = w.copy(payload) catch .null;
    }
};

/// What a client reads from a chat reply, whole or streamed.
const View = struct {
    finish: []const u8 = "",
    reasoning: []const u8 = "",
    content: []const u8 = "",
    calls: usize = 0,
    err: ?json.Value = null,
    loop: ?json.Value = null,
    error_last: bool = false,
};

fn view(a: std.mem.Allocator, w: *const Wire) !View {
    var v: View = .{};
    if (w.status) |code| {
        try t.expectEqual(@as(u16, 200), code);
        const choice = w.whole.get("choices").?.array[0];
        const message = choice.get("message").?;
        v.finish = choice.strField("finish_reason").?;
        v.reasoning = message.strField("reasoning_content") orelse "";
        v.content = message.strField("content") orelse "";
        if (message.get("tool_calls")) |c| v.calls = c.array.len;
        v.err = w.whole.get("error");
        v.loop = w.whole.get("tensorfold").?.get("loop");
        return v;
    }
    try t.expect(w.done);
    var thought: std.ArrayList(u8) = .empty;
    var text: std.ArrayList(u8) = .empty;
    for (w.events.items, 0..) |e, i| {
        if (e.get("error")) |err| {
            v.err = err;
            v.error_last = i + 1 == w.events.items.len;
            continue;
        }
        const choices = e.get("choices").?.array;
        if (choices.len == 0) continue;
        if (choices[0].strField("finish_reason")) |f| {
            v.finish = f;
            v.loop = e.get("tensorfold").?.get("loop");
        }
        const d = choices[0].get("delta") orelse continue;
        try thought.appendSlice(a, d.strField("reasoning_content") orelse "");
        try text.appendSlice(a, d.strField("content") orelse "");
        if (d.get("tool_calls")) |c| for (c.array) |x| {
            v.calls += @intFromBool(x.get("id") != null);
        };
    }
    v.reasoning = thought.items;
    v.content = text.items;
    return v;
}

fn config(guard: bool, stop: bool) server.Config {
    return .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false, .loop_guard = guard, .loop_stop = stop };
}

const tools_field = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";

/// One chat completion through ``openai.run`` on a server with these loop settings.
fn chat(a: std.mem.Allocator, e: *Looping, cfg: server.Config, stream: bool, tools: bool) !View {
    var text: Text = .{};
    var srv = try server.Server.init(t.allocator, t.io, e.engine(), text.text(), cfg, null);
    defer srv.deinit();
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":200,\"stream\":{},\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}]{s}}}", .{ stream, if (tools) tools_field else "" });
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var wire: Wire = .{ .a = a };
    openai.run(srv, a, wire.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
    try t.expectEqual(@as(i64, 0), srv.preparing.load(.acquire));
    return view(a, &wire);
}

test "loop stop: a stopped chat reply carries its reasoning, no content or calls, content_filter and the loop error" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |tools| for ([_]bool{ false, true }) |batched| {
        var e: Looping = .{ .batched = batched };
        const whole = try chat(a, &e, config(true, true), false, tools);
        try t.expect(e.guard and e.stop and e.closes);
        const streamed = try chat(a, &e, config(true, true), true, tools);
        for ([_]View{ whole, streamed }) |v| {
            try t.expectEqualStrings("content_filter", v.finish);
            try t.expectEqualStrings(reasoning, v.reasoning);
            try t.expectEqualStrings("", v.content);
            try t.expectEqual(@as(usize, 0), v.calls);
            try t.expectEqualStrings("reasoning_loop", v.err.?.strField("code").?);
            try t.expectEqualStrings("server_error", v.err.?.strField("type").?);
            try t.expect(std.mem.indexOf(u8, v.err.?.strField("message").?, "3-token cycle") != null);
            try t.expectEqual(@as(?i64, 3), v.loop.?.get("period").?.int64());
            try t.expect(v.loop.?.get("stopped").?.bool);
        }
        try t.expect(streamed.error_last); // after the finish chunk and its usage, before [DONE]
    };
}

test "loop stop: off, the guarded reply closes and answers as before, with no error or stopped mark" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |tools| for ([_]bool{ false, true }) |stream| {
        var e: Looping = .{};
        const v = try chat(a, &e, config(true, false), stream, tools);
        try t.expect(e.guard and !e.stop and e.closes);
        try t.expectEqualStrings(if (tools) "tool_calls" else "stop", v.finish);
        try t.expectEqualStrings(reasoning, std.mem.trimEnd(u8, v.reasoning, "\n")); // a stream sends the close's lead
        try t.expectEqualStrings(if (tools) "" else "done", v.content);
        try t.expectEqual(@as(usize, @intFromBool(tools)), v.calls);
        try t.expect(v.err == null);
        try t.expectEqual(@as(usize, 1), v.loop.?.object.count()); // {"period": 3}, as on main
    };
    // the stop setting does nothing without the guard: the request asks neither
    var e: Looping = .{};
    _ = try chat(a, &e, config(false, true), false, false);
    try t.expect(!e.guard and !e.stop);
}

fn routed(a: std.mem.Allocator, stop: bool, path: []const u8, body: []const u8) !struct { u16, []const u8 } {
    var e: Looping = .{ .batched = true }; // one reasoning event: the stream fits the socket pair's buffer
    var text: Text = .{};
    var srv = try server.Server.init(t.allocator, t.io, e.engine(), text.text(), config(true, stop), null);
    defer srv.deinit();
    const got = try exchange(srv, path, body);
    defer t.allocator.free(got.body);
    return .{ got.status, try a.dupe(u8, got.body) };
}

const responses_tools = ",\"tools\":[{\"type\":\"function\",\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}]";

test "loop stop: a stopped Responses reply is incomplete for content_filter, its output the reasoning alone" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |tools| {
        const extra = if (tools) responses_tools else "";
        const whole = try routed(a, true, "/v1/responses", try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"input\":\"x\"{s}}}", .{extra}));
        try t.expectEqual(@as(u16, 200), whole[0]);
        const r = (try json.parse(a, whole[1])).ok;
        try t.expectEqualStrings("incomplete", r.strField("status").?);
        try t.expectEqualStrings("content_filter", r.get("incomplete_details").?.strField("reason").?);
        try t.expect(r.get("error").? == .null);
        const output = r.get("output").?.array;
        try t.expectEqual(@as(usize, 1), output.len);
        try t.expectEqualStrings("reasoning", output[0].strField("type").?);
        try t.expectEqualStrings(reasoning, output[0].get("content").?.array[0].strField("text").?);
        try t.expect(r.get("tensorfold").?.get("loop").?.get("stopped").?.bool);
        const streamed = try routed(a, true, "/v1/responses", try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"input\":\"x\",\"stream\":true{s}}}", .{extra}));
        try t.expectEqual(@as(u16, 200), streamed[0]);
        const s = streamed[1];
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, s, "event: response.incomplete\n"));
        try t.expect(std.mem.indexOf(u8, s, "\"reason\": \"content_filter\"") != null or std.mem.indexOf(u8, s, "\"reason\":\"content_filter\"") != null);
        try t.expect(std.mem.indexOf(u8, s, "event: response.reasoning_text.delta\n") != null);
        for ([_][]const u8{ "response.failed", "response.completed", "output_text.delta", "function_call" }) |absent|
            try t.expect(std.mem.indexOf(u8, s, absent) == null);
    }
    const off = try routed(a, false, "/v1/responses", "{\"model\":\"m\",\"input\":\"x\"}");
    const r = (try json.parse(a, off[1])).ok;
    try t.expectEqualStrings("completed", r.strField("status").?);
    try t.expect(r.get("tensorfold").?.get("loop").?.get("stopped") == null);
}

test "loop stop: a stopped Messages reply is a refusal, streamed without an error event" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const head = "{\"model\":\"m\",\"max_tokens\":200,\"thinking\":{\"type\":\"adaptive\"},\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]";
    const whole = try routed(a, true, "/v1/messages", head ++ "}");
    try t.expectEqual(@as(u16, 200), whole[0]);
    const m = (try json.parse(a, whole[1])).ok;
    try t.expectEqualStrings("refusal", m.strField("stop_reason").?);
    for (m.get("content").?.array) |block| try t.expectEqualStrings("thinking", block.strField("type").?);
    const streamed = try routed(a, true, "/v1/messages", head ++ ",\"stream\":true}");
    try t.expect(std.mem.indexOf(u8, streamed[1], "refusal") != null);
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, streamed[1], "event: message_stop\n"));
    try t.expect(std.mem.indexOf(u8, streamed[1], "event: error") == null);
}

test "loop stop: --loop-guard-stop parses as a native serve flag" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var usage: server.cli.Usage = .{};
    const args = try server.cli.parse(arena.allocator(), &.{ "model", "--loop-guard-stop" }, &usage);
    try t.expect(args.loop_guard_stop and !args.loop_guard);
    const plain = try server.cli.parse(arena.allocator(), &.{ "model", "--loop-guard" }, &usage);
    try t.expect(plain.loop_guard and !plain.loop_guard_stop);
}
