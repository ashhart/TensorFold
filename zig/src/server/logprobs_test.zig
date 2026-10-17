//! `logprobs`: rows cover the answer as `content` does, the 0.6.6 refusals hold; `prompt_logprobs` on /v1/completions.
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

/// Answers `reply` in two rounds; token i at -(i + 1) / 4, best itself then 'z'; even prompt rows rank their token 1.
const Script = struct {
    reply: []const u32,
    asked: ?u8 = null,
    prompt_rows: bool = true,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const e: *Script = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 64, .logprobs = true, .prompt_logprobs = e.prompt_rows };
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
        if (r.prompt_logprobs) |k| {
            var prompt_rows: [32]api.LogprobRow = undefined;
            for (r.prompt[1..], prompt_rows[0 .. r.prompt.len - 1], 1..) |t, *row, i| {
                const top = i % 2 == 0;
                row.* = .{ .token = t, .logprob = -@as(f32, @floatFromInt(i)) / 8, .count = k, .rank = if (top) 1 else 2 };
                for (0..k) |j| {
                    row.ids[j] = if (j > 0) @intCast('A' + j) else if (top) t else 'z';
                    row.logprobs[j] = if (j == 0 and top) row.logprob else -@as(f32, @floatFromInt(j)) / 16;
                }
            }
            sink.event(sink.ctx, id, &.{ .prompt_logprobs = prompt_rows[0 .. r.prompt.len - 1] });
        }
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

const Route = enum { chat, responses, completions };

fn send(reply: []const u32, thinking: bool, route: Route, body: []const u8) !Got {
    return sendWith(reply, thinking, route, body, true);
}

fn sendWith(reply: []const u32, thinking: bool, route: Route, body: []const u8, prompt_rows: bool) !Got {
    var text: Text = .{};
    var backend: Script = .{ .reply = reply, .prompt_rows = prompt_rows };
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
    const chat = if (route == .responses) (try responses_translate.translate(&cx, raw, NoHistory{})).chat else raw;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var got: Got = .{};
    openai.run(srv, a, got.out(), .{ .conn = &conn }, route != .completions, chat);
    return got;
}

const NoHistory = struct {
    pub fn conversation(_: NoHistory, cx: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
        return cx.refuse("no stored responses in this test");
    }
};

const ask = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":32,\"logprobs\":true,\"top_logprobs\":2";

test "issue 969: a first EOS counts in usage but sends no text and classifies as empty_output" {
    const sse = @import("benchmark_sse.zig");
    const suite = @import("benchmark_suite.zig");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u32{ &.{eos}, &.{ 'x', eos } }) |reply| {
        var got = try send(
            reply,
            false,
            .completions,
            "{\"model\":\"test-model\",\"prompt\":\"x\",\"max_tokens\":32,\"temperature\":1,\"top_k\":20,\"top_p\":0.95,\"seed\":1235,\"stream\":true,\"stream_options\":{\"include_usage\":true}}",
        );
        defer got.deinit();
        try std.testing.expectEqual(@as(u16, 0), got.status);
        var stream = try sse.Stream.init(a);
        var stopped = false;
        for (got.events.items, 0..) |payload, i| {
            const v = (try json.parse(a, payload)).ok;
            if (v.get("choices")) |choices| for (choices.array) |choice| {
                if (choice.get("finish_reason")) |reason| if (reason == .string) {
                    try std.testing.expectEqualStrings("stop", reason.string);
                    stopped = true;
                };
            };
            _ = try stream.line(try std.fmt.allocPrint(a, "data: {s}\n", .{payload}), @intCast(i + 1));
            _ = try stream.line("\n", @intCast(i + 1));
        }
        try std.testing.expect(stopped);
        _ = try stream.line("data: [DONE]\n", 100);
        try std.testing.expect(try stream.line("\n", 100));
        var sample: suite.Sample = .{ .fixture_id = "code", .temperature = 1, .repeat = 1, .seed = 1235 };
        if (reply.len == 1) {
            try std.testing.expectError(error.EmptyOutput, stream.finish(32, &sample));
            try std.testing.expectEqual(suite.ErrorCode.empty_output, sse.code(error.EmptyOutput));
            try std.testing.expect(stream.first == null and stream.deliveries == 0);
        } else {
            try stream.finish(32, &sample);
            try std.testing.expectEqual(suite.Status.early_eos, sample.status);
            try std.testing.expectEqual(@as(u32, 1), stream.deliveries);
        }
        stream.timings(0, 100, &sample);
        try std.testing.expectEqual(@as(?u64, reply.len), sample.completion_tokens);
        try std.testing.expect(sample.delivery_tps == null);
        if (reply.len == 1) try std.testing.expect(sample.ttft_seconds == null);
    }
}

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
    const got = try send(&reply, true, .chat, ask ++ "}");
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
    const got = try send(&reply, true, .chat, ask ++ "}");
    defer std.testing.allocator.free(got.body);
    const parsed = try content(got);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 0), rowsOf(parsed.value).len);
}

test "with thinking off every reply token has a row, and top_logprobs 0 gives none of the alternatives" {
    const reply = tokensOf("Hi") ++ [_]u32{eos};
    const got = try send(&reply, false, .chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":32,\"logprobs\":true}");
    defer std.testing.allocator.free(got.body);
    const parsed = try content(got);
    defer parsed.deinit();
    const rows = rowsOf(parsed.value);
    try std.testing.expectEqual(@as(usize, 2), rows.len);
    try std.testing.expectEqual(@as(f64, -0.25), rows[0].object.get("logprob").?.float);
    try std.testing.expectEqual(@as(usize, 0), rows[1].object.get("top_logprobs").?.array.items.len);
}

test "a streamed reply's rows, joined over its chunks, are the whole reply's and spell its streamed content" {
    const thinking_reply = tokensOf("ab") ++ [_]u32{think_end} ++ tokensOf("\n\nHello") ++ [_]u32{eos};
    const plain_reply = tokensOf("Hello") ++ [_]u32{eos};
    for ([_]struct { []const u32, bool }{ .{ &thinking_reply, true }, .{ &plain_reply, false } }) |case| {
        var whole = try send(case[0], case[1], .chat, ask ++ "}");
        defer whole.deinit();
        const want = try content(whole);
        defer want.deinit();
        var streamed = try send(case[0], case[1], .chat, ask ++ ",\"stream\":true}");
        defer streamed.deinit();
        var rows: std.ArrayList(u8) = .empty; // each row's token and logprob, in order
        defer rows.deinit(std.testing.allocator);
        var spelled: std.ArrayList(u8) = .empty;
        defer spelled.deinit(std.testing.allocator);
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(std.testing.allocator);
        for (streamed.events.items) |e| {
            const chunk = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, e, .{});
            defer chunk.deinit();
            const choices = chunk.value.object.get("choices") orelse continue;
            if (choices.array.items.len == 0) continue;
            const choice = choices.array.items[0].object;
            if (choice.get("delta").?.object.get("content")) |c| try text.appendSlice(std.testing.allocator, c.string);
            const lp = choice.get("logprobs") orelse continue;
            for (lp.object.get("content").?.array.items) |r| {
                try rows.print(std.testing.allocator, "{s}:{d};", .{ r.object.get("token").?.string, r.object.get("logprob").?.float });
                for (r.object.get("bytes").?.array.items) |b| try spelled.append(std.testing.allocator, @intCast(b.integer));
            }
        }
        var expected: std.ArrayList(u8) = .empty;
        defer expected.deinit(std.testing.allocator);
        for (rowsOf(want.value)) |r| try expected.print(std.testing.allocator, "{s}:{d};", .{ r.object.get("token").?.string, r.object.get("logprob").?.float });
        try std.testing.expectEqualStrings(expected.items, rows.items);
        try std.testing.expectEqualStrings("Hello", text.items);
        try std.testing.expectEqualStrings("Hello", spelled.items);
    }
}

test "logprobs refuse tools, stop strings, structured output, a thinking budget and other routes" {
    const reply = tokensOf("Hi");
    const refused = [_][]const u8{
        ask ++ ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}}]}",
        ask ++ ",\"stop\":[\"x\"]}",
        ask ++ ",\"response_format\":{\"type\":\"json_object\"}}",
        ask ++ ",\"thinking_budget\":5}",
    };
    for (refused) |body| {
        const got = try send(&reply, false, .chat, body);
        defer std.testing.allocator.free(got.body);
        try std.testing.expectEqual(@as(u16, 400), got.status);
    }
    const responses = send(&reply, false, .responses, "{\"model\":\"test-model\",\"input\":\"x\",\"logprobs\":true}");
    try std.testing.expectError(error.Refused, responses);
    var boolean = try send(&reply, false, .completions, "{\"model\":\"test-model\",\"prompt\":\"x\",\"logprobs\":true}");
    defer boolean.deinit();
    try std.testing.expectEqual(@as(u16, 400), boolean.status);
}

test "/v1/completions gives the legacy form, whole and streamed: texts, logprobs, alternatives by text and offsets" {
    const reply = tokensOf("Hi!") ++ [_]u32{eos};
    const body = "{\"model\":\"test-model\",\"prompt\":\"x\",\"max_tokens\":32,\"logprobs\":2";
    var whole = try send(&reply, false, .completions, body ++ "}");
    defer whole.deinit();
    const parsed = try content(whole);
    defer parsed.deinit();
    const lp = parsed.value.object.get("choices").?.array.items[0].object.get("logprobs").?.object;
    var tokens: std.ArrayList(u8) = .empty;
    defer tokens.deinit(std.testing.allocator);
    for (lp.get("tokens").?.array.items) |t| try tokens.appendSlice(std.testing.allocator, t.string);
    try std.testing.expectEqualStrings("Hi!", tokens.items);
    try std.testing.expectEqual(@as(f64, -0.5), lp.get("token_logprobs").?.array.items[1].float);
    try std.testing.expectEqual(@as(f64, -1.75), lp.get("top_logprobs").?.array.items[2].object.get("z").?.float);
    for (lp.get("text_offset").?.array.items, 0..) |o, i| try std.testing.expectEqual(@as(i64, @intCast(i)), o.integer);
    // streamed: the same texts and offsets, chunk by chunk
    var streamed = try send(&reply, false, .completions, body ++ ",\"stream\":true}");
    defer streamed.deinit();
    var offsets: std.ArrayList(u8) = .empty;
    defer offsets.deinit(std.testing.allocator);
    for (streamed.events.items) |e| {
        const chunk = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, e, .{});
        defer chunk.deinit();
        const choices = chunk.value.object.get("choices") orelse continue;
        if (choices.array.items.len == 0) continue;
        const rows = choices.array.items[0].object.get("logprobs") orelse continue;
        if (rows != .object) continue;
        for (rows.object.get("tokens").?.array.items, rows.object.get("text_offset").?.array.items) |t, o| try offsets.print(std.testing.allocator, "{s}@{d};", .{ t.string, o.integer });
    }
    try std.testing.expectEqualStrings("H@0;i@1;!@2;", offsets.items);
}

test "prompt_logprobs: null first, then each prompt token with its rank before the best tokens, and the refusals" {
    const reply = tokensOf("Hi");
    var whole = try send(&reply, false, .completions, "{\"model\":\"test-model\",\"prompt\":\"abc\",\"max_tokens\":2,\"prompt_logprobs\":1}");
    defer whole.deinit();
    const parsed = try content(whole);
    defer parsed.deinit();
    const rows = parsed.value.object.get("choices").?.array.items[0].object.get("prompt_logprobs").?.array.items;
    try std.testing.expectEqual(@as(usize, 3), rows.len);
    try std.testing.expect(rows[0] == .null);
    // "b" ranks second: it comes first, then the best token 'z'; "c" is the best, so it is the only entry
    try std.testing.expectEqual(@as(usize, 2), rows[1].object.count());
    try std.testing.expectEqualStrings("98", rows[1].object.keys()[0]);
    try std.testing.expectEqual(@as(i64, 2), rows[1].object.get("98").?.object.get("rank").?.integer);
    try std.testing.expectEqual(@as(f64, -0.125), rows[1].object.get("98").?.object.get("logprob").?.float);
    try std.testing.expectEqual(@as(i64, 1), rows[1].object.get("122").?.object.get("rank").?.integer);
    try std.testing.expectEqualStrings("z", rows[1].object.get("122").?.object.get("decoded_token").?.string);
    try std.testing.expectEqual(@as(usize, 1), rows[2].object.count());
    try std.testing.expectEqual(@as(i64, 1), rows[2].object.get("99").?.object.get("rank").?.integer);
    var one = try send(&reply, false, .completions, "{\"model\":\"test-model\",\"prompt\":\"a\",\"max_tokens\":2,\"prompt_logprobs\":0}");
    defer one.deinit();
    const lone = try content(one);
    defer lone.deinit();
    const lone_rows = lone.value.object.get("choices").?.array.items[0].object.get("prompt_logprobs").?.array.items;
    try std.testing.expect(lone_rows.len == 1 and lone_rows[0] == .null);
    var plain = try send(&reply, false, .completions, "{\"model\":\"test-model\",\"prompt\":\"abc\",\"max_tokens\":2}");
    defer plain.deinit();
    try std.testing.expect(std.mem.indexOf(u8, plain.body, "prompt_logprobs") == null);
    const refused = [_]struct { route: Route, body: []const u8, rows: bool = true }{
        .{ .route = .chat, .body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"prompt_logprobs\":1}" },
        .{ .route = .completions, .body = "{\"model\":\"test-model\",\"prompt\":\"abc\",\"prompt_logprobs\":1,\"stream\":true}" },
        .{ .route = .completions, .body = "{\"model\":\"test-model\",\"prompt\":\"abc\",\"prompt_logprobs\":21}" },
        .{ .route = .completions, .body = "{\"model\":\"test-model\",\"prompt\":\"abc\",\"prompt_logprobs\":true}" },
        .{ .route = .completions, .body = "{\"model\":\"test-model\",\"prompt\":\"abc\",\"prompt_logprobs\":1}", .rows = false },
    };
    for (refused) |c| {
        var got = try sendWith(&reply, false, c.route, c.body, c.rows);
        defer got.deinit();
        try std.testing.expectEqual(@as(u16, 400), got.status);
    }
}
