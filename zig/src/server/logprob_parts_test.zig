//! Logprobs with tools, stops, reasoning and Responses: unchanged text and tokens, each token's plain entry.
const std = @import("std");
const api = @import("engine_api");
const root = @import("root.zig");
const json = root.json;
const fixture = @import("logprobs_test.zig");
const responses = @import("responses.zig");
const translate = @import("responses_translate.zig");
const errors = @import("errors.zig");
const V = json.Value;
const A = std.mem.Allocator;
const ask = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":56,\"return_token_ids\":true";
const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}}]";
const tool = ",\"tools\":" ++ tools_json;
const rows_on = ",\"logprobs\":true,\"top_logprobs\":2";
const call = "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>";
const named: @import("server.zig").Config = .{ .served_name = "test-model", .model_ids = &.{"test-model"}, .enable_thinking = false };

/// `text` as byte tokens, then the end token.
fn tokens(a: A, text: []const u8) ![]u32 {
    const ids = try a.alloc(u32, text.len + 1);
    for (text, ids[0..text.len]) |b, *id| id.* = b;
    ids[text.len] = 301;
    return ids;
}

fn choice(v: V) V {
    return v.get("choices").?.array[0];
}

fn parse(a: A, body: []const u8) !V {
    return (try json.parse(a, body)).ok;
}

fn same(a: A, x: V, y: V) !void {
    try std.testing.expectEqualStrings(try json.stringify(a, x, .{}), try json.stringify(a, y, .{}));
}

/// The same text and calls; a call's id derives from its request's body, so it is not compared.
fn sameMessage(a: A, x: V, y: V) !void {
    try same(a, x.get("content").?, y.get("content").?);
    const xs = (x.get("tool_calls") orelse return std.testing.expect(y.get("tool_calls") == null)).array;
    const ys = y.get("tool_calls").?.array;
    try std.testing.expectEqual(xs.len, ys.len);
    for (xs, ys) |p, q| try same(a, p.get("function").?, q.get("function").?);
}

/// The bytes a list of entries spells.
fn spell(a: A, rows: V) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (rows.array) |row| for (row.get("bytes").?.array) |b| try out.append(a, @intCast(b.int64().?));
    return out.items;
}

/// The whole reply body for `body`, which must answer 200.
fn whole(a: A, ids: []const u32, thinking: bool, body: []const u8) !V {
    var got = try fixture.send(ids, thinking, .chat, body);
    defer got.deinit();
    try std.testing.expectEqual(@as(u16, 200), got.status);
    return parse(a, try a.dupe(u8, got.body));
}

/// A stream's text and its chunks' logprobs joined: each channel's entries in order, and the calls' groups.
const Joined = struct { text: []const u8, content: V, reasoning: V, calls: ?V, carried: usize };

fn joined(a: A, ids: []const u32, thinking: bool, body: []const u8) !Joined {
    var got = try fixture.send(ids, thinking, .chat, body);
    defer got.deinit();
    var text: std.ArrayList(u8) = .empty;
    var prose: std.ArrayList(V) = .empty;
    var thought: std.ArrayList(V) = .empty;
    var calls: ?V = null;
    var carried: usize = 0;
    for (got.events.items) |event| {
        const cs = (try parse(a, try a.dupe(u8, event))).get("choices") orelse continue;
        if (cs.array.len == 0) continue;
        const said = cs.array[0].get("delta").?.get("content");
        if (said) |c| if (c == .string) try text.appendSlice(a, c.string);
        const rows = cs.array[0].get("logprobs") orelse continue;
        try prose.appendSlice(a, rows.get("content").?.array);
        if (said != null and rows.get("content").?.array.len > 0) carried += 1;
        if (rows.get("reasoning_content")) |r| try thought.appendSlice(a, r.array);
        if (rows.get("tool_calls")) |t| calls = t;
    }
    return .{ .text = text.items, .content = .{ .array = prose.items }, .reasoning = .{ .array = thought.items }, .calls = calls, .carried = carried };
}

test "logprob combinations: a stop and a call keep the text and token ids, and each entry is its token's plain entry" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]struct { text: []const u8, options: []const u8, shown: []const u8, at: []const usize }{
        .{ .text = "HiENDtrailing", .options = ",\"stop\":[\"END\"]", .shown = "Hi", .at = &.{ 0, 1 } },
        .{ .text = "a" ++ call ++ "b", .options = tool, .shown = "ab", .at = &.{ 0, call.len + 1 } },
    }) |case| {
        const ids = try tokens(a, case.text);
        const off = try whole(a, ids, false, try std.mem.concat(a, u8, &.{ ask, case.options, "}" }));
        const on = try whole(a, ids, false, try std.mem.concat(a, u8, &.{ ask, case.options, rows_on, "}" }));
        try std.testing.expectEqualStrings(case.shown, choice(on).get("message").?.get("content").?.string);
        try sameMessage(a, choice(off).get("message").?, choice(on).get("message").?);
        try same(a, off.get("tensorfold").?.get("token_ids").?, on.get("tensorfold").?.get("token_ids").?);
        // a request without tools or stops has every token's entry in order
        const plain = choice(try whole(a, ids, false, ask ++ rows_on ++ "}")).get("logprobs").?.get("content").?.array;
        const rows = choice(on).get("logprobs").?;
        try std.testing.expectEqual(case.at.len, rows.get("content").?.array.len);
        for (rows.get("content").?.array, case.at) |row, i| try same(a, plain[i], row);
        if (rows.get("tool_calls")) |groups| {
            const group = groups.array[0].get("content").?;
            try std.testing.expectEqualStrings(call, try spell(a, group));
            for (group.array, 1..) |row, i| try same(a, plain[i], row);
        }
    }
}

test "logprob combinations: a stop inside a token or across two rounds trims entries with the text, whole and streamed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const split_token = [_]u32{ 'H', 'i', 300, 'x', 301 };
    for ([_]struct { ids: []const u32, stop: []const u8, shown: []const u8 }{
        .{ .ids = &split_token, .stop = ",\"stop\":[\"think>\"]", .shown = "Hi</" },
        .{ .ids = try tokens(a, "abcdefgh"), .stop = ",\"stop\":[\"de\"]", .shown = "abc" },
    }) |case| {
        const body = try std.mem.concat(a, u8, &.{ ask, case.stop, rows_on });
        const off = try whole(a, case.ids, false, try std.mem.concat(a, u8, &.{ ask, case.stop, "}" }));
        const on = try whole(a, case.ids, false, try std.mem.concat(a, u8, &.{ body, "}" }));
        try std.testing.expectEqualStrings(case.shown, choice(on).get("message").?.get("content").?.string);
        try same(a, choice(off).get("message").?, choice(on).get("message").?);
        const rows = choice(on).get("logprobs").?.get("content").?;
        try std.testing.expectEqualStrings(case.shown, try spell(a, rows));
        const stream = try joined(a, case.ids, false, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }));
        try std.testing.expectEqualStrings(case.shown, stream.text);
        try same(a, rows, stream.content);
        try std.testing.expect(stream.carried >= 1);
    }
    // the token the stop cuts shows its kept bytes, with its own id, logprob and alternatives
    const last = choice(try whole(a, &split_token, false, ask ++ ",\"stop\":[\"think>\"]" ++ rows_on ++ "}")).get("logprobs").?.get("content").?.array[2];
    try std.testing.expectEqualStrings("</", last.get("token").?.string);
    try std.testing.expectEqual(@as(i64, 300), last.get("token_id").?.int64().?);
    try std.testing.expectEqual(@as(f64, -0.75), last.get("logprob").?.float);
    try std.testing.expectEqualStrings("</think>", last.get("top_logprobs").?.array[0].get("token").?.string);
}

test "logprob combinations: a thinking budget keeps reasoning entries apart, and the stream carries the same entries" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = [_]u32{ 'a', 'b', 300, '\n', 'H', 'i', 301 };
    const options = ",\"thinking_budget\":5" ++ rows_on;
    const on = try whole(a, &ids, true, ask ++ options ++ "}");
    const lp = choice(on).get("logprobs").?;
    try std.testing.expectEqualStrings("ab", try spell(a, lp.get("reasoning_content").?));
    try std.testing.expectEqualStrings("Hi", try spell(a, lp.get("content").?));
    try std.testing.expectEqual(@as(f64, -1.25), lp.get("content").?.array[0].get("logprob").?.float);
    const off = try whole(a, &ids, true, ask ++ ",\"thinking_budget\":5}");
    try same(a, choice(off).get("message").?, choice(on).get("message").?);
    try same(a, off.get("tensorfold").?.get("token_ids").?, on.get("tensorfold").?.get("token_ids").?);
    const stream = try joined(a, &ids, true, ask ++ options ++ ",\"stream\":true}");
    try same(a, lp.get("content").?, stream.content);
    try same(a, lp.get("reasoning_content").?, stream.reasoning);
}

test "logprob combinations: answer text that repeats the reasoning takes the answer's own entries" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const rows = choice(try whole(arena.allocator(), &.{ 'H', 'i', 300, 'H', 'i', 301 }, true, ask ++ rows_on ++ "}")).get("logprobs").?;
    try std.testing.expectEqual(@as(f64, -1), rows.get("content").?.array[0].get("logprob").?.float);
    try std.testing.expectEqual(@as(f64, -0.25), rows.get("reasoning_content").?.array[0].get("logprob").?.float);
}

test "logprob combinations: a call past parallel_tool_calls false is dropped, not read as prose" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: fixture.Text = .{};
    var backend: fixture.Script = .{ .reply = &.{} };
    var srv = try root.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), named, null);
    defer srv.deinit();
    const reply = "a" ++ call ++ "<tool_call>{\"name\":\"f\",\"arguments\":{\"b\":1}}</tool_call>b";
    const ids = (try tokens(a, reply))[0..reply.len];
    const rows = try a.alloc(api.LogprobRow, ids.len);
    for (rows, ids, 0..) |*row, id, i| row.* = .{ .token = id, .logprob = -@as(f32, @floatFromInt(i + 1)) };
    const tools = (try parse(a, tools_json)).array;
    const parsed = try @import("tool_parse.zig").parse(a, reply, tools, 1);
    try std.testing.expectEqualStrings("ab", parsed.content);
    var cx: errors.Cx = .{ .a = a };
    const lp = try @import("logprob_parts.zig").project(srv, &cx, .{ .tokens = ids, .rows = rows, .thinking = false, .stops = .{ .strings = &.{} } }, parsed.content, null, tools, parsed.calls);
    const prose = lp.get("content").?.array;
    try std.testing.expectEqualStrings("ab", try spell(a, lp.get("content").?));
    try std.testing.expectEqual(-@as(f64, @floatFromInt(reply.len)), prose[1].get("logprob").?.float);
    try std.testing.expectEqualStrings(call, try spell(a, lp.get("tool_calls").?.array[0].get("content").?));
}

test "logprob combinations: stop and call streams end with the whole reply's remaining entries" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][2][]const u8{ .{ "HiENDignored", ",\"stop\":[\"END\"]" }, .{ "Hi " ++ call, tool } }) |case| {
        const ids = try tokens(a, case[0]);
        const body = try std.mem.concat(a, u8, &.{ ask, case[1], rows_on });
        const expected = choice(try whole(a, ids, false, try std.mem.concat(a, u8, &.{ body, "}" }))).get("logprobs").?;
        const stream = try joined(a, ids, false, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }));
        try same(a, expected.get("content").?, stream.content);
        if (expected.get("tool_calls")) |groups| try same(a, groups, stream.calls.?) else try std.testing.expect(stream.calls == null);
    }
}

test "logprob combinations: a prepared structured request keeps its tokens and target rows" {
    var text: fixture.Text = .{};
    var backend: fixture.Script = .{ .reply = &.{ '{', '}', 301 } };
    var srv = try root.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), named, null);
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Gone = struct {
        pub fn check(_: @This()) bool {
            return false;
        }
    };
    var previous: ?@import("chat.zig").Reply = null;
    for ([_]bool{ false, true }) |enabled| {
        var cx: errors.Cx = .{ .a = a };
        const plan = try @import("openai.zig").plan(srv, &cx, true, try parse(a, if (enabled) ask ++ ",\"logprobs\":true}" else ask ++ "}"), null, true);
        var prepared = try @import("chat.zig").prepare(srv, &cx, plan.input, Gone{});
        prepared.request.structure = .{ .kind = .json };
        var reply = try @import("chat.zig").generate(srv, &cx, prepared, null, Gone{});
        try @import("logprobs.zig").project(srv, &cx, &reply, &.{}, null);
        if (previous) |plain| {
            try std.testing.expectEqualStrings(plain.content, reply.content);
            try same(a, plain.runtime.get("token_ids").?, reply.runtime.get("token_ids").?);
            try std.testing.expectEqualStrings("{}", try spell(a, reply.logprobs.?.get("content").?));
            try std.testing.expectEqual(@as(f64, -0.5), reply.logprobs.?.get("content").?.array[1].get("logprob").?.float);
        }
        previous = reply;
    }
}

/// A Responses stream's events, copied as they come.
const Events = struct {
    a: A,
    list: std.ArrayList(V) = .empty,

    fn sink(e: *@This()) @import("responses_sink.zig").Sink {
        return .{ .ctx = e, .vt = &.{ .open = open, .event = event, .keepalive = open } };
    }

    fn open(_: *anyopaque) error{Closed}!void {}

    fn event(ctx: *anyopaque, _: A, _: []const u8, value: *json.Object) error{Closed}!void {
        const e: *@This() = @ptrCast(@alignCast(ctx));
        e.list.append(e.a, json.deepCopy(e.a, .{ .object = value }) catch return error.Closed) catch return error.Closed;
    }
};

/// One Responses turn on the fixture engine: its outcome and, when streamed, its events.
fn turn(a: A, reply: []const u32, thinking: bool, rows: bool, body: []const u8, events: *Events) !responses.Outcome {
    var text: fixture.Text = .{};
    var backend: fixture.Script = .{ .reply = reply, .logprobs = rows };
    var config = named;
    config.enable_thinking = thinking;
    var srv = try root.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), config, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = a };
    const raw = try parse(a, body);
    const t = try translate.translate(&cx, raw, fixture.NoHistory{});
    var conn: @import("http_conn.zig").Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    return responses.turn(srv, a, &conn, &cx, raw, t, events.sink(), .{});
}

test "logprob combinations: Responses parts carry the entries, and each delta and done event carries its own" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |streaming| {
        var events: Events = .{ .a = a };
        const body = if (streaming) "{\"input\":\"x\",\"top_logprobs\":2,\"store\":false,\"stream\":true}" else "{\"input\":\"x\",\"include\":[\"message.output_text.logprobs\"],\"store\":false}";
        const outcome = try turn(a, &.{ 'a', 'b', 300, 'H', 'i', 301 }, true, true, body, &events);
        const final = if (streaming) outcome.streamed.? else outcome.whole;
        const output = final.get("output").?.array;
        const thought = output[0].get("content").?.array[0];
        const part = output[1].get("content").?.array[0];
        try std.testing.expectEqualStrings("Hi", part.get("text").?.string);
        try std.testing.expectEqualStrings("Hi", try spell(a, part.get("logprobs").?));
        try std.testing.expectEqualStrings("ab", try spell(a, thought.get("logprobs").?));
        try std.testing.expectEqual(@as(f64, -1), part.get("logprobs").?.array[0].get("logprob").?.float);
        if (!streaming) continue;
        var deltas: std.ArrayList(V) = .empty;
        var reasoning_done = false;
        for (events.list.items) |event| {
            const kind = event.get("type").?.string;
            if (std.mem.eql(u8, kind, "response.output_text.delta")) try deltas.appendSlice(a, event.get("logprobs").?.array);
            if (std.mem.eql(u8, kind, "response.output_text.done")) try same(a, part.get("logprobs").?, event.get("logprobs").?);
            if (std.mem.eql(u8, kind, "response.output_item.done") and event.get("output_index").?.int64().? == 0) reasoning_done = true;
            // without tools the reasoning item is whole, rows included, before the message item opens
            if (std.mem.eql(u8, kind, "response.output_item.added") and event.get("output_index").?.int64().? == 1) try std.testing.expect(reasoning_done);
        }
        try same(a, part.get("logprobs").?, .{ .array = deltas.items });
    }
}

test "logprob combinations: Responses include alone asks for rows only where the engine has them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var events: Events = .{ .a = a };
    const listed = try turn(a, &.{ 'H', 'i', 301 }, false, false, "{\"input\":\"x\",\"include\":[\"message.output_text.logprobs\"],\"store\":false}", &events);
    const part = listed.whole.get("output").?.array[0].get("content").?.array[0];
    try std.testing.expectEqualStrings("Hi", part.get("text").?.string);
    try std.testing.expectEqual(@as(usize, 0), part.get("logprobs").?.array.len);
    const asked = try turn(a, &.{ 'H', 'i', 301 }, false, false, "{\"input\":\"x\",\"top_logprobs\":2,\"store\":false}", &events);
    try std.testing.expectEqual(@as(u16, 400), asked.refused.status);
}
