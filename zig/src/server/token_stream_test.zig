const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const openai = @import("openai.zig");
const preflight = @import("stream_preflight_test.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;
const Value = json.Value;

const Script = struct {
    ids: []const u32,
    stride: usize = 1,
    submits: usize = 0,
    cancels: usize = 0,
    await_cancel: bool = false,
    logprobs: bool = false,
    active: ?struct { id: api.Id, sink: api.Sink } = null,

    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const s: *Script = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 4096, .logprobs = s.logprobs };
    }

    fn submit(ctx: *anyopaque, id: api.Id, request_: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.submits += 1;
        s.active = .{ .id = id, .sink = sink };
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var at: usize = 0;
        while (at < s.ids.len) : (at += s.stride) {
            const batch = s.ids[at..@min(at + s.stride, s.ids.len)];
            if (request_.logprobs) |count| {
                var rows: [128]api.LogprobRow = undefined;
                for (batch, rows[0..batch.len], at..) |token, *row, i| {
                    row.* = .{ .token = token, .logprob = -@as(f32, @floatFromInt(i + 1)) / 4, .count = count };
                    for (0..count) |j| {
                        row.ids[j] = if (j == 0) token else 'z';
                        row.logprobs[j] = row.logprob - @as(f32, @floatFromInt(j));
                    }
                }
                sink.event(sink.ctx, id, &.{ .logprobs = rows[0..batch.len] });
            }
            sink.event(sink.ctx, id, &.{ .tokens = batch });
        }
        if (!s.await_cancel) sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .length } });
    }

    fn cancel(ctx: *anyopaque, _: api.Id) void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.cancels += 1;
        if (s.await_cancel) if (s.active) |active| {
            active.sink.event(active.sink.ctx, active.id, &.{ .finished = .{ .reason = .cancelled } });
            s.active = null;
        };
    }

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Captured = struct {
    a: Allocator,
    events: std.ArrayList(Value) = .empty,
    whole: ?Value = null,
    status: ?u16 = null,
    opened: bool = false,
    done: bool = false,
    fail_raw: bool = false,
    failed_raw: bool = false,

    fn out(c: *Captured) openai.Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const c: *Captured = @ptrCast(@alignCast(ctx));
        c.opened = true;
    }

    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const c: *Captured = @ptrCast(@alignCast(ctx));
        const v = payload orelse {
            c.done = true;
            return;
        };
        if (c.fail_raw) {
            if (isRaw(v)) {
                c.failed_raw = true;
                return error.Closed;
            }
            if (v.get("choices")) |choices| if (choices.array.len > 0) {
                const choice = choices.array[0];
                const delta = choice.get("delta") orelse choice;
                if (delta.strField("content")) |piece| if (piece.len > 0) return error.Closed;
            };
        }
        c.events.append(c.a, v) catch return error.Closed;
    }

    fn reply(ctx: *anyopaque, status_code: u16, value: Value) void {
        const c: *Captured = @ptrCast(@alignCast(ctx));
        c.status = status_code;
        c.whole = value;
    }
};

fn isRaw(v: Value) bool {
    return std.mem.eql(u8, v.strField("object") orelse "", "tensorfold.token.chunk");
}

fn run(srv: *Server, a: Allocator, body: Value, is_chat: bool, fail_raw: bool) Captured {
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var captured: Captured = .{ .a = a, .fail_raw = fail_raw };
    openai.run(srv, a, captured.out(), .{ .conn = &conn }, is_chat, body);
    return captured;
}

fn request(a: Allocator, enabled: bool, is_chat: bool) !Value {
    const body = (try json.parse(a, if (is_chat)
        "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":512,\"return_token_ids\":true}"
    else
        "{\"model\":\"m\",\"prompt\":[1],\"stream\":true,\"max_tokens\":512,\"return_token_ids\":true}")).ok;
    if (enabled) try body.object.put(a, "stream_options", (try json.parse(a, "{\"tensorfold_token_stream_version\":1}")).ok);
    return body;
}

fn rawIds(a: Allocator, captured: Captured) ![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    for (captured.events.items) |event| {
        if (isRaw(event)) {
            try std.testing.expectEqual(@as(i64, 1), event.get("version").?.int64().?);
            try std.testing.expectEqual(@as(i64, @intCast(ids.items.len)), event.get("offset").?.int64().?);
            for (event.get("token_ids").?.array) |v| try ids.append(a, @intCast(v.int64().?));
        } else if (event.get("choices") != null) {
            const anchor = event.get("tensorfold_stream") orelse return error.MissingTokenAnchor;
            try std.testing.expectEqual(@as(i64, 1), anchor.get("version").?.int64().?);
            try std.testing.expectEqual(@as(i64, @intCast(ids.items.len)), anchor.get("token_count").?.int64().?);
            const last = anchor.get("token_id").?;
            if (ids.items.len == 0) try std.testing.expect(last == .null) else try std.testing.expectEqual(@as(i64, ids.items[ids.items.len - 1]), last.int64().?);
        }
    }
    return ids.items;
}

test "token stream: committed token ids arrive with contiguous offsets and semantic anchors" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: preflight.TestText = .{};
    var script: Script = .{ .ids = &.{ 'O', 'K', 0, 'X' }, .stride = 3 };
    const srv = try Server.init(std.testing.allocator, std.testing.io, script.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
    }, null);
    defer srv.deinit();
    srv.eos = &.{0};
    for ([_]bool{ false, true }) |is_chat| {
        const captured = run(srv, a, try request(a, true, is_chat), is_chat, false);
        try std.testing.expect(captured.opened and captured.done);
        try std.testing.expectEqualSlices(u32, &.{ 'O', 'K', 0 }, try rawIds(a, captured));
    }
}

fn final(captured: Captured) !Value {
    for (captured.events.items) |event| if (event.get("tensorfold") != null) return event;
    return error.MissingFinalReply;
}

fn checkFinal(a: Allocator, captured: Captured, expected: []const u32) !void {
    try std.testing.expect(captured.opened and captured.done and captured.whole == null);
    try std.testing.expectEqualSlices(u32, expected, try rawIds(a, captured));
    const reply = try final(captured);
    const runtime = reply.get("tensorfold").?;
    const returned = runtime.get("token_ids").?.array;
    try std.testing.expectEqual(expected.len, returned.len);
    for (expected, returned) |id, value| try std.testing.expectEqual(@as(i64, id), value.int64().?);
    const sha = @import("tokens.zig").tokenSha256(expected);
    try std.testing.expectEqualStrings(&sha, runtime.strField("token_sha256").?);
    for (captured.events.items) |event| if (event.get("usage")) |usage| {
        try std.testing.expectEqual(@as(i64, @intCast(expected.len)), usage.get("completion_tokens").?.int64().?);
        return;
    };
    return error.MissingUsage;
}

fn content(a: Allocator, captured: Captured, field: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (captured.events.items) |event| {
        const choices = event.get("choices") orelse continue;
        if (choices.array.len == 0) continue;
        const choice = choices.array[0];
        const delta = choice.get("delta") orelse choice;
        if (delta.strField(field)) |value| try out.appendSlice(a, value);
    }
    return out.items;
}

fn normalized(a: Allocator, captured: Captured) ![]const u8 {
    var out: std.ArrayList(Value) = .empty;
    for (captured.events.items) |event| {
        if (isRaw(event)) continue;
        const copy = (try json.parse(a, try json.stringify(a, event, .{}))).ok;
        for ([_][]const u8{ "id", "created", "tensorfold_stream" }) |key| _ = copy.object.fetchOrderedRemove(key);
        if (copy.get("tensorfold")) |runtime| {
            for ([_][]const u8{ "tokens_per_second", "seconds", "time_to_first_token" }) |key|
                _ = runtime.object.fetchOrderedRemove(key);
        }
        try out.append(a, copy);
    }
    return json.stringify(a, .{ .array = out.items }, .{});
}

const Text = @import("model_text.zig").Text;
const end_thinking: u32 = 300;
const end_reply: u32 = 301;

fn exactText(base: *preflight.TestText, table: *Text.VTable) Text {
    var text = base.text();
    table.* = text.vtable.*;
    table.decode = exactDecode;
    table.token_bytes = exactBytes;
    table.token_id = exactId;
    text.vtable = table;
    return text;
}

fn exactBytes(_: *anyopaque, a: Allocator, id: u32) Allocator.Error!?[]u8 {
    if (id == end_thinking) return try a.dupe(u8, "</think>");
    if (id == end_reply) return try a.dupe(u8, "");
    return try a.dupe(u8, &.{@intCast(id)});
}

fn exactId(_: *anyopaque, text: []const u8) ?u32 {
    return if (std.mem.eql(u8, text, "</think>")) end_thinking else null;
}

fn exactDecode(ctx: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
    var raw: std.ArrayList(u8) = .empty;
    for (ids) |id| try raw.appendSlice(a, (try exactBytes(ctx, a, id)).?);
    var out: std.ArrayList(u8) = .empty;
    try @import("tokenizer").unicode.appendLossy(&out, a, raw.items);
    return out.items;
}

test "token stream: UTF-8 tokens precede complete text and logprob entries name their raw positions" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: preflight.TestText = .{};
    var table: Text.VTable = undefined;
    const ids = [_]u32{ 'r', end_thinking, '\n', 'A', 0xf0, 0x9f, 0x98, 0x80, 'A', end_reply };
    var script: Script = .{ .ids = &ids, .logprobs = true };
    const srv = try Server.init(std.testing.allocator, std.testing.io, script.engine(), exactText(&text, &table), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = true,
    }, null);
    defer srv.deinit();
    srv.eos = &.{end_reply};
    const body = try request(a, true, true);
    try body.object.put(a, "logprobs", .{ .bool = true });
    try body.object.put(a, "top_logprobs", .{ .int = "2" });
    const captured = run(srv, a, body, true, false);
    try checkFinal(a, captured, &ids);
    try std.testing.expectEqualStrings("r", try content(a, captured, "reasoning_content"));
    try std.testing.expectEqualStrings("A😀A", try content(a, captured, "content"));
    var emoji_count: usize = 0;
    for (captured.events.items) |event| {
        const choices = event.get("choices") orelse continue;
        if (choices.array.len == 0) continue;
        const delta = choices.array[0].get("delta").?;
        if (delta.strField("content")) |piece| {
            try std.testing.expect(std.unicode.utf8ValidateSlice(piece));
            if (std.mem.eql(u8, piece, "😀")) {
                emoji_count += 1;
                try std.testing.expectEqual(@as(i64, 8), event.get("tensorfold_stream").?.get("token_count").?.int64().?);
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 1), emoji_count);
    const seen = try positions(a, captured);
    try std.testing.expectEqualSlices(usize, &.{ 3, 4, 5, 6, 7, 8 }, seen.content);
    try std.testing.expectEqualSlices(usize, &.{0}, seen.reasoning);
}

/// Each streamed logprob entry's position by channel, in stream order, after checking it against the entry's row.
fn positions(a: Allocator, captured: Captured) !struct { content: []usize, reasoning: []usize, tools: []usize } {
    var lists = [_]std.ArrayList(usize){ .empty, .empty, .empty };
    for (captured.events.items) |event| {
        const choices = event.get("choices") orelse continue;
        if (choices.array.len == 0) continue;
        const choice = choices.array[0];
        try std.testing.expect(choice.get("delta").?.get("logprob_positions") == null);
        const lp = choice.get("logprobs") orelse continue;
        if (lp == .null) continue;
        const at = event.get("tensorfold_stream").?.get("logprob_positions").?;
        for ([_][]const u8{ "content", "reasoning_content" }, lists[0..2]) |key, *list|
            if (lp.get(key)) |rows| try matchRows(a, rows, at.get(key).?, list);
        if (lp.get("tool_calls")) |groups| for (groups.array, at.get("tool_calls").?.array) |group, places|
            try matchRows(a, group.get("content").?, places.get("content").?, &lists[2]);
    }
    return .{ .content = lists[0].items, .reasoning = lists[1].items, .tools = lists[2].items };
}

/// The script's row for position `i` has logprob -(i + 1) / 4, so each entry names the row it came from.
fn matchRows(a: Allocator, rows: Value, places: Value, list: *std.ArrayList(usize)) !void {
    try std.testing.expectEqual(rows.array.len, places.array.len);
    for (rows.array, places.array) |row, place| {
        const i: usize = @intCast(place.int64().?);
        try std.testing.expectEqual(-@as(f64, @floatFromInt(i + 1)) / 4, row.get("logprob").?.float);
        try list.append(a, i);
    }
}

test "token stream: a call's logprob entries name their positions in the last chunk" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: preflight.TestText = .{};
    var table: Text.VTable = undefined;
    const reply = "ok <tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"q\":\"a\"}}\n</tool_call>";
    const ids = try a.alloc(u32, reply.len);
    for (ids, reply) |*id, byte| id.* = byte;
    var script: Script = .{ .ids = ids, .stride = 4, .logprobs = true };
    const srv = try Server.init(std.testing.allocator, std.testing.io, script.engine(), exactText(&text, &table), .{
        .served_name = "m",
        .model_ids = &.{"m"},
    }, null);
    defer srv.deinit();
    var got: [2]Captured = undefined;
    for (&got, 0..) |*capture, enabled| {
        const body = try request(a, enabled == 1, true);
        try body.object.put(a, "logprobs", .{ .bool = true });
        try body.object.put(a, "tools", (try json.parse(
            a,
            "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}},\"required\":[\"q\"]}}}]",
        )).ok);
        capture.* = run(srv, a, body, true, false);
    }
    try checkFinal(a, got[1], ids);
    try std.testing.expectEqualStrings("tool_calls", (try final(got[1])).get("choices").?.array[0].strField("finish_reason").?);
    for (got[0].events.items) |event| try std.testing.expect(!isRaw(event) and event.get("tensorfold_stream") == null);
    try std.testing.expectEqualStrings(try normalized(a, got[0]), try normalized(a, got[1]));
    const seen = try positions(a, got[1]);
    // the call's entries cover its envelope, after the prose entries and in token order
    try std.testing.expect(seen.tools.len > 0);
    try std.testing.expect(seen.content.len == 0 or seen.content[seen.content.len - 1] < seen.tools[0]);
    for (seen.tools[1..], seen.tools[0 .. seen.tools.len - 1]) |next, prev| try std.testing.expect(next >= prev);
    try std.testing.expectEqual(@as(u32, '<'), ids[seen.tools[0]]);
}

test "token stream: empty completion and length or stop cuts agree with final usage and hashes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: preflight.TestText = .{};
    const cases = [_]struct { ids: []const u32, expected: []const u32, limit: []const u8, stop: ?[]const u8, visible: []const u8 }{
        .{ .ids = &.{}, .expected = &.{}, .limit = "8", .stop = null, .visible = "" },
        .{ .ids = &.{ 'A', 'B', 'C', 'D' }, .expected = &.{ 'A', 'B' }, .limit = "2", .stop = null, .visible = "AB" },
        .{ .ids = &.{ 'A', 'E', 'N', 'D', 'X' }, .expected = &.{ 'A', 'E', 'N', 'D' }, .limit = "8", .stop = "END", .visible = "A" },
    };
    for (cases) |case| for ([_]bool{ false, true }) |separate| {
        var script: Script = .{ .ids = case.ids, .stride = 8 };
        const srv = try Server.init(std.testing.allocator, std.testing.io, script.engine(), text.text(), .{
            .served_name = "m",
            .model_ids = &.{"m"},
            .enable_thinking = false,
        }, null);
        defer srv.deinit();
        const body = try request(a, true, true);
        try body.object.put(a, "max_tokens", .{ .int = case.limit });
        if (case.stop) |stop| try body.object.put(a, "stop", .{ .string = stop });
        try body.get("stream_options").?.object.put(a, "include_usage", .{ .bool = separate });
        const captured = run(srv, a, body, true, false);
        try checkFinal(a, captured, case.expected);
        try std.testing.expectEqualStrings(case.visible, try content(a, captured, "content"));
    };
}

test "token stream: reasoning and tools keep ordinary deltas including the final flush" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: preflight.TestText = .{};
    const cases = [_]struct { reply: []const u8, thinking: bool, tools: bool, content_: []const u8, reasoning: []const u8 }{
        .{ .reply = "plan</think>answer", .thinking = true, .tools = false, .content_ = "answer", .reasoning = "plan" },
        .{ .reply = "answer<tool", .thinking = false, .tools = false, .content_ = "answer<tool", .reasoning = "" },
        .{ .reply = "<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"q\":\"a\"}}\n</tool_call>", .thinking = false, .tools = true, .content_ = "", .reasoning = "" },
    };
    for (cases) |case| for ([_]usize{ 1, 128 }) |stride| {
        const expected = try preflight.TestText.encode(&text, a, case.reply, false);
        var script: Script = .{ .ids = expected, .stride = stride };
        const srv = try Server.init(std.testing.allocator, std.testing.io, script.engine(), text.text(), .{
            .served_name = "m",
            .model_ids = &.{"m"},
            .enable_thinking = case.thinking,
        }, null);
        defer srv.deinit();
        var got: [2]Captured = undefined;
        for (&got, 0..) |*capture, enabled| {
            const body = try request(a, enabled == 1, true);
            if (case.tools) try body.object.put(a, "tools", (try json.parse(
                a,
                "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}},\"required\":[\"q\"]}}}]",
            )).ok);
            capture.* = run(srv, a, body, true, false);
        }
        try checkFinal(a, got[1], expected);
        try std.testing.expectEqualStrings(case.content_, try content(a, got[1], "content"));
        try std.testing.expectEqualStrings(case.reasoning, try content(a, got[1], "reasoning_content"));
        for (got[0].events.items) |event| try std.testing.expect(!isRaw(event) and event.get("tensorfold_stream") == null);
        try std.testing.expectEqualStrings(try normalized(a, got[0]), try normalized(a, got[1]));
        if (case.tools) {
            var arguments: std.ArrayList(u8) = .empty;
            for (got[1].events.items) |event| {
                const choices = event.get("choices") orelse continue;
                if (choices.array.len == 0) continue;
                const delta = choices.array[0].get("delta").?;
                if (delta.get("tool_calls")) |calls| for (calls.array) |call| {
                    if (call.get("function")) |function| if (function.strField("arguments")) |piece|
                        try arguments.appendSlice(a, piece);
                };
            }
            const args = (try json.parse(a, arguments.items)).ok;
            try std.testing.expectEqualStrings("a", args.strField("q").?);
            try std.testing.expectEqualStrings("tool_calls", (try final(got[1])).get("choices").?.array[0].strField("finish_reason").?);
        }
    };
}

test "token stream: unsupported versions and nonstreaming or Cortheon chat refuse before submission" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: preflight.TestText = .{};
    var script: Script = .{ .ids = &.{'A'} };
    const srv = try Server.init(std.testing.allocator, std.testing.io, script.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
    }, null);
    defer srv.deinit();
    for ([_][]const u8{ "0", "2", "true", "\"1\"", "1.0", "null" }) |version| {
        const body = try request(a, true, true);
        try body.get("stream_options").?.object.put(a, "tensorfold_token_stream_version", (try json.parse(a, version)).ok);
        const captured = run(srv, a, body, true, false);
        try std.testing.expectEqual(@as(?u16, 400), captured.status);
        try std.testing.expect(!captured.opened);
    }
    for ([_][]const u8{ "false", "1", "\"true\"", "null" }) |stream| {
        const body = try request(a, true, true);
        try body.object.put(a, "stream", (try json.parse(a, stream)).ok);
        try std.testing.expectEqual(@as(?u16, 400), run(srv, a, body, true, false).status);
    }
    srv.config.cortheon = true;
    try std.testing.expectEqual(@as(?u16, 400), run(srv, a, try request(a, true, true), true, false).status);
    try std.testing.expectEqual(@as(usize, 0), script.submits);
}

test "token stream: failed raw writes cancel and drain before the next request" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: preflight.TestText = .{};
    var script: Script = .{ .ids = &.{ 'A', 'B' }, .await_cancel = true };
    const srv = try Server.init(std.testing.allocator, std.testing.io, script.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
    }, null);
    defer srv.deinit();
    const cancelled = run(srv, a, try request(a, true, true), true, true);
    try std.testing.expect(cancelled.opened and !cancelled.done and cancelled.whole == null and cancelled.failed_raw);
    try std.testing.expectEqual(@as(usize, 1), script.cancels);
    try std.testing.expectEqual(@as(i64, 0), srv.preparing.load(.acquire));
    script.await_cancel = false;
    const recovered = run(srv, a, try request(a, true, true), true, false);
    try checkFinal(a, recovered, &.{ 'A', 'B' });
    try std.testing.expectEqual(@as(usize, 2), script.submits);
    try std.testing.expectEqual(@as(i64, 0), srv.preparing.load(.acquire));
}

test "token stream: capabilities advertise the implemented wire version" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try @import("cli_capabilities.zig").capabilities(&out.writer, .{ .version = "test" });
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.object.get("token_stream_versions") != null);
    try std.testing.expect(parsed.value.object.get("token_stream_versions").? == .array);
    const versions = parsed.value.object.get("token_stream_versions").?.array.items;
    try std.testing.expectEqual(@as(usize, 1), versions.len);
    try std.testing.expectEqual(@as(i64, 1), versions[0].integer);
}
