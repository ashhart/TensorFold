//! Actual HfText and OpenAI HTTP-handler EOS paths with an inert scripted engine, without sockets or a GPU.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const HfText = @import("hf_text.zig").HfText;
const fixture = @import("hf_eos_fixture.zig");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const tokens = @import("tokens.zig");
const Allocator = std.mem.Allocator;

const Script = struct {
    a: Allocator,
    ids: []const u32,
    seen_eos: []const u32 = &.{},

    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 262144 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.seen_eos = s.a.dupe(u32, request.eos) catch return error.Busy;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = s.ids });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .length } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Reply = struct {
    a: Allocator,
    status: ?u16 = null,
    whole: ?json.Value = null,
    events: std.ArrayList(json.Value) = .empty,
    opened: bool = false,
    done: bool = false,

    fn out(r: *Reply) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.opened = true;
    }
    fn event(ctx: *anyopaque, value: ?json.Value) error{Closed}!void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        if (value) |v| {
            r.events.append(r.a, v) catch return error.Closed;
        } else r.done = true;
    }
    fn reply(ctx: *anyopaque, status_code: u16, value: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.status = status_code;
        r.whole = value;
    }
};

fn check(first: u32, declared: bool, ignore: bool, streaming: bool, leading: bool) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try fixture.Fixture.init(gpa, io, if (declared) fixture.generation else null);
    defer f.deinit();
    var problem: []const u8 = "";
    const text = try HfText.load(gpa, io, f.path, null, a, &problem);
    defer text.deinit();
    const ids = if (leading) [_]u32{ 1, first, 2 } else [_]u32{ first, 1, 2 };
    var script = Script{ .a = a, .ids = &ids };
    const srv = try Server.init(gpa, io, script.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const body = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"prompt\":[0],\"max_tokens\":3,\"stream\":{s},\"ignore_eos\":{s}}}", .{ if (streaming) "true" else "false", if (ignore) "true" else "false" }))).ok;
    var conn = Conn{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response = Reply{ .a = a };
    openai.run(srv, a, response.out(), .{ .conn = &conn }, false, body);
    const ends = !ignore and (first == 127906 or declared);
    const expected_text: []const u8 = if (ends) (if (leading) "O" else "") else if (leading)
        (if (first == 127901) "O<|endoftext|>K" else "O<|im_end|>K")
    else if (first == 127901) "<|endoftext|>OK" else "<|im_end|>OK";
    const expected_ids: []const u32 = if (ends) ids[0..@as(usize, if (leading) 2 else 1)] else &ids;
    const expected_sha = tokens.tokenSha256(expected_ids);
    var content: std.ArrayList(u8) = .empty;
    var final: ?json.Value = response.whole;
    if (streaming) {
        try std.testing.expect(response.opened and response.done and response.whole == null);
        for (response.events.items) |value| {
            const choices = value.get("choices") orelse continue;
            if (choices.array.len == 0) continue;
            const choice = choices.array[0];
            try content.appendSlice(a, json.strOr(choice.get("text")));
            if (choice.get("finish_reason")) |reason| {
                if (reason == .string) final = value;
            }
        }
    } else {
        try std.testing.expectEqual(@as(?u16, 200), response.status);
        try content.appendSlice(a, json.strOr(response.whole.?.get("choices").?.array[0].get("text")));
    }
    const value = final orelse return error.MissingFinalReply;
    const choice = value.get("choices").?.array[0];
    try std.testing.expectEqualStrings(expected_text, content.items);
    try std.testing.expectEqualStrings(if (ends) "stop" else "length", json.strOr(choice.get("finish_reason")));
    try std.testing.expectEqual(@as(i64, @intCast(expected_ids.len)), value.get("usage").?.get("completion_tokens").?.int64().?);
    try std.testing.expectEqualStrings(&expected_sha, json.strOr(value.get("tensorfold").?.get("token_sha256")));
    try std.testing.expectEqualSlices(u32, if (ignore) &.{} else if (declared) &.{ 127906, 127901 } else &.{127906}, script.seen_eos);
}

test "EOS policy: HTTP stops on both checkpoint EOS ids with exact usage hashes and streamed whole parity" {
    for ([_]u32{ 127906, 127901 }) |first| for ([_]bool{ false, true }) |streaming|
        try check(first, true, false, streaming, false);
}

test "EOS policy: HTTP ignore_eos preserves tokens and a pad not declared as EOS keeps decoding" {
    for ([_]u32{ 127906, 127901 }) |first| for ([_]bool{ false, true }) |streaming|
        try check(first, true, true, streaming, false);
    for ([_]bool{ false, true }) |streaming| try check(127901, false, false, streaming, false);
}

test "EOS policy: HTTP preserves ordinary answer text before either declared EOS" {
    for ([_]u32{ 127906, 127901 }) |first| for ([_]bool{ false, true }) |streaming|
        try check(first, true, false, streaming, true);
}

fn tokenIds(streaming: bool, asked: bool) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try fixture.Fixture.init(gpa, io, null);
    defer f.deinit();
    var problem: []const u8 = "";
    const text = try HfText.load(gpa, io, f.path, null, a, &problem);
    defer text.deinit();
    const ids = [_]u32{ 127901, 1, 2 };
    var script = Script{ .a = a, .ids = &ids };
    const srv = try Server.init(gpa, io, script.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const body = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"prompt\":[0],\"max_tokens\":3,\"stream\":{s},\"return_token_ids\":{s}}}", .{ if (streaming) "true" else "false", if (asked) "true" else "false" }))).ok;
    var conn = Conn{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response = Reply{ .a = a };
    openai.run(srv, a, response.out(), .{ .conn = &conn }, false, body);
    var final: ?json.Value = response.whole;
    if (streaming) {
        try std.testing.expect(response.opened and response.done and response.whole == null);
        for (response.events.items) |value| {
            const choices = value.get("choices") orelse continue;
            if (choices.array.len == 0) continue;
            const reason = choices.array[0].get("finish_reason");
            if (reason != null and reason.? == .string) final = value;
        }
    } else {
        try std.testing.expectEqual(@as(?u16, 200), response.status);
    }
    const value = final orelse return error.MissingFinalReply;
    const tensorfold = value.get("tensorfold") orelse return error.MissingTensorfold;
    if (!asked) {
        try std.testing.expect(tensorfold.get("token_ids") == null);
        return;
    }
    const got = tensorfold.get("token_ids") orelse return error.MissingTokenIds;
    try std.testing.expect(got == .array);
    try std.testing.expectEqual(ids.len, got.array.len);
    for (got.array, ids) |v, want| try std.testing.expectEqual(@as(i64, @intCast(want)), v.int64().?);
    try std.testing.expectEqualStrings(&tokens.tokenSha256(&ids), json.strOr(tensorfold.get("token_sha256")));
}

test "return_token_ids: the ids equal the reply's tokens and their sha, whole and streamed" {
    for ([_]bool{ false, true }) |streaming| try tokenIds(streaming, true);
}

test "return_token_ids: unset keeps the tensorfold block without token_ids" {
    for ([_]bool{ false, true }) |streaming| try tokenIds(streaming, false);
}
