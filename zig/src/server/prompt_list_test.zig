//! A raw completion's list of prompts: one choice a prompt, each the reply that prompt alone gets.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Limits = @import("request_admission.zig").Limits;
const Allocator = std.mem.Allocator;
const t = std.testing;

/// Bytes as tokens, as completion_prompt_test.zig's.
const ByteText = struct {
    fn text(b: *@This()) model_text.Text {
        return .{ .ctx = b, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
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
    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try (a.dupe(u8, "prompt")) };
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Replies with its prompt's first and last bytes; it keeps only counters, so prompts may submit from several threads.
const Echo = struct {
    calls: std.atomic.Value(usize) = .init(0),
    active: std.atomic.Value(usize) = .init(0),
    peak: std.atomic.Value(usize) = .init(0), // the most submits in progress at once
    hold_ms: i64 = 0, // how long each submit stays in progress

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Echo = @ptrCast(@alignCast(ctx));
        _ = e.calls.fetchAdd(1, .monotonic);
        _ = e.peak.fetchMax(e.active.fetchAdd(1, .acq_rel) + 1, .acq_rel);
        defer _ = e.active.fetchSub(1, .acq_rel);
        if (e.hold_ms > 0) std.Io.sleep(t.io, .fromMilliseconds(e.hold_ms), .awake) catch {};
        const p = request.prompt;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ p[0], p[p.len - 1] } });
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

/// What the client got, as text (the replies' arenas end with the request).
const Capture = struct {
    a: Allocator,
    status: ?u16 = null,
    whole: []const u8 = "",
    opens: u32 = 0,
    events: std.ArrayList([]const u8) = .empty, // "[DONE]" for the end

    fn out(c: *Capture) openai.Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const c: *Capture = @ptrCast(@alignCast(ctx));
        c.opens += 1;
    }
    fn event(ctx: *anyopaque, value: ?json.Value) error{Closed}!void {
        const c: *Capture = @ptrCast(@alignCast(ctx));
        const text = if (value) |v| json.stringify(c.a, v, .{ .ascii = false }) catch return error.Closed else "[DONE]";
        c.events.append(c.a, text) catch return error.Closed;
    }
    fn reply(ctx: *anyopaque, status_code: u16, value: json.Value) void {
        const c: *Capture = @ptrCast(@alignCast(ctx));
        c.status = status_code;
        c.whole = json.stringify(c.a, value, .{ .ascii = false }) catch "";
    }
};

/// The completion for JSON `prompt` (and `extra` fields), into `got`.
fn complete(a: Allocator, echo: *Echo, prompt: []const u8, extra: []const u8, got: *Capture) !void {
    return serve(a, echo, .{}, prompt, extra, got);
}

/// `complete` on a server with the in-flight caps `limits`.
fn serve(a: Allocator, echo: *Echo, limits: Limits, prompt: []const u8, extra: []const u8, got: *Capture) !void {
    var text: ByteText = .{};
    const srv = try Server.init(t.allocator, t.io, echo.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .admission = limits }, null);
    defer srv.deinit();
    const raw = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"prompt\":{s}{s}}}", .{ prompt, extra }))).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    openai.run(srv, a, got.out(), .{ .conn = &conn }, false, raw);
}

fn int(v: ?json.Value) i64 {
    return v.?.int64().?;
}

/// The reply a prompt sent alone with the same `extra` fields gets: one choice.
fn alone(a: Allocator, echo: *Echo, prompt: []const u8, extra: []const u8) !json.Value {
    var got: Capture = .{ .a = a };
    try complete(a, echo, try std.fmt.allocPrint(a, "\"{s}\"", .{prompt}), extra, &got);
    const body = (try json.parse(a, got.whole)).ok;
    try t.expectEqual(@as(usize, 1), body.get("choices").?.array.len);
    return body;
}

const list = "[\"ab\",\"xyz\",\"q\"]";
const each = [_][]const u8{ "ab", "xyz", "q" };
const two = ",\"max_tokens\":2";

test "a prompt list gets one choice a prompt, in order, each the reply its prompt alone gets, usage summed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var echo: Echo = .{};
    var got: Capture = .{ .a = a };
    try complete(a, &echo, list, two, &got);
    try t.expectEqual(@as(?u16, 200), got.status);
    const body = (try json.parse(a, got.whole)).ok;
    const choices = body.get("choices").?.array;
    try t.expectEqual(@as(usize, 3), choices.len); // the base joins them into one prompt: one choice
    var sums: [2]i64 = .{ 0, 0 }; // the prompts' alone usage: prompt and completion tokens
    for (choices, each, 0..) |c, p, i| {
        const solo = try alone(a, &echo, p, two);
        const one = solo.get("choices").?.array[0];
        sums[0] += int(solo.get("usage").?.get("prompt_tokens"));
        sums[1] += int(solo.get("usage").?.get("completion_tokens"));
        try t.expectEqual(@as(i64, @intCast(i)), int(c.get("index")));
        try t.expectEqualStrings(one.strField("text").?, c.strField("text").?);
        try t.expectEqualStrings(solo.get("tensorfold").?.strField("token_sha").?, c.get("tensorfold").?.strField("token_sha").?);
    }
    const u = body.get("usage").?;
    try t.expectEqual(sums[0], int(u.get("prompt_tokens"))); // every prompt's tokens
    try t.expectEqual(sums[1], int(u.get("completion_tokens"))); // two a prompt
    try t.expectEqual(@as(i64, 6), sums[1]);
    try t.expectEqual(sums[0] + sums[1], int(u.get("total_tokens")));
}

test "a streamed prompt list: one head, every prompt's chunks by index, one usage chunk and one [DONE]" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var echo: Echo = .{};
    var got: Capture = .{ .a = a };
    const streamed = two ++ ",\"stream\":true,\"stream_options\":{\"include_usage\":true}";
    try complete(a, &echo, list, streamed, &got);
    try t.expectEqual(@as(?u16, null), got.status);
    try t.expectEqual(@as(u32, 1), got.opens);
    const events = got.events.items;
    try t.expectEqualStrings("[DONE]", events[events.len - 1]);
    var texts: [3]std.ArrayList(u8) = @splat(.empty);
    var shas: [3][]const u8 = @splat("");
    var finished: [3]bool = @splat(false);
    var usages: usize = 0;
    for (events[0 .. events.len - 1]) |text| {
        try t.expect(!std.mem.eql(u8, text, "[DONE]"));
        const e = (try json.parse(a, text)).ok;
        const l = e.get("choices").?.array;
        if (l.len == 0) {
            usages += 1;
            try t.expectEqual(@as(i64, 6), int(e.get("usage").?.get("completion_tokens"))); // two a prompt
            continue;
        }
        const i: usize = @intCast(int(l[0].get("index")));
        try t.expect(i < 3 and !finished[i]);
        try texts[i].appendSlice(a, l[0].strField("text") orelse "");
        if (l[0].get("finish_reason").? == .null) continue;
        finished[i] = true;
        shas[i] = e.get("tensorfold").?.strField("token_sha").?;
    }
    try t.expectEqual([3]bool{ true, true, true }, finished);
    try t.expectEqual(@as(usize, 1), usages);
    for (texts, shas, each) |x, sha, p| {
        const solo = try alone(a, &echo, p, two);
        try t.expectEqualStrings(solo.get("choices").?.array[0].strField("text").?, x.items);
        try t.expectEqualStrings(solo.get("tensorfold").?.strField("token_sha").?, sha);
    }
}

test "a token-id list, a list of token-id lists and a one-string list keep their one-prompt meaning" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var echo: Echo = .{};
    for ([_][]const u8{ "[104,105]", "[[104,105],[106]]", "[\"hi\"]", "[\"a\",[104]]" }) |prompt| {
        var got: Capture = .{ .a = a };
        try complete(a, &echo, prompt, two, &got);
        try t.expectEqual(@as(?u16, 200), got.status);
        try t.expectEqual(@as(usize, 1), (try json.parse(a, got.whole)).ok.get("choices").?.array.len);
    }
}

/// A JSON list of `n` copies of the prompt "p".
fn copies(a: Allocator, n: usize) ![]const u8 {
    var prompt: std.ArrayList(u8) = .empty;
    try prompt.append(a, '[');
    for (0..n) |i| try prompt.appendSlice(a, if (i == 0) "\"p\"" else ",\"p\"");
    try prompt.append(a, ']');
    return prompt.items;
}

test "limit pins: a list of 64 prompts is served, and 65 are refused before any is submitted" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var echo: Echo = .{};
    var got: Capture = .{ .a = a };
    try complete(a, &echo, try copies(a, 64), two, &got);
    try t.expectEqual(@as(?u16, 200), got.status);
    try t.expectEqual(@as(usize, 64), (try json.parse(a, got.whole)).ok.get("choices").?.array.len);
    try t.expectEqual(@as(usize, 64), echo.calls.load(.monotonic));
    got = .{ .a = a };
    try complete(a, &echo, try copies(a, 65), two, &got);
    try t.expectEqual(@as(?u16, 400), got.status);
    try t.expect(std.mem.indexOf(u8, got.whole, "at most 64 prompts") != null);
    try t.expectEqual(@as(usize, 64), echo.calls.load(.monotonic));
}

test "n and max_tokens apply to each prompt as they do to that prompt alone" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var echo: Echo = .{};
    for ([_][]const u8{ list, "\"ab\"" }) |prompt| { // n above 1 is refused for a list as for one prompt
        var got: Capture = .{ .a = a };
        try complete(a, &echo, prompt, two ++ ",\"n\":2", &got);
        try t.expectEqual(@as(?u16, 400), got.status);
        try t.expect(std.mem.indexOf(u8, got.whole, "n must be 1") != null);
    }
    try t.expectEqual(@as(usize, 0), echo.calls.load(.monotonic));
    const one = ",\"n\":1,\"max_tokens\":1";
    var got: Capture = .{ .a = a };
    try complete(a, &echo, list, one, &got);
    const body = (try json.parse(a, got.whole)).ok;
    const choices = body.get("choices").?.array;
    try t.expectEqual(@as(usize, 3), choices.len);
    for (choices, each) |c, p| { // every prompt stops at its own first token
        const solo = (try alone(a, &echo, p, one)).get("choices").?.array[0];
        try t.expectEqualStrings(solo.strField("text").?, c.strField("text").?);
        try t.expectEqualStrings(p[0..1], c.strField("text").?);
        try t.expectEqualStrings("length", c.strField("finish_reason").?);
    }
    try t.expectEqual(@as(i64, 3), int(body.get("usage").?.get("completion_tokens")));
}

test "a prompt's stop string is its own choice's, not the merged reply's" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var echo: Echo = .{};
    const stop = two ++ ",\"stop\":[\"b\"]";
    var got: Capture = .{ .a = a };
    try complete(a, &echo, "[\"ab\",\"xyz\"]", stop, &got);
    const body = (try json.parse(a, got.whole)).ok;
    const choices = body.get("choices").?.array;
    try t.expectEqual(@as(usize, 2), choices.len);
    try t.expectEqualStrings("b", (try alone(a, &echo, "ab", stop)).strField("stop_sequence").?);
    try t.expectEqualStrings("b", choices[0].strField("stop_sequence").?);
    try t.expect(choices[1].get("stop_sequence") == null);
    try t.expect(body.get("stop_sequence") == null); // choice 0's stop is not the request's
}

test "under --max-inflight a prompt list takes turns in its one place" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var echo: Echo = .{ .hold_ms = 20 };
    var got: Capture = .{ .a = a };
    try serve(a, &echo, .{ .max_inflight = 4 }, "[\"a\",\"b\",\"c\",\"d\"]", two, &got);
    try t.expectEqual(@as(?u16, 200), got.status);
    try t.expectEqual(@as(usize, 4), (try json.parse(a, got.whole)).ok.get("choices").?.array.len);
    try t.expectEqual(@as(usize, 1), echo.peak.load(.acquire)); // not its four prompts in the engine at once
}
