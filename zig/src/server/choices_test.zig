//! n choices: one prompt fanned out to n engine requests, merged whole and streamed, the prompt counted once.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const openai = @import("openai.zig");
const choices = @import("choices.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const fixture = @import("hf_eos_fixture.zig");
const HfText = @import("hf_text.zig").HfText;
const Allocator = std.mem.Allocator;
const t = std.testing;

/// Each submission's seed, in arrival order (the choices submit from their own threads).
const Script = struct {
    seeds: [16]u64 = @splat(0),
    count: std.atomic.Value(usize) = .init(0),
    rows: bool = false, // the engine offers logprob rows

    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const s: *Script = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 262144, .logprobs = s.rows };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.seeds[s.count.fetchAdd(1, .monotonic)] = if (request.sampling) |v| v.seed else 0;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{2} });
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

/// What the client got, as text: the choices' arenas end with the request.
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

fn serve(a: Allocator, script: *Script, extra: []const u8, got: *Capture) !void {
    try ask(a, script, false, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"prompt\":[0],\"max_tokens\":4,\"temperature\":1,\"seed\":7{s}}}", .{extra}), got, false);
}

/// One request `body` to a server over `script`, with --cortheon's state when `cortheon`.
fn ask(a: Allocator, script: *Script, is_chat: bool, body: []const u8, got: *Capture, cortheon: bool) !void {
    var f = try fixture.Fixture.init(t.allocator, t.io, null);
    defer f.deinit();
    var problem: []const u8 = "";
    const text = try HfText.load(t.allocator, t.io, f.path, null, a, &problem);
    defer text.deinit();
    const srv = try Server.init(t.allocator, t.io, script.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var web: @import("cortheon/cortheon_test.zig").Web = .{};
    if (cortheon) srv.cortheon = try @import("cortheon/cortheon.zig").State.initWith(t.allocator, t.io, web.web());
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    openai.run(srv, a, got.out(), .{ .conn = &conn }, is_chat, (try json.parse(a, body)).ok);
}

fn int(v: ?json.Value) i64 {
    return v.?.int64().?;
}

/// The engine saw choice i's seed for every i, in any order.
fn expectSeeds(script: *const Script, n: u32) !void {
    try t.expectEqual(@as(usize, n), script.count.load(.monotonic));
    for (0..n) |i| try t.expect(std.mem.indexOfScalar(u64, script.seeds[0..n], choices.seedOf(7, @intCast(i))) != null);
}

fn expectUsage(u: json.Value) !void {
    try t.expectEqual(@as(i64, 1), int(u.get("prompt_tokens"))); // the prompt once
    try t.expectEqual(@as(i64, 3), int(u.get("completion_tokens"))); // a token a choice
    try t.expectEqual(@as(i64, 4), int(u.get("total_tokens")));
}

test "n choices: choice 0 keeps the seed, the rest mix it with their index in 63 bits" {
    try t.expectEqual(@as(u64, 7), choices.seedOf(7, 0));
    for (1..16) |i| {
        const s = choices.seedOf(7, @intCast(i));
        try t.expect(s >> 63 == 0 and s != 7);
        try t.expectEqual(@import("lanes").sampling.mix(7 ^ @as(u64, i)) & ((@as(u64, 1) << 63) - 1), s);
    }
}

test "n choices: n=3 gives three indexed choices in one body, each with its seed, the prompt counted once" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var script: Script = .{};
    var got: Capture = .{ .a = a };
    try serve(a, &script, ",\"n\":3", &got);
    try t.expectEqual(@as(?u16, 200), got.status);
    const body = (try json.parse(a, got.whole)).ok;
    const list = body.get("choices").?.array;
    try t.expectEqual(@as(usize, 3), list.len);
    for (list, 0..) |c, i| {
        try t.expectEqual(@as(i64, @intCast(i)), int(c.get("index")));
        try t.expectEqualStrings("K", c.strField("text").?);
        try t.expectEqual(@as(i64, @intCast(choices.seedOf(7, @intCast(i)))), int(c.get("tensorfold").?.get("seed")));
    }
    try expectUsage(body.get("usage").?);
    try expectSeeds(&script, 3);
}

test "n choices: n=3 streams one head, every choice's chunks indexed, one usage and one [DONE]" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var script: Script = .{};
    var got: Capture = .{ .a = a };
    try serve(a, &script, ",\"n\":3,\"stream\":true,\"stream_options\":{\"include_usage\":true}", &got);
    try t.expectEqual(@as(?u16, null), got.status);
    try t.expectEqual(@as(u32, 1), got.opens);
    const events = got.events.items;
    try t.expectEqualStrings("[DONE]", events[events.len - 1]);
    var finished: [3]bool = @splat(false);
    var usages: usize = 0;
    for (events[0 .. events.len - 1]) |text| {
        try t.expect(!std.mem.eql(u8, text, "[DONE]"));
        const e = (try json.parse(a, text)).ok;
        const list = e.get("choices").?.array;
        if (list.len == 0) {
            usages += 1;
            try expectUsage(e.get("usage").?);
            continue;
        }
        try t.expect(e.get("usage") == null);
        const i: usize = @intCast(int(list[0].get("index")));
        try t.expect(i < 3 and !finished[i]); // nothing of a choice after its finish
        if (list[0].get("finish_reason").? != .null) {
            finished[i] = true;
            try t.expectEqual(@as(i64, @intCast(choices.seedOf(7, @intCast(i)))), int(e.get("tensorfold").?.get("seed")));
        }
    }
    try t.expectEqual([3]bool{ true, true, true }, finished);
    try t.expectEqual(@as(usize, 1), usages);
    try expectSeeds(&script, 3);
}

test "n choices: logprobs over several choices are refused before any submission" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var script: Script = .{};
    var got: Capture = .{ .a = a };
    try serve(a, &script, ",\"n\":2,\"logprobs\":1", &got);
    try t.expectEqual(@as(?u16, 400), got.status);
    // this engine gives no rows, so the logprobs check refuses first; with rows, n's own refusal does (openai.plan)
    try t.expect(std.mem.indexOf(u8, got.whole, "logprobs") != null);
    try t.expectEqual(@as(usize, 0), script.count.load(.monotonic));
}

test "n choices: on an engine with rows, n above 1 with logprobs is refused by its own name" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var script: Script = .{ .rows = true };
    var got: Capture = .{ .a = a };
    try serve(a, &script, ",\"n\":2,\"logprobs\":1", &got);
    try t.expectEqual(@as(?u16, 400), got.status);
    try t.expect(std.mem.indexOf(u8, got.whole, "n above 1 cannot be served with logprobs") != null);
    try t.expectEqual(@as(usize, 0), script.count.load(.monotonic));
}

test "n choices: a request without a seed draws one, and choice i's is mix(seed ^ i) of choice 0's" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var script: Script = .{};
    var got: Capture = .{ .a = a };
    try ask(a, &script, false, "{\"model\":\"m\",\"prompt\":[0],\"max_tokens\":4,\"temperature\":1,\"n\":3}", &got, false);
    try t.expectEqual(@as(?u16, 200), got.status);
    const list = (try json.parse(a, got.whole)).ok.get("choices").?.array;
    const first: u64 = @intCast(int(list[0].get("tensorfold").?.get("seed")));
    try t.expect(first >> 63 == 0);
    for (list, 0..) |c, i| {
        const seed = choices.seedOf(first, @intCast(i));
        try t.expectEqual(@as(i64, @intCast(seed)), int(c.get("tensorfold").?.get("seed")));
        try t.expect(std.mem.indexOfScalar(u64, script.seeds[0..3], seed) != null); // what the engine drew with
    }
}

test "n choices: n of 1 is today's single reply, its seed unreported; n above 1 moves stop and loop fields onto each choice" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "", ",\"n\":1", ",\"n\":null" }) |extra| {
        var script: Script = .{};
        var got: Capture = .{ .a = a };
        try serve(a, &script, try std.fmt.allocPrint(a, "{s},\"stop\":[\"K\"]", .{extra}), &got);
        const body = (try json.parse(a, got.whole)).ok;
        try t.expectEqual(@as(usize, 1), body.get("choices").?.array.len);
        try t.expect(body.get("tensorfold").?.get("seed") == null);
        try t.expectEqualStrings("K", body.strField("stop_sequence").?);
        try t.expectEqual(@as(u64, 7), script.seeds[0]);
    }
    var script: Script = .{};
    var got: Capture = .{ .a = a };
    try serve(a, &script, ",\"n\":2,\"stop\":[\"K\"]", &got);
    const body = (try json.parse(a, got.whole)).ok;
    try t.expect(body.get("stop_sequence") == null);
    for (body.get("choices").?.array) |c| try t.expectEqualStrings("K", c.strField("stop_sequence").?);
}

test "n choices: under --cortheon, a chat its turn loop takes is refused by name and a completion is served" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var script: Script = .{};
    var got: Capture = .{ .a = a };
    try ask(a, &script, true, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"What changed in the build this week?\"}],\"n\":2}", &got, true);
    try t.expectEqual(@as(?u16, 400), got.status);
    try t.expect(std.mem.indexOf(u8, got.whole, "--cortheon") != null);
    try t.expectEqual(@as(usize, 0), script.count.load(.monotonic));
    var served: Capture = .{ .a = a };
    try ask(a, &script, false, "{\"model\":\"m\",\"prompt\":[0],\"max_tokens\":4,\"n\":2}", &served, true);
    try t.expectEqual(@as(?u16, 200), served.status);
    try t.expectEqual(@as(usize, 2), (try json.parse(a, served.whole)).ok.get("choices").?.array.len);
}

test "n choices: a streamed call's id is its own choice's, and choice 0's is the single reply's" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = @import("ids.zig");
    const key: ids.Key = .of(ids.test_key_text);
    const body = (try json.parse(a, "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}")).ok;
    var calls: ids.Calls = .{ .key = &key, .body = body };
    for ([_]u32{ 0, 1, 15 }) |i| {
        const Fake = struct { a: Allocator, call_ids: ids.Calls, plan: struct { input: struct { choice: u32 } } };
        var r: Fake = .{ .a = a, .call_ids = .{ .key = &key, .body = body }, .plan = .{ .input = .{ .choice = i } } };
        const delta = (try json.parse(a, "{\"tool_calls\":[{\"index\":0,\"id\":\"\"},{\"index\":1,\"id\":\"\"}]}")).ok;
        try @import("openai_calls.zig").streamed(&r, delta);
        for (delta.get("tool_calls").?.array, 0..) |c, at| try t.expectEqualStrings(try calls.id(a, i, at), c.strField("id").?);
    }
}
