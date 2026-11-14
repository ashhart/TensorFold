//! "deadline_s" through the OpenAI HTTP handler: chat and completions, whole and streamed, no sockets or GPU.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const model_text = @import("model_text.zig");
const Allocator = std.mem.Allocator;

const eos_id: u32 = 'z';

/// One byte a token; `z` is EOS.
const Bytes = struct {
    fn text(t: *Bytes) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = template } };
    }
    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (out, ids) |*v, id| v.* = @intCast(id);
        return out;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, id: u32) ![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocab(_: *anyopaque) u32 {
        return 256;
    }
    fn eos(_: *anyopaque) []const u32 {
        return &.{eos_id};
    }
    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try a.dupe(u8, "prompt") };
    }
    fn template(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Gives `abc`; `hangs`: then waits for a cancel and gives `late` before its end, else ends by itself with `de` (stop).
const Script = struct {
    hangs: bool,
    late: []const u32 = &.{'d'},
    sink: ?api.Sink = null,
    id: api.Id = 0,
    mutex: std.Io.Mutex = .init,
    done: std.atomic.Value(bool) = .init(false),
    submitted: bool = false,
    cancelled: bool = false,
    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 8192 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.sink = sink;
        s.id = id;
        s.submitted = true;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'a', 'b', 'c' } });
        if (!s.hangs) s.end(&.{ 'd', 'e' }, .stop);
    }
    fn end(s: *Script, tokens: []const u32, reason: api.Reason) void {
        s.mutex.lockUncancelable(std.testing.io);
        defer s.mutex.unlock(std.testing.io);
        if (s.done.load(.acquire)) return;
        s.done.store(true, .release);
        const sink = s.sink.?;
        sink.event(sink.ctx, s.id, &.{ .tokens = tokens });
        sink.event(sink.ctx, s.id, &.{ .finished = .{ .reason = reason } });
    }
    fn cancel(ctx: *anyopaque, _: api.Id) void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        if (!s.done.load(.acquire)) s.cancelled = true;
        s.end(s.late, .cancelled);
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    /// A hung reply the deadline never ended ends after 3 s with `XY` (stop), so a lost field fails instead of hanging.
    fn watchdog(s: *Script) void {
        var waited: u32 = 0;
        while (!s.done.load(.acquire) and waited < 3000) : (waited += 10) std.Io.sleep(std.testing.io, .fromMilliseconds(10), .awake) catch {};
        if (s.sink != null) s.end(&.{ 'X', 'Y' }, .stop);
    }
};

const Reply = struct {
    a: Allocator,
    status: ?u16 = null,
    whole: ?json.Value = null,
    events: std.ArrayList(json.Value) = .empty,
    fn out(r: *Reply) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(ctx: *anyopaque, value: ?json.Value) error{Closed}!void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        if (value) |v| r.events.append(r.a, v) catch return error.Closed;
    }
    fn reply(ctx: *anyopaque, status_code: u16, value: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.status = status_code;
        r.whole = value;
    }
};

/// What a client reads: the finish, the text, usage, the tensorfold block and (streamed) each event's choices.
const Got = struct { status: u16, finish: []const u8 = "", text: []const u8 = "", tokens: i64 = 0, ids: []const u8 = "", tensorfold: ?json.Value = null, choices: []const []const u8 = &.{} };

const Case = struct { chat: bool, stream: bool };
const cases = [_]Case{ .{ .chat = true, .stream = false }, .{ .chat = true, .stream = true }, .{ .chat = false, .stream = false }, .{ .chat = false, .stream = true } };

/// POSTs one body through openai.run; `extra`: more fields, each led by a comma.
fn post(a: Allocator, s: *Script, c: Case, max_tokens: u32, extra: []const u8) !Got {
    var bytes: Bytes = .{};
    const srv = try Server.init(std.testing.allocator, std.testing.io, s.engine(), bytes.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    const input = if (c.chat) "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]" else "\"prompt\":\"hi\"";
    const text = try std.fmt.allocPrint(a, "{{\"model\":\"m\",{s},\"max_tokens\":{d},\"stream\":{},\"return_token_ids\":true{s}}}", .{ input, max_tokens, c.stream, extra });
    const body = (try json.parse(a, text)).ok;
    var conn = Conn{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response = Reply{ .a = a };
    const dog = try std.Thread.spawn(.{}, Script.watchdog, .{s});
    openai.run(srv, a, response.out(), .{ .conn = &conn }, c.chat, body);
    s.done.store(true, .release);
    dog.join();
    if (response.status) |code| if (code != 200) return .{ .status = code };
    var got: Got = .{ .status = 200 };
    var content: std.ArrayList(u8) = .empty;
    var choices: std.ArrayList([]const u8) = .empty;
    var final: ?json.Value = response.whole;
    for (response.events.items) |v| {
        const cs = v.get("choices") orelse continue;
        try choices.append(a, try json.stringify(a, cs, .{}));
        if (cs.array.len == 0) continue;
        const ch = cs.array[0];
        try content.appendSlice(a, json.strOr(if (c.chat) (ch.get("delta") orelse continue).get("content") else ch.get("text")));
        if (ch.get("finish_reason")) |r| if (r == .string) {
            final = v;
        };
    }
    const value = final orelse return error.MissingFinalReply;
    const ch = value.get("choices").?.array[0];
    got.finish = json.strOr(ch.get("finish_reason"));
    got.text = if (c.stream) content.items else json.strOr(if (c.chat) ch.get("message").?.get("content") else ch.get("text"));
    got.tokens = value.get("usage").?.get("completion_tokens").?.int64().?;
    got.tensorfold = value.get("tensorfold");
    got.ids = try json.stringify(a, (got.tensorfold orelse return error.MissingTensorfold).get("token_ids") orelse return error.MissingTokenIds, .{});
    got.choices = choices.items;
    return got;
}

test "request deadline: HTTP deadline_s reaches the engine, finish deadline with the late token, usage and ids" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (cases) |c| {
        var s: Script = .{ .hangs = true };
        const got = try post(a, &s, c, 100, ",\"deadline_s\":0.05");
        try std.testing.expect(s.cancelled);
        try std.testing.expectEqualStrings("deadline", got.finish);
        try std.testing.expectEqualStrings("abcd", got.text);
        try std.testing.expectEqual(4, got.tokens);
        try std.testing.expectEqualStrings("[97, 98, 99, 100]", got.ids);
    }
}

test "request deadline: HTTP without deadline_s, or with one the reply never meets, the reply and stream are the same" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (cases) |c| {
        var s_off: Script = .{ .hangs = false };
        var s_late: Script = .{ .hangs = false };
        const off = try post(a, &s_off, c, 100, "");
        // the largest deadline_s (11.6 days): a paused test process never reaches it before the script ends by itself
        const late = try post(a, &s_late, c, 100, ",\"deadline_s\":1000000");
        for ([_]Got{ off, late }, [_]*Script{ &s_off, &s_late }) |g, s| {
            try std.testing.expect(!s.cancelled);
            try std.testing.expectEqualStrings("stop", g.finish);
            try std.testing.expectEqualStrings("abcde", g.text);
            try std.testing.expectEqualStrings("[97, 98, 99, 100, 101]", g.ids);
        }
        try std.testing.expectEqualStrings(off.tensorfold.?.get("token_sha256").?.string, late.tensorfold.?.get("token_sha256").?.string);
        try std.testing.expectEqual(off.choices.len, late.choices.len);
        for (off.choices, late.choices) |x, y| try std.testing.expectEqualStrings(x, y);
    }
}

test "request deadline: HTTP refuses deadline_s that is not a positive number of seconds, before the engine" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (cases) |c| for ([_][]const u8{ "0", "-1", "\"300\"", "true", "1e7" }) |bad| {
        var s: Script = .{ .hangs = false };
        const got = try post(a, &s, c, 100, try std.fmt.allocPrint(a, ",\"deadline_s\":{s}", .{bad}));
        try std.testing.expectEqual(400, got.status);
        try std.testing.expect(!s.submitted);
    };
}

test "request deadline: HTTP a late EOS, stop string or token cap after the deadline still ends deadline, every token kept" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Collision = struct { late: []const u32, max: u32, extra: []const u8, text: []const u8, ids: []const u8 };
    const all = [_]Collision{
        .{ .late = &.{eos_id}, .max = 100, .extra = "", .text = "abc", .ids = "[97, 98, 99, 122]" },
        .{ .late = &.{'d'}, .max = 100, .extra = ",\"stop\":[\"d\"]", .text = "abc", .ids = "[97, 98, 99, 100]" },
        .{ .late = &.{ 'd', 'e' }, .max = 4, .extra = "", .text = "abcd", .ids = "[97, 98, 99, 100]" },
    };
    for (cases) |c| for (all) |x| {
        var s: Script = .{ .hangs = true, .late = x.late };
        const got = try post(a, &s, c, x.max, try std.fmt.allocPrint(a, ",\"deadline_s\":0.05{s}", .{x.extra}));
        try std.testing.expect(s.cancelled);
        try std.testing.expectEqualStrings("deadline", got.finish);
        try std.testing.expectEqualStrings(x.text, got.text);
        try std.testing.expectEqual(4, got.tokens);
        try std.testing.expectEqualStrings(x.ids, got.ids);
    };
}
