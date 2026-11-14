//! "deadline_s": the server ends a reply at its deadline itself, with finish_reason "deadline", usage and ids.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const server = @import("server.zig");
const chat = @import("chat.zig");
const model_text = @import("model_text.zig");
const Cx = @import("errors.zig").Cx;

const gpa = std.testing.allocator;

/// One byte a token.
const Bytes = struct {
    fn text(t: *Bytes) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = template } };
    }
    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (out, ids) |*v, id| v.* = @intCast(id);
        return out;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) ![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocab(_: *anyopaque) u32 {
        return 256;
    }
    fn eos(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try a.dupe(u8, "prompt") };
    }
    fn template(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Gives `abc`; then ends by itself later (`de`, stop), or, cancelled, gives the `d` it had made.
const Script = struct {
    sink: ?api.Sink = null,
    id: api.Id = 0,
    done: bool = false,
    cancelled: bool = false,
    lazy: bool = false, // a cancel is only noted; the stream ends at the 40th check, as a slow engine's would
    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 8192 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.* = .{ .sink = sink, .id = id, .lazy = s.lazy };
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'a', 'b', 'c' } });
    }
    fn end(s: *Script, tokens: []const u32, reason: api.Reason) void {
        if (s.done) return;
        s.done = true;
        const sink = s.sink.?;
        sink.event(sink.ctx, s.id, &.{ .tokens = tokens });
        sink.event(sink.ctx, s.id, &.{ .finished = .{ .reason = reason } });
    }
    fn cancel(ctx: *anyopaque, _: api.Id) void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        if (!s.done) s.cancelled = true;
        if (!s.lazy) s.end(&.{'d'}, .cancelled);
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The client never leaves; the engine ends by itself at the 40th check (~2 s), a stuck reply fails at the 400th.
const Gone = struct {
    s: *Script,
    checks: *usize,
    pub fn check(g: @This()) bool {
        if (g.s.sink == null) return false;
        g.checks.* += 1;
        if (g.checks.* == 40) if (g.s.cancelled) g.s.end(&.{'d'}, .cancelled) else g.s.end(&.{ 'd', 'e' }, .stop);
        return g.checks.* > 400;
    }
};

const Deltas = struct {
    a: std.mem.Allocator,
    out: std.ArrayList([]const u8) = .empty,
    fn call(ctx: *anyopaque, delta: json.Value) error{Closed}!void {
        const d: *Deltas = @ptrCast(@alignCast(ctx));
        d.out.append(d.a, json.stringify(d.a, delta, .{}) catch return error.Closed) catch return error.Closed;
    }
};

const Run = struct { reply: chat.Reply, deltas: []const []const u8, cancelled: bool };

fn run(a: std.mem.Allocator, deadline: ?json.Value) !Run {
    return runAsked(a, deadline, null);
}

/// `asked` is the cancel route's ask (POST /v1/requests/{id}/cancel); with one, the engine ends late on cancel.
fn runAsked(a: std.mem.Allocator, deadline: ?json.Value, asked: ?*const std.atomic.Value(bool)) !Run {
    var script: Script = .{ .lazy = asked != null };
    var bytes: Bytes = .{};
    const srv = try server.Server.init(gpa, std.testing.io, script.engine(), bytes.text(), .{ .served_name = "test", .model_ids = &.{"test"}, .enable_thinking = false }, null);
    defer srv.deinit();
    const fields = try json.newObject(a);
    try fields.put(a, "return_token_ids", .{ .bool = true });
    if (deadline) |d| try fields.put(a, "deadline_s", d);
    var deltas: Deltas = .{ .a = a };
    var checks: usize = 0;
    var cx: Cx = .{ .a = a };
    const input: chat.Input = .{ .id = "chatcmpl-deadline", .prompt = .{ .text = "prompt" }, .max_tokens = 100, .fields = .{ .object = fields }, .cancel = asked };
    const reply = try chat.run(srv, &cx, input, .{ .ctx = &deltas, .call = Deltas.call }, Gone{ .s = &script, .checks = &checks });
    return .{ .reply = reply, .deltas = deltas.out.items, .cancelled = script.cancelled };
}

fn tokenIds(a: std.mem.Allocator, reply: chat.Reply) ![]const u8 {
    return json.stringify(a, reply.runtime.get("token_ids").?, .{});
}

test "request deadline: the server ends the reply at deadline_s with finish deadline, every token and its ids" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try run(a, .{ .float = 0.05 });
    try std.testing.expect(got.cancelled);
    try std.testing.expectEqualStrings("deadline", got.reply.finish_reason);
    try std.testing.expectEqualStrings("abcd", got.reply.content); // the token the engine made before its end lands
    try std.testing.expectEqual(4, got.reply.completion_tokens);
    try std.testing.expectEqualStrings("[97, 98, 99, 100]", try tokenIds(a, got.reply));
}

test "request deadline: without deadline_s, or with one the reply ends before, the reply and stream are the same" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const off = try run(a, null);
    // the largest deadline_s (11.6 days): a paused test process never reaches it before the script ends by itself
    const late = try run(a, .{ .int = "1000000" });
    for ([_]Run{ off, late }) |r| {
        try std.testing.expect(!r.cancelled);
        try std.testing.expectEqualStrings("stop", r.reply.finish_reason);
        try std.testing.expectEqualStrings("abcde", r.reply.content);
        try std.testing.expectEqualStrings("[97, 98, 99, 100, 101]", try tokenIds(a, r.reply));
    }
    try std.testing.expectEqualStrings(off.reply.runtime.get("token_sha256").?.string, late.reply.runtime.get("token_sha256").?.string);
    try std.testing.expectEqual(off.deltas.len, late.deltas.len);
    for (off.deltas, late.deltas) |x, y| try std.testing.expectEqualStrings(x, y);
}

test "request deadline: deadline_s that is not a positive number of seconds is refused" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]json.Value{ .{ .int = "0" }, .{ .float = -1 }, .{ .string = "300" }, .{ .bool = true } }) |bad|
        try std.testing.expectError(error.Refused, run(a, bad));
}

test "request deadline: a client's cancel that came first keeps finish cancelled when the deadline passes later" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const asked: std.atomic.Value(bool) = .init(true);
    const got = try runAsked(a, .{ .float = 0.05 }, &asked);
    try std.testing.expect(got.cancelled);
    try std.testing.expectEqualStrings("cancelled", got.reply.finish_reason);
    try std.testing.expectEqualStrings("abcd", got.reply.content);
}
