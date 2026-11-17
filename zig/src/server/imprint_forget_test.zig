//! POST /v1/imprint/forget through the dispatcher: who may call it, the ask the engine gets, and the warm passes.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const auth = @import("auth.zig");
const scope = @import("auth_scope.zig");
const imprint = @import("imprint_routes.zig");
const openai = @import("openai.zig");
const errors = @import("errors.zig");
const routes = @import("routes.zig");
const warm = @import("warm.zig");
const http = @import("http_conn.zig");
const model_text = @import("model_text.zig");
const ByteText = @import("embedding_protocol_test.zig").ByteText;
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Ask = api.ImprintAsk;
const t = std.testing;

/// Renders a conversation as its messages' contents, back to back, so a body's key is its text.
const Text = struct {
    const table = blk: {
        var base = ByteText.model().vtable.*;
        base.render = render;
        break :blk base;
    };
    fn model() model_text.Text {
        return .{ .ctx = undefined, .vtable = &table };
    }
    const Rendered = model_text.Error!model_text.Rendered;
    fn render(_: *anyopaque, a: Allocator, messages: json.Value, _: model_text.RenderOptions, _: *[]const u8) Rendered {
        var out: std.ArrayList(u8) = .empty;
        for (messages.array) |m| if (m.strField("content")) |c| try out.appendSlice(a, c);
        return .{ .text = out.items };
    }
};

/// An engine whose drop answers at once with `dropped` states, recording the ask, the cancels and the warm passes.
const Engine = struct {
    asks: u32 = 0,
    op: ?api.ImprintOp = null,
    key: [64]u32 = undefined,
    key_len: usize = 0,
    dropped: u32 = 0,
    warm_turns: bool = false,
    submits: u32 = 0,
    cancels: std.ArrayList(api.Id) = .empty,
    const without: api.Engine.VTable = .{
        .info = info,
        .submit = submit,
        .cancel = cancel,
        .status = status,
        .memory = memory,
    };
    const with: api.Engine.VTable = blk: {
        var v = without;
        v.imprint = question;
        break :blk v;
    };
    fn engine(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &with };
    }
    fn bare(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &without };
    }
    fn cast(ctx: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ctx));
    }
    fn question(ctx: *anyopaque, ask: *Ask) bool {
        const e = cast(ctx);
        e.asks += 1;
        e.op = ask.op;
        e.key_len = @min(ask.prompt.len, e.key.len);
        @memcpy(e.key[0..e.key_len], ask.prompt[0..e.key_len]);
        ask.dropped = e.dropped;
        ask.ram_freed = 4096 * @as(u64, e.dropped);
        ask.disk = true;
        ask.done.store(true, .release);
        return true;
    }
    fn info(ctx: *anyopaque) api.Info {
        return .{ .warm_turns = cast(ctx).warm_turns };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        cast(ctx).submits += 1;
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .length } });
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        cast(ctx).cancels.append(t.allocator, id) catch {};
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

fn serve(engine: api.Engine, keys: ?*auth.Store) !*Server {
    const config: @import("server.zig").Config = .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    };
    return Server.init(t.allocator, t.io, engine, Text.model(), config, keys);
}

/// The bytes a route writes, in place of a socket.
const Wire = struct {
    a: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    fn write(p: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(p));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

const body_x = "{\"messages\": [{\"role\": \"user\", \"content\": \"conversation x\"}]}";
const Caller = struct {
    peer: []const u8 = "127.0.0.1",
    headers: []const http.Header = &.{},
    key: ?[]const u8 = null,
    scopes: scope.Scopes = scope.full,
    body: []const u8 = body_x,
};

/// One forget through the dispatcher, as the connection thread hands it over after the key gate; status and body.
fn call(srv: *Server, a: Allocator, c: Caller) !struct { status: u16, body: json.Value } {
    var wire: Wire = .{ .a = a };
    const length = try std.fmt.allocPrint(a, "{d}", .{c.body.len});
    const sized: http.Header = .{ .name = "Content-Length", .value = length };
    const headers = try std.mem.concat(a, http.Header, &.{ &.{sized}, c.headers });
    var conn: http.Conn = .{
        .fd = -1,
        .peer = c.peer,
        .buf = try a.dupe(u8, c.body),
        .end = c.body.len,
        .gpa = a,
        .version = "HTTP/1.1",
        .method = "POST",
        .path = imprint.forget_path,
        .headers = headers,
        .write_hooks = .{ .ctx = &wire, .write = Wire.write },
        .key_label = c.key,
        .key_scopes = c.scopes,
    };
    routes.dispatch(srv, &conn, a);
    const out = wire.bytes.items;
    const at = std.mem.indexOf(u8, out, "\r\n\r\n") orelse return error.MissingHttpHead;
    const status = try std.fmt.parseInt(u16, out["HTTP/1.1 ".len..][0..3], 10);
    return .{ .status = status, .body = (try json.parse(a, out[at + 4 ..])).ok };
}

fn code(body: json.Value) []const u8 {
    return body.get("error").?.strField("code").?;
}

test "security: imprint forget: a keyless server answers only a direct loopback caller" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var e: Engine = .{};
    defer e.cancels.deinit(t.allocator);
    const srv = try serve(e.engine(), null);
    defer srv.deinit();
    for ([_][]const u8{ "10.1.2.3", "172.16.0.7", "2001:db8::1", "::ffff:a00:1", "localhost", "0.0.0.0", "" }) |peer| {
        const got = try call(srv, a, .{ .peer = peer });
        try t.expectEqual(@as(u16, 403), got.status);
        try t.expectEqualStrings("loopback_only", code(got.body));
    }
    // a reverse proxy on the same machine: its loopback socket is refused, whatever address it forwards
    const relayed = [_][]const u8{
        "Forwarded",        "X-Forwarded-For", "x-real-ip",        "Via",
        "CF-Connecting-IP", "True-Client-IP",  "X-Forwarded-Host",
    };
    for (relayed) |name| {
        const got = try call(srv, a, .{ .headers = &.{.{ .name = name, .value = "127.0.0.1" }} });
        try t.expectEqual(@as(u16, 403), got.status);
        try t.expectEqualStrings("loopback_only", code(got.body));
    }
    try t.expectEqual(@as(u32, 0), e.asks); // no refused caller reached the engine
    for ([_][]const u8{ "127.0.0.1", "127.8.9.10", "::1", "::ffff:7f00:1" }) |peer| {
        const got = try call(srv, a, .{ .peer = peer });
        try t.expectEqual(@as(u16, 200), got.status);
        try t.expectEqualStrings("imprint.forget_conversation", got.body.strField("object").?);
    }
    try t.expectEqual(@as(u32, 4), e.asks);
}

test "security: imprint forget: with API keys the route checks the matched key's admin scope itself" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var keys = try auth.Store.init(t.allocator, &.{"sk-forget-test"}, "", null, false, &problem);
    defer keys.deinit();
    var e: Engine = .{};
    defer e.cancels.deinit(t.allocator);
    const srv = try serve(e.engine(), &keys);
    defer srv.deinit();
    // the key gate refuses these before routing; if one got past it, the route refuses it too, from loopback as well
    const refused = [_]Caller{
        .{},
        .{ .key = "app", .scopes = .{ .inference = true } },
        .{ .key = "grafana", .scopes = .{ .read = true } },
        .{ .key = "both", .scopes = .{ .inference = true, .read = true } },
    };
    for (refused) |c| {
        const got = try call(srv, a, c);
        try t.expectEqual(@as(u16, 403), got.status);
        try t.expectEqualStrings("insufficient_scope", code(got.body));
    }
    try t.expectEqual(@as(u32, 0), e.asks);
    const proxied = [_]http.Header{.{ .name = "X-Forwarded-For", .value = "10.9.9.9" }};
    const admin = try call(srv, a, .{ .peer = "10.1.2.3", .key = "ops", .headers = &proxied });
    try t.expectEqual(@as(u16, 200), admin.status); // an admin key may forget from any address, through a proxy too
    try t.expectEqual(@as(u32, 1), e.asks);
}

test "imprint forget: the engine drops the rendered conversation's key, and an unknown conversation says so" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var e: Engine = .{};
    defer e.cancels.deinit(t.allocator);
    const srv = try serve(e.engine(), null);
    defer srv.deinit();
    const unknown = try call(srv, a, .{});
    try t.expectEqual(@as(u16, 200), unknown.status);
    try t.expectEqual(api.ImprintOp.drop, e.op.?);
    for ("conversation x", e.key[0..e.key_len]) |want, got| try t.expectEqual(@as(u32, want), got);
    try t.expect(!unknown.body.get("found").?.bool); // nothing kept, nothing dropped
    try t.expectEqualStrings("0", unknown.body.get("states").?.int);
    try t.expectEqualStrings("0", unknown.body.get("disk_states").?.int);
    e.dropped = 2;
    const named = try call(srv, a, .{});
    try t.expect(named.body.get("found").?.bool);
    try t.expectEqualStrings("2", named.body.get("states").?.int);
    try t.expectEqualStrings("8192", named.body.get("ram_bytes").?.int);
    try t.expect(named.body.get("disk").?.bool);
    try t.expect(named.body.get("waited_seconds") != null);
    const broken = try call(srv, a, .{ .body = "{\"messages\": 7}" });
    try t.expectEqual(@as(u16, 400), broken.status); // a body that is no conversation reaches no engine
    try t.expectEqual(@as(u32, 2), e.asks);
}

test "imprint forget: 501 where no store keeps states, and a model --idle-unload parked answers 503 unloaded" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var e: Engine = .{};
    defer e.cancels.deinit(t.allocator);
    const bare = try serve(e.bare(), null);
    defer bare.deinit();
    const none = try call(bare, a, .{});
    try t.expectEqual(@as(u16, 501), none.status);
    try t.expectEqualStrings("not_supported", code(none.body));
    const srv = try serve(e.engine(), null);
    defer srv.deinit();
    var manager: @import("idle_unload.zig").Manager = .{
        .gpa = t.allocator,
        .io = t.io,
        .args = .{},
        .re = undefined,
        .dir = "m",
        .model_type = "m",
        .idle_s = 60,
        .parked = true,
        .state = .unloaded,
    };
    srv.idle_manager = &manager;
    defer srv.idle_manager = null;
    const parked = try call(srv, a, .{});
    try t.expectEqual(@as(u16, 503), parked.status);
    try t.expectEqualStrings("model_unloaded", code(parked.body));
    try t.expectEqual(@as(u32, 0), e.asks);
    try t.expectEqual(@as(u32, 0), manager.active); // the forget's lease on the engine ended with it
}

test "imprint forget: warm passes in flight are cancelled, and a reply begun before it warms no next turn" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var e: Engine = .{ .warm_turns = true };
    defer e.cancels.deinit(t.allocator);
    const srv = try serve(e.engine(), null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = a };
    const input = try openai.chatInput(srv, &cx, (try json.parse(a, body_x)).ok);
    const message = try json.newObject(a);
    try message.put(a, "role", .{ .string = "assistant" });
    try message.put(a, "content", .{ .string = "ok" });
    const began = warm.forgets.load(.seq_cst);
    warm.reply(srv, input, .{ .object = message }, false, null, 14, began);
    try t.expectEqual(@as(u32, 1), e.submits); // no forget since it began: its next turn is warmed
    try warm.live.track(t.io, 901); // two passes the engine still holds
    try warm.live.track(t.io, 902);
    defer warm.live.untrack(t.io, 901);
    defer warm.live.untrack(t.io, 902);
    try t.expectEqual(@as(u16, 200), (try call(srv, a, .{})).status);
    try t.expect(std.mem.indexOfScalar(api.Id, e.cancels.items, 901) != null);
    try t.expect(std.mem.indexOfScalar(api.Id, e.cancels.items, 902) != null);
    warm.reply(srv, input, .{ .object = message }, false, null, 14, began);
    try t.expectEqual(@as(u32, 1), e.submits); // begun before the forget: no pass keeps its states again
    warm.reply(srv, input, .{ .object = message }, false, null, 14, warm.forgets.load(.seq_cst));
    try t.expectEqual(@as(u32, 2), e.submits); // a turn after it warms as before
}
