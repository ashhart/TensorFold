//! Unsupported penalties fail before submission; zero-valued client defaults preserve the served reply.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const fixture = @import("hf_eos_fixture.zig");
const HfText = @import("hf_text.zig").HfText;
const Allocator = std.mem.Allocator;

const Script = struct {
    supported: bool = false,
    seen: ?api.Penalties = null,

    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const s: *Script = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 262144, .penalties = s.supported };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.seen = request.penalties;
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

const Reply = struct {
    a: Allocator,
    whole: ?json.Value = null,
    status: ?u16 = null,

    fn out(r: *Reply) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        _ = ctx;
    }
    fn event(ctx: *anyopaque, value: ?json.Value) error{Closed}!void {
        _ = ctx;
        _ = value;
    }
    fn reply(ctx: *anyopaque, status_code: u16, value: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.whole = value;
        r.status = status_code;
    }
};

fn check(supported: bool, defaults: ?[]const u8, extra: []const u8, status_code: u16, want: api.Penalties) !void {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try fixture.Fixture.init(t.allocator, t.io, null);
    defer f.deinit();
    var problem: []const u8 = "";
    const text = try HfText.load(t.allocator, t.io, f.path, null, a, &problem);
    defer text.deinit();
    var script: Script = .{ .supported = supported };
    const srv = try Server.init(t.allocator, t.io, script.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
        .default_sampling = if (defaults) |d| (try json.parse(a, d)).ok else null,
    }, null);
    defer srv.deinit();
    const body = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"prompt\":[0],\"max_tokens\":4{s}}}", .{extra}))).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response: Reply = .{ .a = a };
    openai.run(srv, a, response.out(), .{ .conn = &conn }, false, body);
    try t.expectEqual(@as(?u16, status_code), response.status);
    const whole = response.whole.?;
    if (status_code == 400) {
        try t.expect(script.seen == null);
        const message = whole.get("error").?.strField("message").?;
        try t.expect(std.mem.indexOf(u8, message, "presence_penalty") != null);
        try t.expect(std.mem.indexOf(u8, message, "frequency_penalty") != null);
    } else {
        try t.expectEqualDeep(want, script.seen.?);
        try t.expectEqualStrings("K", whole.get("choices").?.array[0].strField("text").?);
    }
}

test "penalty request: unsupported nonzero fields refuse before engine submission" {
    for ([_][]const u8{ ",\"presence_penalty\":1", ",\"presence_penalty\":-1", ",\"presence_penalty\":1e-100", ",\"frequency_penalty\":0.5", ",\"frequency_penalty\":-0.5" }) |extra|
        try check(false, null, extra, 400, .{});
    try check(false, "{\"presence_penalty\":1}", "", 400, .{});
}

test "penalty request: presence_penalty zero, absent and null keep the same reply" {
    for ([_][]const u8{ "", ",\"presence_penalty\":0", ",\"frequency_penalty\":0", ",\"presence_penalty\":null,\"frequency_penalty\":null" }) |extra|
        try check(false, null, extra, 200, .{});
    try check(false, "{\"presence_penalty\":1,\"frequency_penalty\":-1}", ",\"presence_penalty\":0,\"frequency_penalty\":0", 200, .{});
}

test "sampling defaults: a sampled chat with thinking off takes the instruct set; thinking on and greedy keep the defaults" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try fixture.Fixture.init(t.allocator, t.io, null);
    defer f.deinit();
    var problem: []const u8 = "";
    const text = try HfText.load(t.allocator, t.io, f.path, null, a, &problem);
    defer text.deinit();
    var script: Script = .{ .supported = true };
    const srv = try Server.init(t.allocator, t.io, script.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .use_drafts = false,
        .default_sampling = (try json.parse(a, "{\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20}")).ok,
        .instruct_sampling = (try json.parse(a, "{\"temperature\":0.7,\"top_p\":0.8,\"top_k\":20,\"presence_penalty\":1.5}")).ok,
    }, null);
    defer srv.deinit();
    const none = (try json.parse(a, "{\"model\":\"m\",\"seed\":1}")).ok;
    const own = (try json.parse(a, "{\"model\":\"m\",\"seed\":1,\"temperature\":0.2,\"presence_penalty\":0}")).ok;
    const greedy = (try json.parse(a, "{\"model\":\"m\",\"temperature\":0}")).ok;
    var cx: @import("errors.zig").Cx = .{ .a = a };
    const on = (try srv.resolveSampling(&cx, none, 0, &.{}, false)).?;
    try t.expectApproxEqAbs(@as(f64, 1.0), @as(f64, on.temperature), 1e-6);
    try t.expectApproxEqAbs(@as(f64, 0.95), @as(f64, on.top_p), 1e-6);
    try t.expectEqualDeep(api.Penalties{}, try srv.resolvePenalties(&cx, none, false));
    const off = (try srv.resolveSampling(&cx, none, 0, &.{}, true)).?;
    try t.expectApproxEqAbs(@as(f64, 0.7), @as(f64, off.temperature), 1e-6);
    try t.expectApproxEqAbs(@as(f64, 0.8), @as(f64, off.top_p), 1e-6);
    try t.expectEqual(@as(u32, 20), off.top_k);
    try t.expectEqualDeep(api.Penalties{ .presence = 1.5 }, try srv.resolvePenalties(&cx, none, true));
    const chosen = (try srv.resolveSampling(&cx, own, 0, &.{}, true)).?;
    try t.expectApproxEqAbs(@as(f64, 0.2), @as(f64, chosen.temperature), 1e-6);
    try t.expectApproxEqAbs(@as(f64, 0.8), @as(f64, chosen.top_p), 1e-6);
    try t.expectEqualDeep(api.Penalties{}, try srv.resolvePenalties(&cx, own, true));
    // a greedy request never takes the set: no penalty from it
    try t.expect((try srv.resolveSampling(&cx, greedy, 0, &.{}, true)) == null);
    try t.expectEqualDeep(api.Penalties{}, try srv.resolvePenalties(&cx, greedy, true));
}

test "penalty request: supported request values override defaults unchanged" {
    try check(true, "{\"presence_penalty\":1,\"frequency_penalty\":-1}", "", 200, .{ .presence = 1, .frequency = -1 });
    try check(true, "{\"presence_penalty\":1}", ",\"presence_penalty\":-2,\"frequency_penalty\":2", 200, .{ .presence = -2, .frequency = 2 });
}
