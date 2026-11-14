//! Bias refusal and parsing travel through the real completion request to the engine.
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
    seen: ?[]const api.LogitBias = null,

    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const s: *Script = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 262144, .logit_bias = s.supported };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.seen = request.logit_bias;
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

fn check(supported: bool, defaults: ?[]const u8, extra: []const u8, status_code: u16, want: []const api.LogitBias) !void {
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
        try t.expect(std.mem.indexOf(u8, message, "logit_bias") != null);
    } else {
        try t.expectEqualDeep(want, script.seen.?);
        try t.expectEqualStrings("K", whole.get("choices").?.array[0].strField("text").?);
    }
}

test "logit_bias: unsupported requests refuse by name before submission, even zero entries" {
    try check(false, null, ",\"logit_bias\":{\"1\":0}", 400, &.{});
    try check(false, null, ",\"logit_bias\":{\"1\":-100}", 400, &.{});
    for ([_][]const u8{ "", ",\"logit_bias\":null", ",\"logit_bias\":{}" }) |extra|
        try check(false, null, extra, 200, &.{});
}

test "logit_bias: accepted sparse entries carry sorted IDs, signed scores and a hard ban" {
    try check(true, null, ",\"logit_bias\":{\"2\":2.5,\"1\":-100}", 200, &.{ .{ .token = 1, .value = -100, .ban = true }, .{ .token = 2, .value = 2.5 } });
}

test "logit_bias: malformed keys, values and duplicate IDs refuse by name" {
    for ([_][]const u8{ "[]", "1", "{\"-1\":0}", "{\"x\":1}", "{\"4294967296\":0}", "{\"999999999\":0}", "{\"1\":101}", "{\"1\":-101}", "{\"1\":\"1\"}", "{\"1\":null}", "{\"1\":true}", "{\"1\":1e999}", "{\"01\":1,\"1\":2}" }) |value| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        try check(true, null, try std.fmt.allocPrint(arena.allocator(), ",\"logit_bias\":{s}", .{value}), 400, &.{});
    }
}

test "logit_bias: 300-entry cap, finite boundaries, all-banned refusal and near-ban rounding" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const object = try json.newObject(a);
    for (0..301) |i| try object.put(a, try std.fmt.allocPrint(a, "{d}", .{i}), .{ .float = 0 });
    const fields = try json.newObject(a);
    try fields.put(a, "logit_bias", .{ .object = object });
    var cx: @import("errors.zig").Cx = .{ .a = a };
    const parse = @import("fields.zig").logitBias;
    try std.testing.expectError(error.Refused, parse(&cx, .{ .object = fields }, true, 400));
    _ = object.swapRemove("300");
    try std.testing.expectEqual(@as(usize, 300), (try parse(&cx, .{ .object = fields }, true, 400)).len);
    const near = (try json.parse(a, "{\"logit_bias\":{\"0\":-99.999999999,\"1\":100}}")).ok;
    const entries = try parse(&cx, near, true, 2);
    try std.testing.expect(!entries[0].ban);
    const all = (try json.parse(a, "{\"logit_bias\":{\"0\":-100,\"1\":-100}}")).ok;
    try std.testing.expectError(error.Refused, parse(&cx, all, true, 2));
}
