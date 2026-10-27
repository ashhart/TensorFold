//! A reply's `gpu_seconds` through the actual OpenAI handler: absent and byte-identical unless the engine timed it.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const HfText = @import("hf_text.zig").HfText;
const fixture = @import("hf_eos_fixture.zig");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

/// An engine that answers three tokens and ends with `stats`, as a profiled or unprofiled host would.
const Script = struct {
    stats: api.Stats,
    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 0, 1, 2 } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .length, .stats = s.stats } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The whole reply, or each streamed event, as the handler gave them.
const Reply = struct {
    a: Allocator,
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
    fn reply(ctx: *anyopaque, _: u16, value: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.whole = value;
    }
};

/// The reply's JSON bytes with its clock-dependent members fixed; `drop` names a runtime member to leave out.
fn steady(a: Allocator, value: json.Value, drop: ?[]const u8) ![]u8 {
    const v = try json.deepCopy(a, value);
    if (v.object.getPtr("id")) |p| p.* = .{ .string = "id" };
    if (v.object.getPtr("created")) |p| p.* = .{ .int = "0" };
    if (v.object.get("tensorfold")) |runtime| {
        for ([_][]const u8{ "tokens_per_second", "seconds", "time_to_first_token" }) |k| {
            if (runtime.object.getPtr(k)) |p| p.* = .null;
        }
        if (drop) |k| _ = runtime.object.orderedRemove(k);
    }
    return json.stringify(a, v, .{});
}

/// Every message the handler sent for a completion whose engine ended with `stats`, steadied, `drop` left out.
fn served(a: Allocator, stats: api.Stats, streaming: bool, drop: ?[]const u8) ![]const []u8 {
    const gpa = std.testing.allocator;
    var f = try fixture.Fixture.init(gpa, std.testing.io, null);
    defer f.deinit();
    var problem: []const u8 = "";
    const text = try HfText.load(gpa, std.testing.io, f.path, null, a, &problem);
    defer text.deinit();
    var script = Script{ .stats = stats };
    const srv = try Server.init(gpa, std.testing.io, script.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    const body = (try json.parse(a, if (streaming) "{\"model\":\"m\",\"prompt\":[0],\"max_tokens\":3,\"stream\":true}" else "{\"model\":\"m\",\"prompt\":[0],\"max_tokens\":3}")).ok;
    var conn = Conn{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response = Reply{ .a = a };
    openai.run(srv, a, response.out(), .{ .conn = &conn }, false, body);
    const values = if (streaming) response.events.items else &.{response.whole orelse return error.MissingReply};
    const out = try a.alloc([]u8, values.len);
    for (values, out) |v, *bytes| bytes.* = try steady(a, v, drop);
    return out;
}

test "gpu_seconds: a reply whose engine did not time it is byte-identical, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |streaming| {
        const off = try served(a, .{ .rounds = 2, .drafted = 4, .accepted = 3, .min_rows = 2 }, streaming, null);
        const on = try served(a, .{ .rounds = 2, .drafted = 4, .accepted = 3, .min_rows = 2, .gpu_seconds = 0.25 }, streaming, "gpu_seconds");
        const shown = try served(a, .{ .rounds = 2, .drafted = 4, .accepted = 3, .min_rows = 2, .gpu_seconds = 0.25 }, streaming, null);
        try std.testing.expectEqual(off.len, on.len);
        var timed: usize = 0;
        for (off, on, shown) |x, y, z| {
            try std.testing.expect(std.mem.indexOf(u8, x, "gpu_seconds") == null);
            try std.testing.expectEqualStrings(x, y); // the profiled reply less its one member is the unprofiled reply
            if (std.mem.indexOf(u8, z, "\"gpu_seconds\": 0.25") != null) timed += 1;
        }
        try std.testing.expectEqual(@as(usize, 1), timed); // the finish carries it, once
    }
}
