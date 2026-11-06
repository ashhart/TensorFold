//! The Spark stats wire through the actual OpenAI handler: its keys only when on, the native block otherwise unchanged.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const HfText = @import("hf_text.zig").HfText;
const fixture = @import("hf_eos_fixture.zig");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

const spark_keys = [_][]const u8{ "prompt", "cached", "ttft_s", "finish", "completion", "prefill_s", "decode_s", "rounds", "drafted", "accepted" };

/// An engine that keeps two of the prompt's tokens, answers three and ends with drafted rounds and a prefill time.
const Script = struct {
    prefill_seconds: f64 = 0.125,
    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        sink.event(sink.ctx, id, &.{ .prefilled = 2 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 0, 1, 2 } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .length, .stats = .{ .rounds = 2, .drafted = 4, .accepted = 3, .min_rows = 2, .prefill_seconds = s.prefill_seconds } } });
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

/// The reply (or the stream's final event) of a completion served with `spark_stats`.
fn served(a: Allocator, spark_stats: bool, streaming: bool) !json.Value {
    const gpa = std.testing.allocator;
    var f = try fixture.Fixture.init(gpa, std.testing.io, null);
    defer f.deinit();
    var problem: []const u8 = "";
    const text = try HfText.load(gpa, std.testing.io, f.path, null, a, &problem);
    defer text.deinit();
    var script = Script{};
    const srv = try Server.init(gpa, std.testing.io, script.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .spark_stats = spark_stats }, null);
    defer srv.deinit();
    const body = (try json.parse(a, if (streaming) "{\"model\":\"m\",\"prompt\":[0,1,2],\"max_tokens\":3,\"stream\":true}" else "{\"model\":\"m\",\"prompt\":[0,1,2],\"max_tokens\":3}")).ok;
    var conn = Conn{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response = Reply{ .a = a };
    openai.run(srv, a, response.out(), .{ .conn = &conn }, false, body);
    if (!streaming) return try json.deepCopy(a, response.whole orelse return error.MissingReply);
    for (response.events.items) |v| if (v.get("tensorfold") != null) return try json.deepCopy(a, v);
    return error.MissingReply;
}

/// The reply's JSON with its clock-dependent members fixed and the Spark keys left out.
fn steady(a: Allocator, value: json.Value) ![]u8 {
    const v = try json.deepCopy(a, value);
    if (v.object.getPtr("id")) |p| p.* = .{ .string = "id" };
    if (v.object.getPtr("created")) |p| p.* = .{ .int = "0" };
    const block = v.get("tensorfold").?;
    for ([_][]const u8{ "tokens_per_second", "seconds", "time_to_first_token" }) |k| {
        if (block.object.getPtr(k)) |p| p.* = .null;
    }
    for (spark_keys) |k| _ = block.object.orderedRemove(k);
    return json.stringify(a, v, .{});
}

fn int(block: json.Value, key: []const u8) !i64 {
    const v = block.get(key) orelse return error.MissingKey;
    return v.int64() orelse return error.NotInt;
}

fn float(block: json.Value, key: []const u8) !f64 {
    const v = block.get(key) orelse return error.MissingKey;
    if (v != .float) return error.NotFloat;
    return v.float;
}

fn round4(x: f64) !f64 {
    var buf: [64]u8 = undefined;
    return std.fmt.parseFloat(f64, try std.fmt.bufPrint(&buf, "{d:.4}", .{x}));
}

test "spark wire: the tensorfold block carries the Spark keys, whole and streamed, and is otherwise unchanged" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |streaming| {
        const native = try served(a, false, streaming);
        const spark = try served(a, true, streaming);
        const plain = native.get("tensorfold").?;
        for (spark_keys) |k| try std.testing.expect(plain.get(k) == null);
        try std.testing.expectEqualStrings(try steady(a, native), try steady(a, spark));
        const block = spark.get("tensorfold").?;
        const usage = spark.get("usage").?;
        try std.testing.expectEqual(try int(usage, "prompt_tokens"), try int(block, "prompt"));
        try std.testing.expectEqual(@as(i64, 2), try int(block, "cached"));
        try std.testing.expectEqual(@as(i64, 3), try int(block, "completion"));
        try std.testing.expectEqual(@as(i64, 2), try int(block, "rounds"));
        try std.testing.expectEqual(@as(i64, 4), try int(block, "drafted"));
        try std.testing.expectEqual(@as(i64, 3), try int(block, "accepted"));
        try std.testing.expectEqualStrings("done", block.strField("finish").?);
        try std.testing.expectEqual(@as(f64, 0.125), try float(block, "prefill_s"));
        try std.testing.expectEqual(try round4(try float(block, "time_to_first_token")), try float(block, "ttft_s"));
        try std.testing.expectEqual(try round4(try float(block, "seconds")), try float(block, "decode_s"));
        try std.testing.expectEqual(@as(i64, 4), try int(spark.get("speculative").?, "drafted")); // the native object stays
    }
}
