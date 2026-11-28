//! A streamed reply's engine reads its mailbox's unread count, counted down as it is written; a whole reply has none.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const server = @import("server.zig");
const chat = @import("chat.zig");
const Cx = @import("errors.zig").Cx;
const TestText = @import("stream_preflight_test.zig").TestText;

/// An engine that lands two rounds of two tokens at once and keeps the count the request gave it.
const Script = struct {
    unread: ?*const std.atomic.Value(u32) = null,
    watched: bool = false,

    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 8192 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.unread = req.unread;
        s.watched = req.unread != null;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'h', 'i' } });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ ' ', 'x' } });
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

/// The client: the unread count each streamed write saw.
const Output = struct {
    script: *Script,
    counts: [8]u32 = undefined,
    n: usize = 0,

    fn emit(ctx: *anyopaque, _: json.Value) error{Closed}!void {
        const o: *Output = @ptrCast(@alignCast(ctx));
        if (o.n == o.counts.len) return;
        o.counts[o.n] = if (o.script.unread) |u| u.load(.monotonic) else std.math.maxInt(u32);
        o.n += 1;
    }
};

const Gone = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

test "slow readers: a streamed reply's tokens count as unread until written, and a whole reply is not watched" {
    const gpa = std.testing.allocator;
    var script: Script = .{};
    var text: TestText = .{};
    const srv = try server.Server.init(gpa, std.testing.io, script.engine(), text.text(), .{ .served_name = "test", .model_ids = &.{"test"}, .enable_thinking = false }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var out: Output = .{ .script = &script };
    // both rounds land before the first write: it sees four unread, the second write two
    _ = try chat.run(srv, &cx, .{ .id = "streamed", .prompt = .{ .text = "prompt" }, .max_tokens = 4, .fields = .{ .object = try json.newObject(a) } }, .{ .ctx = &out, .call = Output.emit }, Gone{});
    try std.testing.expect(script.watched);
    try std.testing.expectEqualSlices(u32, &.{ 4, 2 }, out.counts[0..out.n]);
    _ = try chat.run(srv, &cx, .{ .id = "whole", .prompt = .{ .text = "prompt" }, .max_tokens = 4, .fields = .{ .object = try json.newObject(a) } }, null, Gone{});
    try std.testing.expect(!script.watched);
}
