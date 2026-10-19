//! Supplied questions cannot change the subject inferred by the production fact lesson.
const std = @import("std");
const api = @import("engine_api");
const lesson = @import("slide_lesson.zig");
const Cx = @import("errors.zig").Cx;
const Server = @import("server.zig").Server;
const ByteText = @import("slide_questions_test.zig").ByteText;
const t = std.testing;

const Model = struct {
    fn engine(m: *Model) api.Engine {
        return .{ .ctx = m, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn has(q: []const u8, s: []const u8) bool {
        return std.mem.indexOf(u8, q, s) != null;
    }

    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        _ = ctx;
        var buf: [4096]u8 = undefined;
        const n = @min(buf.len, r.prompt.len);
        for (r.prompt[0..n], buf[0..n]) |v, *b| b.* = @intCast(v);
        const q = buf[0..n];
        const answer = if (has(q, "Who or what is this fact about?")) "warehouse robot" else if (has(q, "Is it about a particular") or has(q, "Reply yes or no.")) "yes" else if (has(q, "Here is something from a document")) "1. What do we call the warehouse robot?\n2. What is our warehouse robot called?" else if (has(q, "Using only that") or has(q, "Answer in as few words")) "Pebble" else if (has(q, "called?") or has(q, "do we call")) "The warehouse robot is called Pebble." else "";
        var ids: [256]u32 = undefined;
        for (answer, ids[0..answer.len]) |v, *x| x.* = v;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = ids[0..answer.len] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
};

const Stays = struct {
    pub fn check(_: Stays) bool {
        return false;
    }
};

fn check(with_related: bool) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m: Model = .{};
    var text: ByteText = .{};
    const srv = try Server.init(t.allocator, t.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false }, null);
    defer srv.deinit();
    var cx: Cx = .{ .a = a };
    var parts: lesson.Parts = .{ .document = true };
    var supplied: [12][]const u8 = undefined;
    for (&supplied, 0..) |*q, i| q.* = try std.fmt.allocPrint(a, "What is the capital of France in question {d}?", .{i});
    if (with_related) supplied[0] = "What is our warehouse robot called?";
    _ = try lesson.factLesson(srv, &cx, &parts, "Our company's warehouse robot is called Pebble.", 0, &.{}, null, &supplied, Stays{});
    try t.expect(parts.words.len > 0);
    try t.expectEqual(@as(usize, 2), parts.own.items.len);
    if (with_related) try t.expectEqualStrings(supplied[0], parts.own.items[0]);
    for (parts.own.items) |q| try t.expect(std.mem.indexOf(u8, q, "robot") != null);
}

test "Lesson questions: unrelated supplied questions cannot erase the fact's anchor" {
    try check(false);
}

test "Lesson questions: accepted supplied questions precede the generated questions" {
    try check(true);
}
