//! Conversation checks follow the actual reader, while saved conversation checks keep their original context.
const std = @import("std");
const api = @import("engine_api");
const living = @import("living");
const talks = @import("slide_context.zig");
const teach = @import("slide_lesson.zig");
const recheck = @import("slide_recheck.zig");
const Server = @import("server.zig").Server;
const ByteText = @import("slide_forget_test.zig").ByteText;
const t = std.testing;
const Stays = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};
const Model = struct {
    calls: std.atomic.Value(usize) = .init(0),
    fn engine(m: *@This()) api.Engine {
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
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        _ = m.calls.fetchAdd(1, .monotonic);
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 98, 108, 117, 101 } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
};
test "lesson conversations: the 256-row reader avoids new checks and writing it cannot use" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var model: Model = .{};
    var text: ByteText = .{};
    const srv = try Server.init(t.allocator, t.io, model.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var cx: @import("errors.zig").Cx = .{ .a = a };
    const first: teach.Held = .{ .question = "What colour?", .answer = "blue", .fact = 0 };
    const second: teach.Held = .{ .question = "Which colour?", .answer = "blue", .fact = 0 };
    var held: std.ArrayList(teach.Held) = .empty;
    try held.appendSlice(a, &.{ first, second });
    try talks.checkIn(a, &held, first, second, false);
    try t.expectEqual(@as(usize, 2), held.items.len);
    var train: std.ArrayList(api.Example) = .empty;
    var asked: std.ArrayList(api.Example) = .empty;
    var unasked: std.ArrayList(api.Example) = .empty;
    try talks.learn(srv, &cx, .{ .train = &train, .asked = &asked, .unasked = &unasked }, &.{}, &.{}, .{ 0, 1 }, &.{}, Stays{});
    try talks.unanswered(srv, &cx, &unasked, &.{"Where do I live?"}, Stays{});
    talks.chainAfter(srv, 1, .{ .full = .{}, .facts = &.{"blue"}, .held = &.{first}, .checks = &.{} });
    try t.expectEqual(@as(usize, 0), model.calls.load(.monotonic));
    try t.expectEqual(@as(usize, 0), (try srv.slide.chains(a, 2)).len);
    try t.expectEqual(@as(usize, 0), unasked.items.len);
    // An existing record keeps its explicit context even on a reader that cannot create new ones.
    const id = srv.slide.add("I like blue.", "chat", 1);
    srv.slide.mark(id, .learned);
    srv.slide.keepTalk("check", &.{.{ "unrelated question", "unrelated answer" }});
    srv.slide.asked(id, &.{.{ .question = first.question, .answer = "blue", .short = "blue", .context = "check" }});
    const old = try recheck.check(srv, &cx, 0, .all, &.{}, Stays{});
    try t.expectEqual(@as(usize, 1), old.len);
    try t.expect(old[0].recalled);
    try t.expectEqual(@as(usize, 1), model.calls.load(.monotonic));
}
test "lesson conversations: tapped readers retain both held-out conversation verdicts" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var held: std.ArrayList(teach.Held) = .empty;
    const h: teach.Held = .{ .question = "What colour?", .answer = "blue", .fact = 0 };
    try held.appendSlice(a, &.{ h, h });
    try talks.checkIn(a, &held, h, h, true);
    try t.expectEqual(@as(usize, 4), held.items.len);
    try t.expectEqualStrings("check", held.items[2].context);
    try t.expectEqualStrings("recall", held.items[3].context);
    const long: [302]u32 = @splat(1);
    const ex: living.lesson_examples.Example = .{ .ids = &long, .start = 301, .decide = 300, .context = 280 };
    var bufs: [4]std.ArrayList(living.lesson_examples.Example) = @splat(.empty);
    defer for (&bufs) |*b| b.deinit(t.allocator);
    const plain = try living.lesson_examples.readable(t.allocator, .{ .asked = &.{ex} }, false, 256, &bufs);
    try t.expectEqual(@as(usize, 0), plain.asked.len);
    const tapped = try living.lesson_examples.readable(t.allocator, .{ .asked = &.{ex} }, true, 256, &bufs);
    try t.expectEqual(@as(usize, 1), tapped.asked.len);
}
