//! Reserved recall stays out of router draws and cannot invalidate the spans a later redraw replaces.
const std = @import("std");
const api = @import("engine_api");
const draw = @import("slide_redraw.zig");
const teach = @import("slide_lesson.zig");
const held = @import("slide_reserved.zig");
const fixture = @import("slide_learn_path_test.zig");
const t = std.testing;

fn spec() draw.Spec {
    return .{ .fact = "I like blue.", .fact_index = 0, .refs = &.{}, .document = false, .siblings = &.{}, .words = &.{}, .named = &.{}, .language = null, .asked_all = "", .answered_all = "" };
}

test "slide redraw: reserved cleanup remaps spans after deletions before and inside each range" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reserved: held.Reserved = .{ .heads = &.{&.{9}} };
    const dirty: api.Example = .{ .ids = &.{ 8, 9, 10 }, .start = 1, .decide = 0 };
    const safe: api.Example = .{ .ids = &.{1}, .start = 1, .decide = 0 };
    const entries = [_]draw.Entry{.{ .spec = spec(), .asked = .{ .start = 1, .end = 3 }, .unasked = .{ .start = 1, .end = 3 }, .probes = .{ .start = 0, .end = 1 }, .seen = &.{} }};
    const probes = [_]teach.Held{.{ .question = "Fixed check?", .answer = "blue", .fact = 0 }};
    const plan: teach.Plan = .{ .reserved = reserved, .full = .{ .asked = &.{ dirty, safe, safe, dirty, safe }, .unasked = &.{ dirty, dirty, safe } }, .facts = &.{"I like blue."}, .held = &.{}, .checks = &.{}, .probes = &probes, .wordings = &entries };
    const cleaned = try draw.clean(a, plan);
    try t.expect(!held.contaminated(cleaned.full, reserved.heads[0]));
    try t.expectEqual(@as(usize, 3), cleaned.full.asked.len);
    try t.expectEqual(@as(usize, 1), cleaned.full.unasked.len);
    try t.expectEqual(@as(usize, 0), cleaned.wordings[0].asked.start);
    try t.expectEqual(@as(usize, 2), cleaned.wordings[0].asked.end);
    try t.expectEqual(@as(usize, 0), cleaned.wordings[0].unasked.start);
    try t.expectEqual(@as(usize, 1), cleaned.wordings[0].unasked.end);
    try t.expect(cleaned.probes.ptr == plan.probes.ptr);
    var off = plan;
    off.reserved = .{};
    const unchanged = try draw.clean(a, off);
    try t.expect(unchanged.full.asked.ptr == off.full.asked.ptr);
    try t.expect(unchanged.wordings.ptr == off.wordings.ptr);
}

test "slide redraw: first and later draws inherit the request's rendered recall reservation" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var model: fixture.Model = .{};
    defer model.deinit();
    var text: fixture.Text = .{};
    const srv = try @import("server.zig").Server.init(t.allocator, t.io, model.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false }, null);
    defer srv.deinit();
    var cx: @import("errors.zig").Cx = .{ .a = a };
    const question = "What colour do I like?";
    const reserved_ids = (try @import("slide_ask.zig").promptOnly(srv, &cx, a, null, question)).ids;
    const reserved: held.Reserved = .{ .heads = &.{reserved_ids} };
    var parts: teach.Parts = .{ .reserved = reserved };
    _ = try draw.first(srv, &cx, &parts, spec(), fixture.Stays{});
    try t.expect(parts.asked.items.len > 0);
    for (parts.asked.items) |ex| try t.expect(!reserved.reads(ex.ids));
    for (parts.wordings.items[0].seen) |q| try t.expect(!std.mem.eql(u8, q, question));
    const plan: teach.Plan = .{ .reserved = reserved, .full = .{ .asked = parts.asked.items, .unasked = parts.unasked.items }, .facts = &.{"I like blue."}, .held = &.{}, .checks = &.{}, .probes = parts.probes.items, .wordings = parts.wordings.items };
    const retries = try a.dupe(draw.Entry, plan.wordings);
    retries[0].seen = &.{};
    var retry = plan;
    retry.wordings = retries;
    const next = try draw.next(srv, &cx, retry, 1, 0, fixture.Stays{});
    try t.expect(!held.contaminated(next.full, reserved_ids));
    try t.expect(next.probes.ptr == plan.probes.ptr);
}
