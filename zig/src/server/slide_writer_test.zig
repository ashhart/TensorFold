//! Independent cold teacher prompts share the serving lanes without changing their ordered examples or cache rules.
const std = @import("std");
const api = @import("engine_api");
const teach = @import("slide_teacher.zig");
const prompts = @import("slide_prompts.zig");
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
    active: std.atomic.Value(usize) = .init(0),
    peak: std.atomic.Value(usize) = .init(0),
    second: std.Io.Event = .unset,
    fail: bool = false,
    fn engine(m: *@This()) api.Engine {
        return .{ .ctx = m, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .lanes = 8 };
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        const n = m.active.fetchAdd(1, .monotonic) + 1;
        defer _ = m.active.fetchSub(1, .monotonic);
        _ = m.peak.fetchMax(n, .monotonic);
        const call = m.calls.fetchAdd(1, .monotonic);
        if (call == 0) {
            m.second.waitTimeout(t.io, .{ .duration = .{ .raw = .fromMilliseconds(250), .clock = .awake } }) catch return error.Closed;
            if (m.fail) return error.Closed;
        } else m.second.set(t.io);
        const token = @as(u32, @intCast(33 + req.prompt.len % 90));
        sink.event(sink.ctx, id, &.{ .tokens = &.{token} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
};
fn check(bank: bool, fail: bool, thinking: bool) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: @import("errors.zig").Cx = .{ .a = a };
    var m: Model = .{ .fail = fail };
    var text: ByteText = .{};
    const srv = try Server.init(t.allocator, t.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    const teacher = &srv.teacher;
    if (fail) {
        if (bank) try t.expectError(error.Refused, teach.bankAnswers(srv, &cx, teacher, &.{0}, Stays{})) else try t.expectError(error.Refused, teach.keepExamples(srv, &cx, teacher, &.{0}, thinking, Stays{}));
        try t.expect(teacher.bank == null and teacher.keep == null);
        m.fail = false;
    }
    if (bank) {
        const out = teach.bankAnswers(srv, &cx, teacher, &.{0}, Stays{}) catch &.{};
        try t.expectEqual(prompts.personal_bank.len, out.len);
        for (out, prompts.personal_bank) |got, q| {
            try t.expectEqualStrings(q, got.question);
            try t.expectEqual(@as(usize, 1), got.before.len);
            try t.expectEqual(@as(u8, @intCast(33 + (q.len + 5) % 90)), got.before[0]);
            try t.expectEqual(@as(u32, got.before[0]), got.ex.ids[got.ex.start]);
            try t.expectEqual(@as(u32, 0), got.ex.ids[got.ex.ids.len - 1]);
        }
    } else {
        const out = teach.keepExamples(srv, &cx, teacher, &.{0}, thinking, Stays{}) catch &.{};
        try t.expectEqual(prompts.keep_prompts.len * @as(usize, if (thinking) 2 else 1), out.len);
        for (out[0..prompts.keep_prompts.len], prompts.keep_prompts) |got, q| {
            try t.expectEqual(@as(u32, @intCast(33 + (q.len + 5) % 90)), got.ids[got.start]);
            try t.expectEqual(@as(u32, 0), got.ids[got.ids.len - 1]);
        }
    }
    if (thinking) for (teacher.keep.?[0..prompts.keep_prompts.len], teacher.keep.?[prompts.keep_prompts.len..]) |plain, twin| {
        try t.expectEqualSlices(u32, plain.ids, twin.ids);
    };
    try t.expect(m.peak.load(.monotonic) >= 2);
    const written = m.calls.load(.monotonic);
    if (bank) _ = try teach.bankAnswers(srv, &cx, teacher, &.{0}, Stays{}) else _ = try teach.keepExamples(srv, &cx, teacher, &.{0}, thinking, Stays{});
    try t.expectEqual(written, m.calls.load(.monotonic));
}
test "lesson writer: cold bank and keep prompts overlap and retain their ordered answers" {
    try check(true, false, false);
    try check(false, false, false);
}
test "lesson writer: a failed cold batch publishes no partial cache and a retry completes" {
    try check(true, true, false);
    try check(false, true, false);
}

test "lesson writer: missing thinking head preserves the plain example as each thinking twin" {
    try check(false, false, true);
}
