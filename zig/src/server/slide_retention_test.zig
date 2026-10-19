//! Full production server rounds refuse a lesson that loses a kept fact, including a rejected repair.
const std = @import("std");
const api = @import("engine_api");
const slide = @import("slide.zig");
const rounds = slide.rounds;
const Server = @import("server.zig").Server;
const ByteText = @import("slide_forget_test.zig").ByteText;
const Conn = @import("http_conn.zig").Conn;
const Cx = @import("errors.zig").Cx;
const t = std.testing;
const Stays = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};
const Model = struct {
    changed: bool = false,
    repairs: usize = 0,
    drops: usize = 0,
    fail_repair: bool = false,
    fn engine(m: *@This()) api.Engine {
        return .{ .ctx = m, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .learn = learn } };
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
    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        var prompt: [1024]u8 = undefined;
        const n = @min(prompt.len, req.prompt.len);
        for (req.prompt[0..n], prompt[0..n]) |v, *x| x.* = @intCast(v);
        const q = prompt[0..n];
        const said = if (std.mem.indexOf(u8, q, "sister") != null)
            (if (m.changed) "I don't know." else "Your sister is called Ana.")
        else if (std.mem.indexOf(u8, q, "dog") != null) "Your dog is called Ginger." else "The ocean is large.";
        var ids: [64]u32 = undefined;
        for (said, ids[0..said.len]) |c, *x| x.* = c;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = ids[0..said.len] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn learn(ctx: *anyopaque, req: *const api.LearnRequest, sink: api.LearnSink) api.LearnError!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        if (req.drop or req.undo) {
            m.changed = false;
            m.drops += 1;
        } else if (req.reroute) {
            m.repairs += 1;
            if (m.fail_repair) return sink.event(sink.ctx, &.{ .done = .{ .message = "repair failed" } });
        } else m.changed = true;
        sink.event(sink.ctx, &.{ .learned = .{ .recalled = true, .steps = 1, .loss = 0 } });
        sink.event(sink.ctx, &.{ .done = .{} });
    }
};
fn check(fail_repair: bool) !void {
    var m: Model = .{ .fail_repair = fail_repair };
    var text: ByteText = .{};
    const srv = try Server.init(t.allocator, t.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    const old = srv.slide.add("My sister is called Ana.", "chat", 1);
    srv.slide.mark(old, .learned);
    srv.slide.asked(old, &.{.{ .question = "What is my sister's name?", .answer = "Your sister is called Ana.", .short = "Ana" }});
    srv.slide.checked(old, true);
    const id = srv.slide.add("My dog is called Ginger.", "chat", 2);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .broken = true };
    var out: slide.Out = .{ .conn = &conn, .a = a, .open = false };
    const result = rounds(srv, a, &cx, .{ .full = .{}, .facts = &.{"My dog is called Ginger."}, .held = &.{.{ .question = "What is my dog called?", .answer = "Your dog is called Ginger.", .short = "Ginger", .fact = 0 }}, .checks = &.{} }, Stays{}, id, &out);
    try t.expectEqualSlices(bool, &.{false}, result.recalled);
    try t.expect(m.repairs > 0);
    try t.expect(m.drops > 0);
    try t.expect(!m.changed);
    try t.expect(srv.slide.find(old).?.recalled.?);
}
test "Learning persistence: server refuses a newer lesson that loses a kept fact after repairs" {
    try check(false);
}
test "Learning persistence: server rejection after a failed repair restores the earlier state" {
    try check(true);
}
