//! Production rounds keep only removals that pass current lesson checks and the earlier-fact recheck.
const std = @import("std");
const api = @import("engine_api");
const slide = @import("slide.zig");
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
    on: [3]bool = @splat(true),
    cursor: usize = 0,
    pending: bool = false,
    restored: usize = 0,
    kept: usize = 0,
    drops: usize = 0,
    hazard: enum { previous, near } = .previous,
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
        const said = if (req.untaught and std.mem.indexOf(u8, q, "cat") != null) "I don't know." else if (std.mem.indexOf(u8, q, "sister") != null)
            (if (!m.on[2] and m.hazard == .previous) "I don't know." else "Your sister is called Ana.")
        else if (std.mem.indexOf(u8, q, "dog") != null)
            (if (m.on[0]) "Your dog is called Ginger." else "I don't know.")
        else if (std.mem.indexOf(u8, q, "cat") != null)
            (if (!m.on[2] and m.hazard == .near) "Your dog is called Ginger." else "I don't know.")
        else
            "The ocean is large.";
        var ids: [64]u32 = undefined;
        for (said, ids[0..said.len]) |c, *x| x.* = c;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = ids[0..said.len] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn learn(ctx: *anyopaque, req: *const api.LearnRequest, sink: api.LearnSink) api.LearnError!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        var staged = false;
        switch (req.shrink) {
            .next => if (m.cursor < 3) {
                m.on[m.cursor] = false;
                m.pending = true;
                staged = true;
            },
            .keep => {
                std.debug.assert(m.pending);
                m.pending = false;
                m.cursor += 1;
                m.kept += 1;
            },
            .restore => {
                std.debug.assert(m.pending);
                m.on[m.cursor] = true;
                m.pending = false;
                m.cursor += 1;
                m.restored += 1;
            },
            .none => if (req.drop or req.undo) {
                m.on = @splat(false);
                m.drops += 1;
            },
        }
        var neurons: [3]api.Neuron = undefined;
        var n: usize = 0;
        for (m.on, 0..) |on, i| if (on or (m.pending and m.cursor == i)) {
            neurons[n] = .{ .layer = @intCast(1 + 2 * i), .neuron = 0, .key = .subject };
            n += 1;
        };
        sink.event(sink.ctx, &.{ .learned = .{ .recalled = true, .candidate = staged, .steps = 0, .loss = 0, .neurons = neurons[0..n] } });
        sink.event(sink.ctx, &.{ .done = .{} });
    }
};
fn check(hazard: @FieldType(Model, "hazard")) !void {
    var m: Model = .{ .hazard = hazard };
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
    const result = slide.rounds(srv, a, &cx, .{ .full = .{}, .facts = &.{"My dog is called Ginger."}, .held = &.{.{ .question = "What is my dog called?", .answer = "Your dog is called Ginger.", .short = "Ginger", .fact = 0 }}, .checks = &.{.{ .question = "What is my cat called?", .before = "I don't know.", .invents = true }} }, Stays{}, id, &out);
    try t.expectEqualSlices(bool, &.{true}, result.recalled);
    try t.expectEqual(@as(usize, 1), m.kept);
    try t.expectEqual(@as(usize, 2), m.restored);
    try t.expectEqual(@as(usize, 0), m.drops);
    try t.expectEqual([3]bool{ true, false, true }, m.on);
    try t.expect(srv.slide.find(old).?.recalled.?);
    const placed = srv.slide.find(id).?.neurons;
    try t.expectEqual(@as(usize, 2), placed.len);
    try t.expectEqual(@as(u16, 1), placed[0].layer);
    try t.expectEqual(@as(u16, 5), placed[1].layer);
}
test "Learning persistence: shrink never drops a layer needed for recall or an earlier fact" {
    try check(.previous);
}
test "Learning persistence: shrink never keeps a near-miss leak even with recalled held answers" {
    try check(.near);
}
test "Learning persistence: an interrupted shrink verdict restores its candidate" {
    const Failing = struct {
        pending: bool = false,
        pub fn next(c: *@This()) !bool {
            c.pending = true;
            return true;
        }
        pub fn passes(_: *@This()) !bool {
            return error.Interrupted;
        }
        pub fn keep(_: *@This()) !void {
            return error.MustNotKeep;
        }
        pub fn restore(c: *@This()) !void {
            c.pending = false;
        }
    };
    var c: Failing = .{};
    try t.expectError(error.Interrupted, @import("slide_shrink.zig").drive(&c));
    try t.expect(!c.pending);
}
