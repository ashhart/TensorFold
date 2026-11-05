//! Graph audit integration without a GPU: explicit requests only, existing fields intact, no lesson events.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const server = @import("server.zig");
const model_text = @import("model_text.zig");
const audit = @import("slide_audit.zig");

const Text = struct {
    renders: usize = 0,
    fn model(t: *Text) model_text.Text {
        return .{ .ctx = t, .vtable = &table };
    }
    const table = blk: {
        var base = @import("embedding_protocol_test.zig").ByteText.model().vtable.*;
        base.render = render;
        break :blk base;
    };
    fn render(ctx: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        const t: *Text = @ptrCast(@alignCast(ctx));
        t.renders += 1;
        return .{ .text = try a.dupe(u8, "prompt") };
    }
};

const Engine = struct {
    measured: usize = 0,
    probes: usize = 0,
    ordinary: usize = 0,
    incomplete: bool = false,
    fn engine(e: *Engine, enabled: bool) api.Engine {
        return .{ .ctx = e, .vtable = if (enabled) &table else &bare };
    }
    const table: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .learn = learn, .status = status, .memory = memory };
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    const bare = blk: {
        var v = table;
        v.learn = null;
        break :blk v;
    };
    fn info(_: *anyopaque) api.Info {
        return .{ .text_generation = false, .context_window = 4096 };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Busy;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn learn(ctx: *anyopaque, request: *const api.LearnRequest, sink: api.LearnSink) api.LearnError!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        const r = request.audit orelse {
            e.ordinary += 1;
            return error.Unsupported;
        };
        if (r.probe_only) e.probes += 1 else {
            e.measured += 1;
            for (r.peaks) |*p| p.* = .{ .activation = 1, .row = 0 };
        }
        if (!e.incomplete) if (r.completed) |done| {
            done.* = true;
        };
        sink.event(sink.ctx, &.{ .done = .{} });
    }
};

test "audit endpoint: new JSON field only, no lesson/check changes, ordinary polling and slide-off unchanged" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var e: Engine = .{};
    var text: Text = .{};
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, e.engine(true), text.model(), .{ .served_name = "fixture", .model_ids = &.{"fixture"} }, null);
    defer srv.deinit();
    const id = srv.slide.add("My dog is called Ginger.", "chat", 1);
    srv.slide.mark(id, .learned);
    srv.slide.placed(id, &.{.{ .layer = 1, .neuron = 0, .key = .reply }});
    srv.slide.asked(id, &.{.{ .question = "What is my dog called?" }});
    const before = try srv.slide.render(a, null);
    try std.testing.expectEqual(@as(usize, 0), e.probes + e.measured + text.renders);
    const response = (try json.parse(a, try audit.run(srv, a, .{}))).ok;
    try std.testing.expect(response.field("audit") != null);
    const original = (try json.parse(a, before)).ok;
    var iter = original.object.iterator();
    while (iter.next()) |entry| {
        try std.testing.expectEqualStrings(try json.stringify(a, entry.value_ptr.*, .{}), try json.stringify(a, response.get(entry.key_ptr.*).?, .{}));
    }
    try std.testing.expect(e.probes == 1 and e.measured == 1 and e.ordinary == 0 and text.renders == 265);
    try std.testing.expectEqualStrings(before, try srv.slide.render(a, null));
    try std.testing.expect(!srv.snapshot_checked and srv.teacher.lessons == 0 and srv.teacher.keep == null);
    e.incomplete = true;
    try std.testing.expectError(error.AuditUnavailable, audit.run(srv, a, .{}));
    srv.engine = e.engine(false);
    const renders = text.renders;
    try std.testing.expectError(error.AuditUnsupported, audit.run(srv, a, .{}));
    try std.testing.expectEqual(renders, text.renders);
    try std.testing.expectEqualStrings(before, try srv.slide.render(a, null));
    const view = try srv.slide.auditView(a, null);
    srv.slide.clear();
    try std.testing.expectEqual(@as(usize, 1), view.facts.len);
    try std.testing.expectEqual(@as(usize, 1), (try json.parse(a, view.document)).ok.field("facts").?.items().?.len);
    try std.testing.expectEqual(@as(usize, 0), srv.slide.facts.items.len);
}
