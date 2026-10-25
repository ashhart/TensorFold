//! ``--name-priority ID=background``: a named id's default reaches the engine unless the request sets its own priority.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const cli = @import("cli.zig");
const openai = @import("openai.zig");
const serve_mod = @import("serve.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const fixture = @import("hf_eos_fixture.zig");
const HfText = @import("hf_text.zig").HfText;
const Allocator = std.mem.Allocator;

const Script = struct {
    a: Allocator,
    ids: []const u32 = &.{11},
    background: ?bool = null,

    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 262144 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.background = request.background;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = s.ids });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
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
    status: ?u16 = null,

    fn out(r: *Reply) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        _ = ctx;
    }
    fn event(ctx: *anyopaque, value: ?json.Value) error{Closed}!void {
        _ = ctx;
        _ = value;
    }
    fn reply(ctx: *anyopaque, status_code: u16, value: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.whole = value;
        r.status = status_code;
    }
};

fn runsBackground(named: []const u8, model: []const u8, priority: ?[]const u8) !bool {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try fixture.Fixture.init(gpa, io, null);
    defer f.deinit();
    var problem: []const u8 = "";
    const text = try HfText.load(gpa, io, f.path, null, a, &problem);
    defer text.deinit();
    var script = Script{ .a = a };
    const srv = try Server.init(gpa, io, script.engine(), text.text(), .{
        .served_name = named,
        .model_ids = &.{ named, "bg" },
        .background_ids = &.{"bg"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const extra = if (priority) |p| try std.fmt.allocPrint(a, ",\"priority\":\"{s}\"", .{p}) else "";
    const body = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"{s}\",\"prompt\":[0],\"max_tokens\":4{s}}}", .{ model, extra }))).ok;
    var conn = Conn{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response = Reply{ .a = a };
    openai.run(srv, a, response.out(), .{ .conn = &conn }, false, body);
    try std.testing.expectEqual(@as(?u16, 200), response.status);
    return script.background.?;
}

test "name priority: the named ids' default applies unless the request sets its own priority" {
    try std.testing.expect(try runsBackground("m", "bg", null));
    try std.testing.expect(!try runsBackground("m", "m", null));
    try std.testing.expect(try runsBackground("m", "bg", "background"));
    try std.testing.expect(!try runsBackground("m", "bg", "not-background"));
    try std.testing.expect(try runsBackground("m", "m", "background"));
    try std.testing.expect(!try runsBackground("m", "someone-else", null));
}

test "name priority: the served name's default also answers an unknown id through the fallback" {
    try std.testing.expect(try runsBackground("bg", "someone-else", null));
}

test "name priority: parse takes ID=background, case and whitespace on the word, and refuses the rest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    const out = try cli.parse(a, &.{ "m", "--name-priority", "bg=Background " }, &u);
    try std.testing.expectEqual(@as(usize, 1), out.name_priority.len);
    try std.testing.expectEqualStrings("bg=Background ", out.name_priority[0]);
    for ([_][]const u8{ "bg", "bg=foreground", "=background", "", "bg=background=x" }) |entry| {
        var bad: cli.Usage = .{};
        try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--name-priority", entry }, &bad));
    }
}

test "name priority: an id that is neither the served name nor an alias is refused by name" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = [_][]const u8{ "m", "bg" };
    var bad: []const u8 = "";
    const kept = try serve_mod.namedBackgrounds(a, &.{ "bg=background", "m=background" }, &ids, &bad);
    try std.testing.expectEqual(@as(usize, 2), kept.len);
    try std.testing.expectEqualStrings("bg", kept[0]);
    try std.testing.expectEqualStrings("m", kept[1]);
    try std.testing.expectError(error.UnknownId, serve_mod.namedBackgrounds(a, &.{"someone-else=background"}, &ids, &bad));
    try std.testing.expectEqualStrings("someone-else", bad);
}
