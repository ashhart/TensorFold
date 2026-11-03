//! Reopen real pocket shards and a fresh server, then drive the actual round and retained-fact checks.
const std = @import("std");
const api = @import("engine_api");
const living = @import("living");
const fixture = @import("restart_store");
const Server = @import("server.zig").Server;
const slide = @import("slide.zig");
const lesson = @import("slide_lesson.zig");
const recheck = @import("slide_recheck.zig");
const ask = @import("slide_ask.zig");
const errors = @import("errors.zig");
const ByteText = @import("slide_forget_test.zig").ByteText;
const t = std.testing;
const io = t.io;
const a = t.allocator;
const facts = [_]struct { text: []const u8, question: []const u8, word: []const u8, answer: []const u8 }{
    .{ .text = "My favourite colour is teal.", .question = "What is my favourite colour?", .word = "colour", .answer = "teal" },
    .{ .text = "My dog's name is Brindle.", .question = "What is my dog's name?", .word = "dog", .answer = "Brindle" },
    .{ .text = "My workshop code is 47.", .question = "What is my workshop code?", .word = "workshop", .answer = "47" },
};
const Stays = struct {
    pub fn check(_: Stays) bool {
        return false;
    }
};
const Model = struct {
    host: *fixture.Host,
    adapter: fixture.Adapter,
    reject: usize = 0,
    requests: std.atomic.Value(usize) = .init(0),
    fn engine(m: *Model) api.Engine {
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
    fn recalled(m: *const Model, fact: usize) bool {
        var x: [64]u16 = @splat(@as(u16, 0x3f80));
        x[fact] = 0x4100;
        var y: [64]f32 = @splat(0);
        for (0..m.host.used) |k| {
            const z = @max(living.pocket_fit.dotBf16(m.host.up[k * 64 ..][0..64], &x), 0);
            for (&y, m.host.down[k * 64 ..][0..64]) |*v, w| v.* += z * z * w;
        }
        var other: f32 = 0.4;
        for (y, 0..) |v, j| if (j != fact) {
            other = @max(other, v);
        };
        return y[fact] > other;
    }
    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        _ = m.requests.fetchAdd(1, .monotonic);
        var prompt: [4096]u8 = undefined;
        if (req.prompt.len > prompt.len) return error.InvalidRequest;
        for (req.prompt, prompt[0..req.prompt.len]) |v, *b| b.* = @intCast(v);
        var answer: []const u8 = "Hello.";
        for (facts, 1..) |f, n| if (std.mem.indexOf(u8, prompt[0..req.prompt.len], f.word) != null) {
            answer = if (!req.untaught and n != m.reject and m.recalled(n)) f.answer else "I do not know.";
            break;
        };
        var ids: [64]u32 = undefined;
        for (answer, ids[0..answer.len]) |b, *v| v.* = b;
        sink.event(sink.ctx, id, &.{ .tokens = ids[0..answer.len] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn learn(ctx: *anyopaque, req: *const api.LearnRequest, sink: api.LearnSink) api.LearnError!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        const hook = m.adapter.hook();
        hook.begin(hook.ctx, req, sink) catch |err| {
            sink.event(sink.ctx, &.{ .done = .{ .message = @errorName(err) } });
            return;
        };
        for (0..10000) |_| if (hook.step(hook.ctx).done) return;
        hook.abort(hook.ctx);
    }
};
fn opened(m: *Model, text: *ByteText, path: []const u8) !*Server {
    return Server.init(a, io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .slide_graph = path }, null);
}
fn check(srv: *Server, al: std.mem.Allocator, n: usize) !void {
    var cx: errors.Cx = .{ .a = al };
    for (facts[0..n]) |f| {
        const r = try ask.ask(srv, &cx, null, f.question, 32, Stays{});
        try t.expectEqualStrings(f.answer, r.content);
    }
    const rs = try recheck.check(srv, &cx, 0, .all, &.{}, Stays{});
    for (rs) |r| try t.expect(r.recalled);
}
fn teach(srv: *Server, al: std.mem.Allocator, n: usize, keep: usize, accepted: bool) !u32 {
    const f = facts[n - 1];
    const id = srv.slide.add(f.text, "chat", @intCast(n));
    srv.slide.mark(id, .learning);
    srv.slide.asked(id, &.{ .{ .question = f.question, .answer = f.answer, .short = f.answer }, .{ .question = f.question, .answer = f.answer, .short = f.answer, .probe = true } });
    const train_ids = [_]u32{ @intCast(n), 2, 3, 4 };
    const held_ids = [_]u32{ @intCast(n), 7, 3, 4 };
    var keep_ids: [3][4]u32 = undefined;
    var keeps: [3]api.Example = undefined;
    for (0..keep) |i| {
        keep_ids[i] = .{ @intCast(i + 1), 2, 3, 4 };
        keeps[i] = .{ .ids = &keep_ids[i], .start = 3, .decide = 2 };
    }
    const plan: lesson.Plan = .{
        .full = .{ .train = &.{.{ .ids = &train_ids, .start = 3, .decide = 2 }}, .held = &.{.{ .ids = &held_ids, .start = 3, .decide = 2 }}, .keep = keeps[0..keep], .steps = 400 },
        .facts = &.{f.text},
        .held = &.{.{ .question = f.question, .answer = f.answer, .short = f.answer, .fact = 0 }},
        .probes = &.{.{ .question = f.question, .answer = f.answer, .short = f.answer, .fact = 0 }},
        .checks = &.{},
    };
    var conn: @import("http_conn.zig").Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = al, .broken = true };
    var out: slide.Out = .{ .conn = &conn, .a = al };
    var cx: errors.Cx = .{ .a = al };
    const got = slide.rounds(srv, al, &cx, plan, Stays{}, id, &out);
    if (got.recalled.len != 1 or got.recalled[0] != accepted) {
        std.debug.print("PROBE lesson {d} accepted={any} why={s}\n", .{ n, got.recalled, got.why orelse "" });
        return error.UnexpectedLessonResult;
    }
    srv.teacher.ended(accepted);
    srv.slide.mark(id, if (accepted) .learned else .missed);
    try t.expectEqual(@as(?[]const u8, null), slide.saveAll(srv, al));
    return id;
}
fn restartCase(retain_graph: bool) !void {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const graph = try std.fs.path.join(a, &.{ dir, "graph.json" });
    defer a.free(graph);
    const empty = try std.fs.path.join(a, &.{ dir, "empty.json" });
    defer a.free(empty);
    try fixture.Store.init(dir);
    var text: ByteText = .{};
    {
        var h: fixture.Host = .{ .dir = dir };
        var m: Model = .{ .host = &h, .adapter = fixture.Adapter.init(a, io, &h) };
        defer m.adapter.deinit();
        const srv = try opened(&m, &text, graph);
        defer srv.deinit();
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        _ = try teach(srv, arena.allocator(), 1, 0, true);
        try check(srv, arena.allocator(), 1);
        try t.expectEqual(@as(usize, 1), h.used);
    }
    {
        var h: fixture.Host = .{ .dir = dir };
        try h.reopen();
        try t.expectEqual(@as(usize, 1), h.used);
        var m: Model = .{ .host = &h, .adapter = fixture.Adapter.init(a, io, &h) };
        defer m.adapter.deinit();
        const srv = try opened(&m, &text, if (retain_graph) graph else empty);
        defer srv.deinit();
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const al = arena.allocator();
        try check(srv, al, 1);
        _ = try teach(srv, al, 2, 1, true);
        try check(srv, al, 2);
        const before = try tmp.dir.readFileAlloc(io, "model.safetensors", al, .limited(1 << 20));
        m.reject = 3;
        _ = try teach(srv, al, 3, 2, false);
        try check(srv, al, 2);
        const after = try tmp.dir.readFileAlloc(io, "model.safetensors", al, .limited(1 << 20));
        try t.expectEqualSlices(u8, before, after);
        std.debug.print("RESTART_CASE graph={any} used={d} submits={d} both_recalled=true refusal_file_equal=true\n", .{ retain_graph, h.used, m.requests.load(.acquire) });
    }
    var last: fixture.Host = .{ .dir = dir };
    try last.reopen();
    var live: Model = .{ .host = &last, .adapter = fixture.Adapter.init(a, io, &last) };
    defer live.adapter.deinit();
    const reopened = try opened(&live, &text, if (retain_graph) graph else empty);
    defer reopened.deinit();
    var final_arena = std.heap.ArenaAllocator.init(a);
    defer final_arena.deinit();
    try check(reopened, final_arena.allocator(), 2);
    std.debug.print("REOPEN_FINAL graph={any} used={d} both_recalled=true\n", .{ retain_graph, last.used });
}
test "restart case: retained graph, new server and learner teach a second fact" {
    try restartCase(true);
}
test "restart case: empty graph, new server and learner teach a second fact" {
    try restartCase(false);
}
