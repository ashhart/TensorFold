//! A duplicate fact needs one explicit transaction, with every retained fact checked and the whole cut reversible.
const std = @import("std");
const api = @import("engine_api");
const errors = @import("errors.zig");
const slide = @import("slide.zig");
const forget = @import("slide_forget.zig");
const graph = @import("slide_graph.zig");
const Conn = @import("http_conn.zig").Conn;
const Server = @import("server.zig").Server;
const ByteText = @import("slide_forget_test.zig").ByteText;
const t = std.testing;

const Model = struct {
    cut: [3]bool = @splat(false),
    saved: [3]bool = @splat(false),
    before: [3]bool = @splat(false),
    breaks: bool = false,
    cuts: usize = 0,
    backs: usize = 0,
    saves: usize = 0,
    fails: usize = 0,
    graph: ?*graph.Graph = null,
    fault: graph.Graph.Fault = .none,

    fn engine(m: *Model) api.Engine {
        return .{ .ctx = m, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .learn = learn } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, s: *api.Status, _: []u32) void {
        s.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        var prompt: [256]u8 = undefined;
        const n = @min(prompt.len, request.prompt.len);
        for (request.prompt[0..n], prompt[0..n]) |x, *b| b.* = @intCast(x);
        const sales = std.mem.indexOf(u8, prompt[0..n], "sales") != null;
        const kept = !m.cut[2] and !(m.breaks and (m.cut[0] or m.cut[1]));
        const said: []const u8 = if (request.untaught) "I don't know." else if (sales)
            (if (!m.cut[0] or !m.cut[1]) "Mira heads sales at Vale." else "I don't know.")
        else
            (if (kept) "You like blue." else "I don't know.");
        var ids: [64]u32 = undefined;
        for (said, ids[0..said.len]) |b, *x| x.* = b;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = ids[0..said.len] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn learn(ctx: *anyopaque, r: *const api.LearnRequest, sink: api.LearnSink) api.LearnError!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        if (r.forget.len > 0 or r.unforget) {
            var n: u32 = 0;
            if (r.unforget) {
                m.backs += 1;
                m.cut = m.before;
            } else {
                m.cuts += 1;
                m.before = m.cut;
                for (r.forget) |x| {
                    if (x.layer >= 3) continue;
                    n += @intFromBool(!m.cut[x.layer]);
                    m.cut[x.layer] = true;
                }
            }
            sink.event(sink.ctx, &.{ .learned = .{ .recalled = n > 0, .steps = n, .loss = 0 } });
        } else if (r.save) {
            m.saves += 1;
            if (m.graph) |g| g.fault = m.fault;
            if (m.fails > 0) {
                m.fails -= 1;
                m.saved[0] = m.cut[0];
                return sink.event(sink.ctx, &.{ .done = .{ .message = "a shard could not be written" } });
            }
            m.saved = m.cut;
        }
        sink.event(sink.ctx, &.{ .done = .{} });
    }
};

const Fixture = struct {
    srv: *Server,
    ids: [3]u32,

    fn init(text: *ByteText, m: *Model, path: ?[]const u8) !Fixture {
        const srv = try Server.init(t.allocator, t.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .slide_graph = path }, null);
        const statements = [_][]const u8{ "The head of sales at Vale is Mira.", "Mira is the head of sales at Vale.", "I like blue." };
        var f: Fixture = .{ .srv = srv, .ids = undefined };
        for (statements, &f.ids, 0..) |s, *id, i| {
            id.* = srv.slide.add(s, "chat", 1);
            srv.slide.mark(id.*, .learned);
            srv.slide.placed(id.*, &.{.{ .layer = @intCast(i), .neuron = 0, .key = .reply }});
            const q = if (i < 2) "Who heads sales at Vale?" else "What colour do I like?";
            srv.slide.asked(id.*, &.{ .{ .question = q, .answer = s, .short = if (i < 2) "Mira" else "blue" }, .{ .question = q, .answer = s, .short = if (i < 2) "Mira" else "blue", .probe = true } });
        }
        return f;
    }

    fn remove(f: Fixture, ats: []const usize) ![3]graph.State {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .broken = true };
        var out: slide.Out = .{ .conn = &conn, .a = a };
        var cx: errors.Cx = .{ .a = a };
        try forget.deleteMany(f.srv, &cx, &out, try f.srv.slide.snapshot(a), ats, "duplicate fact", Stays{});
        const facts = try f.srv.slide.snapshot(a);
        return .{ facts[0].state, facts[1].state, facts[2].state };
    }
};
const Stays = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

test "multi-fact delete: duplicate orientations fail alone and disappear together while the kept fact still answers" {
    var m: Model = .{};
    var text: ByteText = .{};
    const f = try Fixture.init(&text, &m, null);
    defer f.srv.deinit();
    try t.expectEqual([3]graph.State{ .learned, .learned, .learned }, try f.remove(&.{0}));
    try t.expectEqual([3]graph.State{ .learned, .learned, .learned }, try f.remove(&.{1}));
    try t.expectEqual(@as(usize, 0), m.saves);
    try t.expectEqual([3]graph.State{ .deleted, .deleted, .learned }, try f.remove(&.{ 0, 1 }));
    try t.expectEqual([3]bool{ true, true, false }, m.saved);
    try t.expectEqual(@as(usize, 1), m.saves);
}

test "multi-fact delete: a retained regression restores every selected neuron and writes nothing" {
    var m: Model = .{ .breaks = true };
    var text: ByteText = .{};
    const f = try Fixture.init(&text, &m, null);
    defer f.srv.deinit();
    try t.expectEqual([3]graph.State{ .learned, .learned, .learned }, try f.remove(&.{ 0, 1 }));
    try t.expectEqual([3]bool{ false, false, false }, m.cut);
    try t.expectEqual(@as(usize, 1), m.backs);
    try t.expectEqual(@as(usize, 0), m.saves);
}

fn statesOnDisk(path: []const u8) ![3]graph.State {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 20));
    const facts = (try @import("json").parse(a, bytes)).ok.field("facts").?.items().?;
    var out: [3]graph.State = undefined;
    for (&out, facts) |*s, f| s.* = std.meta.stringToEnum(graph.State, f.strField("state").?).?;
    return out;
}

test "multi-fact delete: a failed shard write restores the whole selection and every graph state" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    const path = try std.fs.path.join(t.allocator, &.{ dir, "graph.json" });
    defer t.allocator.free(path);
    var m: Model = .{ .fails = 1 };
    var text: ByteText = .{};
    const f = try Fixture.init(&text, &m, path);
    defer f.srv.deinit();
    const before = [3]graph.State{ .learned, .learned, .learned };
    try t.expectEqual(before, try f.remove(&.{ 0, 1 }));
    try t.expectEqual(before, try statesOnDisk(path));
    try t.expectEqual([3]bool{ false, false, false }, m.cut);
    try t.expectEqual(m.saved, m.cut);
    try t.expectEqual(@as(usize, 2), m.saves);
    try t.expectEqual(@as(usize, 1), m.backs);
}

test "multi-fact delete: graph write failure never publishes half a selection and retry settles both" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    for ([_]graph.Graph.Fault{ .write, .rename }) |fault| {
        const path = try std.fs.path.join(t.allocator, &.{ dir, @tagName(fault) });
        defer t.allocator.free(path);
        var m: Model = .{};
        var text: ByteText = .{};
        const f = try Fixture.init(&text, &m, path);
        defer f.srv.deinit();
        f.srv.slide.fault = fault;
        const before = [3]graph.State{ .learned, .learned, .learned };
        try t.expectEqual(before, try f.remove(&.{ 0, 1 }));
        try t.expectEqual(before, try statesOnDisk(path));
        try t.expectEqual(@as(usize, 0), m.saves);
        m.graph = &f.srv.slide;
        m.fault = fault;
        const pending = [3]graph.State{ .deleting, .deleting, .learned };
        try t.expectEqual(pending, try f.remove(&.{ 0, 1 }));
        try t.expectEqual(pending, try statesOnDisk(path));
        m.fault = .none;
        const deleted = [3]graph.State{ .deleted, .deleted, .learned };
        try t.expectEqual(deleted, try f.remove(&.{ 0, 1 }));
        try t.expectEqual(deleted, try statesOnDisk(path));
        try t.expectEqual(@as(usize, 1), m.saves);
    }
}

test "multi-fact delete: IDs are explicit distinct positive integers and the entire request is validated" {
    var m: Model = .{};
    var text: ByteText = .{};
    const f = try Fixture.init(&text, &m, null);
    defer f.srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const many = @import("slide_forget_many.zig");
    const json = @import("json");
    const facts = try f.srv.slide.snapshot(a);
    for ([_][]const u8{ "{}", "{\"ids\":[]}", "{\"ids\":[1,1]}", "{\"ids\":[0]}", "{\"ids\":[1,999]}", "{\"ids\":[1,2.0]}", "{\"ids\":[true]}", "{\"ids\":[\"1\"]}" }) |body| {
        const parsed = (try json.parse(a, body)).ok;
        if (many.indices(a, parsed, facts)) |_| return error.AcceptedInvalidSelection else |_| {}
    }
    const ats = try many.indices(a, (try json.parse(a, "{\"ids\":[2,1]}")).ok, facts);
    try t.expectEqualSlices(usize, &.{ 1, 0 }, ats);
    const batch = try many.plan(a, facts, ats);
    try t.expectEqualSlices(u32, &.{ 2, 1 }, batch.ids);
    try t.expectEqual(@as(usize, 3), batch.facts.len);
    try t.expectEqual(@as(usize, 2), batch.neurons.len);
    try t.expectEqual(@as(usize, 0), m.cuts);
}

test "multi-fact delete: dispatcher preserves the JSON body and streams both final IDs" {
    try dispatchDelete(true);
    try dispatchDelete(false);
}

fn dispatchDelete(multiple: bool) !void {
    var m: Model = .{};
    var text: ByteText = .{};
    const f = try Fixture.init(&text, &m, null);
    defer f.srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = if (multiple) "{\"ids\":[1,2],\"reason\":\"same statement\"}" else "{\"reason\":\"one statement\"}";
    const Capture = struct {
        bytes: std.ArrayList(u8) = .empty,
        fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            c.bytes.appendSlice(t.allocator, bytes) catch return error.Closed;
        }
    };
    var capture: Capture = .{};
    defer capture.bytes.deinit(t.allocator);
    var headers = [_]@import("http_conn.zig").Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }};
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = try a.dupe(u8, body), .end = body.len, .gpa = a, .method = "DELETE", .path = if (multiple) "/v1/slide/facts" else "/v1/slide/fact/3", .version = "HTTP/1.1", .headers = &headers, .write_hooks = .{ .ctx = &capture, .write = Capture.write } };
    @import("routes.zig").dispatch(f.srv, &conn, a);
    try t.expectEqual(if (multiple) [3]bool{ true, true, false } else [3]bool{ false, false, true }, m.saved);
    try t.expectEqual(@as(usize, 1), m.cuts);
    const wire = capture.bytes.items;
    try t.expect(std.mem.indexOf(u8, wire, "HTTP/1.1 200") != null);
    try t.expectEqual(@as(usize, if (multiple) 2 else 1), std.mem.count(u8, wire, "event: forgot\n"));
    try t.expect(std.mem.endsWith(u8, wire, "data: [DONE]\n\n"));
    const facts = try f.srv.slide.snapshot(a);
    if (multiple) {
        try t.expectEqualStrings("same statement", facts[0].message);
        try t.expectEqualStrings("same statement", facts[1].message);
    } else try t.expectEqualStrings("one statement", facts[2].message);
}

test "multi-fact delete: graph allocation failures restore every selected record before a retry" {
    var failing = t.FailingAllocator.init(t.allocator, .{});
    var g = graph.Graph.init(failing.allocator(), t.io, null);
    defer g.deinit();
    var ids: [2]u32 = undefined;
    for (&ids, 0..) |*id, i| {
        id.* = g.add("A duplicate statement.", "chat", 1);
        g.mark(id.*, .learned);
        g.placed(id.*, &.{.{ .layer = @intCast(i), .neuron = 0, .key = .reply }});
    }
    const why: [64 << 10]u8 = @splat('x');
    var failures: usize = 0;
    while (true) : (failures += 1) {
        failing.fail_index = failing.alloc_index + failures;
        g.removeMany(&ids, 2, &why) catch |e| {
            failing.fail_index = std.math.maxInt(usize);
            try t.expectEqual(error.OutOfMemory, e);
            for (g.facts.items) |f| {
                try t.expectEqual(graph.State.learned, f.state);
                try t.expectEqual(@as(usize, 1), f.neurons.len);
                try t.expectEqual(@as(i64, 0), f.deleted);
            }
            continue;
        };
        break;
    }
    try t.expect(failures >= 3);
    for (g.facts.items) |f| try t.expectEqual(graph.State.deleted, f.state);
}

test "multi-fact delete: any selected orientation still answering fails, with unchanged retained thresholds" {
    const recheck = @import("slide_recheck.zig");
    const asks = [_]recheck.Ask{ .{ .fact = 0, .question = "One orientation?" }, .{ .fact = 1, .question = "Other orientation?" }, .{ .fact = 2, .question = "Kept fact?" } };
    const before = [_]bool{ true, true, true };
    const none = [_]bool{ false, false, false };
    try t.expectEqual(@as(usize, 1), recheck.judgeMany(&asks, &before, &.{ false, true, true }, &none, 3, 2).?.fact);
    try t.expectEqual(@as(usize, 2), recheck.judgeMany(&asks, &before, &none, &none, 3, 2).?.fact);
    try t.expectEqual(@as(?recheck.Miss, null), recheck.judgeMany(&asks, &before, &.{ false, true, true }, &.{ false, true, false }, 3, 2));
}

test "multi-fact delete: failed write and failed restoration leave every selected graph record pending" {
    var m: Model = .{ .fails = 2 };
    var text: ByteText = .{};
    const f = try Fixture.init(&text, &m, null);
    defer f.srv.deinit();
    try t.expectEqual([3]graph.State{ .deleting, .deleting, .learned }, try f.remove(&.{ 0, 1 }));
    try t.expectEqual([3]bool{ false, false, false }, m.cut);
    try t.expectEqual([3]bool{ false, false, false }, m.saved);
    try t.expectEqual([3]graph.State{ .deleted, .deleted, .learned }, try f.remove(&.{ 0, 1 }));
    try t.expectEqual([3]bool{ true, true, false }, m.saved);
}
