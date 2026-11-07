//! Routed gate-off lessons match the current-main graph, event payloads and engine call counts.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const fixture = @import("slide_learn_path_test.zig");
const Server = @import("server.zig").Server;
const t = std.testing;
const Capture = struct { calls: usize, learns: usize, writes: usize, requests_sha256: []const u8, replies_sha256: []const u8, graph: []const u8, events: []const u8 };

const Counted = struct {
    inner: fixture.Model = .{},
    calls: usize = 0,
    learns: usize = 0,
    lock: std.Io.Mutex = .init,
    requests: [2048][32]u8 = undefined,
    replies: [2048][32]u8 = undefined,
    fn engine(m: *@This()) api.Engine {
        return .{ .ctx = m, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .learn = learn } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const m: *@This() = @ptrCast(@alignCast(ctx));
        m.lock.lockUncancelable(t.io);
        const index = m.calls;
        m.calls += 1;
        m.lock.unlock(t.io);
        std.debug.assert(index < m.requests.len);
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(request.prompt), &m.requests[index], .{});
        var tokens = std.crypto.hash.sha2.Sha256.init(.{});
        defer {
            var token_hash: [32]u8 = undefined;
            tokens.final(&token_hash);
            var paired = std.crypto.hash.sha2.Sha256.init(.{});
            paired.update(&m.requests[index]);
            paired.update(&token_hash);
            paired.final(&m.replies[index]);
        }
        const Forward = struct {
            tokens: *std.crypto.hash.sha2.Sha256,
            sink: api.Sink,
            fn event(raw: *anyopaque, which: api.Id, value: *const api.Event) void {
                const f: *@This() = @ptrCast(@alignCast(raw));
                switch (value.*) {
                    .tokens => |ids| {
                        f.tokens.update(std.mem.sliceAsBytes(ids));
                    },
                    else => {},
                }
                f.sink.event(f.sink.ctx, which, value);
            }
        };
        var forward: Forward = .{ .tokens = &tokens, .sink = sink };
        const e = m.inner.engine();
        return e.vtable.submit(e.ctx, id, request, .{ .ctx = &forward, .event = Forward.event });
    }
    fn learn(ctx: *anyopaque, request: *const api.LearnRequest, sink: api.LearnSink) api.LearnError!void {
        const m: *@This() = @ptrCast(@alignCast(ctx));
        m.learns += 1;
        const e = m.inner.engine();
        return e.vtable.learn.?(e.ctx, request, sink);
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

fn events(a: std.mem.Allocator, wire: []const u8) ![]const u8 {
    try t.expect(std.mem.indexOf(u8, wire, "200 OK") != null);
    var out: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, wire, '\n');
    var kind: []const u8 = "";
    var done = false;
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, "\r");
        if (std.mem.startsWith(u8, line, "event: ")) kind = line[7..];
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const data = line[6..];
        if (std.mem.eql(u8, data, "[DONE]")) {
            done = true;
            try out.appendSlice(a, "[DONE]\n");
            continue;
        }
        const value = (try json.parse(a, data)).ok;
        if (value.field("seconds") != null) try value.object.put(a, "seconds", try json.intValue(a, 0));
        try out.print(a, "{s}\n{s}\n", .{ kind, try json.stringify(a, value, .{ .compact = true }) });
    }
    try t.expect(done);
    return out.toOwnedSlice(a);
}

fn capture(a: std.mem.Allocator, case: usize, extra: []const u8) !Capture {
    var m: Counted = .{};
    defer m.inner.deinit();
    if (case == 1) m.inner.gives_up = "I like blue.";
    if (case == 2) {
        m.inner.mode = .empty;
        m.inner.fault_now = true;
    }
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path, "graph.json" });
    var text: fixture.Text = .{};
    const srv = try Server.init(t.allocator, t.io, m.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
        .seed_salt = 19,
        .slide_graph = path,
    }, null);
    defer srv.deinit();
    const body = try std.fmt.allocPrint(a, "{{\"text\":\"I like blue.\"{s}}}", .{extra});
    const wire = try fixture.learnWire(a, srv, body);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 20));
    const graph = (try json.parse(a, bytes)).ok;
    for (graph.field("facts").?.array) |fact| try fact.object.put(a, "at", try json.intValue(a, 0));
    var request_hash: [32]u8 = undefined;
    var reply_hash: [32]u8 = undefined;
    const Less = struct {
        fn less(_: void, x: [32]u8, y: [32]u8) bool {
            return std.mem.order(u8, &x, &y) == .lt;
        }
    };
    // Side questions run concurrently; compare every request/reply pair independently of submission order.
    std.mem.sort([32]u8, m.requests[0..m.calls], {}, Less.less);
    std.mem.sort([32]u8, m.replies[0..m.calls], {}, Less.less);
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(m.requests[0..m.calls]), &request_hash, .{});
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(m.replies[0..m.calls]), &reply_hash, .{});
    return .{ .requests_sha256 = try std.fmt.allocPrint(a, "{x}", .{request_hash}), .replies_sha256 = try std.fmt.allocPrint(a, "{x}", .{reply_hash}), .calls = m.calls, .learns = m.learns, .writes = srv.slide.test_persist_calls, .graph = try json.stringify(a, graph, .{ .compact = true }), .events = try events(a, wire) };
}

test "Slide metrics: off matches the current-main kept refused and early-refused routes" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const expected = try std.json.parseFromSliceLeaky([]const Capture, a, @embedFile("fixtures/slide_metrics_off.json"), .{});
    try t.expectEqual(@as(usize, 3), expected.len);
    for (expected, 0..) |base, case| for ([_][]const u8{
        "", ",\"recall\":false", ",\"recall\":[{\"fact\":\"I like blue.\",\"question\":\"External recall?\",\"answer\":\"blue\"}]",
    }) |extra| {
        const actual = try capture(a, case, extra);
        try t.expectEqual(base.calls, actual.calls);
        try t.expectEqual(base.learns, actual.learns);
        try t.expectEqual(base.writes, actual.writes);
        try t.expectEqualStrings(base.requests_sha256, actual.requests_sha256);
        try t.expectEqualStrings(base.replies_sha256, actual.replies_sha256);
        try t.expectEqualStrings(base.graph, actual.graph);
        try t.expectEqualStrings(base.events, actual.events);
    };
}

test "Slide metrics: off candidate entry leaves the model and graph untouched" {
    var m: Counted = .{};
    defer m.inner.deinit();
    var text: fixture.Text = .{};
    const srv = try Server.init(t.allocator, t.io, m.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: @import("errors.zig").Cx = .{ .a = a };
    const id = srv.slide.add("I like blue.", "chat", 0);
    var kept = [_]bool{false};
    const Stays = fixture.Stays;
    const plan = (try @import("slide_lesson.zig").lesson(srv, &cx, &srv.teacher, &.{"I like blue."}, &kept, false, &.{}, &.{}, Stays{})).?;
    const before = try srv.slide.render(a, null);
    srv.slide.fault = .write;
    m.calls = 0;
    try t.expectEqual(@as(?[]const u8, null), @import("slide_metrics.zig").candidate(srv, &cx, plan, id, Stays{}));
    try t.expectEqual(@as(usize, 0), m.calls);
    try t.expectEqual(@import("slide_graph.zig").Graph.Fault.write, srv.slide.fault);
    try t.expectEqualStrings(before, try srv.slide.render(a, null));
}
